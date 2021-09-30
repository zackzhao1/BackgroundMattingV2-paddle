import paddle
import paddle.vision as vision
from paddle import nn
from paddle.nn import functional as F
from typing import Tuple


class Refiner(nn.Layer):
    """
    Refiner refines the coarse output to full resolution.

    Args:
        mode: area selection mode. Options:
            "full"         - No area selection, refine everywhere using regular Conv2d.
            "sampling"     - Refine fixed amount of pixels ranked by the top most errors.
            "thresholding" - Refine varying amount of pixels that have greater error than the threshold.
        sample_pixels: number of pixels to refine. Only used when mode == "sampling".
        threshold: error threshold ranged from 0 ~ 1. Refine where err > threshold. Only used when mode == "thresholding".
        kernel_size: The convolution kernel_size. Options: [1, 3]
        prevent_oversampling: True for regular cases, False for speedtest.

    Compatibility Args:
        patch_crop_method: the method for cropping patches. Options:
            "unfold"           - Best performance for Pypaddle and paddleScript.
            "roi_align"        - Another way for croping patches.
            "gather"           - Another way for croping patches.
        patch_replace_method: the method for replacing patches. Options:
            "scatter_nd"       - Best performance for Pypaddle and paddleScript.
            "scatter_element"  - Another way for replacing patches.

    Input:
        src: (B, 3, H, W) full resolution source image.
        bgr: (B, 3, H, W) full resolution background image.
        pha: (B, 1, Hc, Wc) coarse alpha prediction.
        fgr: (B, 3, Hc, Wc) coarse foreground residual prediction.
        err: (B, 1, Hc, Hc) coarse error prediction.
        hid: (B, 32, Hc, Hc) coarse hidden encoding.

    Output:
        pha: (B, 1, H, W) full resolution alpha prediction.
        fgr: (B, 3, H, W) full resolution foreground residual prediction.
        ref: (B, 1, H/4, W/4) quarter resolution refinement selection map. 1 indicates refined 4x4 patch locations.
    """

    # For paddleScript export optimization.
    __constants__ = ['kernel_size', 'patch_crop_method', 'patch_replace_method']

    def __init__(self,
                 mode: str,
                 sample_pixels: int,
                 threshold: float,
                 kernel_size: int = 3,
                 prevent_oversampling: bool = True,
                 patch_crop_method: str = 'unfold',
                 patch_replace_method: str = 'scatter_nd'):
        super().__init__()
        assert mode in ['full', 'sampling', 'thresholding']
        assert kernel_size in [1, 3]
        assert patch_crop_method in ['unfold', 'roi_align', 'gather']
        assert patch_replace_method in ['scatter_nd', 'scatter_element']

        self.mode = mode
        self.sample_pixels = sample_pixels
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.prevent_oversampling = prevent_oversampling
        self.patch_crop_method = patch_crop_method
        self.patch_replace_method = patch_replace_method

        channels = [32, 24, 16, 12, 4]
        self.conv1 = nn.Conv2D(channels[0] + 6 + 4, channels[1], kernel_size, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(channels[1])
        self.conv2 = nn.Conv2D(channels[1], channels[2], kernel_size, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(channels[2])
        self.conv3 = nn.Conv2D(channels[2] + 6, channels[3], kernel_size, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(channels[3])
        self.conv4 = nn.Conv2D(channels[3], channels[4], kernel_size, bias_attr=True)
        self.relu = nn.ReLU(True)

    def forward(self,
                src: paddle.Tensor,
                bgr: paddle.Tensor,
                pha: paddle.Tensor,
                fgr: paddle.Tensor,
                err: paddle.Tensor,
                hid: paddle.Tensor):

        # import numpy as np
        # root = '/home/sp/zhaojl/BackgroundMattingV2-master/model/PyTorch/'
        # src = paddle.to_tensor(np.load(root + 'src.npy'))#[0].unsqueeze(0)
        # bgr = paddle.to_tensor(np.load(root + 'bgr.npy'))#[0].unsqueeze(0)
        # pha = paddle.to_tensor(np.load(root + 'pha.npy'))#[0].unsqueeze(0)
        # fgr = paddle.to_tensor(np.load(root + 'fgr.npy'))#[0].unsqueeze(0)
        # err = paddle.to_tensor(np.load(root + 'err.npy'))#[0].unsqueeze(0)
        # hid = paddle.to_tensor(np.load(root + 'hid.npy'))#[0].unsqueeze(0)


        H_full, W_full = src.shape[2:]
        H_half, W_half = H_full // 2, W_full // 2
        H_quat, W_quat = H_full // 4, W_full // 4

        src_bgr = paddle.concat([src, bgr], axis=1)

        if self.mode != 'full':
            err = F.interpolate(err, (H_quat, W_quat), mode='bilinear', align_corners=False)
            ref = self.select_refinement_regions(err)
            idx = paddle.nonzero(ref.squeeze(1))
            idx = idx[:, 0], idx[:, 1], idx[:, 2]

            if idx[0].shape[0] > 0:
                x = paddle.concat([hid, pha, fgr], axis=1)
                x = F.interpolate(x, (H_half, W_half), mode='bilinear', align_corners=False)
                x = self.crop_patch(x, idx, 2, 3 if self.kernel_size == 3 else 0)

                y = F.interpolate(src_bgr, (H_half, W_half), mode='bilinear', align_corners=False)
                y = self.crop_patch(y, idx, 2, 3 if self.kernel_size == 3 else 0)

                x = self.conv1(paddle.concat([x, y], axis=1))
                x = self.bn1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)

                x = F.interpolate(x, (8,8) if self.kernel_size == 3 else (4,4), mode='nearest')
                y = self.crop_patch(src_bgr, idx, 4, 2 if self.kernel_size == 3 else 0)

                x = self.conv3(paddle.concat([x, y], axis=1))
                x = self.bn3(x)
                x = self.relu(x)
                x = self.conv4(x)

                out = paddle.concat([pha, fgr], axis=1)
                out = F.interpolate(out, (H_full, W_full), mode='bilinear', align_corners=False)
                out = self.replace_patch(out, x, idx)
                pha = out[:, :1]
                fgr = out[:, 1:]
            else:
                pha = F.interpolate(pha, (H_full, W_full), mode='bilinear', align_corners=False)
                fgr = F.interpolate(fgr, (H_full, W_full), mode='bilinear', align_corners=False)
        else:
            x = paddle.concat([hid, pha, fgr], axis=1)
            x = F.interpolate(x, (H_half, W_half), mode='bilinear', align_corners=False)
            y = F.interpolate(src_bgr, (H_half, W_half), mode='bilinear', align_corners=False)
            if self.kernel_size == 3:
                x = F.pad(x, [3, 3, 3, 3])
                y = F.pad(y, [3, 3, 3, 3])

            x = self.conv1(paddle.concat([x, y], axis=1))
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

            if self.kernel_size == 3:
                x = F.interpolate(x, (H_full + 4, W_full + 4))
                y = F.pad(src_bgr, [2, 2, 2, 2])
            else:
                x = F.interpolate(x, (H_full, W_full), mode='nearest')
                y = src_bgr

            x = self.conv3(paddle.concat([x, y], axis=1))
            x = self.bn3(x)
            x = self.relu(x)
            x = self.conv4(x)

            pha = x[:, :1]
            fgr = x[:, 1:]
            ref = paddle.ones((src.shape[0], 1, H_quat, W_quat), dtype=src.dtype)

        return pha, fgr, ref

    def select_refinement_regions(self, err: paddle.Tensor):
        """
        Select refinement regions.
        Input:
            err: error map (B, 1, H, W)
        Output:
            ref: refinement regions (B, 1, H, W). FloatTensor. 1 is selected, 0 is not.
        """
        if self.mode == 'sampling':
            # Sampling mode.
            b, _, h, w = err.shape
            err = err.reshape((b, -1))
            idx = err.topk(self.sample_pixels // 16, axis=1, sorted=False)[1]
            ref = paddle.zeros_like(err)
            # ref.scatter_(1, idx, 1.)
            for i in range(b):
                ref[i] = ref[i].scatter(idx[i], paddle.ones_like(idx[i], dtype=ref[i].dtype))
            if self.prevent_oversampling:
                # ref.mul_(err.gt(0).float())
                ref.multiply(err.greater_than(paddle.zeros_like(err)).astype(paddle.float32))
            ref = ref.reshape((b, 1, h, w))
        else:
            # Thresholding mode.
            ref = err.gt(self.threshold).float()
        return ref

    def crop_patch(self,
                   x: paddle.Tensor,
                   idx: Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor],
                   size: int,
                   padding: int):
        """
        Crops selected patches from image given indices.

        Inputs:
            x: image (B, C, H, W).
            idx: selection indices Tuple[(P,), (P,), (P,),], where the 3 values are (B, H, W) index.
            size: center size of the patch, also stride of the crop.
            padding: expansion size of the patch.
        Output:
            patch: (P, C, h, w), where h = w = size + 2 * padding.
        """
        if padding != 0:
            x = F.pad(x, [padding,] * 4)

        if self.patch_crop_method == 'unfold':
            # Use unfold. Best performance for Pypaddle and paddleScript.
            return x.transpose([0, 2, 3, 1]) \
                .unfold(1, size + 2 * padding, size) \
                .unfold(2, size + 2 * padding, size)[idx[0], idx[1], idx[2]]
        elif self.patch_crop_method == 'roi_align':
            # Use roi_align. Best compatibility for ONNX.
            idx = idx[0].astype(x.dtype), idx[1].astype(x.dtype), idx[2].astype(x.dtype)
            b = idx[0]
            x1 = idx[2] * size - 0.5
            y1 = idx[1] * size - 0.5
            x2 = idx[2] * size + size + 2 * padding - 0.5
            y2 = idx[1] * size + size + 2 * padding - 0.5
            boxes = paddle.stack([x1, y1, x2, y2], axis=1)

            import numpy as np
            boxes_num = paddle.to_tensor(np.array([5000], dtype=np.int32))
            return roi_align(x, boxes, boxes_num=boxes_num, output_size=size + 2 * padding, sampling_ratio=1)

            # return paddle.fluid.layers.roi_align(x, boxes, rois_num=size + 2 * padding, sampling_ratio=1)
        else:
            # Use gather. Crops out patches pixel by pixel.
            idx_pix = self.compute_pixel_indices(x, idx, size, padding)
            pat = paddle.gather(x.reshape((-1)), 0, idx_pix.reshape((-1)))
            pat = pat.reshape((-1, x.shape[1], size + 2 * padding, size + 2 * padding))
            return pat

    def replace_patch(self,
                      x: paddle.Tensor,
                      y: paddle.Tensor,
                      idx: Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]):
        """
        Replaces patches back into image given index.

        Inputs:
            x: image (B, C, H, W)
            y: patches (P, C, h, w)
            idx: selection indices Tuple[(P,), (P,), (P,)] where the 3 values are (B, H, W) index.

        Output:
            image: (B, C, H, W), where patches at idx locations are replaced with y.
        """
        xB, xC, xH, xW = x.shape
        yB, yC, yH, yW = y.shape
        if self.patch_replace_method == 'scatter_nd':
            # Use scatter_nd. Best performance for Pypaddle and paddleScript. Replacing patch by patch.
            x = x.reshape((xB, xC, xH // yH, yH, xW // yW, yW)).transpose((0, 2, 4, 1, 3, 5))
            x[idx[0], idx[1], idx[2]] = y
            x = x.transpose((0, 3, 1, 4, 2, 5)).reshape((xB, xC, xH, xW))
            return x
        else:
            # Use scatter_element. Best compatibility for ONNX. Replacing pixel by pixel.
            idx_pix = self.compute_pixel_indices(x, idx, size=4, padding=0)
            return x.reshape((-1)).scatter_(0, idx_pix.reshape((-1)), y.reshape((-1))).reshape((x.shape))

    def compute_pixel_indices(self,
                              x: paddle.Tensor,
                              idx: Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor],
                              size: int,
                              padding: int):
        """
        Compute selected pixel indices in the tensor.
        Used for crop_method == 'gather' and replace_method == 'scatter_element', which crop and replace pixel by pixel.
        Input:
            x: image: (B, C, H, W)
            idx: selection indices Tuple[(P,), (P,), (P,),], where the 3 values are (B, H, W) index.
            size: center size of the patch, also stride of the crop.
            padding: expansion size of the patch.
        Output:
            idx: (P, C, O, O) long tensor where O is the output size: size + 2 * padding, P is number of patches.
                 the element are indices pointing to the input x.view(-1).
        """
        B, C, H, W = x.shape
        S, P = size, padding
        O = S + 2 * P
        b, y, x = idx
        n = b.shape[0]
        c = paddle.arange(C)
        o = paddle.arange(O)
        idx_pat = (c * H * W).reshape((C, 1, 1)).expand([C, O, O]) +\
                  (o * W).reshape((1, O, 1)).expand([C, O, O]) + o.reshape((1, 1, O)).expand([C, O, O])
        idx_loc = b * W * H + y * W * S + x * S
        idx_pix = idx_loc.reshape((-1, 1, 1, 1)).expand([n, C, O, O]) + idx_pat.reshape((1, C, O, O)).expand([n, C, O, O])
        return idx_pix


from paddle.nn import Layer
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype
from paddle.fluid import core, layers
from paddle.fluid.layer_helper import LayerHelper
from paddle.common_ops_import import *


def roi_align(x,
              boxes,
              boxes_num,
              output_size,
              spatial_scale=1.0,
              sampling_ratio=-1,
              aligned=False,
              name=None):
    """
    This operator implements the roi_align layer.
    Region of Interest (RoI) Align operator (also known as RoI Align) is to
    perform bilinear interpolation on inputs of nonuniform sizes to obtain
    fixed-size feature maps (e.g. 7*7), as described in Mask R-CNN.
    Dividing each region proposal into equal-sized sections with the pooled_width
    and pooled_height. Location remains the origin result.
    In each ROI bin, the value of the four regularly sampled locations are
    computed directly through bilinear interpolation. The output is the mean of
    four locations. Thus avoid the misaligned problem.
    Args:
        x (Tensor): Input feature, 4D-Tensor with the shape of [N,C,H,W],
            where N is the batch size, C is the input channel, H is Height,
            W is weight. The data type is float32 or float64.
        boxes (Tensor): Boxes (RoIs, Regions of Interest) to pool over. It
            should be a 2-D Tensor of shape (num_boxes, 4). The data type is
            float32 or float64. Given as [[x1, y1, x2, y2], ...], (x1, y1) is
            the top left coordinates, and (x2, y2) is the bottom right coordinates.
        boxes_num (Tensor): The number of boxes contained in each picture in
            the batch, the data type is int32.
        output_size (int or Tuple[int, int]): The pooled output size(h, w), data
            type is int32. If int, h and w are both equal to output_size.
        spatial_scale (float32): Multiplicative spatial scale factor to translate
            ROI coords from their input scale to the scale used when pooling.
            Default: 1.0
        sampling_ratio (int32): number of sampling points in the interpolation
            grid used to compute the output value of each pooled output bin.
            If > 0, then exactly ``sampling_ratio x sampling_ratio`` sampling
            points per bin are used.
            If <= 0, then an adaptive number of grid points are used (computed
            as ``ceil(roi_width / output_width)``, and likewise for height).
            Default: -1
        aligned (bool): If False, use the legacy implementation. If True, pixel
            shift the box coordinates it by -0.5 for a better alignment with the
            two neighboring pixel indices. This version is used in Detectron2.
            Default: True
        name(str, optional): For detailed information, please refer to :
            ref:`api_guide_Name`. Usually name is no need to set and None by
            default.
    Returns:
        Tensor: The output of ROIAlignOp is a 4-D tensor with shape (num_boxes,
            channels, pooled_h, pooled_w). The data type is float32 or float64.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.ops import roi_align
            data = paddle.rand([1, 256, 32, 32])
            boxes = paddle.rand([3, 4])
            boxes[:, 2] += boxes[:, 0] + 3
            boxes[:, 3] += boxes[:, 1] + 4
            boxes_num = paddle.to_tensor([3]).astype('int32')
            align_out = roi_align(data, boxes, boxes_num, output_size=3)
            assert align_out.shape == [3, 256, 3, 3]
    """

    check_type(output_size, 'output_size', (int, tuple), 'roi_align')
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    pooled_height, pooled_width = output_size
    if in_dygraph_mode():
        assert boxes_num is not None, "boxes_num should not be None in dygraph mode."
        align_out = core.ops.roi_align(
            x, boxes, boxes_num, "pooled_height", pooled_height, "pooled_width",
            pooled_width, "spatial_scale", spatial_scale, "sampling_ratio",
            sampling_ratio, "aligned", aligned)
        return align_out

    else:
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'roi_align')
        check_variable_and_dtype(boxes, 'boxes', ['float32', 'float64'],
                                 'roi_align')
        helper = LayerHelper('roi_align', **locals())
        dtype = helper.input_dtype()
        align_out = helper.create_variable_for_type_inference(dtype)
        inputs = {
            "X": x,
            "ROIs": boxes,
        }
        if boxes_num is not None:
            inputs['RoisNum'] = boxes_num
        helper.append_op(
            type="roi_align",
            inputs=inputs,
            outputs={"Out": align_out},
            attrs={
                "pooled_height": pooled_height,
                "pooled_width": pooled_width,
                "spatial_scale": spatial_scale,
                "sampling_ratio": sampling_ratio,
                "aligned": aligned,
            })
        return align_out
