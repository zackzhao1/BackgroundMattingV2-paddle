#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import random
import typing

import paddle
import numpy as np
import math
import paddle.vision.transforms as T
import paddle.vision.transforms.functional as TF
from paddle.nn import functional as F
from PIL import Image, ImageFilter

import numbers
from collections.abc import Sequence
from typing import Tuple, List, Optional
from paddle import Tensor


"""
Pair transforms are MODs of regular transforms so that it takes in multiple images
and apply exact transforms on all images. This is especially useful when we want the
transforms on a pair of images.

Example:
    img1, img2, ..., imgN = transforms(img1, img2, ..., imgN)
"""

class PairCompose(T.Compose):
    def __call__(self, *x):
        for transform in self.transforms:
            x = transform(*x)
        return x
    

class PairApply:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, *x):
        return [self.transforms(xi) for xi in x]


class PairApplyOnlyAtIndices:
    def __init__(self, indices, transforms):
        self.indices = indices
        self.transforms = transforms
    
    def __call__(self, *x):
        return [self.transforms(xi) if i in self.indices else xi for i, xi in enumerate(x)]


class PairRandomHorizontalFlip(T.RandomHorizontalFlip):
    def __call__(self, *x):
        if paddle.rand([1]) < self.prob:
            x = [TF.hflip(xi) for xi in x]
        return x


class RandomBoxBlur:
    def __init__(self, prob, max_radius):
        self.prob = prob
        self.max_radius = max_radius
    
    def __call__(self, img):
        if paddle.rand([1]) < self.prob:
            fil = ImageFilter.BoxBlur(random.choice(range(self.max_radius + 1)))
            img = img.filter(fil)
        return img


class PairRandomBoxBlur(RandomBoxBlur):
    def __call__(self, *x):
        if paddle.rand([1]) < self.prob:
            fil = ImageFilter.BoxBlur(random.choice(range(self.max_radius + 1)))
            x = [xi.filter(fil) for xi in x]
        return x


class RandomSharpen:
    def __init__(self, prob):
        self.prob = prob
        self.filter = ImageFilter.SHARPEN
    
    def __call__(self, img):
        if paddle.rand([1]) < self.prob:
            img = img.filter(self.filter)
        return img
    
    
class PairRandomSharpen(RandomSharpen):
    def __call__(self, *x):
        if paddle.rand([1]) < self.prob:
            x = [xi.filter(self.filter) for xi in x]
        return x


#----------------------------------------------------------------------

def _get_image_size(img):
    if isinstance(img, paddle.Tensor):
        return [img.shape[-1], img.shape[-2]]
    else:
        return img.size


def _get_inverse_affine_matrix(
        center: List[float], angle: float, translate: List[float], scale: float, shear: List[float]
) -> List[float]:
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    #
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    matrix = [d, -b, 0.0, -c, a, 0.0]
    matrix = [x / scale for x in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += cx
    matrix[5] += cy

    return matrix


def _assert_grid_transform_inputs(
        img: Tensor,
        matrix: Optional[List[float]],
        resample: int,
        fillcolor: Optional[int],
        _interpolation_modes: typing.Dict[int, str],
        coeffs: Optional[List[float]] = None,
):
    if not isinstance(img, paddle.Tensor):
        raise TypeError("Input img should be Tensor Image")

    if matrix is not None and not isinstance(matrix, list):
        raise TypeError("Argument matrix should be a list")

    if matrix is not None and len(matrix) != 6:
        raise ValueError("Argument matrix should have 6 float values")

    if coeffs is not None and len(coeffs) != 8:
        raise ValueError("Argument coeffs should have 8 float values")

    if fillcolor is not None:
        import warnings
        warnings.warn("Argument fill/fillcolor is not supported for Tensor input. Fill value is zero")

    if resample not in _interpolation_modes:
        raise ValueError("Resampling mode '{}' is unsupported with Tensor input".format(resample))


def _gen_affine_grid(
        theta: Tensor, w: int, h: int, ow: int, oh: int,
) -> Tensor:
    # https://github.com/pytorch/pytorch/blob/74b65c32be68b15dc7c9e8bb62459efbfbde33d8/aten/src/ATen/native/
    # AffineGridGenerator.cpp#L18
    # Difference with AffineGridGenerator is that:
    # 1) we normalize grid values after applying theta
    # 2) we can normalize by other image size, such that it covers "extend" option like in PIL.Image.rotate

    d = 0.5
    base_grid = paddle.empty(1, oh, ow, 3, dtype=theta.dtype, device=theta.device)
    x_grid = paddle.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow, device=theta.device)
    base_grid[..., 0].copy_(x_grid)
    y_grid = paddle.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh, device=theta.device).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta = theta.transpose(1, 2) / paddle.to_tensor([0.5 * w, 0.5 * h], dtype=theta.dtype, device=theta.device)
    output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
    return output_grid.view(1, oh, ow, 2)


def _cast_squeeze_in(img: Tensor, req_dtype: paddle.dtype) -> Tuple[Tensor, bool, bool, paddle.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype != req_dtype:
        need_cast = True
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: paddle.dtype):
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        # it is better to round before cast
        img = paddle.round(img).to(out_dtype)

    return img


def _apply_grid_transform(img: Tensor, grid: Tensor, mode: str) -> Tensor:

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, grid.dtype)

    if img.shape[0] > 1:
        # Apply same grid to a batch of images
        grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])
    from paddle.nn.functional import grid_sample
    img = grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


def ft_affine(
        img: Tensor, matrix: List[float], resample: int = 0, fillcolor: Optional[int] = None
) -> Tensor:
    """PRIVATE METHOD. Apply affine transformation on the Tensor image keeping image center invariant.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (Tensor): image to be rotated.
        matrix (list of floats): list of 6 float values representing inverse matrix for affine transformation.
        resample (int, optional): An optional resampling filter. Default is nearest (=0). Other supported values:
            bilinear(=2).
        fillcolor (int, optional): this option is not supported for Tensor input. Fill value for the area outside the
            transform in the output image is always 0.

    Returns:
        Tensor: Transformed image.
    """
    _interpolation_modes = {
        0: "nearest",
        2: "bilinear",
    }

    _assert_grid_transform_inputs(img, matrix, resample, fillcolor, _interpolation_modes)

    dtype = img.dtype if paddle.is_floating_point(img) else paddle.float32
    theta = paddle.to_tensor(matrix, dtype=dtype, device=img.device).reshape(1, 2, 3)
    shape = img.shape
    # grid will be generated on the same device as theta and img
    grid = _gen_affine_grid(theta, w=shape[-1], h=shape[-2], ow=shape[-1], oh=shape[-2])
    mode = _interpolation_modes[resample]
    return _apply_grid_transform(img, grid, mode)


def _parse_fill(fill, img, min_pil_version, name="fillcolor"):
    """PRIVATE METHOD. Helper function to get the fill color for rotate, perspective transforms, and pad.

    .. warning::

        Module ``transforms.functional_pil`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        fill (n-tuple or int or float): Pixel fill value for area outside the transformed
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands.
        img (PIL Image): Image to be filled.
        min_pil_version (str): The minimum PILLOW version for when the ``fillcolor`` option
            was first introduced in the calling function. (e.g. rotate->5.2.0, perspective->5.0.0)
        name (str): Name of the ``fillcolor`` option in the output. Defaults to ``"fillcolor"``.

    Returns:
        dict: kwarg for ``fillcolor``
    """
    from PIL import __version__ as PILLOW_VERSION
    major_found, minor_found = (int(v) for v in PILLOW_VERSION.split('.')[:2])
    major_required, minor_required = (int(v) for v in min_pil_version.split('.')[:2])
    if major_found < major_required or (major_found == major_required and minor_found < minor_required):
        if fill is None:
            return {}
        else:
            msg = ("The option to fill background area of the transformed image, "
                   "requires pillow>={}")
            raise RuntimeError(msg.format(min_pil_version))

    num_bands = len(img.getbands())
    if fill is None:
        fill = 0
    if isinstance(fill, (int, float)) and num_bands > 1:
        fill = tuple([fill] * num_bands)
    if not isinstance(fill, (int, float)) and len(fill) != num_bands:
        msg = ("The number of elements in 'fill' does not match the number of "
               "bands of the image ({} != {})")
        raise ValueError(msg.format(len(fill), num_bands))

    return {name: fill}


def pil_affine(img, matrix, resample=0, fillcolor=None):
    """PRIVATE METHOD. Apply affine transformation on the PIL Image keeping image center invariant.

    .. warning::

        Module ``transforms.functional_pil`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (PIL Image): image to be rotated.
        matrix (list of floats): list of 6 float values representing inverse matrix for affine transformation.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter.
            See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)

    Returns:
        PIL Image: Transformed image.
    """
    if not isinstance(img, Image.Image):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    output_size = img.size
    opts = _parse_fill(fillcolor, img, '5.0.0')
    return img.transform(output_size, Image.AFFINE, matrix, resample, **opts)


def affine(img: Tensor, angle: float, translate: List[int], scale: float, shear: List[float],
            resample: int = 0, fillcolor: Optional[int] = None) -> Tensor:

    if not isinstance(angle, (int, float)):
        raise TypeError("Argument angle should be int or float")

    if not isinstance(translate, (list, tuple)):
        raise TypeError("Argument translate should be a sequence")

    if len(translate) != 2:
        raise ValueError("Argument translate should be a sequence of length 2")

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    if not isinstance(shear, (numbers.Number, (list, tuple))):
        raise TypeError("Shear should be either a single value or a sequence of two values")

    if isinstance(angle, int):
        angle = float(angle)

    if isinstance(translate, tuple):
        translate = list(translate)

    if isinstance(shear, numbers.Number):
        shear = [shear, 0.0]

    if isinstance(shear, tuple):
        shear = list(shear)

    if len(shear) == 1:
        shear = [shear[0], shear[0]]

    if len(shear) != 2:
        raise ValueError("Shear should be a sequence containing two values. Got {}".format(shear))

    img_size = _get_image_size(img)
    if not isinstance(img, paddle.Tensor):
        # center = (img_size[0] * 0.5 + 0.5, img_size[1] * 0.5 + 0.5)
        # it is visually better to estimate the center without 0.5 offset
        # otherwise image rotated by 90 degrees is shifted vs output image of torch.rot90 or F_t.affine
        center = [img_size[0] * 0.5, img_size[1] * 0.5]
        matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
        return pil_affine(img, matrix=matrix, resample=resample, fillcolor=fillcolor)

    translate_f = [1.0 * t for t in translate]
    matrix = _get_inverse_affine_matrix([0.0, 0.0], angle, translate_f, scale, shear)
    return ft_affine(img, matrix=matrix, resample=resample, fillcolor=fillcolor)


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))


def _setup_angle(x, name, req_sizes=(2, )):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


class RandomAffine(paddle.nn.Layer):

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=0, fillcolor=0):
        super().__init__()
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))

        if translate is not None:
            _check_sequence_input(translate, "translate", req_sizes=(2,))
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            _check_sequence_input(scale, "scale", req_sizes=(2,))
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            self.shear = _setup_angle(shear, name="shear", req_sizes=(2, 4))
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(
            degrees: List[float],
            translate: Optional[List[float]],
            scale_ranges: Optional[List[float]],
            shears: Optional[List[float]],
            img_size: List[int]
    ) -> Tuple[float, Tuple[int, int], float, Tuple[float, float]]:
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = float(paddle.uniform([1], min=float(degrees[0]), max=float(degrees[1])).item())
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(round(paddle.uniform([1], min=-max_dx, max=max_dx).item()))
            ty = int(round(paddle.uniform([1], min=-max_dy, max=max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(paddle.uniform([1], min=float(scale_ranges[0]), max=float(scale_ranges[1])).item())
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(paddle.uniform([1], min=float(shears[0]), max=float(shears[1])).item())
            if len(shears) == 4:
                shear_y = float(paddle.uniform([1], min=float(shears[2]), max=float(shears[3])).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear

    def forward(self, img):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        img_size = _get_image_size(img)
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
        return affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)


class PairRandomAffineAndResize:
    def __init__(self, size, degrees, translate, scale, shear, ratio=(3./4., 4./3.), resample=Image.BILINEAR, fillcolor=0):
        self.size = size
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.ratio = ratio
        self.resample = resample
        self.fillcolor = fillcolor

    def __call__(self, *x):
        if not len(x):
            return []

        w, h = x[0].size
        scale_factor = max(self.size[1] / w, self.size[0] / h)

        w_padded = max(w, self.size[1])
        h_padded = max(h, self.size[0])

        pad_h = int(math.ceil((h_padded - h) / 2))
        pad_w = int(math.ceil((w_padded - w) / 2))

        scale = self.scale[0] * scale_factor, self.scale[1] * scale_factor
        translate = self.translate[0] * scale_factor, self.translate[1] * scale_factor
        affine_params = RandomAffine.get_params(self.degrees, translate, scale, self.shear, (w, h))

        def transform(img):
            if pad_h > 0 or pad_w > 0:
                img = TF.pad(img, (pad_w, pad_h))

            img = affine(img, *affine_params, self.resample, self.fillcolor)
            img = TF.center_crop(img, self.size)
            return img

        return [transform(xi) for xi in x]


class RandomAffineAndResize(PairRandomAffineAndResize):
    def __call__(self, img):
        return super().__call__(img)[0]


class PairRandomAffine(RandomAffine):
    def __init__(self, degrees, translate=None, scale=None, shear=None, resamples=None, fillcolor=0):
        super().__init__(degrees, translate, scale, shear, Image.NEAREST, fillcolor)
        self.resamples = resamples

    def __call__(self, *x):
        if not len(x):
            return []
        param = self.get_params(self.degrees, self.translate, self.scale, self.shear, x[0].size)
        resamples = self.resamples or [self.resample] * len(x)
        return [affine(xi, *param, resamples[i], self.fillcolor) for i, xi in enumerate(x)]


#----------------------------------------------------------------------


def random_crop(*imgs):
    H_src, W_src = imgs[0].shape[2:]
    W_tgt = random.choice(range(1024, 2048)) // 4 * 4
    H_tgt = random.choice(range(1024, 2048)) // 4 * 4
    scale = max(W_tgt / W_src, H_tgt / H_src)
    results = []
    for img in imgs:
        img = tensor_resize(img, (int(H_src * scale), int(W_src * scale)))
        img = tensor_center_crop(img, (H_tgt, W_tgt))
        results.append(img)
    return results


def random_crop_base(*imgs):
    w = random.choice(range(256, 512))
    h = random.choice(range(256, 512))
    results = []
    for img in imgs:
        img = tensor_resize(img, (max(h, w), max(h, w)))
        img = tensor_center_crop(img, (h, w))
        results.append(img)
    return results

def tensor_resize(input, size):
    if not isinstance(input, paddle.Tensor) or len(input.shape) < 2:
        raise TypeError(f"Input Error")

    input_size = input.shape[-2:]
    if size == input_size:
        return input
    else:
        return paddle.nn.functional.interpolate(input, size=size, mode='bilinear')


def tensor_center_crop(tensor: paddle.Tensor, size: Tuple[int, int],
                interpolation: str = 'bilinear',
                align_corners: bool = True) -> paddle.Tensor:
    r"""Crop the 2D images (4D tensor) at the center.

    Args:
        tensor (paddle.Tensor): the 2D image tensor with shape (B, C, H, W).
        size (Tuple[int, int]): a tuple with the expected height and width
          of the output patch.
        interpolation (str): Interpolation flag. Default: 'bilinear'.
        align_corners (bool): mode for grid_generation. Default: False. See
          https://pypaddle.org/docs/stable/nn.functional.html#paddle.nn.functional.interpolate for details
    Returns:
        paddle.Tensor: the output tensor with patches.

    Examples:
        >>> input = paddle.tensor([[
                [1., 2., 3., 4.],
                [5., 6., 7., 8.],
                [9., 10., 11., 12.],
                [13., 14., 15., 16.],
             ]])
        >>> kornia.center_crop(input, (2, 4))
        tensor([[[ 5.0000,  6.0000,  7.0000,  8.0000],
                 [ 9.0000, 10.0000, 11.0000, 12.0000]]])
    """
    if not isinstance(tensor, paddle.Tensor):
        raise TypeError("Input tensor type is not a paddle.Tensor. Got {}"
                        .format(type(tensor)))
    if not isinstance(size, (tuple, list,)) and len(size) == 2:
        raise ValueError("Input size must be a tuple/list of length 2. Got {}"
                         .format(size))
    assert len(tensor.shape) == 4, f"Only tensor with shape (B, C, H, W) supported. Got {tensor.shape}."

    # unpack input sizes
    dst_h, dst_w = size
    src_h, src_w = tensor.shape[-2:]

    # compute start/end offsets
    dst_h_half = dst_h / 2
    dst_w_half = dst_w / 2
    src_h_half = src_h / 2
    src_w_half = src_w / 2

    start_x = src_w_half - dst_w_half
    start_y = src_h_half - dst_h_half

    end_x = start_x + dst_w - 1
    end_y = start_y + dst_h - 1
    # [y, x] origin
    # top-left, top-right, bottom-right, bottom-left
    points_src: paddle.Tensor = paddle.to_tensor([[
        [start_x, start_y],
        [end_x, start_y],
        [end_x, end_y],
        [start_x, end_y],
    ]])

    # [y, x] destination
    # top-left, top-right, bottom-right, bottom-left
    points_dst: paddle.Tensor = paddle.to_tensor([[
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1],
    ]])  # .expand(points_src.shape[0], -1, -1)
    points_dst = paddle.expand(points_dst, shape=(points_src.shape[0], -1, -1))
    return crop_by_boxes(tensor,
                         points_src,
                         points_dst,
                         interpolation,
                         align_corners)


def crop_by_boxes(tensor: paddle.Tensor, src_box: paddle.Tensor, dst_box: paddle.Tensor,
                  interpolation: str = 'bilinear', align_corners: bool = False) -> paddle.Tensor:
    """Perform crop transform on 2D images (4D tensor) by bounding boxes.

    Given an input tensor, this function selected the interested areas by the provided bounding boxes (src_box).
    Then the selected areas would be fitted into the targeted bounding boxes (dst_box) by a perspective transformation.
    So far, the ragged tensor is not supported by PyTorch right now. This function hereby requires the bounding boxes
    in a batch must be rectangles with same width and height.

    Args:
        tensor (paddle.Tensor): the 2D image tensor with shape (B, C, H, W).
        src_box (paddle.Tensor): a tensor with shape (B, 4, 2) containing the coordinates of the bounding boxes
            to be extracted. The tensor must have the shape of Bx4x2, where each box is defined in the clockwise
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in x, y order.
        dst_box (paddle.Tensor): a tensor with shape (B, 4, 2) containing the coordinates of the bounding boxes
            to be placed. The tensor must have the shape of Bx4x2, where each box is defined in the clockwise
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in x, y order.
        interpolation (str): Interpolation flag. Default: 'bilinear'.
        align_corners (bool): mode for grid_generation. Default: False. See
          https://pypaddle.org/docs/stable/nn.functional.html#paddle.nn.functional.interpolate for details

    Returns:
        paddle.Tensor: the output tensor with patches.

    Examples:
        >>> input = paddle.arange(16, dtype=paddle.float32).reshape((1, 4, 4))
        >>> src_box = paddle.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ]])  # 1x4x2
        >>> dst_box = paddle.tensor([[
        ...     [0., 0.],
        ...     [1., 0.],
        ...     [1., 1.],
        ...     [0., 1.],
        ... ]])  # 1x4x2
        >>> crop_by_boxes(input, src_box, dst_box, align_corners=True)
        tensor([[[ 5.0000,  6.0000],
                 [ 9.0000, 10.0000]]])

    Note:
        If the src_box is smaller than dst_box, the following error will be thrown.
        RuntimeError: solve_cpu: For batch 0: U(2,2) is zero, singular U.
    """
    # validate_bboxes(src_box)
    # validate_bboxes(dst_box)

    assert len(tensor.shape) == 4, f"Only tensor with shape (B, C, H, W) supported. Got {tensor.shape}."

    # compute transformation between points and warp
    # Note: Tensor.dtype must be float. "solve_cpu" not implemented for 'Long'
    dst_trans_src: paddle.Tensor = get_perspective_transform(src_box, dst_box)
    # simulate broadcasting
    dst_trans_src = paddle.expand(dst_trans_src, shape=(tensor.shape[0], -1, -1))
    # dst_trans_src = dst_trans_src.expand(tensor.shape[0], -1, -1).type_as(tensor)

    bbox = infer_box_shape(dst_box)
    assert (bbox[0] == bbox[0][0]).all() and (bbox[1] == bbox[1][0]).all(), (
        f"Cropping height, width and depth must be exact same in a batch. Got height {bbox[0]} and width {bbox[1]}.")
    patches: paddle.Tensor = warp_affine(
        tensor, dst_trans_src[:, :2, :], (int(bbox[0][0].item()), int(bbox[1][0].item())),
        flags=interpolation, align_corners=align_corners)

    return patches


def warp_affine(src: paddle.Tensor, M: paddle.Tensor,
                dsize: Tuple[int, int], flags: str = 'bilinear',
                padding_mode: str = 'zeros',
                align_corners: bool = False) -> paddle.Tensor:
    r"""Applies an affine transformation to a tensor.

    The function warp_affine transforms the source tensor using
    the specified matrix:

    .. math::
        \text{dst}(x, y) = \text{src} \left( M_{11} x + M_{12} y + M_{13} ,
        M_{21} x + M_{22} y + M_{23} \right )

    Args:
        src (paddle.Tensor): input tensor of shape :math:`(B, C, H, W)`.
        M (paddle.Tensor): affine transformation of shape :math:`(B, 2, 3)`.
        dsize (Tuple[int, int]): size of the output image (height, width).
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        align_corners (bool): mode for grid_generation. Default: False.

    Returns:
        paddle.Tensor: the warped tensor with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia.readthedocs.io/en/latest/
       tutorials/warp_affine.html>`__.
    """
    if not isinstance(src, paddle.Tensor):
        raise TypeError("Input src type is not a paddle.Tensor. Got {}"
                        .format(type(src)))

    if not isinstance(M, paddle.Tensor):
        raise TypeError("Input M type is not a paddle.Tensor. Got {}"
                        .format(type(M)))

    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}"
                         .format(src.shape))

    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError("Input M must be a Bx2x3 tensor. Got {}"
                         .format(M.shape))
    B, C, H, W = src.shape
    dsize_src = (H, W)
    out_size = dsize
    # we generate a 3x3 transformation matrix from 2x3 affine
    M_3x3: paddle.Tensor = convert_affinematrix_to_homography(M)
    dst_norm_trans_src_norm: paddle.Tensor = normalize_homography(
        M_3x3, dsize_src, out_size)
    src_norm_trans_dst_norm = paddle.inverse(dst_norm_trans_src_norm)
    grid = F.affine_grid(src_norm_trans_dst_norm[:, :2, :],
                         [B, C, out_size[0], out_size[1]],
                         align_corners=align_corners)
    return F.grid_sample(src, grid,
                         align_corners=align_corners,
                         mode=flags,
                         padding_mode=padding_mode)


def check_is_tensor(obj):
    """Checks whether the supplied object is a tensor.
    """
    if not isinstance(obj, paddle.Tensor):
        raise TypeError("Input type is not a paddle.Tensor. Got {}".format(type(obj)))


def normalize_homography(dst_pix_trans_src_pix: paddle.Tensor,
                         dsize_src: Tuple[int, int], dsize_dst: Tuple[int, int]) -> paddle.Tensor:
    r"""Normalize a given homography in pixels to [-1, 1].

    Args:
        dst_pix_trans_src_pix (paddle.Tensor): homography/ies from source to destiantion to be
          normalized. :math:`(B, 3, 3)`
        dsize_src (tuple): size of the source image (height, width).
        dsize_dst (tuple): size of the destination image (height, width).

    Returns:
        paddle.Tensor: the normalized homography of shape :math:`(B, 3, 3)`.
    """
    check_is_tensor(dst_pix_trans_src_pix)

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError("Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}"
                         .format(dst_pix_trans_src_pix.shape))

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: paddle.Tensor = normal_transform_pixel(
        src_h, src_w)
    src_pix_trans_src_norm = paddle.inverse(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: paddle.Tensor = normal_transform_pixel(
        dst_h, dst_w)

    # compute chain transformations
    dst_norm_trans_src_norm: paddle.Tensor = (
            dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    )
    return dst_norm_trans_src_norm


def normal_transform_pixel(height: int, width: int) -> paddle.Tensor:
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].

    Args:
        height (int): image height.
        width (int): image width.

    Returns:
        paddle.Tensor: normalized transform with shape :math:`(1, 3, 3)`.
    """
    tr_mat = paddle.to_tensor([[1.0, 0.0, -1.0],
                               [0.0, 1.0, -1.0],
                               [0.0, 0.0, 1.0]])  # 3x3

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / (width - 1.0)
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / (height - 1.0)

    return tr_mat.unsqueeze(0)  # 1x3x3


def convert_affinematrix_to_homography(A: paddle.Tensor) -> paddle.Tensor:
    r"""Function that converts batch of affine matrices from [Bx2x3] to [Bx3x3].

    Examples::

        >>> input = paddle.rand(2, 2, 3)  # Bx2x3
        >>> output = kornia.convert_affinematrix_to_homography(input)  # Bx3x3
    """
    if not isinstance(A, paddle.Tensor):
        raise TypeError("Input type is not a paddle.Tensor. Got {}".format(
            type(A)))
    if not (len(A.shape) == 3 and A.shape[-2:] == [2, 3]):
        raise ValueError("Input matrix must be a Bx2x3 tensor. Got {}"
                         .format(A.shape))
    return _convert_affinematrix_to_homography_impl(A)


def _convert_affinematrix_to_homography_impl(A: paddle.Tensor) -> paddle.Tensor:
    # A = np.pad(A.numpy(),[0, 0, 0, 1], "constant", value=0.)
    A = np.pad(A.numpy(), ((0, 0), (0, 1), (0, 0)), "constant")
    # H: paddle.Tensor = F.pad(A, [0, 0, 0, 1], "constant", value=0.,data_format='NCL')

    A[..., -1, -1] += 1.0
    H: paddle.Tensor = paddle.to_tensor(A)
    return H


def infer_box_shape(boxes: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
    r"""Auto-infer the output sizes for the given 2D bounding boxes.

    Args:
        boxes (paddle.Tensor): a tensor containing the coordinates of the
          bounding boxes to be extracted. The tensor must have the shape
          of Bx4x2, where each box is defined in the following (clockwise)
          order: top-left, top-right, bottom-right, bottom-left. The
          coordinates must be in the x, y order.

    Returns:
        Tuple[paddle.Tensor, paddle.Tensor]:
        - Bounding box heights, shape of :math:`(B,)`.
        - Boundingbox widths, shape of :math:`(B,)`.

    Example:
        >>> boxes = paddle.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ], [
        ...     [1., 1.],
        ...     [3., 1.],
        ...     [3., 2.],
        ...     [1., 2.],
        ... ]])  # 2x4x2
        >>> infer_box_shape(boxes)
        (tensor([2., 2.]), tensor([2., 3.]))
    """
    # validate_bboxes(boxes)
    width: paddle.Tensor = (boxes[:, 1, 0] - boxes[:, 0, 0] + 1)
    height: paddle.Tensor = (boxes[:, 2, 1] - boxes[:, 0, 1] + 1)
    return (height, width)


def get_perspective_transform(src, dst):
    r"""Calculates a perspective transform from four pairs of the corresponding
    points.

    The function calculates the matrix of a perspective transform so that:

    .. math ::

        \begin{bmatrix}
        t_{i}x_{i}^{'} \\
        t_{i}y_{i}^{'} \\
        t_{i} \\
        \end{bmatrix}
        =
        \textbf{map_matrix} \cdot
        \begin{bmatrix}
        x_{i} \\
        y_{i} \\
        1 \\
        \end{bmatrix}

    where

    .. math ::
        dst(i) = (x_{i}^{'},y_{i}^{'}), src(i) = (x_{i}, y_{i}), i = 0,1,2,3

    Args:
        src (Tensor): coordinates of quadrangle vertices in the source image.
        dst (Tensor): coordinates of the corresponding quadrangle vertices in
            the destination image.

    Returns:
        Tensor: the perspective transformation.

    Shape:
        - Input: :math:`(B, 4, 2)` and :math:`(B, 4, 2)`
        - Output: :math:`(B, 3, 3)`
    """
    if not isinstance(src, paddle.Tensor):
        raise TypeError("Input type is not a paddle.Tensor. Got {}"
                        .format(type(src)))
    if not isinstance(dst, paddle.Tensor):
        raise TypeError("Input type is not a paddle.Tensor. Got {}"
                        .format(type(dst)))
    if not src.shape[-2:] == [4, 2]:
        raise ValueError("Inputs must be a Bx4x2 tensor. Got {}"
                         .format(src.shape))
    if not src.shape == dst.shape:
        raise ValueError("Inputs must have the same shape. Got {}"
                         .format(dst.shape))
    if not (src.shape[0] == dst.shape[0]):
        raise ValueError("Inputs must have same batch size dimension. Expect {} but got {}"
                         .format(src.shape, dst.shape))

    # we build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here
    # we could even pass more correspondence
    p = []
    for i in [0, 1, 2, 3]:
        p.append(_build_perspective_param(src[:, i], dst[:, i], 'x'))
        p.append(_build_perspective_param(src[:, i], dst[:, i], 'y'))

    # A is Bx8x8
    A = paddle.stack(p, axis=1)

    # b is a Bx8x1
    b = paddle.stack([
        dst[:, 0:1, 0], dst[:, 0:1, 1],
        dst[:, 1:2, 0], dst[:, 1:2, 1],
        dst[:, 2:3, 0], dst[:, 2:3, 1],
        dst[:, 3:4, 0], dst[:, 3:4, 1],
    ], axis=1)

    # # solve the system Ax = b
    # X, LU = paddle.solve(b, A)
    X = np.linalg.solve(A.numpy(), b.numpy())
    # X = paddle.to_tensor(X)

    # create variable to return
    batch_size = src.shape[0]
    M = paddle.ones((batch_size, 9), dtype=src.dtype)
    M = np.ones((batch_size, 9), dtype=np.float)
    M[..., :8] = np.squeeze(X, axis=-1)
    npm = M.reshape((-1, 3, 3))  # Bx3x3
    return paddle.to_tensor(npm)


def _build_perspective_param(p: paddle.Tensor, q: paddle.Tensor, axis: str) -> paddle.Tensor:
    ones = np.ones_like(p)[..., 0:1]
    # ones = paddle.ones_like(p)[..., 0:1]
    ones = paddle.to_tensor(ones)
    zeros = np.zeros_like(p)[..., 0:1]
    zeros = paddle.to_tensor(zeros)
    # zeros = paddle.zeros_like(p)[..., 0:1]
    if axis == 'x':
        return paddle.concat(
            [p[:, 0:1], p[:, 1:2], ones, zeros, zeros, zeros,
             -p[:, 0:1] * q[:, 0:1], -p[:, 1:2] * q[:, 0:1]
             ], axis=1)

    if axis == 'y':
        return paddle.concat(
            [zeros, zeros, zeros, p[:, 0:1], p[:, 1:2], ones,
             -p[:, 0:1] * q[:, 1:2], -p[:, 1:2] * q[:, 1:2]], axis=1)

    raise NotImplementedError(f"perspective params for axis `{axis}` is not implemented.")