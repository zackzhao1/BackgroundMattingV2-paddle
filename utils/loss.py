import paddle
from paddle.nn import functional as F


def base_compute_loss(pred_pha, pred_fgr, pred_err, true_pha, true_fgr, use_laplacian=True):
    true_err = paddle.abs(pred_pha.detach() - true_pha)
    np_true_pha = true_pha.numpy()
    true_msk = np_true_pha != 0
    np_fgr_msk = pred_fgr.numpy() * true_msk
    np_tru_fgr_msk = true_fgr.numpy() * true_msk

    bmv2_loss = F.l1_loss(pred_pha, true_pha) + \
                F.l1_loss(tensor_sobel(pred_pha), tensor_sobel(true_pha)) + \
                F.l1_loss(paddle.to_tensor(np_fgr_msk), paddle.to_tensor(np_tru_fgr_msk)) + \
                F.mse_loss(pred_err, true_err)
    if use_laplacian:
        return bmv2_loss + laplacian_loss(pred_pha, true_pha)
    else:
        return bmv2_loss


def refine_compute_loss(pred_pha_lg, pred_fgr_lg, pred_pha_sm, pred_fgr_sm, pred_err_sm, true_pha_lg, true_fgr_lg, use_laplacian=True):
    true_pha_sm = tensor_resize(true_pha_lg, pred_pha_sm.shape[2:])
    true_fgr_sm = tensor_resize(true_fgr_lg, pred_fgr_sm.shape[2:])
    true_msk_lg = (true_pha_lg != 0).astype(paddle.float32)
    true_msk_sm = (true_pha_sm != 0).astype(paddle.float32)
    bmv2_loss = F.l1_loss(pred_pha_lg, true_pha_lg) + \
                F.l1_loss(pred_pha_sm, true_pha_sm) + \
                F.l1_loss(tensor_sobel(pred_pha_lg), tensor_sobel(true_pha_lg)) + \
                F.l1_loss(tensor_sobel(pred_pha_sm), tensor_sobel(true_pha_sm)) + \
                F.l1_loss(pred_fgr_lg * true_msk_lg, true_fgr_lg * true_msk_lg) + \
                F.l1_loss(pred_fgr_sm * true_msk_sm, true_fgr_sm * true_msk_sm) + \
                F.mse_loss(tensor_resize(pred_err_sm, true_pha_lg.shape[2:]), \
                          paddle.fluid.layers.elementwise_sub(tensor_resize(pred_pha_sm, true_pha_lg.shape[2:]),true_pha_lg).abs())
    if use_laplacian:
        return bmv2_loss + \
               laplacian_loss(pred_pha_lg, true_pha_lg) + \
               laplacian_loss(pred_pha_sm, true_pha_sm)
    else:
        return bmv2_loss


def tensor_resize(input, size):
    if not isinstance(input, paddle.Tensor) or len(input.shape) < 2:
        raise TypeError(f"Input Error")

    input_size = input.shape[-2:]
    if size == input_size:
        return input
    else:
        return paddle.nn.functional.interpolate(input, size=size, mode='bilinear')


def spatial_gradient(input):
    # allocate kernel
    kernel_x = paddle.to_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    kernel_y = kernel_x.transpose([1, 0])
    kernel = paddle.stack([kernel_x, kernel_y])
    norm = kernel.abs().sum(-1).sum(-1)
    kernel /= (norm.unsqueeze(-1).unsqueeze(-1))

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: paddle.Tensor = kernel.detach().unsqueeze(1).unsqueeze(1)

    # convolve input tensor with sobel kernel
    kernel_flip: paddle.Tensor = tmp_kernel.flip(-3)

    # Pad with "replicate for spatial dims, but with zeros for channel
    spatial_pad = [kernel.shape[1] // 2, kernel.shape[1] // 2, kernel.shape[2] // 2, kernel.shape[2] // 2]
    padded_inp: paddle.Tensor = F.pad(input.reshape((b * c, 1, h, w)), spatial_pad, 'replicate').unsqueeze(2)

    return F.conv3d(padded_inp, kernel_flip, padding=0).reshape((b, c, 2, h, w))


def tensor_sobel(input, eps: float = 1e-6):
    if not isinstance(input, paddle.Tensor) or len(input.shape) != 4:
        raise TypeError(f"Input Error")

    # comput the x/y gradients
    edges: paddle.Tensor = spatial_gradient(input)

    # unpack the edges
    gx: paddle.Tensor = edges[:, :, 0]
    gy: paddle.Tensor = edges[:, :, 1]

    # compute gradient maginitude
    magnitude: paddle.Tensor = paddle.sqrt(gx * gx + gy * gy + eps)

    return magnitude


def matting_loss(pred_fgr, pred_pha, true_fgr, true_pha):
    """
    Args:
        pred_fgr: Shape(B, T, 3, H, W)
        pred_pha: Shape(B, T, 1, H, W)
        true_fgr: Shape(B, T, 3, H, W)
        true_pha: Shape(B, T, 1, H, W)
    """
    loss = dict()
    # Alpha losses
    loss['pha_l1'] = F.l1_loss(pred_pha, true_pha)
    loss['pha_laplacian'] = laplacian_loss(pred_pha.flatten(0, 1), true_pha.flatten(0, 1))
    loss['pha_coherence'] = F.mse_loss(pred_pha[:, 1:] - pred_pha[:, :-1],
                                       true_pha[:, 1:] - true_pha[:, :-1]) * 5
    # Foreground losses
    true_msk = true_pha.gt(0)
    pred_fgr = pred_fgr * true_msk
    true_fgr = true_fgr * true_msk
    loss['fgr_l1'] = F.l1_loss(pred_fgr, true_fgr)
    loss['fgr_coherence'] = F.mse_loss(pred_fgr[:, 1:] - pred_fgr[:, :-1],
                                       true_fgr[:, 1:] - true_fgr[:, :-1]) * 5
    # Total
    loss['total'] = loss['pha_l1'] + loss['pha_coherence'] + loss['pha_laplacian'] \
                  + loss['fgr_l1'] + loss['fgr_coherence']
    return loss


# ------------------------  Laplacian Loss ------------------------
def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.place, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels

def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid

def gauss_kernel(device='cpu', dtype=paddle.float32):
    kernel = paddle.to_tensor([[1,  4,  6,  4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1,  4,  6,  4, 1]], place=device, dtype=dtype)
    kernel /= 256
    kernel= paddle.unsqueeze(kernel, axis=[0, 1])
    return kernel

def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = paddle.reshape(img, [B * C, 1, H, W])
    img = F.pad(img, [2, 2, 2, 2], mode='reflect')
    img = F.conv2d(img, kernel)
    img = paddle.reshape(img, [B, C, H, W])
    return img

def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img

def upsample(img, kernel):
    B, C, H, W = img.shape
    out = paddle.zeros((B, C, H * 2, W * 2), dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out

def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]


if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    file = '/home/sp/zhaojl/Crowd/ProcessedData/collect/CgAH6F_hj9mAMYu0AAHEsFs7nX0357.jpg'
    img = Image.open(file)
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)
    img_org = img.copy().astype(np.uint8)
    img = img.transpose((2, 0, 1))  # 转换成CHW
    img = img[(2, 1, 0), :, :] / 255.0 # 转换成BGR
    img = np.expand_dims(img, axis=0)
    data = paddle.to_tensor(img)
    # data = paddle.ones([2, 3, 112, 112], dtype=paddle.float32)
    kernel = gauss_kernel()
    pyramid = laplacian_pyramid(data, kernel, max_levels=5)
    print(kernel.shape)

    diff = pyramid[0]
    import matplotlib.pyplot as plt
    d_tmp = (diff.cpu().detach().numpy()[0] * 255).astype(np.uint8)
    d_tmp = d_tmp.transpose((1, 2, 0))
    d_img = Image.fromarray(d_tmp)
    plt.imshow(d_img)
    plt.show()

    lap_img = Image.fromarray(img_org)
    plt.imshow(lap_img)
    plt.show()
    lap_img = Image.fromarray(img_org - d_img)
    plt.imshow(lap_img)
    plt.show()

