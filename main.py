import paddle


if __name__ == '__main__':
# test MattingBase model
    from model import MattingBase
    src = paddle.rand([4, 3, 256, 200])
    bgr = paddle.rand([4, 3, 256, 200])
    model = MattingBase('resnet50')
    x, *x_short = model(src, bgr)
    print(x.shape)

# YOLOV5 foucs moudle
    import numpy as np
    foucs = np.linspace(0, 47, num=48, dtype=int)
    foucs = foucs.reshape([1, 3, 4, 4])
    x = [foucs[..., ::2, ::2], foucs[..., 1::2, ::2], foucs[..., ::2, 1::2], foucs[..., 1::2, 1::2]]