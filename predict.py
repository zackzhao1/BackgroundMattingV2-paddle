import os
import argparse
import paddle
from PIL import Image
import numpy as np
from model import MattingBase, MattingRefine


parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='./data/test85/')
parser.add_argument('--model-path', type=str, default='./model/weights/stage2.pdparams')
args = parser.parse_args()


def predict():
    src_path = './image/01_src.jpg'
    src_img = Image.open(src_path)
    size = src_img.size
    src_img = src_img.resize((2048, 2048), Image.ANTIALIAS)
    src_img = np.array(src_img).astype(np.float32)
    src_img = src_img.transpose((2, 0, 1))  # 转换成CHW
    src_img /= 255.0  # 转换成BGR
    src_data = paddle.to_tensor(src_img)[0:3]
    src_data = src_data.unsqueeze(0)

    bgr_path = './image/01_bgr.jpg'
    bgr_img = Image.open(bgr_path)
    bgr_img = bgr_img.resize((2048, 2048), Image.ANTIALIAS)
    bgr_img = np.array(bgr_img).astype(np.float32)
    bgr_img = bgr_img.transpose((2, 0, 1))  # 转换成CHW
    bgr_img /= 255.0  # 转换成BGR
    bgr_data = paddle.to_tensor(bgr_img)[0:3]
    bgr_data = bgr_data.unsqueeze(0)

    model = MattingRefine('resnet50', 0.25, 'sampling', 80_000, 0.7, 3)
    # weights = paddle.load(os.path.join(args.model_path, 'stage2.pdparams'))
    weights = paddle.load(args.model_path)
    model.load_dict(weights)

    model.eval()
    with paddle.no_grad():
        pred_pha, *_ = model(src_data, bgr_data)

    ret = (pred_pha[0][0].cpu().numpy() * 255).astype(np.uint8)
    ret = Image.fromarray(ret)
    ret = ret.resize(size, Image.ANTIALIAS)
    ret.save('./image/01_pred.jpg')


if __name__ == '__main__':
    predict()
