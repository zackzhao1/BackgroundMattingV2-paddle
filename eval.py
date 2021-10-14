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


import os
import argparse
import paddle
from paddle.io import DataLoader
from tqdm import tqdm
import paddle.vision.transforms as T
from utils.data_path import DATA_PATH
from dataset import ImagesDataset, ZipDataset, VideoDataset, SampleDataset
from dataset import augmentation as A
from model import MattingBase, MattingRefine
from utils.metric import *


# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='./data/test85/')
parser.add_argument('--model-path', type=str, default='./model/weights/stage2.pdparams')
args = parser.parse_args()
args.batch_size = 1
args.device = 'cuda:7'


# --------------- Loading ---------------
def eval():
    dataset_valid = ZipDataset([
        ZipDataset([
            ImagesDataset(os.path.join(args.data_path, 'pha'), mode='L'),
            ImagesDataset(os.path.join(args.data_path, 'fgr'), mode='RGB')
        ], transforms=A.PairCompose([
            A.PairRandomAffineAndResize((2048, 2048), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.55, 0.9), shear=(-5, 5)),
            A.PairApply(T.ToTensor())
        ]), assert_equal_length=True),
        ImagesDataset(DATA_PATH['backgrounds']['valid'], mode='RGB', transforms=T.Compose([
            A.RandomAffineAndResize((2048, 2048), degrees=(-5, 5), translate=(0.1, 0.1), scale=(1, 1.2), shear=(-5, 5)),
            T.ToTensor()
        ])),
    ])
    dataset_valid = SampleDataset(dataset_valid, 85)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, num_workers=16)
    gen_data = []
    for (true_pha, true_fgr), true_bgr in dataloader_valid:
        gen_data.append([true_pha.cpu().detach().numpy(),
                         true_fgr.cpu().detach().numpy(),
                         true_bgr.cpu().detach().numpy()])

    pd_sad, pd_mse = paddle_valid(gen_data)
    print(f'paddle output:  SAD: {pd_sad / len(gen_data)}, MSE: {pd_mse / len(gen_data)}')


# --------------- utils ---------------
def paddle_valid(dataloader):
    # model = MattingBase('resnet50')
    model = MattingRefine('resnet50', 0.25, 'sampling', 80_000, 0.7, 3)
    # weights = paddle.load(os.path.join(args.model_path, 'stage2.pdparams'))
    weights = paddle.load(args.model_path)
    model.load_dict(weights)

    model.eval()
    loss_count = 0
    sad = 0
    mse = 0
    with paddle.no_grad():
        for true_pha, true_fgr, true_bgr in tqdm(dataloader):
            true_pha = paddle.to_tensor(true_pha)
            true_fgr = paddle.to_tensor(true_fgr)
            true_bgr = paddle.to_tensor(true_bgr)
            true_src = true_pha * true_fgr + (1 - true_pha) * true_bgr

            pred_pha, *_ = model(true_src, true_bgr)

            img = true_pha[0][0].cpu().numpy()
            trimap = gen_trimap(img)
            mask_pha = paddle.to_tensor([trimap]).unsqueeze(1)

            sad += BatchSAD(pred_pha, true_pha, mask_pha)
            mse += BatchMSE(pred_pha, true_pha, mask_pha)
            loss_count += 1

    return sad, mse


# --------------- Start ---------------
if __name__ == '__main__':
    eval()
