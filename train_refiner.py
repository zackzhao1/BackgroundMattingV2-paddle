"""
Train MattingBase

You can download pretrained DeepLabV3 weights from <https://github.com/VainF/DeepLabV3Plus-Pytorch>

Example:

    CUDA_VISIBLE_DEVICES=0 python train_base.py \
        --dataset-name videomatte240k \
        --model-backbone resnet50 \
        --model-name mattingbase-resnet50-videomatte240k \
        --model-pretrain-initialization "pretraining/best_deeplabv3_resnet50_voc_os16.pth" \
        --epoch-end 8

"""

import argparse
# import kornia
import paddle
import os
import random

from torch import nn
from paddle.nn import functional as F
from paddle.amp import auto_cast, GradScaler
# from torch.utils.tensorboard import SummaryWriter
from paddle.io import DataLoader
# from torchvision.utils import make_grid
from tqdm import tqdm
import paddle.vision.transforms as T
from PIL import Image

from utils.metric import *
from utils.loss import base_compute_loss, refine_compute_loss
from utils.data_path import DATA_PATH
from dataset import ImagesDataset, ZipDataset, VideoDataset, SampleDataset
from dataset import augmentation as A
from dataset.augmentation import random_crop
from model import MattingBase, MattingRefine

# from model.utils import load_matched_state_dict


# --------------- Arguments ---------------


parser = argparse.ArgumentParser()
# parser.add_argument('--dataset-name', type=str, required=True, choices=DATA_PATH.keys())
parser.add_argument('--dataset-name', type=str, choices=DATA_PATH.keys())
parser.add_argument('--model-backbone', type=str, choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--model-name', type=str)
parser.add_argument('--model-pretrain-initialization', type=str, default=None)
parser.add_argument('--model-last-checkpoint', type=str, default=None)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--num-workers', type=int, default=16)
parser.add_argument('--epoch-start', type=int, default=0)
parser.add_argument('--epoch-end', type=int)
parser.add_argument('--log-train-loss-interval', type=int, default=10)
parser.add_argument('--log-train-images-interval', type=int, default=2000)
parser.add_argument('--log-valid-interval', type=int, default=5000)
parser.add_argument('--checkpoint-interval', type=int, default=5000)
args = parser.parse_args()

args.log_train_loss_interval = 1000
args.log_train_images_interval = 1000
args.log_valid_interval = 1000
args.checkpoint_interval = 5000

args.dataset_name = 'videomatte240k'
args.model_backbone = 'resnet50'
args.model_name = '/home/sp/zhaojl/BackgroundMattingV2-master/model/PyTorch/'
args.epoch_end = 1
args.batch_size = 2
args.num_workers = 2
# distributed_num_gpus = 2
args.device = 'cuda:7'
# distributed_num_gpus = torch.cuda.device_count()
# assert args.batch_size % distributed_num_gpus == 0
os.environ["CUDA_VISIBLE_DEVICES"] = args.device


# --------------- Loading ---------------


def train():
    # Training DataLoader
    dataset_train = ZipDataset([
        ZipDataset([
            ImagesDataset(DATA_PATH[args.dataset_name]['train']['pha'], mode='L'),
            ImagesDataset(DATA_PATH[args.dataset_name]['train']['fgr'], mode='RGB'),
        ],
            transforms=A.PairCompose([
                A.PairRandomAffineAndResize((2048, 2048), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.3, 1), shear=(-5, 5)),
                A.PairRandomHorizontalFlip(),
                A.PairRandomBoxBlur(0.1, 5),
                A.PairRandomSharpen(0.1),
                A.PairApplyOnlyAtIndices([1], T.ColorJitter(0.15, 0.15, 0.15, 0.05)),
                A.PairApply(T.ToTensor())
            ]),
            assert_equal_length=True),
        ImagesDataset(DATA_PATH['backgrounds']['train'], mode='RGB', transforms=T.Compose([
            A.RandomAffineAndResize((2048, 2048), degrees=(-5, 5), translate=(0.1, 0.1), scale=(1, 2), shear=(-5, 5)),
            T.RandomHorizontalFlip(),
            A.RandomBoxBlur(0.1, 5),
            A.RandomSharpen(0.1),
            T.ColorJitter(0.15, 0.15, 0.15, 0.05),
            T.ToTensor()
        ])),
    ])
    dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    # Validation DataLoader
    dataset_valid = ZipDataset([
        ZipDataset([
            ImagesDataset(DATA_PATH[args.dataset_name]['valid']['pha'], mode='L'),
            ImagesDataset(DATA_PATH[args.dataset_name]['valid']['fgr'], mode='RGB')
        ], transforms=A.PairCompose([
            A.PairRandomAffineAndResize((2048, 2048), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.3, 1),
                                        shear=(-5, 5)),
            A.PairApply(T.ToTensor())
        ]), assert_equal_length=True),
        ImagesDataset(DATA_PATH['backgrounds']['valid'], mode='RGB', transforms=T.Compose([
            A.RandomAffineAndResize((2048, 2048), degrees=(-5, 5), translate=(0.1, 0.1), scale=(1, 1.2), shear=(-5, 5)),
            T.ToTensor()
        ])),
    ])
    dataset_valid = SampleDataset(dataset_valid, 2)
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

    # Model
    # model = MattingBase(args.model_backbone)
    model = MattingRefine('resnet50', 0.25, 'full', 80_000, 0.7, 3)
    weights = paddle.load(os.path.join(args.model_name, 'paddle_resnet50_last.pdparams'))
    model.load_dict(weights)
    optimizer = paddle.optimizer.Adam(1e-5, parameters=model.parameters())
    scaler = GradScaler()

    # Logging and checkpoints
    if not os.path.exists(f'checkpoint/{args.model_name}'):
        os.makedirs(f'checkpoint/{args.model_name}')

    model.train()
    # Run loop
    for epoch in range(args.epoch_start, args.epoch_end):
        for i, ((true_pha, true_fgr), true_bgr) in enumerate(tqdm(dataloader_train)):
            step = epoch * len(dataloader_train) + i
            # true_pha = true_pha
            # true_fgr = true_fgr
            # true_bgr = true_bgr
            true_pha, true_fgr, true_bgr = random_crop(true_pha, true_fgr, true_bgr)
            true_src = true_bgr.clone()

            # # Augment with shadow
            # aug_shadow_idx = torch.rand(len(true_src)) < 0.3
            # if aug_shadow_idx.any():
            #     aug_shadow = true_pha[aug_shadow_idx].mul(0.3 * random.random())
            #     aug_shadow = T.RandomAffine(degrees=(-5, 5), translate=(0.2, 0.2), scale=(0.5, 1.5), shear=(-5, 5))(aug_shadow)
            #     aug_shadow = kornia.filters.box_blur(aug_shadow, (random.choice(range(20, 40)),) * 2)
            #     true_src[aug_shadow_idx] = true_src[aug_shadow_idx].sub_(aug_shadow).clamp_(0, 1)
            #     del aug_shadow
            # del aug_shadow_idx

            # Composite foreground onto source
            true_src = true_fgr * true_pha + true_src * (1 - true_pha)
            #
            #             # Augment with noise
            #             aug_noise_idx = torch.rand(len(true_src)) < 0.4
            #             if aug_noise_idx.any():
            #                 true_src[aug_noise_idx] = true_src[aug_noise_idx].add_(torch.randn_like(true_src[aug_noise_idx]).mul_(0.03 * random.random())).clamp_(0, 1)
            #                 true_bgr[aug_noise_idx] = true_bgr[aug_noise_idx].add_(torch.randn_like(true_bgr[aug_noise_idx]).mul_(0.03 * random.random())).clamp_(0, 1)
            #             del aug_noise_idx
            #
            #             # Augment background with jitter
            #             aug_jitter_idx = torch.rand(len(true_src)) < 0.8
            #             if aug_jitter_idx.any():
            #                 true_bgr[aug_jitter_idx] = kornia.augmentation.ColorJitter(0.18, 0.18, 0.18, 0.1)(true_bgr[aug_jitter_idx])
            #             del aug_jitter_idx
            #
            #             # Augment background with affine
            #             aug_affine_idx = torch.rand(len(true_bgr)) < 0.3
            #             if aug_affine_idx.any():
            #                 true_bgr[aug_affine_idx] = T.RandomAffine(degrees=(-1, 1), translate=(0.01, 0.01))(true_bgr[aug_affine_idx])
            #             del aug_affine_idx
            # #

            with auto_cast():
                # pred_pha, pred_fgr, pred_err = model(true_src, true_bgr)[:3]
                # loss = base_compute_loss(pred_pha, pred_fgr, pred_err, true_pha, true_fgr)
                pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm, _ = model(true_src, true_bgr)
                loss = refine_compute_loss(pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm, true_pha, true_fgr)

            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.minimize(optimizer, scaled)
            optimizer.clear_grad()

            if (i + 1) % args.log_train_loss_interval == 0:
                print(f'step:{step} loss:{loss.item()}')

            # del true_pha, true_fgr, true_bgr
            # del pred_pha, pred_fgr, pred_err
            del true_pha, true_fgr, true_src, true_bgr
            del pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm

            if (i + 1) % args.log_valid_interval == 0:
                valid(model, dataloader_valid, step)

            if (step + 1) % args.checkpoint_interval == 0:
                paddle.save(model.state_dict(), f'{args.model_name}/epoch-{epoch}-iter-{step}.pdparams')

            paddle.save(model.state_dict(), f'{args.model_name}/epoch-{epoch}.pdparams')


def valid(model, dataloader, step):
    model.eval()
    loss_total = 0
    loss_count = 0
    metric_SAD = 0
    metric_MAD = 0
    metric_MSE = 0
    with torch.no_grad():
        for (true_pha, true_fgr), true_bgr in tqdm(dataloader):
            batch_size = true_pha.shape[0]
            #
            # true_pha = true_pha
            # true_fgr = true_fgr
            # true_bgr = true_bgr
            true_src = true_pha * true_fgr + (1 - true_pha) * true_bgr

            # pred_pha, pred_fgr, pred_err = model(true_src, true_bgr)[:3]
            # loss = base_compute_loss(pred_pha, pred_fgr, pred_err, true_pha, true_fgr)
            pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm, _ = model(true_src, true_bgr)
            loss = refine_compute_loss(pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm, true_pha, true_fgr)

            loss_total += loss.cpu().item() * batch_size
            loss_count += batch_size

            metric_SAD += MetricSAD(pred_pha, true_pha).item()
            metric_MAD += MetricMAD(pred_pha, true_pha).item()
            metric_MSE += MetricMSE(pred_pha, true_pha).item()

            del true_pha, true_fgr, true_src, true_bgr
            del pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm

    print(f'valid_loss: {loss_total / loss_count}, step: {step}')
    print(f'valid MAD: {metric_MAD / loss_count}, SAD: {metric_SAD / loss_count}, MSE: {metric_MSE / loss_count}')
    model.train()


# --------------- Start ---------------
if __name__ == '__main__':
    train()
