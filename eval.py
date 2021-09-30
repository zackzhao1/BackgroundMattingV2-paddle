import os
import argparse
import paddle
import torch
from paddle.io import DataLoader
from tqdm import tqdm
import paddle.vision.transforms as T
from utils.data_path import DATA_PATH
from dataset import ImagesDataset, ZipDataset, VideoDataset, SampleDataset
from dataset import augmentation as A
from model import MattingBase, MattingRefine
from model_torch import MattingRefine_pt


# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='/home/sp/zhaojl/BackgroundMattingV2-master/data/test85/')
parser.add_argument('--model-path', type=str, default='/home/sp/zhaojl/BackgroundMattingV2-master/model/PyTorch/')
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
            A.PairRandomAffineAndResize((2048, 2048), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.3, 1), shear=(-5, 5)),
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
    # pt_sad, pt_mse = pytorch_valid(gen_data)

    print('Compare with the official model!')
    print(f'paddle output:  SAD: {pd_sad / len(gen_data)}, MSE: {pd_mse / len(gen_data)}')
    # print(f'pytorch output:  SAD: {pt_sad / len(gen_data)}, MSE: {pt_mse / len(gen_data)}')


# --------------- utils ---------------
def paddle_valid(dataloader):
    # model = MattingBase('resnet50')
    model = MattingRefine('resnet50', 0.25, 'sampling', 80_000, 0.7, 3)
    weights = paddle.load(os.path.join(args.model_path, 'paddle_resnet50_last.pdparams'))
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


def pytorch_valid(dataloader):
    model = MattingRefine_pt('resnet50', 0.25, 'sampling', 80_000, 0.7, 3)
    load_matched_state_dict(model, torch.load(os.path.join(args.model_path, 'pytorch_resnet50.pth')))
    model = model.to('cuda')

    model.eval()
    loss_count = 0
    sad = 0
    mse = 0
    with torch.no_grad():
        for true_pha, true_fgr, true_bgr in tqdm(dataloader):
            true_pha = torch.tensor(true_pha).to('cuda')
            true_fgr = torch.tensor(true_fgr).to('cuda')
            true_bgr = torch.tensor(true_bgr).to('cuda')
            true_src = true_pha * true_fgr + (1 - true_pha) * true_bgr

            pred_pha, *_ = model(true_src, true_bgr)

            img = true_pha[0][0].cpu().numpy()
            trimap = gen_trimap(img)
            mask_pha = torch.tensor([trimap]).unsqueeze(1)

            sad += BatchSAD_pt(pred_pha, true_pha, mask_pha)
            mse += BatchMSE_pt(pred_pha, true_pha, mask_pha)
            loss_count += 1

    return sad, mse


def load_matched_state_dict(model, state_dict, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            curr_state_dict[key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')


def gen_trimap(alpha, ksize=3, iterations=10):
    import cv2
    import numpy as np
    alpha = alpha * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(alpha, kernel, iterations=iterations)
    eroded = cv2.erode(alpha, kernel, iterations=iterations)
    trimap = np.zeros(alpha.shape) + 128
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    return trimap


def BatchSAD(pred, target, mask, scale=2):
    # function loss = compute_sad_loss(pred, target, trimap)
    # error_map = (pred - target).abs() / 255.
    # batch_loss = (error_map * mask).view(B, -1).sum(dim=-1)
    # batch_loss = batch_loss / 1000.
    # return batch_loss.data.cpu().numpy()
    B = target.shape[0]
    error_map = (pred - target).abs()
    batch_loss = (error_map.cpu() * (mask == 128)).reshape((B, -1)).sum(axis=-1)
    batch_loss = batch_loss / 1000.
    return batch_loss.item()*scale


def BatchMSE(pred, target, mask, scale=2):
    # function loss = compute_mse_loss(pred, target, trimap)
    # error_map = (single(pred) - single(target)) / 255;
    # loss = sum(sum(error_map. ^ 2. * single(trimap == 128))) / sum(sum(single(trimap == 128)));
    B = target.shape[0]
    error_map = (pred - target)
    batch_loss = (error_map.pow(2).cpu() * (mask == 128)).reshape((B, -1)).sum(axis=-1)
    batch_loss = batch_loss / ((mask == 128).astype(float).reshape((B, -1)).sum(axis=-1) + 1.)
    batch_loss = batch_loss * 1000.
    return batch_loss.item()*scale


def BatchSAD_pt(pred, target, mask, scale=2):
    # function loss = compute_sad_loss(pred, target, trimap)
    # error_map = (pred - target).abs() / 255.
    # batch_loss = (error_map * mask).view(B, -1).sum(dim=-1)
    # batch_loss = batch_loss / 1000.
    # return batch_loss.data.cpu().numpy()
    B = target.size(0)
    error_map = (pred - target).abs()
    batch_loss = (error_map.cpu() * (mask==128)).view(B, -1).sum(dim=-1)
    batch_loss = batch_loss / 1000.
    return batch_loss.data.numpy()*scale


def BatchMSE_pt(pred, target, mask, scale=2):
    # function loss = compute_mse_loss(pred, target, trimap)
    # error_map = (single(pred) - single(target)) / 255;
    # loss = sum(sum(error_map. ^ 2. * single(trimap == 128))) / sum(sum(single(trimap == 128)));
    B = target.size(0)
    error_map = (pred - target)
    batch_loss = (error_map.pow(2).cpu() * (mask==128)).view(B, -1).sum(dim=-1)
    batch_loss = batch_loss / ((mask==128).view(B, -1).sum(dim=-1) + 1.)
    batch_loss = batch_loss * 1e3
    return batch_loss.data.numpy()*scale


# --------------- Start ---------------
if __name__ == '__main__':
    eval()
