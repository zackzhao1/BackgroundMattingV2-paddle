import torch

# true_pha = torch.from_numpy(true_pha).cuda(non_blocking=True).float().div_(255), cv2.IMREAD_GRAYSCALE)

def MetricSAD(pred_pha, true_pha):
    return (pred_pha - true_pha).abs().sum() * 1e3


def MetricMAD(pred_pha, true_pha):
    return (pred_pha - true_pha).abs().mean() * 1e3


def MetricMSE(pred_pha, true_pha):
    return ((pred_pha - true_pha) ** 2).mean() * 1e3


