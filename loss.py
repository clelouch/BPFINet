import torch
import math
import torch.nn.functional as F
from torch import nn


def gaussian(window_size, sigma):
    import math
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def get_weight(data, window_size=31, sigma=4, channel=1, weight_type='average'):
    if weight_type == 'average':
        weight = torch.abs(F.avg_pool2d(data, kernel_size=window_size, stride=1, padding=window_size // 2) - data)
    elif weight_type == 'normal':
        x = torch.Tensor([math.exp(-(i - window_size // 2) ** 2 / float(2 * sigma ** 2)) for i in range(window_size)])
        x = x.unsqueeze(1) / x.sum()
        normal_kernel = x.mm(x.t()).unsqueeze(0).unsqueeze(0)
        normal_kernel = normal_kernel.expand(channel, 1, window_size, window_size)
        normal_kernel = normal_kernel.cuda()
        weight = torch.abs(F.conv2d(data, normal_kernel, stride=1, padding=window_size // 2) - data)
    return weight


def iou_loss(pred, mask, weight):
    inter = ((pred * mask) * weight).sum(dim=(2, 3))
    union = ((pred + mask) * weight).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wiou


def rev_iou_loss(pred, mask, weight):
    pred = 1 - pred
    mask = 1 - mask
    rev_wiou = iou_loss(pred, mask, weight)
    return rev_wiou


def wbce_loss(pred, mask, weight):
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weight * wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    return wbce


class TriLoss(nn.Module):
    def __init__(self, window_size=31, sigma=4, weight_type='normal', channel=1, n=3):
        super(TriLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.weight_type = weight_type
        self.channel = channel
        self.n = n

    def forward(self, pred, mask):
        # print("mask shape: ", mask.shape)
        weight = get_weight(mask, self.window_size, self.sigma, self.channel, 'normal')
        weight = 1 + self.n * weight
        wbce_ = wbce_loss(pred, mask, weight)

        pred = torch.sigmoid(pred)
        iou_ = iou_loss(pred, mask, weight)
        rev_iou_ = rev_iou_loss(pred, mask, weight)

        return wbce_, iou_, rev_iou_
