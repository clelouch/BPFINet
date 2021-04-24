import torch
import math
import torch.nn.functional as F
from torch import nn


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
    def __init__(self, window_size=31, sigma=4, n=5):
        super(TriLoss, self).__init__()
        self.window_size = window_size
        self.sigma = nn.Parameter(torch.Tensor([sigma]), requires_grad=True)
        self.n = n

    def forward(self, pred, mask):
        up = torch.Tensor([-(i - self.window_size // 2) ** 2 for i in range(self.window_size)])
        x = torch.exp(up / (2 * self.sigma ** 2))
        x = x.unsqueeze(1) / x.sum()
        normal_kernel = x.mm(x.t()).unsqueeze(0).unsqueeze(0)
        normal_kernel = normal_kernel.expand(1, 1, self.window_size, self.window_size)
        normal_kernel = normal_kernel.requires_grad_(True).cuda()
        weight = torch.abs(F.conv2d(mask, normal_kernel, stride=1, padding=self.window_size // 2) - mask)

        weight = 1 + self.n * weight
        wbce_ = wbce_loss(pred, mask, weight)

        pred = torch.sigmoid(pred)
        iou_ = iou_loss(pred, mask, weight)
        rev_iou_ = rev_iou_loss(pred, mask, weight)

        return wbce_, iou_, rev_iou_
