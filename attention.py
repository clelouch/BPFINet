import torch
from torch import nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ChannelAttention, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(nn.Linear(out_channel, out_channel // 4, bias=False), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(out_channel // 4, out_channel, bias=False), nn.Sigmoid())

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 1, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(out_channel))

    def forward(self, x):
        n, c, h, w = x.shape
        x = self.conv(x)
        y1 = self.pool(x)
        y1 = y1.reshape((n, -1))
        y1 = self.linear1(y1)
        y1 = self.linear2(y1)
        y1 = y1.reshape((n, self.out_channel, 1, 1))

        y1 = y1.expand_as(x).clone()
        y = x * y1
        return F.relu(y + self.conv2(y))
