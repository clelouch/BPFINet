import torch
from torch import nn
import torch.nn.functional as F
from attention import ChannelAttention as CA
from deeplab_resnet import resnet50

config_resnet = {'convert': [[64, 256, 512, 1024, 2048], [128, 256, 256, 512, 512]],
                 'score': 128}


class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up = []
        for i in range(len(list_k[0])):
            up.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False),
                                    nn.BatchNorm2d(list_k[1][i]),
                                    nn.ReLU(inplace=True)))
        self.convert0 = nn.ModuleList(up)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl


class BasicConv(nn.Module):
    def __init__(self, channel, stride, padding=1, dilate=1):
        super(BasicConv, self).__init__()
        self.channel = channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, 3, stride=stride, padding=padding, dilation=dilate, bias=False),
            nn.BatchNorm2d(self.channel),)
            # nn.ReLU()

    def forward(self, x):
        return self.conv(x)


class USRM3(nn.Module):
    def __init__(self, channel):
        super(USRM3, self).__init__()
        self.channel = channel
        self.conv1 = BasicConv(self.channel, 2, 1, 1)
        self.conv2 = BasicConv(self.channel, 2, 1, 1)
        self.conv3 = BasicConv(self.channel, 1, 2, 2)

        self.conv_rev1 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev2 = BasicConv(self.channel, 1, 1, 1)

        self.conv_sum = BasicConv(self.channel, 1, 1, 1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)

        y2up = F.interpolate(y3, y2.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv_rev1(y2 + y2up)
        y1up = F.interpolate(y2, y1.shape[2:], mode='bilinear', align_corners=True)
        y1 = self.conv_rev2(y1 + y1up)
        y = F.interpolate(y1, x.shape[2:], mode='bilinear', align_corners=True)
        return self.conv_sum(F.relu(x + y))


class USRM4(nn.Module):
    def __init__(self, channel):
        super(USRM4, self).__init__()
        self.channel = channel
        self.conv1 = BasicConv(self.channel, 2, 1, 1)
        self.conv2 = BasicConv(self.channel, 2, 1, 1)
        self.conv3 = BasicConv(self.channel, 2, 1, 1)
        self.conv4 = BasicConv(self.channel, 1, 2, 2)

        self.conv_rev1 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev2 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev3 = BasicConv(self.channel, 1, 1, 1)

        self.conv_sum = BasicConv(self.channel, 1, 1, 1)

    def forward(self, x, gi):
        # gi means global information
        y1 = self.conv1(x)
        y1 = y1 + F.interpolate(gi, y1.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)

        y3up = F.interpolate(y4, y3.shape[2:], mode='bilinear', align_corners=True)
        y3 = self.conv_rev1(y3 + y3up)
        y2up = F.interpolate(y3, y2.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv_rev2(y2 + y2up)
        y1up = F.interpolate(y2, y1.shape[2:], mode='bilinear', align_corners=True)
        y1 = self.conv_rev3(y1 + y1up)
        y = F.interpolate(y1, x.shape[2:], mode='bilinear', align_corners=True)
        return self.conv_sum(F.relu(x + y))


class USRM5(nn.Module):
    def __init__(self, channel):
        super(USRM5, self).__init__()
        self.channel = channel
        self.conv1 = BasicConv(self.channel, 2, 1, 1)
        self.conv2 = BasicConv(self.channel, 2, 1, 1)
        self.conv3 = BasicConv(self.channel, 2, 1, 1)
        self.conv4 = BasicConv(self.channel, 2, 1, 1)
        self.conv5 = BasicConv(self.channel, 1, 2, 2)

        self.conv_rev1 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev2 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev3 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev4 = BasicConv(self.channel, 1, 1, 1)

        self.conv_sum = BasicConv(self.channel, 1, 1, 1)

    def forward(self, x, high, gi):
        # gi means global information
        y1 = self.conv1(x)
        y1 = y1 + F.interpolate(high, y1.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv2(y1)
        y2 = y2 + F.interpolate(gi, y2.shape[2:], mode='bilinear', align_corners=True)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)

        y4up = F.interpolate(y5, y4.shape[2:], mode='bilinear', align_corners=True)
        y4 = self.conv_rev1(y4 + y4up)
        y3up = F.interpolate(y4, y3.shape[2:], mode='bilinear', align_corners=True)
        y3 = self.conv_rev2(y3 + y3up)
        y2up = F.interpolate(y3, y2.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv_rev3(y2 + y2up)
        y1up = F.interpolate(y2, y1.shape[2:], mode='bilinear', align_corners=True)
        y1 = self.conv_rev4(y1 + y1up)
        y = F.interpolate(y1, x.shape[2:], mode='bilinear', align_corners=True)
        return self.conv_sum(F.relu(x + y))


class USRM5_2(nn.Module):
    def __init__(self, channel):
        super(USRM5_2, self).__init__()
        self.channel = channel
        self.conv1 = BasicConv(self.channel, 2, 1, 1)
        self.conv2 = BasicConv(self.channel, 2, 1, 1)
        self.conv3 = BasicConv(self.channel, 2, 1, 1)
        self.conv4 = BasicConv(self.channel, 2, 1, 1)
        self.conv5 = BasicConv(self.channel, 1, 2, 2)

        self.conv_rev1 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev2 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev3 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev4 = BasicConv(self.channel, 1, 1, 1)

        self.conv_sum = BasicConv(self.channel, 1, 1, 1)

    def forward(self, x, high, gi):
        # gi means global information
        y1 = self.conv1(x)
        y1 = y1 + F.interpolate(high, y1.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y3 = y3 + F.interpolate(gi, y3.shape[2:], mode='bilinear', align_corners=True)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)

        y4up = F.interpolate(y5, y4.shape[2:], mode='bilinear', align_corners=True)
        y4 = self.conv_rev1(y4 + y4up)
        y3up = F.interpolate(y4, y3.shape[2:], mode='bilinear', align_corners=True)
        y3 = self.conv_rev2(y3 + y3up)
        y2up = F.interpolate(y3, y2.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv_rev3(y2 + y2up)
        y1up = F.interpolate(y2, y1.shape[2:], mode='bilinear', align_corners=True)
        y1 = self.conv_rev4(y1 + y1up)
        y = F.interpolate(y1, x.shape[2:], mode='bilinear', align_corners=True)
        return self.conv_sum(F.relu(x + y))


class ScoreLayers(nn.Module):
    def __init__(self, channel_list):
        super(ScoreLayers, self).__init__()
        self.channel_list = channel_list
        scores = []
        for channel in self.channel_list:
            scores.append(nn.Conv2d(channel, 1, 1, 1))
        self.scores = nn.ModuleList(scores)

    def forward(self, x, x_size=None):
        for i in range(len(x)):
            x[i] = self.scores[i](x[i])
        if x_size is not None:
            for i in range(len(x)):
                x[i] = F.interpolate(x[i], x_size[2:], mode='bilinear', align_corners=True)
        return x


def extra_layer(base_model_cfg, resnet):
    config = config_resnet
    convert_layers, score_layers = [], []
    convert_layers = ConvertLayer(config['convert'])
    score_layers = ScoreLayers(config['convert'][1])
    return resnet, convert_layers, score_layers


class BPFINet(nn.Module):
    def __init__(self, base_model_cfg, base, convert_layers, score_layers):
        super(BPFINet, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.base = base
        self.score = score_layers
        self.config = config_resnet
        self.convert = convert_layers
        self.usrm3_1 = USRM3(self.config['convert'][1][4])
        self.usrm3_2 = USRM3(self.config['convert'][1][3])
        self.usrm4 = USRM4(self.config['convert'][1][2])
        self.usrm5_1 = USRM5(self.config['convert'][1][1])
        self.usrm5_2 = USRM5_2(self.config['convert'][1][0])

        self.ca43 = CA(self.config['convert'][1][3], self.config['convert'][1][2])
        self.ca42 = CA(self.config['convert'][1][3], self.config['convert'][1][1])
        self.ca41 = CA(self.config['convert'][1][3], self.config['convert'][1][0])
        self.ca32 = CA(self.config['convert'][1][2], self.config['convert'][1][1])
        self.ca21 = CA(self.config['convert'][1][1], self.config['convert'][1][0])

    def forward(self, x):
        x_size = x.size()
        C1, C2, C3, C4, C5 = self.base(x)
        if self.base_model_cfg == 'resnet':
            C1, C2, C3, C4, C5 = self.convert([C1, C2, C3, C4, C5])

        C5 = self.usrm3_1(C5)
        C5 = F.interpolate(C5, C4.shape[2:], mode='bilinear', align_corners=True)
        C4 = self.usrm3_2(C4 + C5)
        C4_att_3 = self.ca43(C4)
        C4_att_2 = self.ca42(C4)
        C4_att_1 = self.ca41(C4)
        C3 = self.usrm4(C3, C4_att_3)
        C3_att_2 = self.ca32(C3)
        C2 = self.usrm5_1(C2, C3_att_2, C4_att_2)
        C2_att_1 = self.ca21(C2)
        C1 = self.usrm5_2(C1, C2_att_1, C4_att_1)

        C1, C2, C3, C4, C5 = self.score([C1, C2, C3, C4, C5], x_size)
        return C1, C2, C3, C4, C5


def build_model(base_model_cfg='resnet'):
    return BPFINet(base_model_cfg, *extra_layer(base_model_cfg, resnet50()))


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()


def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print(model)
    print("The number of parameters: {}".format(num_params))
