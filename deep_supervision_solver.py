import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam
from torch.backends import cudnn
from deep_supervision_model import build_model, weights_init
import scipy.misc as sm
import numpy as np
import os
import torchvision.utils as vutils
import cv2
import math
import time
from loss import TriLoss

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)
    return dn


Dataset_dict = {'e': 'ECSSD', 'p': 'PASCALS', 'd': 'DUTOMRON', 'h': 'HKU-IS','s': 'SOD','t': 'DUTS_TE'}


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.lr_decay_epoch = [20, ]
        self.build_model()
        self.loss = TriLoss(sigma=self.config.sigma, n=self.config.weight)
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            if self.config.cuda:
                self.net.load_state_dict(torch.load(self.config.model))
            else:
                self.net.load_state_dict(torch.load(self.config.model, map_location='cpu'))
            self.net.eval()

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def deep_supervision_loss(self, preds, gt):
        losses = []
        for pred in preds:
            wbce_, iou_, rev_iou_ = self.loss(pred, gt)
            losses.append((wbce_ + 0.5 * (iou_ + rev_iou_)).mean())
        sum_loss = losses[0] + losses[1] / 2 + losses[2] / 4 + losses[3] /8 + losses[4] / 8
        return sum_loss

    # build the network
    def build_model(self):
        self.net = build_model(self.config.arch)
        if self.config.cuda:
            self.net = self.net.cuda()

        self.net.train()
        # self.net.eval()  # use_global_stats = True
        self.net.apply(weights_init)
        if self.config.load == '':
            self.net.base.load_state_dict(torch.load(self.config.pretrained_model))
        else:
            self.net.load_state_dict(torch.load(self.config.load))

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr,
                              weight_decay=self.wd)
        # self.print_network(self.net, 'AGBNet Structure')

    def test(self):
        self.net.eval()
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            im_size = im_size[1], im_size[0]
            # print(im_size)
            with torch.no_grad():
                if self.config.cuda:
                    images = images.cuda()
                preds = self.net(images)[0]
                preds = normPRED(torch.sigmoid(preds))
                pred = np.squeeze(preds.cpu().data.numpy())
                multi_fuse = 255 * pred
                multi_fuse = cv2.resize(multi_fuse, dsize=im_size, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '.png'), multi_fuse)
        print(Dataset_dict[self.config.sal_mode] + ' Test Done!')

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        for epoch in range(self.config.epoch):
            self.net.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_label = data_batch['sal_image'], data_batch['sal_label']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                if self.config.cuda:
                    # cudnn.benchmark = True
                    sal_image, sal_label = sal_image.cuda(), sal_label.cuda()

                sal_preds = self.net(sal_image)
                sum_loss = self.deep_supervision_loss(sal_preds, sal_label)

                self.optimizer.zero_grad()
                sum_loss.backward()
                self.optimizer.step()

                if i % (self.show_every // self.config.batch_size) == 0:
                    if i == 0:
                        x_showEvery = 1
                    print(
                        'epoch: [%2d/%2d], iter: [%5d/%5d]  ||  sum_loss : %10.4f' % (
                            epoch, self.config.epoch, i, iter_num, sum_loss.data))
                    print('Learning rate: ' + str(self.lr))

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr,
                                      weight_decay=self.wd)

        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_folder)
