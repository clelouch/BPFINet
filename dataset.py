import os
from PIL import Image
import cv2
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import numpy as np
import random


class ImageDataTrain(data.Dataset):
    def __init__(self, data_root, data_list, img_size=352):
        self.sal_root = data_root
        self.sal_source = data_list

        # resize
        self.img_size = img_size

        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.sal_list)

    def __getitem__(self, item):
        # sal data loading
        im_name = self.sal_list[item % self.sal_num].split()[0]
        gt_name = self.sal_list[item % self.sal_num].split()[1]
        sal_image = load_image(os.path.join(self.sal_root, im_name), img_size=self.img_size)
        sal_label = load_sal_label(os.path.join(self.sal_root, gt_name), img_size=self.img_size)
        sal_image, sal_label = cv_random_flip(sal_image, sal_label)

        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)

        sample = {'sal_image': sal_image, 'sal_label': sal_label}
        return sample

    def __len__(self):
        return self.sal_num


class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list, img_size=352):
        self.data_root = data_root
        self.data_list = data_list
        self.img_size = img_size
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size = load_image_test(os.path.join(self.data_root, self.image_list[item]), img_size=self.img_size)
        image = torch.Tensor(image)

        return {'image': image, 'name': self.image_list[item % self.image_num], 'size': im_size}

    def __len__(self):
        return self.image_num


def get_loader(config, mode='train', pin=False):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(config.train_root, config.train_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle,
                                      num_workers=config.num_thread, pin_memory=pin)
    else:
        dataset = ImageDataTest(config.test_root, config.test_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle,
                                      num_workers=config.num_thread, pin_memory=pin)
    return data_loader


def load_image(path, img_size=None):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    if img_size:
        in_ = cv2.resize(in_, dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
    in_ = in_.transpose((2, 0, 1))
    return in_


def load_image_test(path, img_size=None):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    if img_size:
        in_ = cv2.resize(in_, dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
    in_ = in_.transpose((2, 0, 1))
    return in_, im_size


def load_sal_label(path, img_size=None):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    label = np.array(im, dtype=np.float32)

    if len(label.shape) == 3:
        label = label[:, :, 0]

    if img_size:
        label = cv2.resize(label, dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)

    label = label / 255.
    label = label[np.newaxis, ...]
    return label


def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img[:, :, ::-1].copy()
        label = label[:, :, ::-1].copy()
    return img, label
