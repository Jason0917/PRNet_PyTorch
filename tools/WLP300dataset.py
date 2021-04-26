# -*- coding: utf-8 -*-
"""
    @author: samuel ko
    @date: 2019.07.18
    @readme: The implementation of PRNet Network DataLoader.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

import cv2
from glob import glob
import random
import numbers
import numpy as np
from PIL import Image
from skimage import io

from config.config import FLAGS

data_transform = {'train': transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}


class PRNetDataset(Dataset):
    """Pedestrian Attribute Landmarks dataset.""" 

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.dict = dict()
        self._max_idx()
        self.stack_size = FLAGS["stack_size"]
        self.len = len(os.listdir(self.root_dir))
        print('lennnn',self.len)
        

    def get_img_path(self, img_id):
        img_id = self.dict.get(img_id)
        print("img_id",img_id)
        if img_id>=0:
            original = os.path.join(self.root_dir, str(img_id), 'original.jpg')
            # fixme: Thanks to mj, who fix an important bug!
            uv_map_path = glob(os.path.join(self.root_dir, str(img_id), "*.npy"))
            uv_map = uv_map_path[0]

            return original, uv_map

    def _max_idx(self):
        _tmp_lst = map(lambda x: int(x), os.listdir(self.root_dir))
        _sorted_lst = sorted(_tmp_lst)
        for idx, item in enumerate(_sorted_lst):
            self.dict[idx] = item

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        print('lennn',self.len)
        print("idx",idx)
        #offset = (self.stack_size - 1) / 2
        offset = 1
        if idx == self.len - 1:
            print('1111')
            pre_idx = idx - 2 * offset
            next_idx = idx
            idx = idx - offset
        elif idx-offset >= 0:
            pre_idx = idx - offset
            next_idx = idx + offset
        elif idx == 0 :
            pre_idx = idx
            idx = idx + offset
            next_idx = idx + offset

        print("pre_idx",pre_idx)
        print("next_idx",next_idx)

        pre_original, pre_uv_map = self.get_img_path(pre_idx)
        original, uv_map = self.get_img_path(idx)
        next_original, next_uv_map = self.get_img_path(next_idx)

        pre_origin = cv2.imread(pre_original)
        origin = cv2.imread(original)
        next_origin = cv2.imread(next_original)

        pre_uv_map = np.load(pre_uv_map)
        uv_map = np.load(uv_map)
        next_uv_map = np.load(next_uv_map)

        sample = {'uv_map': uv_map, 'origin': origin,
                  'pre_uv_map': pre_uv_map, 'pre_origin': pre_origin,
                  'next_uv_map': next_uv_map, 'next_origin': next_origin}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        pre_uv_map, pre_origin = sample['pre_uv_map'], sample['pre_origin']
        uv_map, origin = sample['uv_map'], sample['origin']
        next_uv_map, next_origin = sample['next_uv_map'], sample['next_origin']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        pre_uv_map = pre_uv_map.transpose((2, 0, 1))
        uv_map = uv_map.transpose((2, 0, 1))
        next_uv_map = next_uv_map.transpose((2, 0, 1))
        pre_origin = pre_origin.transpose((2, 0, 1))
        origin = origin.transpose((2, 0, 1))
        next_origin = next_origin.transpose((2, 0, 1))

        pre_uv_map = pre_uv_map.astype("float32") / 255.
        pre_uv_map = np.clip(pre_uv_map, 0, 1)
        pre_origin = pre_origin.astype("float32") / 255.
        uv_map = uv_map.astype("float32") / 255.
        uv_map = np.clip(uv_map, 0, 1)
        origin = origin.astype("float32") / 255.
        next_uv_map = next_uv_map.astype("float32") / 255.
        next_uv_map = np.clip(next_uv_map, 0, 1)
        next_origin = next_origin.astype("float32") / 255.

        return {'pre_uv_map': torch.from_numpy(pre_uv_map), 'pre_origin': torch.from_numpy(pre_origin),
                'uv_map': torch.from_numpy(uv_map), 'origin': torch.from_numpy(origin),
                'next_uv_map': torch.from_numpy(next_uv_map), 'next_origin': torch.from_numpy(next_origin)}


class ToNormalize(object):
    """Normalized process on origin Tensors."""

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        pre_uv_map, pre_origin = sample['pre_uv_map'], sample['pre_origin']
        uv_map, origin = sample['uv_map'], sample['origin']
        next_uv_map, next_origin = sample['uv_map'], sample['origin']
        pre_origin = F.normalize(pre_origin, self.mean, self.std, self.inplace)
        origin = F.normalize(origin, self.mean, self.std, self.inplace)
        next_origin = F.normalize(next_origin, self.mean, self.std, self.inplace)
        return {'pre_uv_map': pre_uv_map, 'pre_origin': pre_origin,
                'uv_map': uv_map, 'origin': origin,
                'next_uv_map': next_uv_map, 'next_origin': next_origin}
