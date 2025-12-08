import os

import numpy as np
import torch
from PIL import Image as Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

from .data_augment import (PairCompose, 
                          PairRandomCrop, 
                          PairRandomHorizontalFilp, 
                          PairToTensor)


class DeblurDataset(Dataset):
    def __init__(self, image_dir, patch_size=None, mode='ITS', is_valid=False, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'hazy/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.is_test = is_test
        self.is_valid = is_valid
        self.mode = mode
        self.ps = patch_size

        if self.is_valid or self.is_test:
            self.transform = None
        else:
            self.transform = PairCompose(
            [
                PairRandomCrop(patch_size),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'hazy', self.image_list[idx])).convert('RGB')
        if self.is_valid or self.is_test:
            if self.mode in ['Dense', 'NH']:
                label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'_GT.png')).convert('RGB')
            else:
                label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'.png')).convert('RGB')
        else:
            if self.mode == 'ITS':
                label = Image.open(os.path.join(self.image_dir, 'clear', self.image_list[idx].split('_')[0]+'.png')).convert('RGB')
            elif self.mode == 'OTS':
                label = Image.open(os.path.join(self.image_dir, 'clear', self.image_list[idx].split('_')[0]+'.jpg')).convert('RGB')
            elif self.mode in ['Dense', 'NH']:
                label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'_GT.png')).convert('RGB')
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Expected 'ITS' or 'OTS' or 'Dense'.")
        

        if self.ps is not None and not (self.is_valid or self.is_test):
            width, height = image.size
            if height < self.ps or width < self.ps:
                scale = max(self.ps / height, self.ps / width)
                new_height = max(self.ps, int(height * scale))
                new_width = max(self.ps, int(width * scale))
                image = image.resize((new_width, new_height), Image.BILINEAR)
                label = label.resize((new_width, new_height), Image.BILINEAR)


        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)

        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError
