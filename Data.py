import torch
from torch.utils.data import Dataset, DataLoader
from utils.ImageTrans import *
import numpy as np
import cv2
import os

class HPADataset(Dataset):
    def __init__(self, fnames, mask_path, image_path, transform=None):
        '''
        generate HPADataset instance
        :param fnames(list): filename list
        :param mask_path(str): mask file path
        :param image_path(str): image file path
        :param transform:
        '''
        self.fnames = fnames
        self.transform = transform
        self.image_path = image_path
        self.mask_path = mask_path

    def __len__(self):
        '''
        get length
        :return(int): files length
        '''
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.image_path, fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_path, fname), cv2.IMREAD_GRAYSCALE)
        dataset = {'image': img, 'mask': mask}
        if self.transfer is not None:
            dataset = self.transform(dataset)
        return dataset