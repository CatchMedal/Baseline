import torch
from torch.utils.data import Dataset, DataLoader
from utils.ImageTrans import *
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.model_selection import KFold
import json

with open("config/config.json", "r") as f:
    config = json.load(f)

num_folds = config["Data"]["num_folds"]
SEED = config["Data"]["SEED"]

Labels = config["path"]["LABELS"]
image_path = config["path"]["tile_image_path"]
mask_path = config["path"]["tile_mask_path"]


class HPADataset(Dataset):
    def __init__(self, fold, train, transform=None):
        '''
        generate HPADataset instance
        :param fnames(list): filename list
        :param mask_path(str): mask file path
        :param image_path(str): image file path
        :param transform:
        '''
        ids = pd.read_csv(Labels).id.astype(str).values
        kf = KFold(n_splits=num_folds, random_state=SEED, shuffle=True)
        ids = set(ids[list(kf.split(ids))[fold][0 if train else 1]])
        self.fnames = [fname for fname in os.listdir(image_path) if fname.split('_')[0] in ids]
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
        dataset = (img, mask)
        if self.transform is not None:
            dataset = self.transform(dataset)
        return dataset