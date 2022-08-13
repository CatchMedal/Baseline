import torch
import torchvision.transforms as transforms
import albumentations as A
import numpy as np

def img2tensor(img, dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))

def transform_module(mean, std):
    trans = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std),
         A.HorizontalFlip(p=0.5),
         ]
    )
    return trans