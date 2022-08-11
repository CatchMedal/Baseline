import torch
import numpy as np

def img2tensor(img, dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))