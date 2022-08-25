import torch
import torchvision.transforms as transforms
import albumentations as A
import numpy as np
import cv2

def img2tensor(img, dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))

def transform_module():
    return A.Compose(
        [
         A.HorizontalFlip(p=0.5),
         ]
    )

def save_img(data, name, out):
    data = data.float().cpu().numpy()
    # 왜 인코딩 하는지 확인 필요하며 어떻게 인코딩 되는지도 확인이 필요함
    img = cv2.imencode('.png', (data*255).astype(np.uint8))[1]
    out.writestr(name, img)