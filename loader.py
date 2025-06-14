import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class PairedImageDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, img_size=(32, 32)):
        self.lr_paths = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir)])
        self.hr_paths = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir)])
        self.img_size = img_size

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr_img = cv2.imread(self.lr_paths[idx], cv2.IMREAD_GRAYSCALE)
        hr_img = cv2.imread(self.hr_paths[idx], cv2.IMREAD_GRAYSCALE)
        lr_img = cv2.resize(lr_img, self.img_size).astype(np.float32) / 255.0
        hr_img = cv2.resize(hr_img, (300, 300)).astype(np.float32) / 255.0
        return lr_img[np.newaxis, ...], hr_img[np.newaxis, ...]
