import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import colorsys
import torchvision.transforms as transforms

import cv2
import numpy as np 
import matplotlib.pyplot as plt
class ImageDataSet(Dataset):

    def __init__(self, root, transforms_ = None, mode = "train"):
        self.transform = transforms.Compose(transforms_)
        self.hz_img = sorted(glob.glob(os.path.join(root, 'data/hz') + '/*.*'))
        self.gt_img = sorted(glob.glob(os.path.join(root, 'data/gt') + '/*.*'))
        # print(len(self.hz_img))
        # print(len(self.gt_img))
    def __getitem__(self, index):
        item_hz = self.transform(Image.open(self.hz_img[index % len(self.hz_img)]))
        item_gt = self.transform(Image.open(self.gt_img[index % len(self.gt_img)]))

        return {'hz' : item_hz, 'gt' : item_gt, "index":  os.path.basename(self.hz_img[index % len(self.hz_img)])  }
    def __len__(self):
        return max(len(self.hz_img), len(self.gt_img))
