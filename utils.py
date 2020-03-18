import random
import time
import sys

from torch.autograd import Variable
import torch
import numpy as np

def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))

    return image.astype(np.uint8)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)