import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class ResidualBlockBottleNeck(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlockBottleNeck, self).__init__()

        conv_block = [

            nn.Conv2d(in_features, in_features // 4 , 1),
            nn.GroupNorm(in_features // 16, in_features// 4),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features // 4, in_features // 4, 3),
            nn.GroupNorm(in_features // 16, in_features // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 4, in_features, 1),
            nn.GroupNorm(in_features // 4, in_features),
            nn.ReLU(inplace=True),
        ]
        self.conv_block = nn.Sequential(*conv_block)


    def forward(self, x):
        return x + self.conv_block(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.GroupNorm(in_features // 4, in_features),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.GroupNorm(in_features // 4, in_features),
            nn.ReLU(inplace=True),
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

# GroupNorm&UNet&BottleNeck
class Net1(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_block = 16):
        super(Net1, self).__init__()

        in_features = 64
        out_features = in_features
        # Init Convolution block
        init_block = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, in_features, 7),
            nn.GroupNorm(in_features // 4, in_features),
            nn.Tanh()
        ]
        self.init_block = nn.Sequential(*init_block)
        # Downsampling
        downSampleing1 = [
            nn.Conv2d(in_features, out_features * 2, 3, stride=2, padding=1),
            nn.GroupNorm(out_features // 2, out_features * 2),
            nn.ReLU(inplace=True)
        ]
        downSampleing2 = [
            nn.Conv2d(in_features * 2, out_features * 4, 3, stride=2, padding=1),
            nn.GroupNorm(out_features, out_features * 4),
            nn.ReLU(inplace=True)
        ]

        self.downSampleing1 = nn.Sequential(*downSampleing1)
        self.downSampleing2 = nn.Sequential(*downSampleing2)

        # Residual Blocks
        residualBlock3_list = []
        for _ in range(n_residual_block):
            residualBlock3_list += [ResidualBlockBottleNeck(in_features * 4)]

        self.residualBlock3 = nn.Sequential(*residualBlock3_list)
       
        # Upsampling
        upSampling1 = [
            nn.ConvTranspose2d(in_features * 8, out_features * 2, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(out_features // 2, out_features * 2),
            nn.ReLU(inplace=True)
            ]           
        upSampling2 = [
            nn.ConvTranspose2d(in_features * 4, out_features, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(out_features // 4, out_features),
            nn.ReLU(inplace=True)
            ]
        self.upSampling1 = nn.Sequential(*upSampling1)
        self.upSampling2 = nn.Sequential(*upSampling2)

        # Output layer
        out_block = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, 3, 7),
            nn.Tanh()
            ]
        self.out_block = nn.Sequential(*out_block)

        self.tanh = nn.Tanh()
    def forward(self, x):
        init_block = self.init_block(x)
        downSampleing1 = self.downSampleing1(init_block)
        downSampleing2 = self.downSampleing2(downSampleing1)
        
        residualBlock3 = self.residualBlock3(downSampleing2)

        upSampling1 = self.upSampling1(torch.cat([downSampleing2, residualBlock3], 1))          # 128
        upSampling2 = self.upSampling2(torch.cat([upSampling1, downSampleing1], 1))          # 64

        out_block = self.out_block(upSampling2)

        out = x + out_block
        return out