#####################################
#                                   #
#   For Histogram Analysis Code     #
#                                   #
#####################################

import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math 
import os

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
            # nn.Tanh()
        ]
        self.init_block = nn.Sequential(*init_block)
        # Downsampling
        downSampleing1 = [
            nn.Conv2d(in_features, out_features * 2, 3, stride=2, padding=1),
            nn.GroupNorm(out_features // 2, out_features * 2),
            # nn.ReLU(inplace=True)
        ]
        downSampleing2 = [
            nn.Conv2d(in_features * 2, out_features * 4, 3, stride=2, padding=1),
            nn.GroupNorm(out_features, out_features * 4),
            # nn.ReLU(inplace=True)
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
            # nn.ReLU(inplace=True)
            ]           
        upSampling2 = [
            nn.ConvTranspose2d(in_features * 4, out_features, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(out_features // 4, out_features),
            # nn.ReLU(inplace=True)
            ]
        self.upSampling1 = nn.Sequential(*upSampling1)
        self.upSampling2 = nn.Sequential(*upSampling2)

        # Output layer
        out_block = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, 3, 7),
            # nn.Tanh()
            ]
        self.out_block = nn.Sequential(*out_block)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
    def forward(self, x):

        histogram(x[0], "input")
        idx = 0
        # for layer in x[0]:
        #     idx = idx + 1
        #     histogram(layer, "input_{}".format(idx), "input")

        init_block = self.init_block(x)
        histogram(init_block, "pre_tanh_init_block")
        init_block = self.tanh(init_block)

        idx = 0
        histogram(init_block, "post_tanh_init_block")
        # for layer in init_block[0]:
        #     idx = idx + 1
        #     histogram(layer, "init_block_{}".format(idx), "init_block")

            # print(idx)
        #     writer.add_histogram("post_init_block", layer.clone().cpu().data.numpy(), idx)

        downSampleing1 = self.downSampleing1(init_block)

        histogram(downSampleing1, "pre_relu_downSampling1")
        downSampleing1 = self.relu(downSampleing1)
        histogram(downSampleing1, "post_relu_downSampling1")

        # print(downSampleing1.shape)
        idx = 0

        # for layer in downSampleing1[0]:
        #     idx = idx + 1
        #     histogram(layer, "downSampling1_{}".format(idx), "downSampling1")

        #     writer.add_histogram("post_encoder_block1", layer.clone().cpu().data.numpy(), idx)

        downSampleing2 = self.downSampleing2(downSampleing1)

        histogram(downSampleing2, "pre_relu_downSampling2")
        downSampleing2 = self.relu(downSampleing2)
        histogram(downSampleing2, "post_relu_downSampling2")
        # idx = 0
        # for layer in downSampleing2[0]:
        #     idx = idx + 1
        #     histogram(layer, "downSampling2_{}".format(idx), "downSampling2")
            
            # writer.add_histogram("post_encoder_block2", layer.clone().cpu().data.numpy(), idx)

        residualBlock3 = self.residualBlock3(downSampleing2)
        histogram(residualBlock3, "residualBlock3")

        # idx = 0
        # for layer in residualBlock3[0]:
        #     idx = idx + 1
        #     histogram(layer, "residualBlock3_{}".format(idx), "residualBlock3")

        #     writer.add_histogram("post_residualmodule", layer.clone().cpu().data.numpy(), idx)

        upSampling1 = self.upSampling1(torch.cat([downSampleing2, residualBlock3], 1))          # 128
        histogram(upSampling1, "pre_relu_upSampling1")
        upSampling1 = self.relu(upSampling1)
        histogram(upSampling1, "post_rele_upSampling1")
        # idx = 0
        # for layer in upSampling1[0]:
        #     idx = idx + 1
        #     histogram(layer, "upSampling1_{}".format(idx), "upSampling1")
        #     writer.add_histogram("post_decoder_block1", layer.clone().cpu().data.numpy(), idx)

        upSampling2 = self.upSampling2(torch.cat([upSampling1, downSampleing1], 1))          # 64
        histogram(upSampling2, "pre_relu_upSampling2")
        upSampling2 = self.relu(upSampling2)
        histogram(upSampling2, "post_relu_upSampling2")
        # idx = 0
        # for layer in upSampling2[0]:
        #     idx = idx + 1
        #     histogram(layer, "upSampling2_{}".format(idx), "upSampling2")
            # print(layer.shape)
        #     writer.add_histogram("post_decoder_block2", layer.clone().cpu().data.numpy(), idx)

        out_block = self.out_block(upSampling2)
        histogram(out_block, "pre_tanh_out_block")
        out_block = self.tanh(out_block)
        histogram(out_block, "post_tanh_out_block")
        # idx = 0
        # for layer in out_block[0]:
        #     idx = idx + 1
        #     histogram(layer, "out_block_{}".format(idx), "out_block")

        #     writer.add_histogram("post_out_block", layer.clone().cpu().data.numpy(), idx)

        out = x + out_block

        # J = I + R
        histogram(out, "output")
        hist1 = torch.histc(x, bins=100, min=-1, max=1).clone().cpu().data.numpy()
        hist2 = torch.histc(out_block, bins=100, min=-1, max=1).clone().cpu().data.numpy()
        fig = plt.figure()
        plt.bar(np.arange(100), hist1, color='b')
        plt.bar(np.arange(100), hist2, color='r', bottom=hist1)
        plt.xticks(np.arange(100, step=10), ["%.2f" %x for x in np.arange(-1, 1, 0.2)], rotation=30)
        plt.savefig("./histogram/x_out_block.png", dpi=300)
        plt.close()

        return out

def histogram(tensor, name, dir=""):

    path = "./histogram/{}".format(dir)
    if not(os.path.isdir(path)):
        os.makedirs(os.path.join(path))

    f = open(path + "/{}.txt".format(name), 'w')

    print(tensor.shape)
    
    MAX = torch.max(tensor).cpu().data
    MIN = torch.min(tensor).cpu().data

    print(MAX, MIN)
    BINS = 100
    print(tensor.shape)
    
    hist = torch.histc(tensor, bins=BINS, min=MIN, max=MAX).clone().cpu().data.numpy()

    line = "{}\n".format(name)
    for _, i in enumerate(hist):
        line = line + "{}, ".format(i)
    line = line + "\nMIN: {}, MAX: {} BINS: {}".format(MIN, MAX, BINS)
    # f.write(line)

    fig = plt.figure()
    rng = np.arange(len(hist))

    plt.bar(rng, hist)

    if MAX - MIN == 0:
        temp = MIN.clone().cpu().data.numpy()
        plt.xticks(np.arange(len(hist), step=len(hist)/(BINS//10)), ["","","","", temp - 1, temp, temp + 1,"","",""])
    else:
        plt.xticks(np.arange(len(hist), step=len(hist)/(BINS//10)), ["%.2f" %x for x in np.arange(MIN, MAX, (MAX-MIN)/(BINS//10))], rotation=30)

    plt.savefig(path+"/{}.png".format(name), dpi=300)
    plt.close()
