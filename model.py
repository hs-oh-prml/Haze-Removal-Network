import torch.nn as nn
import torch.nn.functional as F
import torch
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True)            
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_block = 9):
        super(Generator, self).__init__()

        # Init Convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual Blocks
        for _ in range(n_residual_block):
            model += [ResidualBlock(in_features)]
        
        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size):
        super(EncoderBlock, self).__init__()
        conv_block = [
            nn.Conv2d(in_features, out_features, kernel_size),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),            
        ]
        self.conv_block = nn.Sequential(*conv_block)
    def forward(self, x):
        return self.conv_block(x)

class DownSampling(nn.Module):
    def __init__(self, in_features, out_features, kernel_size):
        super(Downsampling, self).__init_()

        
    def forward(self, x):
        return 

class UNet(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_block = 9):
        super(Unet, self).__init__()

        # Init Convolution block
        # model = [
        #     nn.ReflectionPad2d(3),
        #     nn.Conv2d(input_nc, 64, 7),
        #     nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True)
        # ]
        self.EncoderBlock1 = EncoderBlock(3, 64, 3)
        self.EncoderBlock2 = EncoderBlock(64, 128, 3)
        self.EncoderBlock3 = EncoderBlock(125, 256, 3)
        self.EncoderBlock3 = EncoderBlock(256, 512, 3)
        self.EncoderBlock3 = EncoderBlock(512, 1024, 3)

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual Blocks
        for _ in range(n_residual_block):
            model += [ResidualBlock(in_features)]
        
        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        model = [       
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)]

        self.model = nn.Sequential(
            *model,
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)