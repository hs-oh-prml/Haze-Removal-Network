import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [

            nn.Conv2d(in_features, in_features // 4 , 1),
            # nn.InstanceNorm2d(in_features // 4),
            nn.GroupNorm(in_features // 16, in_features// 4),

            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features // 4, in_features // 4, 3),
            # nn.InstanceNorm2d(in_features),
            nn.GroupNorm(in_features // 16, in_features // 4),

            nn.ReLU(inplace=True),
            # nn.Tanh(),

            nn.Conv2d(in_features // 4, in_features, 1),
            nn.GroupNorm(in_features // 4, in_features),

            # nn.InstanceNorm2d(in_features),

            nn.ReLU(inplace=True),
            # nn.Tanh(),
            # nn.Dropout(0.5)
        ]
        self.conv_block = nn.Sequential(*conv_block)


    def forward(self, x):
        return x + self.conv_block(x)


class DenseBlock(nn.Module):
    def __init__(self, in_features):
        super(DenseBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(in_features),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_features + 32 * 1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(in_features + 32)            
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_features + 32 * 2, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(in_features + 32 * 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_features + 32 * 3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(in_features + 32 * 3)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_features + 32 * 4, in_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out1 = torch.cat([x, out1], 1)

        out2 = self.layer2(out1)
        out2 = torch.cat([out1, out2], 1)

        out3 = self.layer3(out2)
        out3 = torch.cat([out2, out3], 1)

        out4 = self.layer4(out3)
        out4 = torch.cat([out3, out4], 1)

        out5 = self.layer5(out4)
       
        return out5


class InceptionBlock(nn.Module):
    def __init__(self, in_features):
        super(InceptionBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.instanceNorm = nn.InstanceNorm2d(in_features)

        self.conv3 = nn.Conv2d(in_features, in_features, 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(in_features, in_features, 5, stride = 1, padding = 2)
        self.conv7 = nn.Conv2d(in_features, in_features, 7, stride = 1, padding = 3)

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_features * 2, in_features, 3, 1, 1),
            nn.InstanceNorm2d(in_features),
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.InstanceNorm2d(in_features),
        )
        # self.maxpool = nn.maxpool(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # print(x.shape)
        conv3 = self.conv3(x)
        conv5 = self.conv5(x)
        conv7 = self.conv7(x)

        # print(conv1.shape)
        # print(conv3.shape)
        # print(conv5.shape)

        cat1 = torch.cat((conv3, conv5), 1)
        cat2 = torch.cat((conv3, conv7), 1)
        cat3 = torch.cat((conv5, conv7), 1)

        cat1 = self.tanh(self.conv3_2(cat1))
        cat2 = self.tanh(self.conv3_2(cat2))
        cat3 = self.tanh(self.conv3_2(cat3))

        out = self.tanh(self.conv3_3(cat1 + cat2 + cat3))

        return out + x

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

            # nn.InstanceNorm2d(in_features),
            nn.Tanh()
        ]
        self.init_block = nn.Sequential(*init_block)
        # Downsampling
        # downSampleing = []
        # for _ in range(2):
        #     downSampleing += [
        #         nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
        #         nn.InstanceNorm2d(out_features),
        #         nn.ReLU(inplace=True)
        #         # nn.Tanh(),
        #     ]
 
            # in_features = out_features
            # out_features = in_features * 2
        downSampleing1 = [
            nn.Conv2d(in_features, out_features * 2, 3, stride=2, padding=1),
            nn.GroupNorm(out_features // 2, out_features * 2),

            # nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
            # nn.Tanh(),
        ]
        # in_features = out_features
        # out_features = in_features * 2        
        downSampleing2 = [
            nn.Conv2d(in_features * 2, out_features * 4, 3, stride=2, padding=1),
            nn.GroupNorm(out_features, out_features * 4),

            # nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
            # nn.Tanh(),
        ]
        # in_features = out_features
        # out_features = in_features * 2        
        # self.downSampleing1 = nn.Sequential(*downSampleing)
        self.downSampleing1 = nn.Sequential(*downSampleing1)
        self.downSampleing2 = nn.Sequential(*downSampleing2)

        
        # Residual Blocks
        residualBlock3_list = []
        # dense_block = []

        for _ in range(n_residual_block):
            residualBlock3_list += [ResidualBlock(in_features * 4)]
            # dense_block += [
            #     DenseBlock(in_features),
            #     ResidualBlock(in_features)
            # ]

        self.residualBlock3 = nn.Sequential(*residualBlock3_list)
        # self.dense_block = nn.Sequential(*dense_block)

        # Upsampling
        # out_features = in_features // 2

        upSampling = []
        # for _ in range(2):
        #     upSampling += [
        #         nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
        #         nn.InstanceNorm2d(out_features),
        #         nn.ReLU(inplace=True)
        #         # nn.Tanh()

        #         ]
        upSampling1 = [
            nn.ConvTranspose2d(in_features * 8, out_features * 2, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(out_features // 2, out_features * 2),

            # nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
            # nn.Tanh()
            ]
        # in_features = out_features
        # out_features = in_features // 2
            
        upSampling2 = [
            nn.ConvTranspose2d(in_features * 4, out_features, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(out_features // 4, out_features),

            # nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
            # nn.Tanh()

            ]

        # in_features = out_features
        # out_features = in_features // 2
        self.upSampling1 = nn.Sequential(*upSampling1)
        self.upSampling2 = nn.Sequential(*upSampling2)

        # Output layer
        out_block = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, 3, 7),
            nn.Tanh()
            ]
        self.out_block = nn.Sequential(*out_block)

        # self.model = nn.Sequential(*model)

        self.tanh = nn.Tanh()
    def forward(self, x):
#        print(x.shape)
        # print(self.model(x).shape)
#        intensity = [[(x[0][0] + x[0][1] + x[0][2]) / 3]]

        init_block = self.init_block(x)                         # 64
        downSampleing1 = self.downSampleing1(init_block)        # 128
        downSampleing2 = self.downSampleing2(downSampleing1)        # 256
        # print(downSampleing1.shape)
        # print(downSampleing2.shape)

        residualBlock3 = self.residualBlock3(downSampleing2)    # 256

        # print(residualBlock3)
        # residualBlock5 = self.residualBlock5(downSampleing)
        # residualBlock7 = self.residualBlock7(downSampleing)
        # dense_block = self.dense_block(downSampleing)

        upSampling1 = self.upSampling1(torch.cat([downSampleing2, residualBlock3], 1))          # 128
        upSampling2 = self.upSampling2(torch.cat([upSampling1, downSampleing1], 1))          # 64

        out_block = self.out_block(upSampling2)                  # 3

        # out = [[x[0][0] * out_block, x[0][1] * out_block, x[0][2] * out_block]]
        out = x + out_block
        # out = (out_block * x) - out_block + 1
        # Default

        # out = x + self.model(x)
        # out = self.tanh(out)

        # out = self.model(x)
        # Linear 
        # a = self.model(x)
        # b = self.model(x)
        # out = a * x + b

        # out = torch.clamp(out, -1, 1)

        # out = self.tanh(out)


        # test = (out_block.cpu().detach().numpy() + 1) / 2
        # test2 = (x.cpu().detach().numpy() + 1) / 2

        # plt.subplot(1,2,1)
        # plt.imshow(np.transpose(test[0], (1,2,0)), interpolation="bicubic")
        # plt.subplot(1,2,2)
        # plt.imshow(np.transpose(test2[0], (1,2,0)), interpolation="bicubic")
        # plt.show()

        # normalize [-1, 1]
        return out 

class EncoderBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size):
        super(EncoderBlock, self).__init__()
        conv_block = [
            nn.Conv2d(in_features, out_features, kernel_size, stride=2, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),            
        ]
        self.conv_block = nn.Sequential(*conv_block)
    def forward(self, x):
        return self.conv_block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size):
        super(DecoderBlock, self).__init__()
        conv_block = [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)          
        ]
        self.conv_block = nn.Sequential(*conv_block)
    def forward(self, x):
        return self.conv_block(x)

class DecoderBlockX(nn.Module):
    def __init__(self, in_features, out_features, kernel_size):
        super(DecoderBlockX, self).__init__()
        conv_block = [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)          
        ]
        self.conv_block = nn.Sequential(*conv_block)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x1, x2):

        x1 = self.up(x1)
         # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        x = torch.cat([x2, x1], dim=1)

        return self.conv_block(x)

class Net2(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_block = 11):
        super(Net2, self).__init__()

        in_features = 16
        out_features = in_features * 2
        # Init Convolution block
        model = [
            # nn.ReflectionPad2d(3),
            nn.ReplicationPad2d(3),
            nn.Conv2d(input_nc, in_features, 7),
            nn.InstanceNorm2d(in_features),
            nn.Tanh()
        ]

        # Downsampling        
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),       # 16 32, 32 64, 64 128
                nn.InstanceNorm2d(out_features),
                nn.Tanh()
            ]
            # print(in_features)
            # print(out_features)

            in_features = out_features
            out_features = in_features * 2
        
        # Residual Blocks
        for _ in range(n_residual_block):
            # print(in_features)
            # print(out_features)
            model += [ResidualBlock(in_features)]
        
        # Upsampling
        out_features = in_features // 2
    
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.Tanh()
                ]

            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            # nn.ReflectionPad2d(3),
            nn.ReplicationPad2d(3),
            nn.Conv2d(in_features, output_nc, 7),
            nn.Tanh(), 
            ]
        self.model = nn.Sequential(*model)

        self.tanh = nn.Tanh()

    def forward(self, x):
        # print(x.shape)
        # print(self.model(x).shape)
        out = x + self.model(x)
        out = self.tanh(out)


        # test = (self.model(x).cpu().detach().numpy() + 1) / 2
        # test2 = (x.cpu().detach().numpy() + 1) / 2

        # plt.subplot(1,2,1)
        # plt.imshow(np.transpose(test[0], (1,2,0)), interpolation="bicubic")
        # plt.subplot(1,2,2)
        # plt.imshow(np.transpose(test2[0], (1,2,0)), interpolation="bicubic")
        # plt.show()

        # normalize [-1, 1]
        return out 

class Network(nn.Module):

    def __init__(self, input_nc, output_nc, n_residual_block = 5):
        super(Network, self).__init__()

        # Init Convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                # nn.Tanh(inplace=True)
                nn.Tanh()

            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual Blocks
        for _ in range(n_residual_block):
            model += [InceptionBlock(in_features)]
        
        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh()
                    ]
        self.model = nn.Sequential(*model)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = x + self.model(x)
        out = self.tanh(out)
        return out 



# def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
#     """Create a generator
#     Parameters:
#         input_nc (int) -- the number of channels in input images
#         output_nc (int) -- the number of channels in output images
#         ngf (int) -- the number of filters in the last conv layer
#         netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
#         norm (str) -- the name of normalization layers used in the network: batch | instance | none
#         use_dropout (bool) -- if use dropout layers.
#         init_type (str)    -- the name of our initialization method.
#         init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
#         gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
#     Returns a generator
#     Our current implementation provides two types of generators:
#         U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
#         The original U-Net paper: https://arxiv.org/abs/1505.04597
#         Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
#         Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
#         We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).
#     The generator has been initialized by <init_net>. It uses RELU for non-linearity.
#     """
#     net = None
#     norm_layer = get_norm_layer(norm_type=norm)

#     if netG == 'resnet_9blocks':
#         net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
#     elif netG == 'resnet_6blocks':
#         net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
#     elif netG == 'unet_128':
#         net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
#     elif netG == 'unet_256':
#         net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
#     else:
#         raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
#     return init_net(net, init_type, init_gain, gpu_ids)


# class UnetGenerator(nn.Module):
#     """Create a Unet-based generator"""

#     def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
#         """Construct a Unet generator
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             output_nc (int) -- the number of channels in output images
#             num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
#                                 image of size 128x128 will become of size 1x1 # at the bottleneck
#             ngf (int)       -- the number of filters in the last conv layer
#             norm_layer      -- normalization layer
#         We construct the U-Net from the innermost layer to the outermost layer.
#         It is a recursive process.
#         """
#         super(UnetGenerator, self).__init__()
#         # construct unet structure
#         unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
#         for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
#             unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
#         # gradually reduce the number of filters from ngf * 8 to ngf
#         unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#         self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

#     def forward(self, input):
#         """Standard forward"""
#         return self.model(input)


# class UnetSkipConnectionBlock(nn.Module):
#     """Defines the Unet submodule with skip connection.
#         X -------------------identity----------------------
#         |-- downsampling -- |submodule| -- upsampling --|
#     """

#     def __init__(self, outer_nc, inner_nc, input_nc=None,
#                  submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
#         """Construct a Unet submodule with skip connections.
#         Parameters:
#             outer_nc (int) -- the number of filters in the outer conv layer
#             inner_nc (int) -- the number of filters in the inner conv layer
#             input_nc (int) -- the number of channels in input images/features
#             submodule (UnetSkipConnectionBlock) -- previously defined submodules
#             outermost (bool)    -- if this module is the outermost module
#             innermost (bool)    -- if this module is the innermost module
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers.
#         """
#         super(UnetSkipConnectionBlock, self).__init__()
#         self.outermost = outermost
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#         if input_nc is None:
#             input_nc = outer_nc
#         downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
#                              stride=2, padding=1, bias=use_bias)
#         downrelu = nn.LeakyReLU(0.2, True)
#         downnorm = norm_layer(inner_nc)
#         uprelu = nn.ReLU(True)
#         upnorm = norm_layer(outer_nc)

#         if outermost:
#             upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1)
#             down = [downconv]
#             up = [uprelu, upconv, nn.Tanh()]
#             model = down + [submodule] + up
#         elif innermost:
#             upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downrelu, downconv]
#             up = [uprelu, upconv, upnorm]
#             model = down + up
#         else:
#             upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downrelu, downconv, downnorm]
#             up = [uprelu, upconv, upnorm]

#             if use_dropout:
#                 model = down + [submodule] + up + [nn.Dropout(0.5)]
#             else:
#                 model = down + [submodule] + up

#         self.model = nn.Sequential(*model)

#     def forward(self, x):
#         if self.outermost:
#             return self.model(x)
#         else:   # add skip connections
#             return torch.cat([x, self.model(x)], 1)
