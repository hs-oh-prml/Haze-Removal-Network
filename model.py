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

class InceptionBlock(nn.Module):
    def __init__(self, in_features):
        super(InceptionBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.instanceNorm = nn.InstanceNorm2d(in_features)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, in_features, 1, stride = 1),
        )
        
        self.conv3 =  nn.Sequential(
            nn.Conv2d(in_features, in_features, 1, stride = 1),
            nn.Conv2d(in_features, in_features, 3, stride = 1, padding = 1),
        )
        self.conv5 =  nn.Sequential(
            nn.Conv2d(in_features, in_features, 1, stride = 1),
            nn.Conv2d(in_features, in_features, 5, stride = 1, padding = 2),
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_features * 2, in_features, 3, 1, 1),
            nn.InstanceNorm2d(in_features),
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.InstanceNorm2d(in_features),
        )
        # self.maxpool = nn.maxpool(kernel_size=3, stride=1, padding=1)
        # self.conv7x7 = nn.Conv2d(in_features, out_features, 7, 1, 1)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv3 = self.conv3(x)
        conv5 = self.conv5(x)
        # print(conv1.shape)
        # print(conv3.shape)
        # print(conv5.shape)

        cat1 = torch.cat((conv1, conv3), 1)
        cat2 = torch.cat((conv1, conv5), 1)
        cat3 = torch.cat((conv3, conv5), 1)

        cat1 = self.relu(self.conv3_2(cat1))
        cat2 = self.relu(self.conv3_2(cat2))
        cat3 = self.relu(self.conv3_2(cat3))

        out = self.relu(self.conv3_3(cat1 + cat2 + cat3))
        return out + self.conv1(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_block = 9):
        super(Generator, self).__init__()

        # Init Convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.Tanh()
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
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = x + self.model(x)
        out = self.tanh(out)
        return out 



class Network(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_block = 5):
        super(Network, self).__init__()

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

class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseBlock, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.instanceNorm = nn.InstanceNorm2d(in_features)

        self.conv1 = nn.Conv2d(in_features, out_features, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_features, out_features, 3, 1, 1)
        self.conv3 = nn.Conv2d(out_features * 2, out_features, 3, 1, 1)

        self.conv4 = nn.Conv2d(out_features * 3, out_features, 3, 1, 1)
        self.conv5 = nn.Conv2d(out_features * 4, out_features, 3, 1, 1)

        self.out = nn.Conv2d(out_features * 3, out_features, 3, 1, 1)
    def forward(self, x):
        instanceNorm = self.instanceNorm(x)
        conv1 = self.relu(self.conv1(instanceNorm))
        # conv1 = self.instanceNorm(conv1)

        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        # conv2 = self.instanceNorm(conv2)

        c2_dense = self.relu(torch.cat([conv1, conv2], 1))

        conv3 = self.conv3(c2_dense)
        conv3 = self.relu(conv3)
        # conv3 = self.instanceNorm(conv3)

        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))

        # conv4 = self.relu(self.conv4(c3_dense))
        # c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))


        out = self.relu(self.out(c3_dense))

        # conv5 = self.relu(self.conv5(c4_dense))
        # c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        return out

class DownSampling(nn.Module):
    def __init__(self, in_features, out_features, kernel_size):
        super(Downsampling, self).__init_()

        
    def forward(self, x):
        return 

class UNet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(UNet, self).__init__()

        # Init Convolution block
        self.init_layer = EncoderBlock(3, 64, 7)

        # Downsampling
        self.EncoderBlock1 = EncoderBlock(64, 128, 3)
#        self.EncoderBlock2 = EncoderBlock(128, 256, 3)

        # DenseBlock
        d_block = []
        for _ in range(5):
            d_block += [
                DenseBlock(128, 128)                
            ]
        self.denseblock = nn.Sequential(*d_block)
        
        DenseBlock(128, 128)
        # self.d2 = DenseBlock(256, 256)
        # self.d3 = DenseBlock(256, 256)
        # self.d4 = DenseBlock(256, 256)
        # self.d5 = DenseBlock(256, 256)
        # self.d6 = DenseBlock(256, 256)
        # self.d7 = DenseBlock(256, 256)
        # self.d8 = DenseBlock(256, 256)
        # self.d9 = DenseBlock(256, 256)


        # Upsampling
#        self.DecoderBlock1 = DecoderBlock(256, 128, 3)
        self.DecoderBlock2 = DecoderBlock(128, 64, 3)
        self.DecoderBlock3 = DecoderBlock(64, 32, 3)

        self.output_layer = [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.ReLU() ]

        # # Residual Blocks
        # for _ in range(n_residual_block):
        #     model += [ResidualBlock(in_features)]
        
       

    def forward(self, x):
        encode = self.init_layer(x)
        encode = self.EncoderBlock1(encode)
#        encode = self.EncoderBlock2(encode)

        d = self.denseblock(encode)
        # d1 = self.denseblock(d)
        # d2 = self.denseblock(d1)
        # d3 = self.denseblock(d2)
        # d4 = self.denseblock(d3)
        # d5 = self.denseblock(d4)
        # d6 = self.denseblock(d5 + d3)
        # d7 = self.denseblock(d6 + d2)
        # d8 = self.denseblock(d7 + d1)

        decode = self.DecoderBlock2(d)
        decode = self.DecoderBlock3(decode)
#        decode = self.DecoderBlock3(decode)
        out = self.output_layer(decode)
        return out