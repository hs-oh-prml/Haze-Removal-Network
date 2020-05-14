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
            nn.Tanh(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.Tanh()
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

class Net1(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_block = 9):
        super(Net1, self).__init__()

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
                nn.Tanh()
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
                        nn.Tanh()
                        ]

            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() 
                    ]
        self.model = nn.Sequential(*model)
        # final = [
        #     nn.Conv2d(3, 3, kernel_size=1),
        #     nn.Tanh()
        # ]
        # self.final_block = nn.Sequential(*final)
        self.tanh = nn.Tanh()
        

    def forward(self, x):
        # print(x.shape)
        # print(self.model(x).shape)
        out = x + self.model(x)
        out = self.tanh(out)
        # out = self.final_block(out)
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

        # Init Convolution block
        init = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.Tanh()
        ]
        self.init_block = nn.Sequential(*init)

        # Downsampling
        in_features = 64
        out_features = in_features * 2

        self.encoder1 = EncoderBlock(64, 128, 3)
        self.encoder2 = EncoderBlock(128, 256, 3)
        self.encoder3 = EncoderBlock(256, 512, 3)

        # Residual Blocks
        res = []
        for _ in range(n_residual_block):
            res += [ResidualBlock(512)]
        self.residual = nn.Sequential(*res)
        
        # Upsampling
        self.decoder1 = DecoderBlock(512, 256, 3)
        self.decoder2 = DecoderBlock(256, 128, 3)
        self.decoder3 = DecoderBlock(128, 64, 3)

        # Output layer
        last = [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]
        self.last_block = nn.Sequential(*last)

        self.tanh = nn.Tanh()

    def forward(self, x):
        init = self.init_block(x)                # 3, 64
        encode1 = self.encoder1(init)            # 64, 128
        # print(encode1.shape)
        encode2 = self.encoder2(encode1)        # 128, 256
        # print(encode2.shape)
        encode3 = self.encoder3(encode2)        # 256, 512
        # print(encode3.shape)

        residual = self.residual(encode3)       # 512
        # print(residual.shape)

        out = self.decoder1(residual) + encode2           # 512, 512, 256
        # print(out.shape)
        out = self.decoder2(out) + encode1                # 256, 256, 128
        # print(out.shape)
        out = self.decoder3(out) + init               # 128, 128, 64
        # print(out.shape)
        out = self.last_block(out)              # 64, 3
        # print(out.shape)
        out = self.tanh(x + out)        
        # print(out.shape)

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