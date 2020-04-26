import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
import torch

from PIL import Image
import numpy as np

from model import Generator
from model import UNet
from model import Network


from datasets import ImageDataSet

import time
from options import opt

from utils import weights_init
import pytorch_ssim
import itertools

import time



def save_images(batches_done, dataloader, net):
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor    

    """Saves a generated sample from the validation set"""
    imgs = next(iter(dataloader))
    hz = Variable(imgs["hz"].type(Tensor))
    gt = Variable(imgs["gt"].type(Tensor))
    dehaze = net(hz)
    img_sample = torch.cat((hz.data, gt.data, dehaze.data), -2)
    save_image(img_sample, "images/%s_{}.png".format(imgs["index"]) % (batches_done), nrow=5, normalize=True)


def train():

    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # net = Network(opt.input_nc, opt.output_nc)
    net = Generator(opt.input_nc, opt.output_nc)
    # UNet = UNet(opt.input_nc, opt.output_nc)

    if opt.epoch != 0:
        net.load_state_dict(torch.load('./checkpoints/generator_{}.pth'.format(opt.epoch)))
        print("Load Model Epoch : {}".format(opt.epoch))
        net.to(device)

    if torch.cuda.is_available():
        print("Use GPU")
        net = net.cuda()
    net.apply(weights_init)
    # Loss
    criterion_content = torch.nn.MSELoss().to(device)
    criterion_pixel = torch.nn.L1Loss().to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # Input
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor    
    input_hz = Tensor(opt.batch_size, opt.input_nc)

    # Dataset Loader
    transforms_ = [
        transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]
    dataloader = DataLoader(
        ImageDataSet(opt.dataroot, transforms_=transforms_),
        batch_size=opt.batch_size, 
        shuffle=True,
        num_workers=opt.num_workers
    )
    print("Number of training images: {}".format(len(dataloader)))

    start_time = time.time() 
    prev_time = start_time
    for epoch in range(opt.epoch, opt.epoch + opt.n_epochs):
        print('Epoch {}/{}'.format(epoch + 1, opt.n_epochs))
        print('-' * 70)
        for i, batch in enumerate(dataloader):
            
            # input
            gt = Variable(batch['gt'].type(Tensor))
            hz = Variable(batch['hz'].type(Tensor))
            
            #############################
            #         NetWork         #
            #############################
            optimizer.zero_grad()
            dehaze = net(hz)

            loss_l1 = criterion_pixel(dehaze, gt)
            # loss_mse = criterion_content(dehaze, gt)
            loss_ssim = 1 - pytorch_ssim.ssim(dehaze, gt)            

            loss = loss_ssim + loss_l1
            loss.backward()
            optimizer.step()

            #############################
            #            Log            #
            #############################
            batches_done = epoch * len(dataloader) + i
            time_left = time.time() - prev_time

            print( "[Epoch %d/%d] [Data %d/%d] [SSIM Loss: %f L1 Loss: %f TOTAL Loss: %f] [TIME: %.4s]"
            % (
                (1 + epoch),
                opt.n_epochs,
                i,
                len(dataloader),
                loss_ssim.item(),
                loss_l1.item(),
                loss,
                time_left
            ))
            prev_time = time.time()
            
            if batches_done % 100 == 0:
                save_images(batches_done, dataloader, net)

        # Save model checkpoints
        torch.save(net.state_dict(), "./checkpoints/checkpoint_%d.pth" % (epoch + 1))
        print("Save model Epoch{}".format(epoch))

    end_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))


if __name__ == '__main__':
    train()