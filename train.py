import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
import torch

from PIL import Image
import numpy as np
from model import Generator
from model import Discriminator

from datasets import ImageDataSet

import time
from options import opt

from utils import weights_init
import pytorch_ssim
import itertools

import time



def save_images(batches_done, dataloader, generator):
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor    

    """Saves a generated sample from the validation set"""
    imgs = next(iter(dataloader))
    hz = Variable(imgs["hz"].type(Tensor))
    gt = Variable(imgs["gt"].type(Tensor))
    dehaze = generator(hz)
    img_sample = torch.cat((hz.data, gt.data, dehaze.data), -2)
    save_image(img_sample, "images/%s.png" % (batches_done), nrow=5, normalize=True)


def train():

    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    netG = Generator(opt.input_nc, opt.output_nc)
    netD = Discriminator(opt.input_nc)

    # if opt.epoch != 0:
    #     netG.load_state_dict(torch.load('C:/Users/user/Desktop/dehaze/checkpoints/try3_epoch100_of/generator_{}.pth'.format(opt.epoch)))
    #     print("Load Model Epoch : {}".format(opt.epoch))
    #     netG.to(device)

    if torch.cuda.is_available():
        print("Use GPU")
        netG = netG.cuda()
        netD = netD.cuda()

    netG.apply(weights_init)
    netD.apply(weights_init)
    # Loss
    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_content = torch.nn.MSELoss().to(device)
    criterion_pixel = torch.nn.L1Loss().to(device)
    
    # Optimizer
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))

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
            
            # Adversarial ground truths

            # valid = Variable(Tensor(gt.size(0), 1).fill_(1.0), requires_grad=False)
            # fake = Variable(Tensor(hz.size(0), 1).fill_(0.0), requires_grad=False)
            valid = Variable(Tensor(np.ones((gt.size(0), 1, 30, 40))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((hz.size(0), 1, 30, 40))), requires_grad=False)

            #############################
            #         Generator         #
            #############################
            optimizer_G.zero_grad()
            dehaze = netG(hz)

            loss_pixel = criterion_pixel(dehaze, gt)
            loss_mse = criterion_content(dehaze, gt)
            # pred_real = netD(dehaze)
            # pred_fake = netD(gt)

            # print(pred_real.shape)
            # print(valid.shape)
            
            # loss_gan = criterion_GAN(netD(dehaze), valid)
            loss_ssim = 1 - pytorch_ssim.ssim(dehaze, gt)


            # loss_G = loss_gan + loss_pixel + loss_ssim
            loss_G = loss_pixel + loss_ssim + loss_mse

            loss_G.backward()
            optimizer_G.step()

            # #############################
            # #       Discriminator       #
            # #############################

            # optimizer_D.zero_grad()
                
            # # pred_gt = netD(gt)
            # # pred_hz = netD(dehaze.detach())
            
            # loss_gt = criterion_GAN(netD(gt), valid)
            # loss_hz = criterion_GAN(netD(dehaze.detach()), fake)
            
            # # Total loss
            # loss_D = 0.5 * (loss_gt + loss_hz)

            # loss_D.backward()
            # optimizer_D.step()

            #############################
            #            Log            #
            #############################
            batches_done = epoch * len(dataloader) + i
            time_left = time.time() - prev_time
            # print( "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f SSIM: %f] Time: %.3s"
            print( "[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %f SSIM Loss: %f MSE Loss: %f]"
            % (
                (1 + epoch),
                opt.n_epochs,
                i,
                len(dataloader),
                # loss_D.item(),
                # loss_G.item(),
                loss_pixel.item(),
                loss_ssim.item(),
                loss_mse.item(),

            ))
            prev_time = time.time()
            
            if batches_done % 100 == 0:
                save_images(batches_done, dataloader, netG)


        # if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(netG.state_dict(), "./checkpoints/generator_%d.pth" % (epoch))
        # torch.save(netD.state_dict(), "./checkpoints/discriminator_%d.pth" % (epoch))
        print("Save model Epoch{}".format(epoch))

    end_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))


if __name__ == '__main__':
    train()