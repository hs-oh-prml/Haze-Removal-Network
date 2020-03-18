import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image

from PIL import Image
import torch

from model import Generator
from model import Discriminator

from datasets import ImageDataSet

import time
from options import opt

from utils import weights_init
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

    netG = Generator(opt.input_nc, opt.output_nc)
    netD = Discriminator(opt.input_nc)
    if torch.cuda.is_available():
        print("Use GPU")
        netG = netG.cuda()
        netD = netD.cuda()

    netG.apply(weights_init)
    netD.apply(weights_init)
    # Loss
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    criterion_bce = torch.nn.BCELoss()
    
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

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        print('Epoch {}/{}'.format(epoch, opt.n_epochs - 1))
        print('-' * 10)
        for i, batch in enumerate(dataloader):
            
            # input
            gt = Variable(batch['gt'].type(Tensor))
            hz = Variable(batch['hz'].type(Tensor))
            
            # Adversarial ground truths

            valid = Variable(Tensor(gt.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(hz.size(0), 1).fill_(0.0), requires_grad=False)
            
            # Generator
            optimizer_G.zero_grad()
            dehaze = netG(hz)
            pred_dehaze = netD(dehaze)

            # loss_gan = criterion_GAN(pred_dehaze, valid)
            # loss_pixel = criterion_pixelwise(dehaze, gt)
            # loss_G = loss_gan + 0.5 * loss_pixel
            loss_G = criterion_GAN(dehaze, gt)
            loss_G.backward()
            optimizer_G.step()

            # Discriminator
            optimizer_D.zero_grad()
                
            # Real loss
            pred_gt = netD(gt)
            # loss_real = criterion_GAN(pred_real, valid)

            loss_gt = criterion_GAN(pred_gt, valid)


            # Fake loss
            pred_hz = netD(dehaze.detach())
            # loss_fake = criterion_GAN(pred_fake, fake)
            loss_hz = criterion_GAN(pred_hz, fake)

            # Total loss
            loss_D = 0.5 * (loss_gt + loss_hz)

            loss_D.backward()
            optimizer_D.step()

            batches_done = epoch * len(dataloader) + i
            time_left = time.time() - prev_time
            print( "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                time_left
            ))
            prev_time = time.time()
            
            if batches_done % 10 == 0:
                save_images(batches_done, dataloader, netG)


        # if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(netG.state_dict(), "./checkpoints/generator_%d.pth" % (epoch))
        torch.save(netD.state_dict(), "./checkpoints/discriminator_%d.pth" % (epoch))


    end_time = time.time() - prev_time
    print('Training complete in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))


if __name__ == '__main__':
    train()