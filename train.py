import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from model import Generator
from model import Discriminator

from datasets import ImageDataSet

import time
from options import opt

from utils import weights_init
import itertools

def train():
    since = time.time()

    netG = Generator(opt.input_nc, opt.output_nc)
    netD = Discriminator(opt.input_nc)
    if opt.device == 'cuda':
        netG.cuda()
        netD.cuda()

    netG.apply(weights_init)
    netD.apply(weights_init)
    # Loss
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    
    # Optimizer
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # Input
    Tensor = torch.cuda.FloatTensor if opt.device == 'cuda' else torch.FloatTensor    
    input_hz = Tensor(opt.batch_size, opt.input_nc)

    # Dataset Loader
    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]
    dataloader = DataLoader(
        ImageDataSet(opt.dataroot, transforms_=transforms_, unaligned=True),
        batch_size=opt.batch_size, 
        shuffle=True, 
        num_workers=opt.num_workers
        )

    #print('Number of training images = %d' % len(dataset))

    for epoch in range(opt.epoch, opt.n_epochs):
        print('Epoch {}/{}'.format(epoch, opt.n_epochs - 1))
        print('-' * 10)
        for i, batch in enumerate(dataloader):
            
            # input
            gt = Variable(batch['gt'].type(Tensor))
            hz = Variable(batch['hz'].type(Tensor))
            
            # Adversarial ground truths
            valid = Variable(Tensor(batch.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(batch.size(0), 1).fill_(0.0), requires_grad=False)
            
            # Generator
            optimizer_G.zero_grad()
            dehaze = netG(hz)
            pred_dehaze = netD(dehaze, gt)

            loss_gan = criterion_GAN(pred_dehaze, valid)
            loss_pixel = criterion_pixelwise(dehaze, gt)
            loss_g = loss_gan + 0.5 * loss_pixel
            loss_g.backward()
            optimizer_G.step()

            # Discriminator
            optimizer_D.zero_grad()
                
            # Real loss
            pred_real = netD(gt)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = netD(loss_gan.detach())
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(netG.state_dict(), "./checkpoints/generator_%d.pth" % (epoch))
            torch.save(netD.state_dict(), "./checkpoints/discriminator_%d.pth" % (epoch))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    train()