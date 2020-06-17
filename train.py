import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torch.autograd import Variable
from torchsummary import summary
import torch
import torch.nn as nn 
import torch.nn.functional as F

from PIL import Image
import numpy as np

from model import Net1
from model import Net2
from model import Net3
from model import Net4
from model import Net5
from model import Net6


from datasets import ImageDataSet

import time
from options import opt

import pytorch_ssim
import itertools

import matplotlib.pyplot as plt

import cv2
import os
import colorsys
# Information of Train
train_info = ""

if train_info != ""
    checkpoint_dir = "./checkpoints/cp_{}".format(train_info)
    sample_dir = "./images/sample_{}".format(train_info)
    log_dir = "./runs/log_{}".format(train_info)
else:
    checkpoint_dir = "./checkpoints/"
    sample_dir = "./images/"
    log_dir = "./runs/"

if not(os.path.isdir(checkpoint_dir)):
    os.makedirs(os.path.join(checkpoint_dir))
if not(os.path.isdir(sample_dir)):
    os.makedirs(os.path.join(sample_dir))
if not(os.path.isdir(log_dir)):
    os.makedirs(os.path.join(log_dir))

writer = SummaryWriter(log_dir)

def save_images(batches_done, dataloader, net, hz, gt, dehaze):
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor    

    """Saves a generated sample from the validation set"""

    # Edge Detection Filter
    LoG5x5 = torch.Tensor(np.array([[
                [
                [ 0, 0, -1, 0, 0],
                [ 0,-1, -2,-1, 0],
                [-1,-2, 16,-2,-1],
                [ 0,-1, -2,-1, 0],
                [ 0, 0, -1, 0, 0],
                ],
                [
                [ 0, 0, -1, 0, 0],
                [ 0,-1, -2,-1, 0],
                [-1,-2, 16,-2,-1],
                [ 0,-1, -2,-1, 0],
                [ 0, 0, -1, 0, 0],
                ],
                [
                [ 0, 0, -1, 0, 0],
                [ 0,-1, -2,-1, 0],
                [-1,-2, 16,-2,-1],
                [ 0,-1, -2,-1, 0],
                [ 0, 0, -1, 0, 0],
                ]
            ]], dtype=np.float64)).cuda()


    edge_hz = F.conv2d(hz, LoG5x5, padding=2) 
    edge_dhz = F.conv2d(dehaze, LoG5x5, padding=2) 
    edge_gt = F.conv2d(gt, LoG5x5, padding=2)
  
    edge_hz = torch.cat([edge_hz[0], edge_hz[0], edge_hz[0]])
    edge_dhz = torch.cat([edge_dhz[0], edge_dhz[0], edge_dhz[0]])
    edge_gt = torch.cat([edge_gt[0], edge_gt[0], edge_gt[0]])

    img_sample1 = torch.cat((hz.data, gt.data, dehaze.data), -2)
    img_sample2 = torch.cat((edge_hz.data, edge_gt.data, edge_dhz.data), -2)

    result = torch.cat((img_sample1[0], img_sample2), -1)
    save_image(result, sample_dir +"/%s.png" % (batches_done))

def train():
    
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net1(opt.input_nc, opt.output_nc)

    if opt.epoch != 0:
        net.load_state_dict(torch.load(checkpoint_dir + '/checkpoint_{}.pth'.format(opt.epoch)))
        print("Load Model Epoch : {}".format(opt.epoch))
        net.to(device)

    if torch.cuda.is_available():
        print("Use GPU")
        net = net.cuda()

    # Loss
    criterion_edge = torch.nn.L1Loss().to(device)
    criterion_pixel = torch.nn.L1Loss().to(device)
    criterion_mse = torch.nn.MSELoss().to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    parmeter = net.parameters()
    # Input
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor    
    input_hz = Tensor(opt.batch_size, opt.input_nc)

    # Dataset Loader
    transforms_ = [
        transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        transforms.ToTensor()
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

    loss_list = []

    for epoch in range(opt.epoch, opt.epoch + opt.n_epochs):
        temp_loss_list = []        
        print('Epoch {}/{}'.format(epoch + 1, opt.epoch + opt.n_epochs))
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

            # Compute Edge Loss
            LoG5x5 = torch.Tensor(np.array([[
                [
                [ 0, 0, -1, 0, 0],
                [ 0,-1, -2,-1, 0],
                [-1,-2, 16,-2,-1],
                [ 0,-1, -2,-1, 0],
                [ 0, 0, -1, 0, 0],
                ],
                [
                [ 0, 0, -1, 0, 0],
                [ 0,-1, -2,-1, 0],
                [-1,-2, 16,-2,-1],
                [ 0,-1, -2,-1, 0],
                [ 0, 0, -1, 0, 0],
                ],
                [
                [ 0, 0, -1, 0, 0],
                [ 0,-1, -2,-1, 0],
                [-1,-2, 16,-2,-1],
                [ 0,-1, -2,-1, 0],
                [ 0, 0, -1, 0, 0],
                ]
            ]], dtype=np.float64)).cuda()

            edge_dhz = F.conv2d(dehaze, LoG5x5, padding=2)
            edge_gt = F.conv2d(gt, LoG5x5, padding=2)

            edge_dhz = edge_dhz * 255
            edge_gt = edge_gt * 255
            norm_dz = dehaze * 255
            norm_gt = gt * 255

            loss_edge = criterion_edge(edge_dhz, edge_gt)
            loss_mse = criterion_mse(norm_dz, norm_gt)

            loss = loss_mse + loss_edge    # Nomalize [0, 1] 

            loss.backward()
            optimizer.step()

            #############################
            #            Log            #
            #############################
            batches_done = epoch * len(dataloader) + i
            time_left = time.time() - prev_time

            print( "[Epoch %d/%d] [Data %d/%d] [MSE: %f Edge Loss: %f TOTAL Loss: %f] [TIME: %.4s]"
            % (
                (1 + epoch),
                (opt.epoch + opt.n_epochs),
                i + 1,
                len(dataloader),
                loss_mse.item(),
                loss_edge.item(),
                loss,
                time_left
            ))
            prev_time = time.time()
            if batches_done % 100 == 0:
                save_images(batches_done, dataloader, net, hz, gt, dehaze)
            temp_loss_list.append(loss.item())               
                            

        # Save model checkpoints
        torch.save(net.state_dict(), checkpoint_dir + "/checkpoint_%d.pth" % (epoch + 1))
        print("Save model Epoch{}".format(epoch + 1))

        loss_list.append(np.mean(temp_loss_list))

    end_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))

    # Loss Graph
    plt.plot(loss_list)
    plt.show()


if __name__ == '__main__':
    train()