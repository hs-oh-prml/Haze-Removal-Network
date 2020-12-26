import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
from torchsummary import summary
import torch
import torch.nn as nn 
import torch.nn.functional as F

from PIL import Image
import numpy as np

from model_analysis import Net1

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
train_info = "LoG5x5_analysis"

if train_info != "":
    checkpoint_dir = "./checkpoints/cp_{}".format(train_info)
    sample_dir = "./images/sample_{}".format(train_info)
else:
    checkpoint_dir = "./checkpoints/"
    sample_dir = "./images/"

if not(os.path.isdir(checkpoint_dir)):
    os.makedirs(os.path.join(checkpoint_dir))
if not(os.path.isdir(sample_dir)):
    os.makedirs(os.path.join(sample_dir))

# Edge Detection Filter
# LoG3x3 = torch.Tensor(np.array([[
#                 [
#                 [ 0, -1, -1],
#                 [ -1, 4, -1],
#                 [0,-1, 0],
#                 ],
#                 [
#                 [ 0, -1, -1],
#                 [ -1, 4, -1],
#                 [0,-1, 0],
#                 ],
#                 [
#                 [ 0, -1, -1],
#                 [ -1, 4, -1],
#                 [0,-1, 0],
#                 ]
#             ]], dtype=np.float64)).cuda()
# LoG5x5 = torch.Tensor(np.array([[
#                 [
#                 [ 0, 0, -1, 0, 0],
#                 [ 0,-1, -2,-1, 0],
#                 [-1,-2, 16,-2,-1],
#                 [ 0,-1, -2,-1, 0],
#                 [ 0, 0, -1, 0, 0],
#                 ],
#                 [
#                 [ 0, 0, -1, 0, 0],
#                 [ 0,-1, -2,-1, 0],
#                 [-1,-2, 16,-2,-1],
#                 [ 0,-1, -2,-1, 0],
#                 [ 0, 0, -1, 0, 0],
#                 ],
#                 [
#                 [ 0, 0, -1, 0, 0],
#                 [ 0,-1, -2,-1, 0],
#                 [-1,-2, 16,-2,-1],
#                 [ 0,-1, -2,-1, 0],
#                 [ 0, 0, -1, 0, 0],
#                 ]
#             ]], dtype=np.float64)).cuda()
# LoG7x7 = torch.Tensor(np.array([[
#                 [
#                 [ 0, 0,-1,-1,-1, 0, 0],
#                 [ 0,-1,-3,-3,-3,-1, 0],
#                 [-1,-3, 0, 7, 0,-3,-1],
#                 [-1,-3, 7,24, 7,-3,-1],
#                 [-1,-3, 0, 7, 0,-3,-1],
#                 [ 0,-1,-3,-3,-3,-1, 0],
#                 [ 0, 0,-1,-1,-1, 0, 0],
#                 ],
#                 [
#                 [ 0, 0,-1,-1,-1, 0, 0],
#                 [ 0,-1,-3,-3,-3,-1, 0],
#                 [-1,-3, 0, 7, 0,-3,-1],
#                 [-1,-3, 7,24, 7,-3,-1],
#                 [-1,-3, 0, 7, 0,-3,-1],
#                 [ 0,-1,-3,-3,-3,-1, 0],
#                 [ 0, 0,-1,-1,-1, 0, 0],
#                 ],
#                 [
#                 [ 0, 0,-1,-1,-1, 0, 0],
#                 [ 0,-1,-3,-3,-3,-1, 0],
#                 [-1,-3, 0, 7, 0,-3,-1],
#                 [-1,-3, 7,24, 7,-3,-1],
#                 [-1,-3, 0, 7, 0,-3,-1],
#                 [ 0,-1,-3,-3,-3,-1, 0],
#                 [ 0, 0,-1,-1,-1, 0, 0],
#                 ],
#             ]], dtype=np.float64)).cuda()

def save_images(batches_done, dataloader, net, hz, gt, dehaze):
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor    

    """Saves a generated sample from the validation set"""
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

    # edge_hz = F.conv2d(hz, LoG3x3, padding=1) 
    # edge_dhz = F.conv2d(dehaze, LoG3x3, padding=1) 
    # edge_gt = F.conv2d(gt, LoG3x3, padding=1)
    
    edge_hz = F.conv2d(hz, LoG5x5, padding=2) 
    edge_dhz = F.conv2d(dehaze, LoG5x5, padding=2) 
    edge_gt = F.conv2d(gt, LoG5x5, padding=2)

    # edge_hz = F.conv2d(hz, LoG7x7, padding=3) 
    # edge_dhz = F.conv2d(dehaze, LoG7x7, padding=3) 
    # edge_gt = F.conv2d(gt, LoG7x7, padding=3)
  
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
    params = 0
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        params = params + param
        print(name, param)
    print(params)
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

    analysis = False
    for epoch in range(opt.epoch, opt.epoch + opt.n_epochs):

        if not analysis: analysis = True

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
            dehaze = net(hz, "", False)

            if analysis : 
                analysis_img = "C:/Users/IVP/Documents/GitHub/Haze-Removal-Network/test/hz_indoor_1032.jpg"
                a_image = Image.open(analysis_img).convert('RGB')
                w, h = a_image.size
                t_info = [
                        transforms.Resize((h, w), Image.BICUBIC),
                        transforms.ToTensor(),
                    ]        
                transform = transforms.Compose(t_info)
                        
                a_image = transform(a_image).unsqueeze_(0)
                a_image = Variable(a_image.type(Tensor))
                a_image = a_image.cuda()
                name = "Epoch_{}_{}".format(epoch, analysis_img.split('/')[-1])
                net(a_image, name, analysis)
                analysis = False

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
            # Compute Edge Loss
            # edge_dhz = F.conv2d(dehaze, LoG5x5, padding=1)
            # edge_gt = F.conv2d(gt, LoG5x5, padding=1)

            edge_dhz = F.conv2d(dehaze, LoG5x5, padding=2)
            edge_gt = F.conv2d(gt, LoG5x5, padding=2)

            # edge_dhz = F.conv2d(dehaze, LoG7x7, padding=3)
            # edge_gt = F.conv2d(gt, LoG7x7, padding=3)


            loss_edge = criterion_edge(edge_dhz, edge_gt)
            loss_mse = criterion_mse(dehaze, gt)

            a = 0.01                    # lambda
            loss_edge = a * loss_edge

            loss = loss_mse + loss_edge
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