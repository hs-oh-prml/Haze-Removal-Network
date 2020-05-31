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
from model import Network
from model import Net2


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
# train_info = "LoG3x3_l1_19_rb16_kernel16"
# train_info = "LoG3x3_l1_19_rb16_kernel16"
# train_info = "ReLU_LoG3x3_l1_19_rb16_kernel11"
# train_info = "ReLU_LoG3x3_l1_19_rb16_kernel11_3"
# train_info = "ReLU_LoG3x3_l1_19_rb32_kernel5"
# train_info = "LoG3x3_l1_19_rb16_kernel16_2_double_ed_2"
# train_info = "Reverse_ED"
# train_info = "origin"
# train_info = "DoubleED_rb24"
# train_info = "origin_2"
# train_info = "bottle_neck_64_unet_mse"
# train_info = "bottle_neck_64_unet_mse_group_norm"
train_info = "./"
checkpoint_dir = "./checkpoints/cp_{}".format(train_info)
sample_dir = "./images/sample_{}".format(train_info)
log_dir = "./runs/log_{}".format(train_info)
# 'runs/log_all_tanh_LoG3x3_l1_19_rb16'
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
    sobel = torch.Tensor(np.array([[
                [
                [-1,0,1],
                [-2,0,2],                
                [-1,0,1]
                ],
                [
                [-1,0,1],
                [-2,0,2],                
                [-1,0,1]
                ],
                [
                [-1,0,1],
                [-2,0,2],                
                [-1,0,1]
                ],
            ]], dtype=np.float64)).cuda()

    LoG3x3 = torch.Tensor(np.array([[
                [
                [-1,-1,-1],
                [-1,8,-1],                
                [-1,-1,-1]
                ],
                [
                [-1,-1,-1],
                [-1,8,-1],                
                [-1,-1,-1]
                ],
                [
                [-1,-1,-1],
                [-1,8,-1],                
                [-1,-1,-1]
                ],
            ]], dtype=np.float64)).cuda()

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
    # threshold = nn.Threshold(0.1, 1, inplace=False)
    # edge_hz = threshold(edge_hz)
    # edge_dhz = threshold(edge_dhz)
    # edge_gt = threshold(edge_gt)

    # edge_hz = F.conv2d(hz, sobel, padding=1) 
    # edge_dhz = F.conv2d(dehaze, sobel, padding=1) 
    # edge_gt = F.conv2d(gt, sobel, padding=1) 

    edge_hz = torch.cat([edge_hz[0], edge_hz[0], edge_hz[0]])
    edge_dhz = torch.cat([edge_dhz[0], edge_dhz[0], edge_dhz[0]])
    edge_gt = torch.cat([edge_gt[0], edge_gt[0], edge_gt[0]])

    img_sample1 = (torch.cat((hz.data, gt.data, dehaze.data), -2) + 1) / 2
    img_sample2 = torch.cat((edge_hz.data, edge_gt.data, edge_dhz.data), -2)
    # img_sample1 = torch.cat((hz.data, gt.data, dehaze.data), -2)
    # img_sample2 = torch.cat((edge_hz.data, edge_gt.data, edge_dhz.data), -2)
    # print(img_sample1.shape)
    # print(img_sample2.shape)

    result = torch.cat((img_sample1[0], img_sample2), -1)
    # result = np.transpose(result.cpu().detach().numpy(), (1,2,0))
    # result = Image.fromarray(result, 'HSV').convert('RGB')
    # result = colorsys.hsv_to_rgb(result[0].all(), result[1].all(), result[2].all())

    # imgs["index"]
    # result.save(sample_dir +"/%s.jpg" % (batches_done))
    save_image(result, sample_dir +"/%s.png" % (batches_done))
    # writer.add_image("samples/%s.png" % (batches_done), result)

def train():
    
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = Net2(opt.input_nc, opt.output_nc)
    # net = Network(opt.input_nc, opt.output_nc)
    net = Net1(opt.input_nc, opt.output_nc)
    # UNet = UNet(opt.input_nc, opt.output_nc)

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
    # print("Parameters: {}".format(parmeter.size()))
    # print("Parameters: {}".format(net.summary()))
    # summary(net, input_size=(3, 480, 640))

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

    loss_list = []

    for epoch in range(opt.epoch, opt.epoch + opt.n_epochs):
        temp_loss_list = []        
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

            # Compute Edge Loss
            sobel = torch.Tensor(np.array([[
                [
                [-1,0,1],
                [-2,0,2],                
                [-1,0,1]
                ],
                [
                [-1,0,1],
                [-2,0,2],                
                [-1,0,1]
                ],
                [
                [-1,0,1],
                [-2,0,2],                
                [-1,0,1]
                ],
            ]], dtype=np.float64)).cuda()

            Gaussian3x3 = torch.Tensor(np.array([[
                [
                [1/16,2/16,1/16],
                [2/16,4/16,2/16],
                [1/16,2/16,1/16],
                ],
                [
                [1/16,2/16,1/16],
                [2/16,4/16,2/16],
                [1/16,2/16,1/16],
                ],
                [
                [1/16,2/16,1/16],
                [2/16,4/16,2/16],
                [1/16,2/16,1/16],
                ],
            ]], dtype=np.float64)).cuda()
            LoG3x3 = torch.Tensor(np.array([[
                [
                [-1,-1,-1],
                [-1,8,-1],                
                [-1,-1,-1]
                ],
                [
                [-1,-1,-1],
                [-1,8,-1],                
                [-1,-1,-1]
                ],
                [
                [-1,-1,-1],
                [-1,8,-1],                
                [-1,-1,-1]
                ],
            ]], dtype=np.float64)).cuda()

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
            edge_dhz = ((edge_dhz + 1) / 2) * 255
            edge_gt = ((edge_gt + 1) / 2) * 255

            # threshold = nn.Threshold(0.1, 1, inplace=False)
            # edge_dhz = threshold(edge_dhz)
            # edge_gt = threshold(edge_gt)
            # edge_dhz = F.conv2d(dehaze, LoG5x5, padding=2)
            # edge_gt = F.conv2d(gt, LoG5x5, padding=2)
            norm_dz = ((dehaze + 1) / 2) * 255
            norm_gt = ((gt + 1) / 2) * 255

            loss_l1 = criterion_pixel(dehaze, gt)
            ssim = pytorch_ssim.ssim(dehaze, gt)
            loss_edge = criterion_edge(edge_dhz, edge_gt)
            loss_mse = criterion_mse(norm_dz, norm_gt)
            # loss_l1 = loss_l1                   # L1 Loss: [0, 1]   
            # loss_ssim = loss_ssim               # SSIM Loss: [0, 1]
                                                  # 0 : 1 : 1
            # alpha = 0.9
            alpha = 0.95

            # if epoch > 0:
            #     previous_loss = loss_list[-1]
            #     if previous_loss < 0.1:
            #         alpha = 0.95
            #     if previous_loss < 0.05:
            #         alpha = 1

            loss = alpha * loss_mse + (1 - alpha) * loss_edge    # Nomalize [0, 1] 
            # loss = loss_mse
            loss.backward()
            optimizer.step()

            #######################################################################
            ############################### To Test ############################### 
            ############################## Show Edge ############################## 
            #######################################################################
            # edge_hz = F.conv2d(hz, LoG5x5, padding=2)
            # edge_hz = F.conv2d(hz, LoG3x3, padding=1)
            # edge_hz = F.conv2d(hz, sobel, padding=1)

            # threshold = nn.Threshold(-0.99, 1, inplace=False)
            # edge_dhz = threshold(edge_dhz)
            # edge_hz = threshold(edge_hz)
            # edge_gt = threshold(edge_gt)

            # threshold2 = nn.Threshold(-0.9, 1, inplace=False)
            # edge_dhz = threshold2(edge_dhz)
            # edge_hz = threshold2(edge_hz)
            # edge_gt = threshold2(edge_gt)


            # test_hz = edge_hz.cpu().detach().numpy()
            # test_hz = np.array([test_hz[0][0], test_hz[0][0], test_hz[0][0]])
            # test_dhz = edge_dhz.cpu().detach().numpy()
            # test_dhz = np.array([test_dhz[0][0], test_dhz[0][0], test_dhz[0][0]])
            # test_gt = edge_gt.cpu().detach().numpy()
            # test_gt = np.array([test_gt[0][0], test_gt[0][0], test_gt[0][0]])

            # print(loss_edge.data)
            # print(loss_l1.data)

            # plt.subplot(2,3,1)
            # plt.imshow(np.transpose(test_hz, (1,2,0)), interpolation="bicubic")
            # plt.subplot(2,3,2)
            # plt.imshow(np.transpose(test_dhz, (1,2,0)), interpolation="bicubic")
            # plt.subplot(2,3,3)
            # plt.imshow(np.transpose(test_gt, (1,2,0)), interpolation="bicubic")
            # plt.subplot(2,3,4)
            # plt.imshow((np.transpose(hz.cpu().detach().numpy()[0], (1,2,0)) + 1) / 2, interpolation="bicubic")
            # plt.subplot(2,3,5)
            # plt.imshow((np.transpose(dehaze.cpu().detach().numpy()[0], (1,2,0)) + 1) / 2, interpolation="bicubic")
            # plt.subplot(2,3,6)
            # plt.imshow((np.transpose(gt.cpu().detach().numpy()[0], (1,2,0)) + 1) / 2, interpolation="bicubic")
            # plt.show()
            # break
            #######################################################################

            #############################
            #            Log            #
            #############################
            batches_done = epoch * len(dataloader) + i
            time_left = time.time() - prev_time

            print( "[Epoch %d/%d] [Data %d/%d] [L1 Loss: %f MSE Loss: %f Edge Loss: %f TOTAL Loss: %f] [TIME: %.4s]"
            % (
                (1 + epoch),
                opt.n_epochs,
                i + 1,
                len(dataloader),
                # loss_ssim.item(),
                loss_l1.item(),
                loss_mse.item(),
                
                loss_edge.item(),
                loss,
                time_left
            ))
            prev_time = time.time()
            if batches_done % 100 == 0:
                save_images(batches_done, dataloader, net, hz, gt, dehaze)
                writer.add_scalar('Training Loss(L1: 9, Edge: 1)',
                                loss,
                                epoch * len(dataloader) + i)
                writer.add_scalar('L1 Loss',
                                loss_l1,
                                epoch * len(dataloader) + i)
                writer.add_scalar('Edge Loss',
                                loss_edge,
                                epoch * len(dataloader) + i)
                writer.add_scalar('SSIM',
                                ssim,
                                epoch * len(dataloader) + i)

            temp_loss_list.append(loss.item())               
                            

        # Save model checkpoints
        torch.save(net.state_dict(), checkpoint_dir + "/checkpoint_%d.pth" % (epoch + 1))
        print("Save model Epoch{}".format(epoch + 1))

        loss_list.append(np.mean(temp_loss_list))
        # loss_list.append(temp_loss_list)



    end_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))

    # Validate


    # Loss Graph
    # plt.plot(loss_list)
    plt.plot(loss_list)
    plt.show()


if __name__ == '__main__':
    train()