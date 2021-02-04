import argparse
import os
import torch
import data

parser = argparse.ArgumentParser()
# parser.add_argument("--dataroot", type=str, default = "C:/Users/users/Desktop/", help='data folder path')
parser.add_argument("--dataroot", type=str, default = "C:/Users/IVP/Desktop/", help='data folder path')
parser.add_argument("--epoch", type=int, default=0, help='starting epoch')
parser.add_argument("--n_epochs", type=int, default=100, help='number of epoch')
parser.add_argument("--batch_size", type=int, default=1, help='size of the batches')
parser.add_argument("--lr", type=float, default=0.002, help='initial learning rate')
parser.add_argument("--size", type=int, default=640, help='size of the data crop')
parser.add_argument("--input_nc", type=int, default=3, help='number of channels of input data')
parser.add_argument("--output_nc", type=int, default=3, help='number of channels of output data')
parser.add_argument("--num_workers", type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument("--img_height", type=int, default=480, help='scale image hieght')
parser.add_argument("--img_width", type=int, default=640, help='scale image width')

opt = parser.parse_args()
opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
