import os
import glob
import random
import shutil


val_gt = 'C:/Users/user/Desktop/data/val_gt/'
val_hz = 'C:/Users/user/Desktop/data/val_hz/'
gt = 'C:/Users/user/Desktop/data/gt'
hz = 'C:/Users/user/Desktop/data/hz'

hz_img = sorted(glob.glob(hz+'/*.*'))
gt_img = sorted(glob.glob(gt+'/*.*'))
size = len(hz_img)
if len(hz_img) != len(gt_img):
    print("Error: Not match ")

data_list = []
for i in range(size):
    data_list.append((hz_img[i], gt_img[i]))

for i in range(10):
    select = random.choice(data_list)
    print(select[0])
    shutil.move(select[0], val_hz)
    print(select[1])
    shutil.move(select[1], val_gt)
    