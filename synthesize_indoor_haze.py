import math
import os
import random
import sys
import numpy as np
from PIL import Image
from skimage.transform import resize
import imageio
import h5py

nyu_depth = h5py.File("C:/Users/user/Desktop/nyu_depth_v2_labeled.mat", 'r')

image = nyu_depth['images']
depth = nyu_depth['depths']

saveimgdir = "C:/Users/user/Desktop/data"

for index in range(1445):
    gt_image = (image[index, :, :, :]).astype(float)
    gt_image = np.swapaxes(gt_image, 0, 2)
    gt_image = resize(gt_image, (480, 640)).astype(float)
    gt_image = gt_image / 255

    gt_depth = depth[index, :, :]
    maxhazy = gt_depth.max()
    minhazy = gt_depth.min()
    gt_depth = (gt_depth) / (maxhazy)

    gt_depth = np.swapaxes(gt_depth, 0, 1)

    #beta
    beta = random.uniform(0.4, 1.6)
    tx1 = np.exp(-beta * gt_depth)
    
    #A
    a = random.uniform(0.6, 1.0)
    A = [a,a,a]

    m = gt_image.shape[0]
    n = gt_image.shape[1]

    rep_atmosphere = np.tile(np.reshape(A, [1, 1, 3]), [m, n, 1])
    tx1 = np.reshape(tx1, [m, n, 1])

    max_transmission = np.tile(tx1, [1, 1, 3])

    haze_image = gt_image * max_transmission + rep_atmosphere * (1 - max_transmission)

    imageio.imwrite(saveimgdir+'/ih/hz_indoor_{}.jpg'.format(index), haze_image)
    imageio.imwrite(saveimgdir+'/ig/gt_indoor_{}.jpg'.format(index), gt_image)
    print(index)