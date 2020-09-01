from skimage.measure import compare_ssim

import imutils
import numpy as np
import math
import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image

# input: 2 Image, output: MSE, PSNR, SSIM
def compare_image(origin, compare):

    MSE = np.mean( (origin - compare) ** 2 )
    if MSE == 0:
            PSNR = 100
    else:
        
        PIXEL_MAX = np.max(origin)
        PSNR = 20 * math.log10(PIXEL_MAX) - 10 * math.log10(MSE)

    Oimg = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
    Cimg = cv2.cvtColor(compare, cv2.COLOR_BGR2GRAY)

    SSIM = ssim(Oimg, Cimg, data_range=Oimg.max() - Oimg.min())

    return MSE, PSNR, SSIM
