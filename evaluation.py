import cv2
import os
import sys
import glob
import numpy as np
import compare_util

if __name__ == '__main__':
    year = "2020"
    model = "".format(year)
    gt_path = "".format(year)
    hz_path = "".format(year)
    dhz_path = "".format(model)
    save_path = "".format(model)

    hz_list = glob.glob(hz_path)
    gt_list = glob.glob(gt_path)
    img_list = glob.glob(dhz_path)
    
    filedir = save_path
    if not os.path.isdir(filedir):
        os.mkdir(filedir)

    if not os.path.exists("result_compare.txt"):
        with open(filedir+"result_compare.txt",'w') : pass
    
    f = open("{}/result_compare.txt".format(save_path), 'w')

    i = 0 
    mse_list = []
    psnr_list = []
    ssim_list = []


    while i < len(gt_list):

        hz_image = cv2.imread(hz_list[i], cv2.IMREAD_COLOR)
        o_image = cv2.imread(gt_list[i], cv2.IMREAD_COLOR)
        image = cv2.imread(img_list[i], cv2.IMREAD_COLOR)

        h, w, _ = o_image.shape
        image = cv2.resize(image, dsize=(w,h), interpolation=cv2.INTER_AREA)
        mse, psnr, ssim = compare_util.compare_image(o_image, image)

        result = cv2.hconcat([hz_image, image])
        cv2.imwrite("{}".format(save_path) +"/result{}.jpg".format(i), result)

        line = "mse:{}, psnr:{}, ssim: {}\n ".format(
            mse, psnr, ssim
            )

        f.write(line)
        mse_list.append(mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        print(line)

        i = i + 1
    mse_mean = np.mean(np.array(mse_list, dtype=np.float32))
    psnr_mean = np.mean(np.array(psnr_list, dtype=np.float32))
    ssim_mean = np.mean(np.array(ssim_list, dtype=np.float32))

    mean = """
    MSE: {} PSNR: {} SSIM: {} 
    """.format(
        mse_mean,
        psnr_mean,
        ssim_mean,
        )
    print(mean)
    f.write(mean)