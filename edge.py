import cv2 
import numpy as np
import glob
import os

path = "C:/Users/user/Desktop/"
hz_list = sorted(glob.glob(os.path.join(path, 'data/hz') + '/*.*'))
gt_list = sorted(glob.glob(os.path.join(path, 'data/gt') + '/*.*'))

oh_img = cv2.imread(hz_list[450], 1)
og_img = cv2.imread(gt_list[450], 1)

h_img = cv2.imread(hz_list[450], cv2.IMREAD_GRAYSCALE)
h_sobel = cv2.Sobel(h_img, cv2.CV_8U, 1, 0, 3)

g_img = cv2.imread(gt_list[450], cv2.IMREAD_GRAYSCALE)
g_sobel = cv2.Sobel(g_img, cv2.CV_8U, 1, 0, 3)

con1 = cv2.vconcat([h_img, h_sobel])
con2 = cv2.vconcat([g_img, g_sobel])
result = cv2.hconcat([con1, con2])


r2 = cv2.hconcat([oh_img, og_img])
# cv2.imshow("Result", r2)

cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()