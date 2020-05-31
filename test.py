import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms
from torchvision.utils import save_image

from model import Network
from model import Net1
from model import Net2

import argparse
import glob

import os



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', default='C:/Users/user/Desktop/test/canyon1.jpg', type=str, help='test image name')
    parser.add_argument('--model_name', default='check_point_100.pth', type=str, help='generator model epoch name')
    opt = parser.parse_args()

    TEST_MODE = True if torch.cuda.is_available() else False

    
    # Information of Train
    # train_info = "all_tanh_sobel_l1_19"
    # train_info = "TEST"
    # train_info = "all_tanh_LoG3x3_l1_19_rb96"

    # train_info = "all_tanh_LoG3x3_l1_19_rb16"
    # train_info = "ReLU_LoG3x3_l1_19_rb16_kernel11"
    # train_info = "ReLU_LoG3x3_l1_19_rb16_kernel11_linear"
    # train_info = "ReLU_LoG3x3_l1_19_rb32_kernel5"
    # train_info = "LoG3x3_l1_19_rb16_kernel16_2_double_ed"
    # train_info = "LoG3x3_l1_19_rb16_kernel16_2_double_Ied_4"
    # train_info = "LoG3x3_l1_19_ib9_kernel3"
    # train_info = "Reverse_ED"
    # train_info = "bottle_neck_64_unet_mse_group_norm"
    # train_info = ""
    # checkpoint_dir = "./checkpoints/cp_{}".format(train_info)
    checkpoint_dir = "./checkpoints"

    # RESULT_DIR = "./result/result_{}".format(train_info)
    RESULT_DIR = "./result"

    if not(os.path.isdir(checkpoint_dir)):
        os.makedirs(os.path.join(checkpoint_dir))

    if not(os.path.isdir(RESULT_DIR)):
        os.makedirs(os.path.join(RESULT_DIR))

    MODEL_NAME = opt.model_name
    IMAGE_NAME = opt.image_name
    test_list = glob.glob( 'test' + '/*.*')
    # test_list = glob.glob( 'C:/Users/user/Desktop/RTTS/RTTS/JPEGImages' + '/*.*')
    # test_list = glob.glob( 'C:/Users/user/Desktop/RTTS/RTTS/selected' + '/*.*')
    # test_list = glob.glob( 'C:/Users/user/Desktop/data/val_hz' + '/*.*')

    net = Net1(3, 3)

    model = net.eval()
    # model.load_state_dict(torch.load('./checkpoints/' + MODEL_NAME))

    model.load_state_dict(torch.load(checkpoint_dir + '/checkpoint_100.pth'))

    model.cuda()
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor    

    count = 0
    for i, img in enumerate(test_list):
        count = count + 1
        # if count == 50: break

        image = Image.open(img).convert('RGB')
        # print(image.size)
        # .convert('RGB')

        w, h = image.size
        # print(h)
        # print(w)
        if w > 2000:
            w = w // 2
            h = h // 2
        if h > 2000:
            w = w // 2
            h = h // 2

        bias = 4
        if w % bias != 0:
            w = w - (w % bias)
        if h % bias != 0:
            h = h - (h % bias)
        
        # print(h)
        # print(w)
        t_info = [
                # transforms.Resize((480, 640), Image.BICUBIC),
                transforms.Resize((h, w), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
            ]

        
        # w, h = image.size
        # if w > 2000 or h > 2000:
        #     t_info = [
        #         transforms.Resize((h // 2, w // 2), Image.BICUBIC),
        #         transforms.ToTensor(),                 
        #         transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        #     ]
            
        #     print('Resize {} x {}'.format(w // 2, h // 2))
        
        transform = transforms.Compose(t_info)
                

        image = transform(image).unsqueeze_(0)
        image = Variable(image.type(Tensor))
        image = image.cuda()

        with torch.no_grad():
            dehaze = model(image)

        result = torch.cat((image, dehaze), -1)
        img_name = os.path.basename(img)        
        # save_image(dehaze, RESULT_DIR + '/dehaze_{}.png'.format(img_name), nrow=5, normalize=True)
        save_image(result, RESULT_DIR + '/dehaze_{}.png'.format(img_name), nrow=5, normalize=True)
        # save_image(dehaze, RESULT_DIR + '/dehaze_{}.png'.format(img_name), nrow=5, normalize=True)
        print(img + " Done")