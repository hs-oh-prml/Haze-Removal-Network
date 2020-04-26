import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms
from torchvision.utils import save_image

from model import Network
from model import Generator

import argparse
import glob

import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', default='C:/Users/user/Desktop/test/canyon1.jpg', type=str, help='test image name')
    parser.add_argument('--model_name', default='check_point_100.pth', type=str, help='generator model epoch name')
    opt = parser.parse_args()

    TEST_MODE = True if torch.cuda.is_available() else False


    MODEL_NAME = opt.model_name
    IMAGE_NAME = opt.image_name
    RESULT_DIR = 'result'
    test_list = glob.glob( 'test' + '/*.*')
    # test_list = glob.glob( 'C:/Users/user/Desktop/RTTS/RTTS/JPEGImages' + '/*.*')

    # test_list = ['test/tiananmen2.png']
    # test_list = ['test/tiananmen3.jpg']
    # net = Network(3, 3)
    net = Generator(3, 3)

    model = net.eval()
    # model.load_state_dict(torch.load('./checkpoints/' + MODEL_NAME))
    model.load_state_dict(torch.load('./checkpoints/checkpoint_100.pth'))
    model.cuda()
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor    

    count = 0
    for i, img in enumerate(test_list):
        count = count + 1
        # if count == 50: break

        # transform = transforms.Compose([
        #     transforms.ToTensor(), 
        #     transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        #     ]
        # )

        image = Image.open(img).convert('RGB')
        # print(image.size)
        # .convert('RGB')

        w, h = image.size

        t_info = [
                transforms.Resize((480, 640), Image.BICUBIC),
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
        
        # print(img)
        # print(image.shape)
        # print(dehaze.shape)
        _, _, o_height, o_width = image.shape
        _, _, d_height, d_width = dehaze.shape
        # print("Origin size: ({}, {}), Dehaze size: ({}, {})".format(o_width, o_height, d_width, d_height))

        if o_height != d_height:
            row = torch.zeros((1, 3, 1, o_width)).cuda()
            gap = d_height - o_height            
            for i in range(gap):
                image = torch.cat((image, row), 2)
            # print("Resize height: ({})".format(image.shape))
            _, _, o_height, o_width = image.shape
            
        if o_width != d_width:
            col = torch.zeros((1, 3, o_height, 1)).cuda()
            gap = d_width - o_width
            
            for i in range(gap):
                image = torch.cat((image, col), 3)
            # print("Resize width: ({})".format(image.shape))
            _, _, o_height, o_width = image.shape
            
        # if o_hieght != d_hieght:

        result = torch.cat((image, dehaze), -1)
        img_name = os.path.basename(img)        
        save_image(result, RESULT_DIR + '/dehaze_{}.png'.format(img_name), nrow=5, normalize=True)
        # save_image(dehaze, RESULT_DIR + '/dehaze_{}.png'.format(img_name), nrow=5, normalize=True)
        print(img + " Done")