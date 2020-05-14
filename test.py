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


    MODEL_NAME = opt.model_name
    IMAGE_NAME = opt.image_name
    RESULT_DIR = 'result'
    # test_list = glob.glob( 'test' + '/*.*')
    # test_list = glob.glob( 'C:/Users/user/Desktop/RTTS/RTTS/JPEGImages' + '/*.*')
    # test_list = glob.glob( 'C:/Users/user/Desktop/RTTS/RTTS/selected' + '/*.*')
    test_list = glob.glob( 'C:/Users/user/Desktop/data/val_hz' + '/*.*')

    # test_list = ['test/tiananmen2.png']
    # test_list = ['test/tiananmen3.jpg']
    # net = Network(3, 3)
    net = Net1(3, 3)
    # net = Net2(3, 3)

    model = net.eval()
    # model.load_state_dict(torch.load('./checkpoints/' + MODEL_NAME))
    # model.load_state_dict(torch.load('./checkpoints/checkpoint_100.pth'))
    model.load_state_dict(torch.load('./checkpoints/checkpoint_90.pth'))

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
        # print(h)
        # print(w)

        if w % 4 != 0:
            w = w - (w % 4)
        if h % 4 != 0:
            h = h - (h % 4)

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