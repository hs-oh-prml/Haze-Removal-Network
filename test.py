import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms
from torchvision.utils import save_image

from model import Net1

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
    train_info = ""
    checkpoint_dir = "./checkpoints"
    RESULT_DIR = "./result"
    if not(os.path.isdir(checkpoint_dir)):
        os.makedirs(os.path.join(checkpoint_dir))

    if not(os.path.isdir(RESULT_DIR)):
        os.makedirs(os.path.join(RESULT_DIR))

    MODEL_NAME = opt.model_name
    IMAGE_NAME = opt.image_name
    test_list = glob.glob('test/*.*')

    net = Net1(3, 3)

    model = net.eval()
    model.load_state_dict(torch.load(checkpoint_dir + '/checkpoint_100.pth'))
    model.cuda()
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor    

    count = 0
    for i, img in enumerate(test_list):
        count = count + 1

        image = Image.open(img).convert('RGB')
        w, h = image.size
        if w > 2000:
            w = w // 4
            h = h // 4
        if h > 2000:
            w = w // 4
            h = h // 4

        bias = 4
        if w % bias != 0:
            w = w - (w % bias)
        if h % bias != 0:
            h = h - (h % bias)
        
        t_info = [
                transforms.Resize((h, w), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
            ]
        
        transform = transforms.Compose(t_info)
                

        image = transform(image).unsqueeze_(0)
        image = Variable(image.type(Tensor))
        image = image.cuda()

        with torch.no_grad():
            dehaze = model(image)

        result = torch.cat((image, dehaze), -1)
        img_name = os.path.basename(img)        
        save_image(result, RESULT_DIR + '/dehaze_{}.png'.format(img_name), nrow=5, normalize=True)
        print(img + " Done")