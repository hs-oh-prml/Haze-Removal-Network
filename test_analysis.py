#####################################
#                                   #
#   For Histogram Analysis Code     #
#                                   #
#####################################

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms
from torchvision.utils import save_image

from model_analysis import Net1

import argparse
import glob

import os
import random

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', default='C:/Users/user/Desktop/test/canyon1.jpg', type=str, help='test image name')
    parser.add_argument('--model_name', default='check_point_100.pth', type=str, help='generator model epoch name')
    opt = parser.parse_args()

    TEST_MODE = True if torch.cuda.is_available() else False

    # Information of Train

    train_info = "LoG5x5_analysis"
    if train_info != "":
        checkpoint_dir = "./checkpoints/cp_{}".format(train_info)
        RESULT_DIR = "./result/result_{}".format(train_info)

    else:
        checkpoint_dir = "./checkpoints/"
        RESULT_DIR = "./result/"
    
    if not(os.path.isdir(checkpoint_dir)):
        os.makedirs(os.path.join(checkpoint_dir))

    if not(os.path.isdir(RESULT_DIR)):
        os.makedirs(os.path.join(RESULT_DIR))

    MODEL_NAME = opt.model_name
    IMAGE_NAME = opt.image_name

    testset = "test"
    test_list = glob.glob(testset+ '/*.*')

    net = Net1(3, 3)
    model = net.eval()
    model.load_state_dict(torch.load(checkpoint_dir + '/checkpoint_100.pth'))
    model.cuda()
    
    params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        params = params + param
        print(name, param)
    print(params)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor    

    # img = random.choice(test_list)
    # 'img' is Test image Path 
    img = "C:/Users/IVP/Documents/GitHub/Haze-Removal-Network/test/hz_indoor_1032.jpg"
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
            # transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ]        
    transform = transforms.Compose(t_info)
            
    image = transform(image).unsqueeze_(0)
    image = Variable(image.type(Tensor))
    image = image.cuda()

    name = "Test_{}".format(img.split('/')[-1])
    with torch.no_grad():
        # dehaze = model(image)
        dehaze = model(image, name, True)
    img_name = os.path.basename(img)

    save_image(image, './histogram/{}/input.png'.format(name))
    save_image(dehaze, './histogram/{}/output.png'.format(name), normalize=True)

    print(img + " Done")
