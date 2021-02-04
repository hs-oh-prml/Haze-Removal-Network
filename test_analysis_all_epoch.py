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

import analysis_weight
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

    for i in range(100):
        epoch = i + 1

        net = Net1(3, 3)
        model = net.eval()
        model.load_state_dict(torch.load(checkpoint_dir + '/checkpoint_{}.pth'.format(epoch)))
        model.cuda()
        
        # Analysis_weight(model, "Epoch_{}".format(epoch))
        
        # params = 0
        # for name, parameter in model.named_parameters():
        #     if not parameter.requires_grad: 
        #         continue
        #     param = parameter.numel()
        #     temp_name = name.split(".")
        #     if temp_name[0] == "init_block" or temp_name[0] == "downSampleing1" or temp_name[0] == "downSampleing2" or temp_name[0] == "upSampling1" or temp_name[0] == "upSampling2" or temp_name[0] == "out_block" or len(temp_name) >= 5:
        #         continue
        #     if temp_name[-1] == "bias": continue
        #     if temp_name[-2] == "gn" or temp_name[-2] == "gn1" or temp_name[-2] == "gn2" or temp_name[-2] == "gn3": continue
        #     if temp_name[0] == "init_gn" or temp_name[0] == "downSampleing1_gn" or temp_name[0] == "downSampleing2_gn" or temp_name[0] == "upSampling1_gn" or temp_name[0] == "upSampling2_gn": continue

        #     # line = "{}\n".format(name)
        #     # line += "{}".format(parameter)
        #     print(parameter.shape)
        #     for idx_in, i in enumerate(parameter):
        
        #         path = "./weight/Epoch_{}/{}/{}".format(epoch, temp_name[0], idx_in)
        #         if not(os.path.isdir(path)):
        #             os.makedirs(os.path.join(path))
        
        #         for idx_out, j in enumerate(i):
        #             line = ""
        #             filename = ""
        #             if len(temp_name) > 2:
        #                 filename = "{}_{}_{}_{}.txt".format(temp_name[0],temp_name[1], temp_name[2], idx_out)
        #             else: 
        #                 filename = "{}_{}.txt".format(temp_name[0], idx_out)
        #             f = open(path + "/{}".format(filename), 'w')            

        #             for _, k in enumerate(j):
        #                 for _, l in enumerate(k):
        #                     line += "{} ".format(l)
        #                 line += "\n"
        #             f.write(line)

        #         #     line += "\n"
        #         # line += "\n"
                    

        #     # # for _, i in enumerate(parameter):
        #     f.write(line)

        #     params = params + param
        #     print(name, param)
        # print(params)

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

        name = "Epoch_{}_{}".format(epoch, img.split('/')[-1])
        img_path = "./histogram/".format(name)
        if not(os.path.isdir(img_path)):
            os.makedirs(os.path.join(img_path))

        with torch.no_grad():
            # dehaze = model(image)
            dehaze = model(image, name, True)
        img_name = os.path.basename(img)

        save_image(image, img_path + '/input.png'.format(img_path))
        save_image(dehaze, img_path + '/output.png'.format(img_path), normalize=True)
        print(img + " Done")
