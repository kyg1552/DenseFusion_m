import _init_paths
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import copy
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from lib.build_BiSeNet import BiSeNet
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, cal_miou
import time

import cv2
#from cv_bridge import CvBridge, CvBridgeError

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset_root', type=str, default = 'datasets/ycb/YCB_Video_Dataset', help='dataset root dir')
#parser.add_argument('--model', type=str, default = 'trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth',  help='resume PoseNet model')
#parser.add_argument('--refine_model', type=str, default = 'trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth',  help='resume PoseRefineNet model')
parser.add_argument('--checkpoint_path', type=str, default='/home/young/project/DenseFusion/trained_checkpoints/ycb/best_dice_loss.pth', help='The path to the pretrained weights of model')
parser.add_argument('--num_classes', type=int, default=22, help='num of object classes (with void)')
parser.add_argument('--context_path', type=str, default="resnet18", help='The context path model you are using.')
opt = parser.parse_args()

num_obj = 21

print('load BiseNet')
start_time = time.time()
bise_model = BiSeNet(opt.num_classes, opt.context_path)
bise_model = bise_model.cuda()
bise_model.load_state_dict(torch.load(opt.checkpoint_path), strict=False)
print('Done!')
print("Load time : {}".format(time.time() - start_time))


def seg_predict(image):
    global bise_model
   
    with torch.no_grad():
        bise_model.eval()
        h,w,_ = image.shape
        to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        image = to_tensor(image)
        image = image.unsqueeze_(0)
        image = image.cuda()
        predict = bise_model(image).squeeze()
        predict = reverse_one_hot(predict)
        predict = predict.cpu().numpy()
        predict = np.array(predict)
        predict = np.resize(predict,[h,w])
        print(np.unique(predict)[1:])
        zzzz = cv2.cvtColor(np.uint8(predict), cv2.COLOR_GRAY2BGR)
        cv2.imwrite('/home/young/project/DenseFusion/tools/Seg_img/segmentation_image.png', zzzz)
        return predict
   

if __name__ == "__main__":
    img = cv2.imread('/home/young/project/DenseFusion/datasets/ycb/YCB_Video_Dataset/data/0000/000001-color.png',cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    label = seg_predict(img)
    # cv2.imshow('label', label)
    # cv2.waitKey(0)
