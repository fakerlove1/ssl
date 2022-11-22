import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from utils import helpers
from PIL import Image
import datetime


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def initialize_weights(*models, a=0):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight, a=a)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def data_rotate(im, angle):
    M = cv2.getRotationMatrix2D((im.shape[1] // 2, im.shape[0] // 2), angle, 1.0)
    im = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_NEAREST)
    return im

def log(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        # print(time_stamp + " " + X)
        print(X)
    else:
        f.write(time_stamp + " " + X)


if __name__ == '__main__':
    ...