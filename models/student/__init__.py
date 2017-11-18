import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np
from torch.autograd import Variable
from utils import AverageMeter
from utils import precision
import utils
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 3x32x32 ; 32x16x16 ; 64x8x8 ; 128x4x4 ; 256x2x2 ; 512x1x1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.PReLU())
        self.conv6 = nn.Sequential(
            nn.Conv2d(32, 10, kernel_size=1, padding=0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #print(x.size())
        x = self.conv5(x)
        x = self.conv6(x)
        #print(x.size())
        x = x.view(x.size(0), -1)

        return x

def GetStudent():
    return CNN()
