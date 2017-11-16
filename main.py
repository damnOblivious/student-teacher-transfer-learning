import numpy as np
import os
import argparse
import copy
import opts
import utils
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as modelzoo
import models
import datasets.data_loader as loader
import back
#from tensorboard_logger import Logger

parser = opts.myargparser()


def main():
    global opt, best_studentprec1

    teachers = []

    opt = parser.parse_args()
    opt.logdir = opt.logdir + '/' + opt.name
   # logger = Logger(opt.logdir)

    print(opt)

    for t in opt.teacher:
        print('Loading models...')
        teacher = models.teacherLoader[t](opt.cuda)
        print("Done loading from other file")
        print(teacher)
        teachers.append(teacher)

    #         print("=> no checkpoint found at '{}'".format(opt.resume))
    dataloader = loader.loadCIFAR10(opt)
    print(dataloader)
    train_loader = dataloader['train_loader']
    val_loader = dataloader['val_loader']
    back.teacherStudent(train_loader, val_loader, teachers, opt)


if __name__ == '__main__':
    main()
