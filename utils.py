#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

import torch
import torch.nn as nn
import copy
import time
import shutil
import operator
import numpy as np
import random
import math

from PIL import Image, ImageOps
from torchvision import transforms

class AverageMeter():
    """Computes and stores the average and current value"""
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

def precision(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(opt, optimizer, epoch):
    epoch = copy.deepcopy(epoch)
    lr = opt.maxlr
    wd = opt.weightDecay
    if opt.learningratescheduler == 'decayschedular':
        while epoch >= opt.decayinterval:
            lr = lr/opt.decaylevel
            epoch = epoch - opt.decayinterval
    elif opt.learningratescheduler == 'imagenetschedular':
        lr = lr * (0.1 ** (epoch // 30))
    elif opt.learningratescheduler == 'cifarschedular':
        lr = opt.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))

    lr = max(lr,opt.minlr)
    opt.lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = wd

def get_mean_and_std(dataloader):
    '''Compute the mean and std value of dataset.'''
    mean = torch.zeros(3)
    std = torch.zeros(3)
    len_dataset = 0
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        len_dataset += 1
        for i in range(len(inputs[0])):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len_dataset)
    std.div_(len_dataset)
    return mean, std

def weights_init(model, opt):
    '''Add your favourite weight initializations.'''
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            #c  = math.sqrt(2.0 / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels))
            #m.weight.data = torch.randn(m.weight.data.size()).cuda() * c
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.affine == True:
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data = nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            #c =  math.sqrt(2.0 / m.weight.data.size(1));
            #m.weight.data = torch.randn(m.weight.data.size()).cuda() * c
            # TODO: Check bias
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
