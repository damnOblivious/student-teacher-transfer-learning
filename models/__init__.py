import torch
import torch.nn as nn
import torch.optim as optim
import models.teacher as teacher1
import utils
import os
import shutil


def setup(model, opt):
    if opt.weight_init:
        utils.weights_init(model, opt)

    return model


