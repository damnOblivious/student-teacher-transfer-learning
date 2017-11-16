import torch
import torch.nn as nn
import torch.optim as optim
import models.teacher as teacher1
import utils
import os
from . import vgg11_1
from . import vgg11_2
import shutil

teacherLoader = {
    "vgg11_1": vgg11_1.load_model,
    "vgg11_2": vgg11_2.load_model,
}



def setup(model, opt):
    if opt.weight_init:
        utils.weights_init(model, opt)

    return model


