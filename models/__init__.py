import torch
import torch.nn as nn
import torch.optim as optim
import models.teacher as teacher1
import utils
import os
import shutil


def setup(model, opt, type):
    if opt.weight_init:
        utils.weights_init(model, opt)

    return model


def load_model(opt, type):
    if type == "teacher":
	print opt.teacher_filedir
	if os.path.isfile(opt.teacher_filedir):
        	checkpoint = torch.load(opt.teacher_filedir)
        model = teacher1.vgg11()
	model.features = torch.nn.DataParallel(model.features)
        if opt.cuda:
            model = model.cuda()
        model.load_state_dict(checkpoint['state_dict'])
    return model
