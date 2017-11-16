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

torch.manual_seed(1)    # reproducible
LR = 0.0008              # learning rate


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


'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=5,            # n_filters
                kernel_size=(5,8),              # filter size
                stride=(1,6),                   # filter movement/step
                padding=(18,0),                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
         #   nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(5, 3, (1,3), (1,1), (0,0)),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
          #  nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.conv3 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(3, 3, 1, 1, 0),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
          #  nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.conv4 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(3, 3, 1, 1, 0),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
          #  nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.conv5 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(3, 3, 1, 1, 0),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
          #  nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.conv6 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(3, 1, (5,5), (1,1), (20,1)),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
          #  nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(1 *100 * 1, 10)
    self.softMax = nn.Softmax()   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
    
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.out(x)
    output = self.softMax(x)
        return output    # return x for visualization
'''


def teacherStudent(train_loader, test_loader, teachers, opt):
    student = CNN()
    print(student)  # net architecture

    # optimize all student parameters
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    # the target label is not one-hotted
    hardLossCriterion = nn.CrossEntropyLoss()
    softLossCriterion = nn.L1Loss()
    if opt.cuda:
        hardLossCriterion = hardLossCriterion.cuda()
        student = student.cuda()
        softLossCriterion = nn.L1Loss().cuda()

    for epoch in range(opt.epochs):
        # gives batch data, normalize x when iterate train_loader
        print "epoch : ", epoch
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)
            if opt.cuda:
                b_x = b_x.cuda()   # batch x
                b_y = b_y.cuda()   # batch y

            studentOutput = student(b_x)
            softLoss = None
            for teacherNo in range(len(teachers)):
                teacherOutput = teachers[teacherNo](b_x)
                if softLoss is None:
                    softLoss = opt.wstudSim[teacherNo] * \
                        softLossCriterion(
                            studentOutput, teacherOutput.detach())
                else:
                    softLoss = softLoss + \
                        opt.wstudSim[teacherNo] * \
                        softLossCriterion(
                            studentOutput, teacherOutput.detach())

            #studtotalLoss = self.computenlogStud(student_out, teacher_out, studentgrad_params, teachergrad_params, y_discriminator, target, isReal, isFakeTeacher)

            hardLoss = hardLossCriterion(
                studentOutput, b_y)   # cross entropy lossi
            TotalLoss = hardLoss + softLoss
            optimizer.zero_grad()           # clear gradients for this training step
            TotalLoss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
