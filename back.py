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


def getAccuracy(studentOutput, label):
    (maxOutput, student_output) = torch.max(studentOutput, 1)
    isAccurate = (student_output == label)
    Sum = torch.sum(isAccurate)
    return Sum.data[0]

def runValidation(network, dataLoader,opt):
    accurate_results = 0.0
    total = 0.0
    for _, (x, y) in enumerate(dataLoader):
        b_x = Variable(x, volatile=True)
        b_y = Variable(y, volatile=True)
        if opt.cuda:
            b_x = b_x.cuda(async=True)
            b_y = b_y.cuda(async=True)
        studentOutput = network(b_x)
        accurate_results = accurate_results + getAccuracy(studentOutput, b_y)
        total = total + studentOutput.size()[0]

    return accurate_results, total


def teacherStudent(train_loader, test_loader, teachers, opt):
    student = CNN()

    # optimize all student parameters
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    hardLossCriterion = nn.CrossEntropyLoss()
    softLossCriterion = nn.L1Loss()
    derivativeCriterion = nn.L1Loss()

    if opt.cuda:
        hardLossCriterion = hardLossCriterion.cuda()
        student = student.cuda()
        softLossCriterion = softLossCriterion.cuda()
        derivativeCriterion = derivativeCriterion.cuda()

    for epoch in range(opt.epochs):

        for _, (x, y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)
            if opt.cuda:
                b_x = b_x.cuda()   # batch x
                b_y = b_y.cuda()   # batch y

            studentOutput = student(b_x)
            softLoss = None
            derivativeLoss = None
            for teacherNo, teacher in enumerate(teachers):
                teacherOutput = teacher(b_x)

                teachersimLoss = opt.wstudSim[teacherNo] * \
                    softLossCriterion(teacherOutput, studentOutput.detach())
                studentsimLoss = opt.wstudSim[teacherNo] * \
                    softLossCriterion(studentOutput, teacherOutput.detach())

                teachergrad_params = torch.autograd.grad(
                    teachersimLoss, teacher.parameters(), create_graph=True)
                studentgrad_params = torch.autograd.grad(
                    studentsimLoss, student.parameters(), create_graph=True)

                teachergrad_params, studentgrad_params = teachergrad_params[-1], studentgrad_params[-1]

                curDerivativeLoss = opt.wstudDeriv[teacherNo] * derivativeCriterion(
                    studentgrad_params, teachergrad_params.detach())

                if derivativeLoss is None:
                    derivativeLoss = curDerivativeLoss
                else:
                    derivativeLoss = derivativeLoss + curDerivativeLoss

                if softLoss is None:
                    softLoss = studentsimLoss
                else:
                    softLoss = softLoss + studentsimLoss

            hardLoss = hardLossCriterion(
                studentOutput, b_y)   # cross entropy lossi
            TotalLoss = hardLoss + softLoss + derivativeLoss
            optimizer.zero_grad()           # clear gradients for this training step
            TotalLoss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
        
        accurate_results, total = runValidation(student, train_loader,opt)
        print "On epoch", epoch, "Accuracy = ", accurate_results / total

    accurate_results, total = runValidation(student, test_loader,opt)
    print 'validating on test samples of size = ', total
    print 'Accuracy = ', accurate_results / total
