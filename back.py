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
import models.student

torch.manual_seed(1)    # reproducible


def getAccuracy(studentOutput, label):
    (_, student_output) = torch.max(studentOutput, 1)
    isAccurate = (student_output == label)
    Sum = torch.sum(isAccurate)
    return Sum.data[0]


def getOptim(opt, model):
    if opt.studentoptimType == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(
        ), lr=opt.lr, momentum=opt.momentum, nesterov=opt.nesterov, weight_decay=opt.weightDecay)
    if opt.studentoptimType == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=opt.maxlr, weight_decay=opt.weightDecay)
    return optimizer


def runValidation(network, dataLoader, opt):
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


def teacherStudent(train_loader, test_loader, teachers, student, opt):
    # optimize all student parameters
    # optimizer = getOptim(opt, student)
    optimizer = torch.optim.Adam(student.parameters(), lr=opt.lr)
    hardLossCriterion = nn.CrossEntropyLoss()
    softLossCriterion = nn.KLDivLoss()
    derivativeCriterion = nn.L1Loss()

    temp = 0
    for index, weight in enumerate(opt.wstudSim):
        temp = temp+weight
    for index, weight in enumerate(opt.wstudSim):
        opt.wstudSim[index] = opt.wstudSim[index] / temp

    temp = opt.softWeight+opt.hardWeight
    opt.softWeight, opt.hardWeight = opt.softWeight / temp, opt.hardWeight / temp

    if opt.cuda:
        hardLossCriterion = hardLossCriterion.cuda()
        softLossCriterion = softLossCriterion.cuda()
        derivativeCriterion = derivativeCriterion.cuda()

    for epoch in range(opt.epochs):

        # utils.adjust_learning_rate(opt, optimizer, epoch)
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)
            if opt.cuda:
                b_x = b_x.cuda(async=True)   # batch x
                b_y = b_y.cuda(async=True)   # batch y

            studentOutput = student(b_x)
            softStudentOutput = torch.nn.functional.softmax(studentOutput)
            # normalize student output
            meanStudent, stdStudent = studentOutput.mean(), studentOutput.std()
            studentOutput = (studentOutput - meanStudent) / stdStudent

            # meanSoftStudent, stdSoftStudent = softStudentOutput.mean(), softStudentOutput.std()
            # softStudentOutput = (softStudentOutput -
            #                      meanSoftStudent) / stdSoftStudent

            softLoss = None
            derivativeLoss = None
            for teacherNo, teacher in enumerate(teachers):
                teacherOutput = teacher(b_x)

                # normalize teacher output
                # meanTeacher, stdTeacher = teacherOutput.mean(), teacherOutput.std()
                # teacherOutput = (teacherOutput - meanTeacher) / stdTeacher

                # teachersimLoss = softLossCriterion(teacherOutput, studentOutput.detach())
                studentsimLoss = softLossCriterion(softStudentOutput, teacherOutput.detach())

                # teachergrad_params = torch.autograd.grad(
                #     teachersimLoss, teacher.parameters(), create_graph=True)
                # studentgrad_params = torch.autograd.grad(
                #     studentsimLoss, student.parameters(), create_graph=True)
                # teachergrad_params, studentgrad_params = teachergrad_params[-1], studentgrad_params[-1]

                studentsimLoss = opt.wstudSim[teacherNo] * studentsimLoss

                # Normalization of gradients - Check if they were mismatched first
                # meanTeachergrad, stdTeachergrad = teachergrad_params.mean(), teachergrad_params.std()
                # meanStudentgrad, stdStudentgrad = studentgrad_params.mean(), studentgrad_params.std()
                # teachergrad_params = (teachergrad_params -
                #                       meanTeachergrad) / stdTeachergrad
                # studentgrad_params = (studentgrad_params -
                #                       meanStudentgrad) / stdStudentgrad

                # curDerivativeLoss = opt.wstudDeriv[teacherNo] * derivativeCriterion(
                #     studentgrad_params, teachergrad_params.detach())

                # if derivativeLoss is None:
                #     derivativeLoss = curDerivativeLoss
                # else:
                #     derivativeLoss = derivativeLoss + curDerivativeLoss

                if softLoss is None:
                    softLoss = studentsimLoss
                else:
                    softLoss = softLoss + studentsimLoss

            hardLoss = hardLossCriterion(
                studentOutput, b_y)   # cross entropy loss
            TotalLoss = opt.hardWeight*hardLoss + opt.softWeight*softLoss #+ derivativeLoss
            optimizer.zero_grad()           # clear gradients for this training step
            TotalLoss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

        accurate_results, total = runValidation(student, train_loader, opt)
        print "On epoch", epoch, "train Accuracy = ", accurate_results / total

    accurate_results, total = runValidation(student, test_loader, opt)
    print 'validating on test samples of size = ', total
    print 'Accuracy = ', accurate_results / total
