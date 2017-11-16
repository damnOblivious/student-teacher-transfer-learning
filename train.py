import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np
torch.manual_seed(1)    # reproducible
LR = 0.0008              # learning rate

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


def teacherStudent(train_loader,test_loader,teacher,epochs,opt):
	cnn = CNN()
	print(cnn)  # net architecture

	optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
	loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

	if opt.cuda:
		loss_func = loss_func.cuda()
		cnn = cnn.cuda()


	for epoch in range(epochs):
	    for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
		b_x = Variable(x)
		b_y = Variable(y)
		if opt.cuda:
			b_x = b_x.cuda()   # batch x
			b_y = b_y.cuda()   # batch y

		studentOutput = cnn(b_x)
		teacherOutput = teacher(b_x)
		print("teacher Output")
		print(teacherOutput)
		print("student output")
		print(studentOutput)               # cnn output
		hardLoss = loss_func(studentOutput, b_y)   # cross entropy loss
		optimizer.zero_grad()           # clear gradients for this training step
		hardLoss.backward()                 # backpropagation, compute gradients
		optimizer.step()                # apply gradients
