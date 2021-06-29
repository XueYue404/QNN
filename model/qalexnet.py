import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import math
from torch.utils.data import DataLoader
from torch.autograd import Function

#从本项目中import
from model.quant import QuantConv2d
from model.quant import QuantLinear
from model.quant import QuantGrad

'''
AlexNet分类器，用于CIFAR10
用法：
    network = AlexNet(isquant=True,bitA=4,bitW=8)
'''
class AlexNet(nn.Module):
    def __init__(self,isquant=True,bitA=8,bitW=16,bitG=16):
        super(AlexNet, self).__init__()
        self.isquant = isquant
        self.bitA = bitA
        self.bitW = bitW
        self.bitG = bitG
        if self.isquant == True:
            self.cnn = nn.Sequential(
            # 卷积层1，3通道输入，96个卷积核，核大小7*7，步长2，填充2
            # 经过该层图像大小变为32-7+2*2 / 2 +1，15*15
            # 经3*3最大池化，2步长，图像变为15-3 / 2 + 1， 7*7
            QuantConv2d(3, 96, 7, 2, 2,a_bits=self.bitA,w_bits=self.bitW),
            QuantGrad(bitG = self.bitG),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0),

            # 卷积层2，96输入通道，256个卷积核，核大小5*5，步长1，填充2
            # 经过该层图像变为7-5+2*2 / 1 + 1，7*7
            # 经3*3最大池化，2步长，图像变为7-3 / 2 + 1， 3*3
            QuantConv2d(96, 256, 5, 1, 2,a_bits=self.bitA,w_bits=self.bitW),
            QuantGrad(bitG = self.bitG),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0),

            # 卷积层3，256输入通道，384个卷积核，核大小3*3，步长1，填充1
            # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
            QuantConv2d(256, 384, 3, 1, 1,a_bits=self.bitA,w_bits=self.bitW),
            QuantGrad(bitG = self.bitG),
            nn.ReLU(inplace=True),

            # 卷积层3，384输入通道，384个卷积核，核大小3*3，步长1，填充1
            # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
            QuantConv2d(384, 384, 3, 1, 1,a_bits=self.bitA,w_bits=self.bitW),
            QuantGrad(bitG = self.bitG),
            nn.ReLU(inplace=True),

            # 卷积层3，384输入通道，256个卷积核，核大小3*3，步长1，填充1
            # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
            QuantConv2d(384, 256, 3, 1, 1,a_bits=self.bitA,w_bits=self.bitW),
            QuantGrad(bitG = self.bitG),
            nn.ReLU(inplace=True)
            )

            self.fc = nn.Sequential(
                # 256个feature，每个feature 3*3
                QuantLinear(256*3*3, 1024,a_bits=self.bitA,w_bits=self.bitW),
                QuantGrad(bitG = self.bitG),
                nn.ReLU(),
                QuantLinear(1024, 512,a_bits=self.bitA,w_bits=self.bitW),
                QuantGrad(bitG = self.bitG),
                nn.ReLU(),
                QuantLinear(512, 10,a_bits=self.bitA,w_bits=self.bitW)
            )
        else:
            self.cnn = nn.Sequential(
            nn.Conv2d(3, 96, 7, 2, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0),

            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0),

            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True)
            )

            self.fc = nn.Sequential(
                nn.Linear(256*3*3, 1024),
                nn.ReLU(),

                nn.Linear(1024, 512),
                nn.ReLU(),

                nn.Linear(512, 10)
            )
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x