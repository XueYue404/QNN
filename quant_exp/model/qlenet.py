"""
目前的输入尺寸符合Cifar-10，如果要实验mnist需要改代码
"""

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

class DorefaNet(nn.Module):
    # Using Modified AlexNet, baseline accuracy = 75%
    def __init__(self,bitA,bitW,bitG):
        super(DorefaNet, self).__init__()
        self.bitA,self.bitW,self.bitG = bitA,bitW,bitG
        self.cnn = nn.Sequential(
            QuantConv2d(3,96,7,stride=2,padding=2,a_bits=self.bitA,w_bits=self.bitW),
            QuantGrad(bitG = self.bitG),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0),
            QuantConv2d(96,256,5,stride=1,padding=2,a_bits=self.bitA,w_bits=self.bitW),
            QuantGrad(bitG = self.bitG),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0),
            QuantConv2d(256,384,3,stride=1,padding=1,a_bits=self.bitA,w_bits=self.bitW),
            QuantGrad(bitG = self.bitG),
            nn.ReLU(inplace=True),
            QuantConv2d(384,384,3,stride=1,padding=1,a_bits=self.bitA,w_bits=self.bitW),
            QuantGrad(bitG = self.bitG),
            nn.ReLU(inplace=True),
            QuantConv2d(384,256,3,stride=1,padding=1,a_bits=self.bitA,w_bits=self.bitW),
            QuantGrad(bitG = self.bitG),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            QuantLinear(256*3*3,1024,a_bits=self.bitA,w_bits=self.bitW),
            QuantGrad(bitG = self.bitG),
            nn.ReLU(inplace=True),
            QuantLinear(1024,512,a_bits=self.bitA,w_bits=self.bitW),
            QuantGrad(bitG = self.bitG),
            nn.ReLU(inplace=True),
            QuantLinear(512,10,a_bits=self.bitA,w_bits=self.bitW)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
        