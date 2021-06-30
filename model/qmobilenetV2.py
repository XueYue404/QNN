"""
MobileNetV2 in PyTorch. from github.com/kuangliu/pytorch-cifar
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.quant import QuantConv2d
from model.quant import QuantLinear
from model.quant import QuantGrad

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride,bitW=4,bitA=8,bitG=8):
        super(Block, self).__init__()
        self.stride = stride
        self.bitW=bitW
        self.bitA=bitA
        self.bitG=bitG
        planes = expansion * in_planes
        self.conv1 = QuantConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False,a_bits=self.bitA,w_bits=self.bitW)
        self.quantgrad1 = QuantGrad(bitG = self.bitG)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuantConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False,a_bits=self.bitA,w_bits=self.bitW)
        self.quantgrad2 = QuantGrad(bitG = self.bitG)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = QuantConv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False,a_bits=self.bitA,w_bits=self.bitW)
        self.quantgrad3 = QuantGrad(bitG = self.bitG)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                QuantConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False,a_bits=self.bitA,w_bits=self.bitW),
                QuantGrad(bitG = self.bitG),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.quantgrad1(self.conv1(x))))
        out = F.relu(self.bn2(self.quantgrad2(self.conv2(out))))
        out = self.bn3(self.quantgrad3(self.conv3(out)))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class QMobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10,bitW=4,bitA=8,bitG=8):
        super(QMobileNetV2, self).__init__()
        self.bitW=bitW
        self.bitA=bitA
        self.bitG=bitG

        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = QuantConv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False,a_bits=self.bitA,w_bits=self.bitW)
        self.quantgrad1 = QuantGrad(bitG = self.bitG)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = QuantConv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False,a_bits=self.bitA,w_bits=self.bitW)
        self.quantgrad2 = QuantGrad(bitG = self.bitG)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = QuantLinear(1280, num_classes,a_bits=self.bitA,w_bits=self.bitW)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride,bitW=self.bitW,bitA=self.bitA,bitG=self.bitG))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.quantgrad1(self.conv1(x))))
        out = self.layers(out)
        out = F.relu(self.bn2(self.quantgrad2(self.conv2(out))))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = QMobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()