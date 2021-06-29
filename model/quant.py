"""
该文件提供了各种量化操作的库，包括量化卷积层、量化全连接层、量化梯度等等
直方图功能被注释掉了，TODO！！
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


# 取整(ste)
class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

# A(特征)量化
class ActivationQuantizer(nn.Module):
    def __init__(self, a_bits):
        super(ActivationQuantizer, self).__init__()
        self.a_bits = a_bits

    # 取整(ste)
    def round(self, input):
        output = Round.apply(input)
        return output

    # 量化/反量化
    def forward(self, input):
        if self.a_bits == 32:
            output = input
        elif self.a_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.a_bits != 1
        else:
            output = torch.clamp(input * 0.1, 0, 1)      # 特征A截断前先进行缩放（* 0.1），以减小截断误差
            scale = 1 / float(2 ** self.a_bits - 1)      # scale
            output = self.round(output / scale) * scale  # 量化/反量化
        return output

# W(权重)量化
class WeightQuantizer(nn.Module):
    def __init__(self, w_bits):
        super(WeightQuantizer, self).__init__()
        self.w_bits = w_bits

    # 取整(ste)
    def round(self, input):
        output = Round.apply(input)
        return output

    # 量化/反量化
    def forward(self, input):
        if self.w_bits == 32:
            output = input
        elif self.w_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.w_bits != 1
        else:
            output = torch.tanh(input)
            output = output / 2 / torch.max(torch.abs(output)) + 0.5  # 归一化-[0,1]
            scale = 1 / float(2 ** self.w_bits - 1)                   # scale
            output = self.round(output / scale) * scale               # 量化/反量化
            output = 2 * output - 1
        return output


class QuantConv2d(nn.Conv2d):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros',
                a_bits=8,
                w_bits=8,
                quant_inference=False):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                        bias, padding_mode)
        self.quant_inference = quant_inference
        self.activation_quantizer = ActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.conv2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.dilation,
                        self.groups)
        return output


class QuantLinear(nn.Linear):
    def __init__(self,
                in_features,
                out_features,
                bias=True,
                a_bits=8,
                w_bits=8,
                quant_inference=False):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        self.activation_quantizer = ActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.linear(quant_input, quant_weight, self.bias)
        return output

class QuantGrad(nn.Module):
    def __init__(self,bitG):
        super(QuantGrad, self).__init__()
        self.bitG = bitG
    def forward(self, input):
        # output = QuantGrad(input)
        return input
    def backward(self,grad_output):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # if make_hist_plot == True:
        #     before_quant = grad_output[0]
        #     before_quant = before_quant.detach()
        #     before_quant = before_quant.view(-1)
        #     # sns.displot(before_quant.detach().numpy())
        #     plt.hist(before_quant.numpy(),bins=4 * int(2**self.bitG - 1),density=True)
        #     plt.title('before_quant')
        #     plt.show()

        x = grad_output.clone()
        rank = x.dim()
        maxx,_ = torch.max(torch.abs(x),dim=1,keepdims = True)
        for i in range(2,rank):
            maxx,_ = torch.max(torch.abs(maxx),dim=i,keepdims = True)
        x = x/maxx
        n = float(2**self.bitG - 1)
        Nk = ((torch.rand(x.size(),device=device)-0.5)/n)
        # print(Nk.type())
        # Nk.to('cuda:0')
        # print(Nk.type())    
        x = x * 0.5 + 0.5 + Nk
        x = torch.clamp(x,0.0,1.0)
        # x = Round.apply(x) -0.5
        x = torch.round(x*n)/n -0.5
        grad_input = x * maxx * 2
        # grad_input = grad_output

        afterquant = grad_input[0]
        afterquant = afterquant.detach()
        afterquant = afterquant.view(-1)
        # if make_hist_plot == True:
        #     plt.hist(afterquant.numpy(),bins=4 * int(2**self.bitG - 1),density=True)
        #     plt.title('after_quant')
        #     plt.show()

        grad_input.to(device)
        return grad_input
