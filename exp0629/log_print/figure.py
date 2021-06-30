'''
画图
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import math
import json

from torch.utils.data import DataLoader
from torch.autograd import Function

#本项目的import
from logger import Logger

if __name__ == "__main__":
    # 这一段代码需要根据作图来修改！！
    PATH1 = '../alexnet-10epoch/log'
    PATH2 = '../qalexnet_22epoch/log'
    PATH3 = '../baseline_mobilenet_result/log'
    log1,log2,log3 = Logger(),Logger(),Logger()
    log1.load_logger(PATH1)
    log2.load_logger(PATH2)
    log3.load_logger(PATH3)

    logs = [log3]
    legend = ['mobilenet']

    #=================================

    # 通用代码
    # 取出训练误差，由于要画在一张图上，所以截断
    train_list_length,test_list_length = len(logs[0].train_counter),len(logs[0].test_counter)
    for i in range(1,len(logs)):
        if len(logs[i].train_counter) < train_list_length:
            train_list_length = len(logs[i].train_counter)
        if len(logs[i].test_counter) < test_list_length:
            test_list_length = len(logs[i].test_counter)

    # 取出训练误差、预测精度
    train_losses,train_counters,test_accs,test_counters = [],[],[],[]
    for i in range(0,len(logs)):
        train_losses.append(logs[i].train_loss[0:train_list_length])
        test_accs.append(logs[i].test_accuracy[0:test_list_length])
        train_counters.append(logs[i].train_counter[0:train_list_length])
        test_counters.append(logs[i].test_counter[0:test_list_length])

    # 下面开始画图
    fig = plt.figure(dpi=1000)
    train_plot = fig.add_subplot(2,1,1)
    for i in range(0,len(logs)):
        train_plot.plot(train_counters[i],train_losses[i])
    train_plot.legend(legend,loc='upper right')
    train_plot.set_xlabel('number of training examples')
    train_plot.set_ylabel('cross entropy loss')

    test_plot = fig.add_subplot(2,1,2)
    for i in range(0,len(logs)):
        test_plot.plot(test_counters[i],test_accs[i])
    test_plot.legend(legend,loc='best')
    test_plot.set_xlabel('epochs')
    test_plot.set_ylabel('accuracy')
    
    plt.tight_layout() # 自动调整子图间距
    plt.savefig('对比baseline和bitG=16')
    # plt.show()