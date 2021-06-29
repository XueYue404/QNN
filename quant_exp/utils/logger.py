"""
通用的logger类，用于记录超参数和训练/测试结果
"""

import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import seaborn as sns
# import math
import json

# from torch.utils.data import DataLoader
# from torch.autograd import Function


class Logger():
    def __init__(self) -> None:
        self.hyper_param = {}
        self.train_counter = []
        self.train_loss = []
        self.test_accuracy = []
        self.test_counter = []
        self.epoch = []
    def save_logger(self,save_path):
        current_info = {
            'hyper_param':self.hyper_param,
            'train_counter':self.train_counter,
            'train_loss':self.train_loss,
            'test_accuracy':self.test_accuracy,
            'test_counter':self.test_counter,
            'epoch':self.epoch
        }
        # new_info = {}
        with open(save_path + '/log.json','w') as f_obj: #读写模式，指针位于文件头
            # prev_info = json.load(f_obj) #先读之前的内容，然后追加
            # new_info['hyper_param'] = self.hyper_param
            # if bool(prev_info): #如果字典不为空
            #     new_info['train_counter'] = prev_info['train_counter'] + self.train_counter
            #     new_info['train_loss'] = prev_info['train_counter'] + self.train_loss
            #     new_info['test_accuracy'] = prev_info['test_accuracy'] + self.test_accuracy
            #     new_info['test_counter'] = prev_info['test_counter'] + self.test_counter
            #     new_info['epoch'] = prev_info['epoch'] + self.epoch
            # else:
            #     new_info = current_info
            # json.dump(new_info,f_obj)
            json.dump(current_info,f_obj)

    def load_logger(self,load_path):
        with open(load_path + '/log.json','r') as f_obj:
            info = json.load(f_obj)
            self.hyper_param = info['hyper_param']
            self.train_counter = info['train_counter']
            self.train_loss = info['train_loss']
            self.test_accuracy = info['test_accuracy']
            self.test_counter = info['test_counter']
            self.epoch = info['epoch']
    

if __name__ == "__main__":
    PATH = '../test'
    logger = Logger()
    logger.hyper_param = {
        'net' : [1,2,3],
        'score' : 5,
        'name' : 'mike'
    }
    logger.test_accuracy.append([100,200])
    logger.save_logger(PATH)

    logger.hyper_param = {
        'net' : [4,5,6],
        'score' : 7,
        'name' : 'mike'
    }
    logger.test_accuracy =[300,400]
    logger.save_logger(PATH)

    logger.load_logger(PATH)
    pass
