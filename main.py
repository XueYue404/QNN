# 解决MAC的怪异问题
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from seaborn.utils import load_dataset
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


#本项目的import
from utils.cifar10_prepare import cifar10_data_loader
from utils.logger import Logger
from model.qalexnet import AlexNet
from model.mobilenetV2 import MobileNetV2
from model.qmobilenetV2 import QMobileNetV2

# 定义超参数和设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hp = {
  'n_epochs' : 30,
  'bitW' : 4,
  'bitA' : 8,
  'bitG' : 8,
  'batch_size_train' : 64,
  'batch_size_test' : 512,
  'train_log_interval' : 1024,
  'learning_rate' : 0.01,
  'momentum' : 0.5,
  'log_interval' : 10
}
logger = Logger()# 记录器

# 加载数据集
CIFAR10_PATH = './dataset'# 存放cifar10数据集的目录
train_loader,test_loader,classes = cifar10_data_loader(CIFAR10_PATH,hp['batch_size_train'],hp['batch_size_test'])

# 定义各种目录
CHECKPOINT_PATH = './result/checkpoint'
BESTMODEL_PATH = './result/bestmodel_param'
LOG_RESULT_PATH= './result/log'

# 全局变量
best_acc = 0

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
      data,target = data.to(device),target.to(device)
      optimizer.zero_grad()
      output = network(data)
      loss = criterion(output,target)
      # 如果损失计算出nan,则停止训练，并加载此前的模型参数
      if math.isnan(loss.item()):
        print("=====get nan loss!=====")
        checkpoint = torch.load(CHECKPOINT_PATH + '/ckpt.pth')
        network.load_state_dict(checkpoint['network'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        break
      # 记录检查点
      state = {
        'network':network.state_dict(),
        'optimizer':optimizer.state_dict(),
        'epoch':epoch,
        'loss':loss
      }
      torch.save(state,CHECKPOINT_PATH + '/ckpt.pth')
      loss.backward()
      optimizer.step()

      # 每当间隔train_log_interval个训练样本，记录到logger
      num_of_train_samples_this_epoch = batch_idx * len(data)
      if num_of_train_samples_this_epoch % hp['train_log_interval'] == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch,num_of_train_samples_this_epoch, len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
          logger.train_loss.append(loss.item())
          logger.train_counter.append(
              num_of_train_samples_this_epoch + ((epoch-1)*len(train_loader.dataset)))
      # break  

def test(epoch):
    network.eval()
    test_loss = 0
    correct = 0
    acc = 0
    with torch.no_grad():
      for data, target in test_loader:
        data,target = data.to(device),target.to(device)
        output = network(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset) * len(data)
    acc = correct / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * acc))

    # logger记录
    logger.test_counter.append(epoch)
    logger.test_accuracy.append(acc.item())
 

    # 记录最佳精度，用到全局变量
    global best_acc
    if acc > best_acc:
      best_acc = acc
      state = {
        'network':network.state_dict(),
        'optimizer':optimizer.state_dict(),
        'epoch':epoch,
        'loss':test_loss
      }
      torch.save(state,BESTMODEL_PATH + '/best_model_state.pth')


# 设置随机种子，使每次网络构造初始值相同
random_seed = 1
torch.manual_seed(random_seed)

# network = AlexNet(isquant = False).to(device)
# network = MobileNetV2(num_classes=10).to(device)
network = QMobileNetV2(
    num_classes=10,
    bitW=hp['bitW'],
    bitA=hp['bitA'],
    bitG=hp['bitG']
  ).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters())


# logger记录超参数
logger.hyper_param = hp
#从checkpoint恢复训练
start_epoch = 1
end_epoch = hp['n_epochs'] + 1
load_check_point = False
if load_check_point == True:
  print("=====Loading Checkpoint====>")
  checkpoint = torch.load(CHECKPOINT_PATH + '/ckpt.pth')
  network.load_state_dict(checkpoint['network'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  start_epoch = checkpoint['epoch']

for epoch in range(start_epoch,end_epoch):
  train(epoch)
  test(epoch)   
  logger.epoch.append(epoch)
  logger.save_logger(LOG_RESULT_PATH)