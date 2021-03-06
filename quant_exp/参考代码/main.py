# -*- coding: utf-8 -*-
"""Cifar10-dorefa-AlexNet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1T5m2NcRMnPWy59htE0KxnjdovDnETCh7
"""

from google.colab import drive
drive.mount('/content/gdrive')

!nvidia-smi

"""Prepare Dataset"""

from gdrive.MyDrive.utils.cifar10_prepare import cifar10_data_loader
PATH = './content/gdrive/MyDrive/CIFAR10_dataset/data'
batch_size_train,batch_size_test = 64,512
train_loader,test_loader,classes = cifar10_data_loader(PATH,batch_size_train,batch_size_test)

'''
现在的版本是全精度模型和量化模型的对比
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
from torch.utils.data import DataLoader
from torch.autograd import Function

# from torch.utils.tensorboard import SummaryWriter

make_hist_plot = False

#调整不同超参数，记录不同曲线
train_counters_log = []
train_losses_log = []
test_accuracy_log = []
test_counter_log = []
hps = []
bitGs = [16,16]
log_test = False #是否每隔一定间隔去测试一下？
jy_log_interval = 1024 #每512个样本记录画图

for bitG in bitGs:
  n_epochs = 10
  # bitG = 2
  bitA = 16
  bitW = 8
  batch_size_train = 64
  batch_size_test = 1000

  hps.append(f'bitG={bitG}')

  hp_log = f'\nn_epochs = {n_epochs}\nbitG = {bitG}\nbitA = {bitA}\nbitW = {bitW}\nbatch_size_train = {batch_size_train}\n'
  print(hp_log)

  learning_rate = 0.01
  momentum = 0.5
  log_interval = 10


  random_seed = 1
  torch.manual_seed(random_seed)

  # ********************* quantizers（量化器，量化） *********************
  #******************quantization functions begin******************
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


  #******************量化梯度,rbjy!!!!******************
  #******************量化梯度,rbjy!!!!******************
  class QuantGrad(torch.autograd.Function):
      @staticmethod
      def forward(ctx,input):
          # ctx.save_for_backward(input)
          return input
      #   output = input
      #   return output
      @staticmethod
      def backward(ctx,grad_output):

          if make_hist_plot == True:
              before_quant = grad_output[0]
              before_quant = before_quant.detach()
              before_quant = before_quant.view(-1)
              # sns.displot(before_quant.detach().numpy())
              plt.hist(before_quant.numpy(),bins=4 * int(2**bitG - 1),density=True)
              plt.title('before_quant')
              plt.show()


      #   bitG = 8 #假设G的位宽为8位
          x = grad_output.clone()
          # x.to('cpu')
          # print(x.type())

          rank = x.dim()
          # maxx = torch.max(torch.abs(grad_output),keep_dims=True)
          maxx,_ = torch.max(torch.abs(x),dim=1,keepdims = True)
          for i in range(2,rank):
              maxx,_ = torch.max(torch.abs(maxx),dim=i,keepdims = True)
          x = x/maxx
          n = float(2**bitG - 1)
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
          if make_hist_plot == True:
              plt.hist(afterquant.numpy(),bins=4 * int(2**bitG - 1),density=True)
              plt.title('after_quant')
              plt.show()
          # sns.displot(afterquant.detach().numpy())
          grad_input.to(device)
          return grad_input

        # x = x * 0.5 + 0.5 + tf.random_uniform(
        #               tf.shape(x), minval=-0.5 / n, maxval=0.5 / n)
        # x = tf.clip_by_value(x, 0.0, 1.0)
        # x = quantize(x, bitG) - 0.5
        # return x * maxx * 2

  class QuantGradCLS(nn.Module):
    def __init__(self):
      super(QuantGradCLS, self).__init__()
    def forward(self, input):
      # output = QuantGrad(input)
      return input
    def backward(self,grad_output):
      if make_hist_plot == True:
        before_quant = grad_output[0]
        before_quant = before_quant.detach()
        before_quant = before_quant.view(-1)
        # sns.displot(before_quant.detach().numpy())
        plt.hist(before_quant.numpy(),bins=4 * int(2**bitG - 1),density=True)
        plt.title('before_quant')
        plt.show()
  #   bitG = 8 #假设G的位宽为8位
      x = grad_output.clone()
      # x.to('cpu')
      # print(x.type())

      rank = x.dim()
      # maxx = torch.max(torch.abs(grad_output),keep_dims=True)
      maxx,_ = torch.max(torch.abs(x),dim=1,keepdims = True)
      for i in range(2,rank):
          maxx,_ = torch.max(torch.abs(maxx),dim=i,keepdims = True)
      x = x/maxx
      n = float(2**bitG - 1)
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
      if make_hist_plot == True:
          plt.hist(afterquant.numpy(),bins=4 * int(2**bitG - 1),density=True)
          plt.title('after_quant')
          plt.show()
      # sns.displot(afterquant.detach().numpy())
      grad_input.to(device)
      return grad_input

  #******************quantization functions end******************
  #******************quantization functions end******************
  #******************quantization functions end******************
  class AlexNet(nn.Module):
    def __init__(self,isquant=True):
        super(AlexNet, self).__init__()
        self.isquant = isquant
        if self.isquant == True:
          self.cnn = nn.Sequential(
            # 卷积层1，3通道输入，96个卷积核，核大小7*7，步长2，填充2
            # 经过该层图像大小变为32-7+2*2 / 2 +1，15*15
            # 经3*3最大池化，2步长，图像变为15-3 / 2 + 1， 7*7
            QuantConv2d(3, 96, 7, 2, 2,a_bits=bitA,w_bits=bitW),
            QuantGradCLS(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0),

            # 卷积层2，96输入通道，256个卷积核，核大小5*5，步长1，填充2
            # 经过该层图像变为7-5+2*2 / 1 + 1，7*7
            # 经3*3最大池化，2步长，图像变为7-3 / 2 + 1， 3*3
            QuantConv2d(96, 256, 5, 1, 2,a_bits=bitA,w_bits=bitW),
            QuantGradCLS(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0),

            # 卷积层3，256输入通道，384个卷积核，核大小3*3，步长1，填充1
            # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
            QuantConv2d(256, 384, 3, 1, 1,a_bits=bitA,w_bits=bitW),
            QuantGradCLS(),
            nn.ReLU(inplace=True),

            # 卷积层3，384输入通道，384个卷积核，核大小3*3，步长1，填充1
            # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
            QuantConv2d(384, 384, 3, 1, 1,a_bits=bitA,w_bits=bitW),
            QuantGradCLS(),
            nn.ReLU(inplace=True),

            # 卷积层3，384输入通道，256个卷积核，核大小3*3，步长1，填充1
            # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
            QuantConv2d(384, 256, 3, 1, 1,a_bits=bitA,w_bits=bitW),
            QuantGradCLS(),
            nn.ReLU(inplace=True)
          )

          self.fc = nn.Sequential(
              # 256个feature，每个feature 3*3
              QuantLinear(256*3*3, 1024,a_bits=bitA,w_bits=bitW),
              QuantGradCLS(),
              nn.ReLU(),
              QuantLinear(1024, 512,a_bits=bitA,w_bits=bitW),
              QuantGradCLS(),
              nn.ReLU(),
              QuantLinear(512, 10,a_bits=bitA,w_bits=bitW)
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

        # x.size()[0]: batch size
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x

  class DorefaNet(nn.Module):
    # Using Modified AlexNet, baseline accuracy = 75%
    def __init__(self):
        super(DorefaNet, self).__init__()

        # self.conv1 = QuantConv2d(3,6,5,padding=same,a_bits=bitA,w_bits=bitW)
        # self.conv2 = QuantConv2d(6,16,5,a_bits=bitA,w_bits=bitW)
        # self.fc1 = QuantLinear(16*5*5,120,a_bits=bitA,w_bits=bitW)
        # self.fc2 = QuantLinear(120,84,a_bits=bitA,w_bits=bitW)
        # self.fc3 = QuantLinear(84,10,a_bits=bitA,w_bits=bitW)
        # 卷积层1，3通道输入，96个卷积核，核大小7*7，步长2，填充2

        self.conv1 = QuantConv2d(3,96,7,stride=2,padding=2,a_bits=bitA,w_bits=bitW)
        self.conv2 = QuantConv2d(96,256,5,stride=1,padding=2,a_bits=bitA,w_bits=bitW)
        self.conv3 = QuantConv2d(256,384,3,stride=1,padding=1,a_bits=bitA,w_bits=bitW)
        self.conv4 = QuantConv2d(384,384,3,stride=1,padding=1,a_bits=bitA,w_bits=bitW)
        self.conv5 = QuantConv2d(384,256,3,stride=1,padding=1,a_bits=bitA,w_bits=bitW)    

        self.fc1 = QuantLinear(256*3*3,1024,a_bits=bitA,w_bits=bitW)
        self.fc2 = QuantLinear(1024,512,a_bits=bitA,w_bits=bitW)
        self.fc3 = QuantLinear(512,10,a_bits=bitA,w_bits=bitW)

    def forward(self, x):
        quantgrad = QuantGrad.apply
        # x = self.conv1(x)
        # x = quantgrad(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)

        # x = self.conv2(x)
        # x = quantgrad(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)

        # x = x.view(-1, 16*5*5)

        # x = self.fc1(x)
        # x = quantgrad(x)
        # x = F.relu(x)

        # x = self.fc2(x)
        # x = quantgrad(x)
        # x = F.relu(x)

        # x = self.fc3(x)
        x = self.conv1(x)
        x = quantgrad(x)
        x = F.relu(x)
        x = F.max_pool2d(x,3,2,0)

        x = self.conv2(x)
        x = quantgrad(x)
        x = F.relu(x)
        x = F.max_pool2d(x,3,2,0)

        x = self.conv3(x)
        x = quantgrad(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = quantgrad(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = quantgrad(x)
        x = F.relu(x)

        x = x.view(-1)

        x = self.fc1(x)
        x = quantgrad(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = quantgrad(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x
        
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  network = AlexNet(isquant = False).to(device)
  # network = QuantNet()
  # network = JustQuantGrad()
  # optimizer = optim.SGD(network.parameters(), lr=learning_rate,
  #                       momentum=momentum)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(network.parameters())

  # writer = SummaryWriter('runs/MNIST_tensorboard')
  # images,labels=next(iter(train_loader))
  # grid=torchvision.utils.make_grid(images)
  # writer.add_image('images', grid)
  # writer.add_graph(model=network,input_to_model=images)
  # tb.close()


  train_losses = []
  train_counter = []

  test_accuracy = []
  test_counter = []

  # test_losses = []
  # test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

  def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
      data = data.to(device)
      target = target.to(device)
      optimizer.zero_grad()
      output = network(data)
      # loss = F.nll_loss(output, target)
      loss = criterion(output,target)
      if math.isnan(loss.item()):
        print("get nan loss!")
        network.load_state_dict(torch.load('./model.pth'))
        break
      torch.save(network.state_dict(), './model.pth')
      torch.save(optimizer.state_dict(), './optimizer.pth')
      loss.backward()
      optimizer.step()

      # 相等的样本间隔记录、画图
      if (batch_size_train * batch_idx) % jy_log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
          train_losses.append(loss.item())
          train_counter.append(
              (batch_size_train * batch_idx) + ((epoch-1)*len(train_loader.dataset)))
          if log_test == True:
              tl,acc = test()
              test_accuracy.append(acc.item())
              test_counter.append(
                  (batch_size_train * batch_idx) + ((epoch-1)*len(train_loader.dataset)))


      # if batch_idx % log_interval == 0:
      #   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
      #     epoch, batch_idx * len(data), len(train_loader.dataset),
      #     100. * batch_idx / len(train_loader), loss.item()))
      #   train_losses.append(loss.item())
      #   train_counter.append(
      #     (batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))



      #   writer.add_scalar('Train/Loss', loss.item(), (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      #   writer.flush()

  def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = network(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
  #   test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))
    return test_loss,100. * correct / len(test_loader.dataset)



  for epoch in range(1, n_epochs + 1):
    train(epoch)
    
    # writer.add_scalar('Train/Loss', loss.item(), (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
  #   writer.flush()
    test_loss,accuracy = test()
  # import matplotlib.pyplot as plt

  train_counters_log.append(train_counter)
  train_losses_log.append(train_losses)
  test_counter_log.append(test_counter)
  test_accuracy_log.append(test_accuracy)

  result_log = f'\ntrain_counter = {train_counter}\ntrain_losses = {train_losses}'
  
  if log_test:
    result_log = result_log + f'\ntest_accuracy = {test_accuracy}'

  file = open('run_results',mode='a+')
  file.write(hp_log)
  file.write(result_log)
  file.close()

# end hp explore

fig = plt.figure(dpi=1000)
for i in range(len(bitGs)):
  
  if log_test:
    trl = fig.add_subplot(2,1,1) #训练误差
    trl.plot(train_counters_log[i], train_losses_log[i])
    trl.legend(hps,loc='upper right')
    trl.set_xlabel('number of training examples')
    trl.set_ylabel('cross entropy loss')

    tea = fig.add_subplot(2,1,2) #测试精度
    tea.plot(test_counter_log[i], test_accuracy_log[i])
    tea.legend(hps,loc='upper right')
    tea.set_xlabel('number of training examples')
    tea.set_ylabel('test accuracy')
  else:
    plt.plot(train_counters_log[i], train_losses_log[i])
    plt.legend(hps,loc='upper right')
    plt.xlabel('number of training examples')
    plt.ylabel('cross entropy loss')

plt.savefig('exp')
plt.show()