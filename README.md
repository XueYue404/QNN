# NN_quant
神经网络量化训练

##文件结构:
###主目录:
main.py --> 训练、测试、超参数选择

###model文件夹: 存放量化器，以及量化后的各种NN模型
  quant.py --> 量化器，封装成各种nn.Module
  qalexnet.py --> 针对Cifar-10设置形状
  qlenet.py

###utils文件夹: 各种与量化任务无关的helper functions
  logger.py --> 用来记录不同epoch的训练、测试结果，以及超参数
  cifar10_prepare.py --> 准备cifar10数据集，返回train_loader和test_loader供使用
  figure.py --> 用来画图，注意要改该文件里的路径
  
###dataset文件夹:(repository里没显示)
  用于存放数据集

###result文件夹：存放运行结果
  checkpoint
  
