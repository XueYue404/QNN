'''
This file download and prepare cifar10 dataset
'''
import torch
import torchvision
import torchvision.transforms as transforms

def cifar10_data_loader(path,batch_size_train,batch_size_test):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                            shuffle=True)

    testset = torchvision.datasets.CIFAR10(root=path, train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                            shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader,test_loader,classes