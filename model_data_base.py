import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import math
from torchvision import models
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
class Net_cifar100_s(nn.Module):
    def __init__(self):
        super(Net_cifar100_s, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 100)
        # self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x
mycfg = {
    'CNN7':  [64, 'M', 256, 'M', 512, 'M', 512, 'M'], 
}
class Net_cifar100_m(nn.Module):
    def __init__(self, CNN_name='CNN7', dropout=False):
        super(Net_cifar100_m, self).__init__()
        self.query_num = 0
        self.features = self._make_layers(mycfg[CNN_name])
        if dropout:
            self.classifier = nn.Sequential(
                nn.Dropout(0.6),
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.Linear(256, 100) )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.Linear(256, 100) )
        
    def forward(self, x):
        self.query_num += 1
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x, track_running_stats=True),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

class Net_mnist(nn.Module):
    def __init__(self):
        super(Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class Net_cifar10(nn.Module):
    def __init__(self):
        super(Net_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def dataloader(data_name = 'cifar100'):
    if data_name == 'cifar100':
        trainset = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        testset = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    if data_name == 'mnist':
        trainset = datasets.MNIST('./data/MNIST', train=True, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
        testset = datasets.MNIST('./data/MNIST', train=False, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
    if data_name == 'cifar10':
        trainset = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        testset = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = DataLoader(trainset, batch_size=64, shuffle=False, **kwargs)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False, **kwargs)
    split_index = [int(trainset.__len__() / 5)] * 4
    split_index.append(
        int(trainset.__len__() - int(trainset.__len__() / 5) * 4))
    client_dataset = torch.utils.data.random_split(trainset, split_index)
    client_loaders = []
    for ii in range(5):
        client_loaders.append(
            DataLoader(client_dataset[ii], 64, shuffle=False, **kwargs))
    return client_loaders, train_loader, test_loader
