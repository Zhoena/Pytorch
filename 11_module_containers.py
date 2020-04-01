# -*- coding: utf-8 -*-
"""
# @date       : 2020-03-31 22:30
# @brief      : 模型容器——Sequential, ModuleList, ModuleDict
"""

import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict


# =========================  Sequential  =============================

# Sequential也继承module，也有8个有序字典
class LeNetSequential(nn.Module):
    def __init__(self, classes):
        super(LeNetSequential, self).__init__()
        self.features = nn.Sequential(  # 只能通过序号索引
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),  # 全连接层
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes),
        )

    def forward(self, x):
        # 2个子模块
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x


class LeNetSequentialOrderDict(nn.Module):
    def __init__(self, classes):
        super(LeNetSequentialOrderDict, self).__init__()

        self.features = nn.Sequential(OrderedDict({  # 可以通过名称Key索引
            'conv1': nn.Conv2d(3, 6, 5),
            'relu1': nn.ReLU(inplace=True),
            'pool1': nn.MaxPool2d(kernel_size=2, stride=2),

            'conv2': nn.Conv2d(6, 16, 5),
            'relu2': nn.ReLU(inplace=True),
            'pool2': nn.MaxPool2d(kernel_size=2, stride=2),
        }))

        self.classifier = nn.Sequential(OrderedDict({
            'fc1': nn.Linear(16*5*5, 120),
            'relu3': nn.ReLU(),

            'fc2': nn.Linear(120, 84),
            'relu4': nn.ReLU(inplace=True),

            'fc3': nn.Linear(84, classes),
        }))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x


# net = LeNetSequential(classes=2)
# net = LeNetSequentialOrderDict(classes=2)

# 用标准正态分布创建一个随机张量，作为图片输入
# fake_img = torch.randn((4, 3, 32, 32), dtype=torch.float32)

# output = net(fake_img)
# print(net)
# print(output)


# =========================  ModuleList  =============================

class ModuleList(nn.Module):
    def __init__(self):
        super(ModuleList, self).__init__()

        # 构建20个全连接层，每个全连接层是10个神经元  的网络  nn.Linear是全连接层
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(20)])

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
        return x


# net = ModuleList()
# print(net)
#
# fake_data = torch.ones((10, 10))
#
# output = net(fake_data)
# print(output)


# =========================  ModuleDict  =============================

class ModuleDict(nn.Module):
    def __init__(self):
        super(ModuleDict, self).__init__()
        self.choices = nn.ModuleDict({
            'conv': nn.Conv2d(10, 10, 3),
            'pool': nn.MaxPool2d(3)
        })

        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'prelu': nn.PReLU()
        })

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x


net = ModuleDict()

fake_img = torch.randn((4, 10, 32, 32))

# 两个参数，第一个参数选择choices中选哪个，第二个参数选择activations中选哪个
# 通过名字选择网络层，实现forward
output = net(fake_img, 'conv', 'relu')

print(output)


# 4 AlexNet

alexnet = torchvision.models.AlexNet()





