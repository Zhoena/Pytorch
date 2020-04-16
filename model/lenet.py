# -*- coding: utf-8 -*-

"""
# @brief      : lenet模型定义
"""

import torch.nn as nn
import torch.nn.functional as F


# 模型都继承自字nn.Module
class LeNet(nn.Module):
    def __init__(self, classes):
        """
        在__init__中构建网络层子模块
        定义网络需要的操作算子，比如卷积、全连接算子等等
        :param classes: 要划分的类个数
        """
        super(LeNet, self).__init__()  # 父类函数调用 - 即调用nn.Module的__init__()  -  构建8个有序字典
        # nn.Conv2d的第一个参数是输入的channel数量，第二个是输出的channel数量，第三个是kernel size
        # nn.Linear的第一个参数是输入的维数，第二个参数是输出的维数
        # 输入 32 * 32 * 3

        self.conv1 = nn.Conv2d(3, 6, 5)  # 得到(32-5+1)*(32-5+1))*6 = 28*28*6
        # nn.Conv2d(3, 6, 5)是一个module，会记录在_modules的字典当中，
        # 并且也有8个有序字典，由于没有子层，所有_modules为空，但会记录可学习参数，_parameters中有weight、bias
        # parameters类继承于张量，有data、dvice...

        self.conv2 = nn.Conv2d(6, 16, 5)
        # module会拦截类属性赋值，module会判断值是什么类型（setattr），
        # 判断是parameters还是modules，并根据类型赋值到相应的有序字典中，比如，是module还是parameter

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        """
        在forward中拼接网络层子模块
        :param x: inputs - 模型的输入
        :return: outputs - 模型的输出
        """
        out = F.relu(self.conv1(x))  # 得到(32-5+1)*(32-5+1))*6 = 28*28*6
        out = F.max_pool2d(out, 2)  # 14*14*6
        out = F.relu(self.conv2(out))  # (14-5+1)*(14-5+1)*16 = 10*10*16
        out = F.max_pool2d(out, 2)  # 5*5*16 = 400
        out = out.view(out.size(0), -1)  # 将高维数据压缩成低维
        out = F.relu(self.fc1(out))  # 400 -> 120
        out = F.relu(self.fc2(out))  # 120 -> 84
        out = self.fc3(out)  # 84
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()


class LeNet2(nn.Module):
    def __init__(self, classes):
        super(LeNet2, self).__init__()
        self.features = nn.Sequential(  # 一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)  # 将高维数据压缩成低维
        x = self.classifier(x)
        return x


class LeNet_bn(nn.Module):
    def __init__(self, classes):
        super(LeNet_bn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(num_features=6)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(num_features=120)

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = F.max_pool2d(out, 2)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.bn3(out)
        out = F.relu(out)

        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 1)
                m.bias.data.zero_()