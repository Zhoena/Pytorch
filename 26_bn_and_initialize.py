# -*- coding: utf-8 -*-
"""
# @date       : 2020-04-14 17:00
# @brief      : bn与权值初始化
"""

import torch
import numpy as np
import torch.nn as nn
from tools.common_tools import set_seed

set_seed(1)


class MLP(nn.Module):
    def __init__(self, neural_num, layers=100):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([
            nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(neural_num) for i in range(layers)
        ])
        self.neural_num = neural_num

    def forward(self, x):

        for (i, linear), bn in zip(enumerate(self.linears), self.bns):
            x = linear(x)
            x = bn(x)  # 要在激活函数之前使用
            x = torch.relu(x)

            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break

            print("layers:{}, std:{}".format(i, x.std().item()))

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):

                # method 1  normal:nean=0, std=1  35层nan
                # nn.init.normal_(m.weight.data, std=1)

                # method 2 kaiming
                nn.init.kaiming_normal_(m.weight.data)


neural_nums = 256
layer_nums = 100
batch_size = 16

net = MLP(neural_nums, layer_nums)
# net.initialize()

inputs = torch.randn((batch_size, neural_nums))

output = net(inputs)
print(output)