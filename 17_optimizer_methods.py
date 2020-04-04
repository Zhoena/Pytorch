# -*- coding: utf-8 -*-
"""
# @date       : 2020-04-04 11:30
# @brief      : optimizer's methods
"""

import os
import torch
import torch.optim as optim
from tools.common_tools import set_seed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前执行脚本的完整路径

set_seed(1)

weight = torch.randn((2, 2), requires_grad=True)
weight.grad = torch.ones((2, 2))

optimizer = optim.SGD([weight], lr=0.1)


# ------------------- step -------------------
flag = 0
# flag = 1
if flag:
    print("weight before step:{}".format(weight.data))
    optimizer.step()  # 修改lr=1 0.1观察结果
    print("weight after step:{}".format(weight.data))


# ------------------- zero_grad -------------------
# 清空所管理参数的梯度
flag = 0
# flag = 1
if flag:

    print("weight before step:{}".format(weight.data))
    optimizer.step()  # 修改lr=1 0.1 观察结果
    print("weight after step:{}".format(weight.data))

    # 优化器当中参数地址和真实参数地址是一样的，复制地址，节省内存消耗
    print("weight in optimizer:{}\nweight in weight:{}\n".format(id(optimizer.param_groups[0]['params'][0]), id(weight)))

    print("weight.grad is {}".format(weight.grad))
    optimizer.zero_grad()
    print("after optimizer.zero_grad(), weight.grad is {}".format(weight.grad))


# ------------------- add_param_group -------------------
# 添加参数组
flag = 0
# flag = 1
if flag:

    print("optimizer.param_groups is\n{}".format(optimizer.param_groups))

    w2 = torch.randn((3, 3), requires_grad=True)

    optimizer.add_param_group(({"params": w2, 'lr': 0.0001}))  # 微调非常实用，不同参数不同学习率

    print("optimizer.param_groups is\n{}".format(optimizer.param_groups))


# ------------------- state_dict -------------------
# 获取优化器当前状态信息字典
flag = 0
# flag = 1
if flag:

    optimizer = optim.SGD([weight], lr=0.1, momentum=0.9)
    opt_state_dict = optimizer.state_dict()

    print("state_dict before step:\n", opt_state_dict)

    for i in range(10):
        optimizer.step()

    print("state_dict after step:\n", optimizer.state_dict())

    torch.save(optimizer.state_dict(), os.path.join(BASE_DIR, "optimizer_state_dict.pkl"))  # 保存字典


# ------------------- state_dict -------------------
# 加载状态信息字典
# flag = 0
flag = 1
if flag:

        optimizer = optim.SGD([weight], lr=0.1, momentum=0.9)
        state_dict = torch.load(os.path.join(BASE_DIR, "optimizer_state_dict.pkl"))

        print("state_dict before load state:\n", optimizer.state_dict())
        optimizer.load_state_dict(state_dict)
        print("state_dict after load state:\n", optimizer.state_dict())


