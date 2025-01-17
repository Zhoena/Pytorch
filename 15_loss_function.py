# -*- coding: utf-8 -*-
"""
# @date       : 2020-04-02 19:00
# @brief      : 1. nn.CrossEntropyLoss
                2. nn.NLLLoss
                3. BCELoss
                4. BCEWithLogitsLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# fake data
inputs = torch.tensor([[1, 2], [1, 3], [1, 3]], dtype=torch.float)
target = torch.tensor([0, 1, 1], dtype=torch.long)  # label为长整型，类别数从0开始


# ========================  CrossEntropy loss: reduction  =======================
# flag = 1
flag = 0
if flag:
    # def loss function
    loss_f_none = nn.CrossEntropyLoss(weight=None, reduction='none')
    loss_f_sum = nn.CrossEntropyLoss(weight=None, reduction='sum')
    loss_f_mean = nn.CrossEntropyLoss(weight=None, reduction='mean')

    # forward
    loss_none = loss_f_none(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    # view
    print("Cross Entropy Loss:\n", loss_none, loss_sum, loss_mean)


# ========================  compute by hand  =======================
# flag = 1
flag = 0
if flag:

    idx = 0

    input_1 = inputs.detach().numpy()[idx]  # [1, 2]
    target_1 = target.numpy()[idx]  # [0]  转化为np array

    # 第一项
    x_class = input_1[target_1]

    # 第二项
    sigma_exp_x = np.sum(list(map(np.exp, input_1)))
    log_sigma_exp_x = np.log(sigma_exp_x)

    # 输出loss
    loss_1 = -x_class + log_sigma_exp_x

    print("第一个样本loss为：", loss_1)


# ========================  weight  =======================
# flag = 1
flag = 0
if flag:
    # def loss function
    weights = torch.tensor([1, 2], dtype=torch.float)  # 每个类别都要设置weight，不想关注的设置为1
    # weights = torch.tensor([0.7, 0.3], dtype=torch.float)

    loss_f_none_w = nn.CrossEntropyLoss(weight=weights, reduction='none')
    loss_f_sum_w = nn.CrossEntropyLoss(weight=weights, reduction='sum')
    loss_f_mean_w = nn.CrossEntropyLoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target)
    loss_sum_w = loss_f_sum_w(inputs, target)
    loss_mean_w = loss_f_mean_w(inputs, target)

    # view
    print("\nweights:", weights)
    print(loss_none_w, loss_sum_w, loss_mean_w)


# ========================  compute by hand  =======================
# flag = 1
flag = 0
if flag:
    weights = torch.tensor([1, 2], dtype=torch.float)
    weights_all = np.sum(list(map(lambda x: weights.numpy()[x], target.numpy())))  #[0, 1, 1] -> [1 2 2]

    mean = 0
    loss_sep = loss_none.detach().numpy()
    for i in range(target.shape[0]):
        x_class = target.numpy()[i]
        tmp = loss_sep[i] * (weights.numpy()[x_class] / weights_all)
        mean += tmp

    print(mean)


# ========================  2 NLLLoss  =======================
# flag = 1
flag = 0
if flag:

    weights = torch.tensor([1, 1], dtype=torch.float)  # 相当于没设置

    loss_f_none_w = nn.NLLLoss(weight=weights, reduction='none')
    loss_f_sum_w = nn.NLLLoss(weight=weights, reduction='sum')
    loss_f_mean_w = nn.NLLLoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target)
    loss_sum_w = loss_f_sum_w(inputs, target)
    loss_mean_w = loss_f_mean_w(inputs, target)

    # view
    print("\nweights:", weights)
    print("NLL Loss", loss_none_w, loss_sum_w, loss_mean_w)


# ========================  3 BCE Loss  =======================
# flag = 1
flag = 0
if flag:
    inputs = torch.tensor([[1, 2], [2, 2], [3, 4], [4, 5]], dtype=torch.float)
    target = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)  # 一一对应的神经元计算Loss

    target_bce = target

    # itarget 将输出值压缩到0~1之间，符合概率取值
    inputs = torch.sigmoid(inputs)

    weights = torch.tensor([1, 1], dtype=torch.float)

    loss_f_none_w = nn.BCELoss(weight=weights, reduction='none')
    loss_f_sum_w = nn.BCELoss(weight=weights, reduction='sum')
    loss_f_mean_w = nn.BCELoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target)
    loss_sum_w = loss_f_sum_w(inputs, target)
    loss_mean_w = loss_f_mean_w(inputs, target)

    # view
    print("\nweights:", weights)
    print("BCE Loss", loss_none_w, loss_sum_w, loss_mean_w)  # 每个神经元一一对应计算Loss


# ========================  compute by hand  =======================
# flag = 1
flag = 0
if flag:

    idx = 0
    x_i = inputs.detach().numpy()[idx, idx]
    y_i = target.numpy()[idx, idx]

    # loss
    # l_i = -[y_i * np.log(x_i) + (1-y_i) * np.log(1-y_i)]  # np.log(0) = nan
    l_i = -y_i * np.log(x_i) if y_i else -(1-y_i) * np.log(1-x_i)

    # 输出loss
    print("BCE inputs: ", inputs)
    print("第一个loss为: ", l_i)


# ========================  4 BCE with Logis Loss  =======================
# flag = 1
flag = 0
if flag:
    inputs = torch.tensor([[1, 2], [2, 2], [3, 4], [4, 5]], dtype=torch.float)
    target = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)  # 一一对应的神经元计算Loss

    target_bce = target

    weights = torch.tensor([1, 1], dtype=torch.float)

    loss_f_none_w = nn.BCEWithLogitsLoss(weight=weights, reduction='none')
    loss_f_sum_w = nn.BCEWithLogitsLoss(weight=weights, reduction='sum')
    loss_f_mean_w = nn.BCEWithLogitsLoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target)
    loss_sum_w = loss_f_sum_w(inputs, target)
    loss_mean_w = loss_f_mean_w(inputs, target)

    # view
    print("\nweights:", weights)
    print("BCE with Logis Loss", loss_none_w, loss_sum_w, loss_mean_w)  # 每个神经元一一对应计算Loss


# ========================  pos weight  =======================
# flag = 1
flag = 0
if flag:
    inputs = torch.tensor([[1, 2], [2, 2], [3, 4], [4, 5]], dtype=torch.float)
    target = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)

    target_bce = target

    weights = torch.tensor([1], dtype=torch.float)
    pos_w = torch.tensor([3], dtype=torch.float)  # 3

    loss_f_none_w = nn.BCEWithLogitsLoss(weight=weights, reduction='none', pos_weight=pos_w)
    loss_f_sum = nn.BCEWithLogitsLoss(weight=weights, reduction='sum', pos_weight=pos_w)
    loss_f_mean = nn.BCEWithLogitsLoss(weight=weights, reduction='mean', pos_weight=pos_w)

    # forward
    loss_none_w = loss_f_none_w(inputs, target_bce)
    loss_sum = loss_f_sum(inputs, target_bce)
    loss_mean = loss_f_mean(inputs, target_bce)

    # view
    print("\npos_weights: ", pos_w)
    print(loss_none_w, loss_sum, loss_mean)












