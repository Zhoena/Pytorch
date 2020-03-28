# -*- coding: utf-8 -*-
"""
@brife      ：逻辑回归示例
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(10)


# ===================== step 1/5 生成数据 =====================
sample_nums = 100
mean_value = 1.7
bias = 1
n_data = torch.ones(sample_nums, 2)  # 生成100行2列（100 × 2）的全1张量
x0 = torch.normal(mean_value * n_data, 1) + bias  # 正态分布， 类别 0  数据 shape=(100,2)
y0 = torch.zeros(sample_nums)  # 类别 0  标签 shape=(100,1)
x1 = torch.normal(-mean_value * n_data, 1) + bias  # 正态分布， 类别 1  数据 shape=(100,2)
y1 = torch.ones(sample_nums)  # 类别 1  标签 shape=(100,1)
train_x = torch.cat((x0, x1), 0)  # 将两个类别的自变量进行拼接
train_y = torch.cat((y0, y1), 0)


# ===================== step 2/5 选择模型 =====================
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x


lr_net = LR()  # 实例化逻辑回归模型


# ===================== step 3/5 选择损失函数 =====================
loss_fn = nn.BCELoss()  # 二分类交叉熵函数


# ===================== step 4/5 选择优化器 =====================
lr = 0.01  # 学习率
# 随机梯度下降法
optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9)


# ===================== step 5/5 模型训练 =====================
for iteration in range(1000):

    # 前向传播
    y_pred = lr_net(train_x)

    # 计算 loss
    loss = loss_fn(y_pred.squeeze(), train_y)  # 将模型的输出与标签输入给损失函数

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 绘图
    if iteration % 20 == 0:

        # ge()比较函数，>=0.5为true：float() 改变数据类型；squeeze() 压缩数据维度
        mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
        correct = (mask == train_y).sum()  # 计算正确预测的样本个数
        acc = correct.item() / train_y.size(0)  # 计算分类准确率

        # 图例
        plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
        plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

        w0, w1 = lr_net.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        plot_b = float(lr_net.features.bias[0].item())
        plot_x = np.arange(-6, 6, 0.1)
        plot_y = (-w0 * plot_x - plot_b) / w1

        plt.xlim(-5, 7)
        plt.xlim(-7, 7)
        plt.plot(plot_x, plot_y)

        plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.title("Iteration: {}\nw0:{:.2f} w1:{:.2f} b:{:.2f} accuracy:{:.2%}".format(iteration, w0, w1, plot_b, acc))
        plt.legend()

        plt.show()
        plt.pause(0.5)

        if acc > 0.99:
            break
