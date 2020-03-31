# -*- coding: utf-8 -*-
"""
# @brief        :自定义一个transforms方法 - 椒盐噪声
"""

import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tools.public_my_dataset import ObjectDataset
from torch.utils.data import DataLoader
from tools.common_tools import transform_invert
from matplotlib import pyplot as plt


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed(1)  # 设置随机种子

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1
object_label = {"brain": 0, "chair": 1}


class AddPepperNoise(object):
    """
    增加椒盐噪声
    Args:
        snr (float) : Signal Noise Rate - 信噪比
        p (float) : 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        :param img: PIL image
        :return: PIL image
        """
        if random.uniform(0, 1) < self.p:  # 依概率
            img_ = np.array(img).copy()  # 将PIL image 转成np.array
            h, w, c = img_.shape
            signal_pct = self.snr  # 设置信噪比，保存self.snr为原始图像，默认为90%
            noise_pct = (1 - self.snr)  # 噪声 10%
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])  # 0 原始图像，1 盐噪声 2 椒噪声
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255  # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')  # 转化为PIL image类型
        else:
            return img


# ========================== step 1/5 数据 ===================================
split_dir = os.path.join("data", "object_split")
train_dir = os.path.join(split_dir, "train")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 输出PIL image
    AddPepperNoise(0.9, p=0.5),  # 输入PIL image，输出PIL image
    transforms.ToTensor(),  # 接收PIL image
    transforms.Normalize(norm_mean, norm_std),
])

# 构建MyDataset实例
train_data = ObjectDataset(data_dir=train_dir, transform=train_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


# ========================== step 5/5 训练 ===================================
for epoch in range(MAX_EPOCH):
    for i, data in enumerate(train_loader):

        inputs, labels = data  # B C H W

        img_tensor = inputs[0, ...]  # C H W
        img = transform_invert(img_tensor, train_transform)
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()
