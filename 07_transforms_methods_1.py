# -*- coding: utf-8 -*-
"""
# @brief    :transforms方法
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
from tools.public_my_dataset import ObjectDataset


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed(1)  # 设置随机种子


# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 12
LR = 0.01
log_interval = 10
val_interval = 1
object_label = {"brain": 0, "chair": 1}


def transform_invert(img_, transform_train):
    """
    将data进行反transform操作，使得我们可以观察到模型长什么样子
    将张量数据变换为PIL image，我们就可以对其可视化
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        # normalize是 减去均值 除以方差
        # 这里我们对normalize进行反操作 *方差 +均值
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W -> H*W*C
    img_ = np.array(img_)*255  # 0-1 转换到 0-255

    # 根据3通道、1通道，将nampy.array的形式转换为PIL image
    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_


# ====================== step 1/5 数据 ======================
split_dir = os.path.join("data", "object_split")
# train_dir = os.path.join("..")
train_dir = os.path.join(split_dir, "train")
# valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

# 图像预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 1 CenterCrop
    # transforms.CenterCrop(196),     # 512（外圈填充全黑）  尝试比原尺寸大和比原尺寸小

    # 2 RandomCrop
    # transforms.RandomCrop(224, padding=16),
    # transforms.RandomCrop(224, padding=(16, 64)),
    # transforms.RandomCrop(224, padding=16, fill=(255, 0, 0)),
    # transforms.RandomCrop(512, pad_if_needed=True),   # 当size大于图像尺寸时，必须打开，pad_if_needed=True
    # transforms.RandomCrop(224, padding=64, padding_mode='edge'),
    # transforms.RandomCrop(224, padding=64, padding_mode='reflect'),
    # transforms.RandomCrop(1024, padding=1024, padding_mode='symmetric'),

    # 3 RandomResizedCrop
    transforms.RandomResizedCrop(size=224, scale=(0.5, 0.5)),

    # 4 FiveCrop
    # transforms.FiveCrop(112),
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 5 TenCrop
    # transforms.TenCrop(112, vertical_flip=False),
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 1 Horizontal Flip
    # transforms.RandomHorizontalFlip(p=1),

    # 2 Vertical Flip
    # transforms.RandomVerticalFlip(p=0.5),

    # 3 RandomRotation
    # transforms.RandomRotation(90),
    # transforms.RandomRotation((90), expand=True),
    # transforms.RandomRotation(30, center=(0, 0)),
    # transforms.RandomRotation(30, center=(0, 0), expand=True),   # expand only for center rotation


    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# valid_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(norm_mean, norm_std),
# ])

# 构建MyDataset实例
train_data = ObjectDataset(data_dir=train_dir, transform=train_transform)
# valid_data = ObjectDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoder
# 对batch批数据进行操作
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)


# ====================== step 5/5 训练 ======================

for epoch in range(MAX_EPOCH):
    for i, data in enumerate(train_loader):

        inputs, labels = data  # B C H W

        img_tensor = inputs[0, ...]  # C H W
        img = transform_invert(img_tensor, train_transform)
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()
        break
    break


