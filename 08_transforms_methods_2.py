# -*- coding: utf-8 -*-
"""
# @brief    :transforms方法二
             transforms方法部分已注释，使用时取消注释
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


set_seed(2)  # 设置随机种子


# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 1
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

    # 根据3通道、1通道，将numpy.array的形式转换为PIL image
    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_


# ====================== step 1/5 数据 ======================
split_dir = os.path.join("data", "object_split")
train_dir = os.path.join(split_dir, "train")
# valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

# 图像预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 1 Pad
    # transforms.Pad(padding=32, fill=(255, 0, 0), padding_mode='constant'),
    # transforms.Pad(padding=(8, 64), fill=(255, 0, 0), padding_mode='constant'),
    # transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='constant'),
    # transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='symmetric'),  # 此时fill不起作用

    # 2 ColorJitter
    # transforms.ColorJitter(brightness=0.5),
    # transforms.ColorJitter(contrast=0.5),
    # transforms.ColorJitter(saturation=0.5),
    # transforms.ColorJitter(hue=0.5),

    # 3 Grayscale
    # transforms.Grayscale(num_output_channels=3),

    # 4 Affine
    # transforms.RandomAffine(degrees=30),
    # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), fillcolor=(255, 0, 0)),
    # transforms.RandomAffine(degrees=0, scale=(0.7, 0.7)),
    # transforms.RandomAffine(degrees=0, shear=(0, 0, 0, 45)),
    # transforms.RandomAffine(degrees=0, shear=90, fillcolor=(255, 0, 0)),

    # 5 Erasing - 在张量上进行
    # transforms.ToTensor(),
    # transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254/255, 0, 0)),  # ratio=(0.3, 3.3)是论文中给出的
    # transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),  # value可以是任意的字符串

    # 1 RandomChoice
    # transforms.RandomChoice([transforms.RandomVerticalFlip(p=1), transforms.RandomHorizontalFlip(p=1)]),

    # 2 RandomApply
    # transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=45, fillcolor=(255, 0, 0)),
    #                         transforms.Grayscale(num_output_channels=3)], p=0.5),

    # 3 RandomOrder
    transforms.RandomOrder([transforms.RandomRotation(15),
                           transforms.Pad(padding=32),
                            transforms.RandomAffine(degrees=0, translate=(0.01, 0.1), scale=(0.9, 1.1))]),



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

        # bs, ncrops, c, h, w = inputs.shape
        # for n in range(ncrops):
        #     img_tensor = inputs[0, n, ...]  # C H W
        #     img = transform_invert(img_tensor, train_transform)
        #     plt.imshow(img)
        #     plt.show()
        #     plt.pause(1)

        break
    break


