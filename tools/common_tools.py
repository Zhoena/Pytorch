# -*- coding: utf-8 -*-
"""
# @brief        :通用函数
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


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

    if 'ToTensor' in str(transform_train):
        img_ = np.array(img_)*255  # 0-1 转换到 0-255

    # 根据3通道、1通道，将numpy.array的形式转换为PIL image
    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_