# -*- coding:utf-8 -*-
"""
@date       : 2020-04-09 17:40
@brief      : 卷积核和特征图的可视化
"""

import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tools.common_tools import set_seed
import torchvision.models as models

set_seed(1)


# ---------------- kernel visualization -----------------
# flag = 0
flag = 1
if flag:

    writer = SummaryWriter(comment="test_your_comment", filename_suffix="_test_your_filename_suffix")

    alexnet = models.alexnet(pretrained=True)  # 获取预训练的alexnet，参数在imagenet上训练过的

    kernel_num = -1  # 当前第几个卷积层
    vis_max = 1  # 最大可视化层

    for sub_module in alexnet.modules():  # 每个模型都有modules，它可以递归地返回子modules
        if isinstance(sub_module, nn.Conv2d):
            kernel_num += 1
            if kernel_num > vis_max:
                break
            kernels = sub_module.weight
            c_out, c_int, k_w, k_h = tuple(kernels.shape)  # 4D

            for o_idx in range(c_out):
                kernel_idx = kernels[o_idx, :, :, :].unsqueeze(1)  # make_grid需要 BCHW，这里拓展C维度
                kernel_grid = vutils.make_grid(kernel_idx, normalize=True, scale_each=True, nrow=c_int)
                writer.add_image('{}_Convlayer_split_in_channel'.format(kernel_num), kernel_grid, global_step=o_idx)

            kernel_all = kernels.view(-1, 3, k_h, k_w)  # 3, h, w
            kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=8)  # c, h, w
            writer.add_image('{}_all'.format(kernel_num), kernel_grid, global_step=322)

            print("{}_convlayer shape:{}".format(kernel_num, tuple(kernels.shape)))

        writer.close()


# ---------------- feature map visualization -----------------
# flag = 0
flag = 1
if flag:
    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

    # 数据
    path_img = "./lena.png"
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]

    norm_transform = transforms.Normalize(normMean, normStd)
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm_transform
    ])

    img_pil = Image.open(path_img).convert('RGB')
    if img_transforms is not None:
        img_tensor = img_transforms(img_pil)
    img_tensor.unsqueeze_(0)  # chw --> bchw

    # 模型
    alexnet = models.alexnet(pretrained=True)

    # forward
    convlayer1 = alexnet.features[0]
    fmap_1 = convlayer1(img_tensor)

    # 预处理
    fmap_1.transpose_(0, 1)  # bchw=(1, 64, 55, 55) --> (64, 1, 55, 55)
    fmap_1_grid = vutils.make_grid(fmap_1, normalize=True, scale_each=True, nrow=8)

    writer.add_image('feature map in conv1', fmap_1_grid, global_step=322)
    writer.close()