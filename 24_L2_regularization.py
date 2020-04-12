# -*- coding:utf-8 -*-
"""
@date       : 2020-04-12 22:00
@brief      : weight decay使用实验
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tools.common_tools import set_seed
from torch.utils.tensorboard import SummaryWriter

set_seed(1)
