# -*- coding:utf-8 -*-
"""
@date       : 2020-04-06
@brief      : 测试tensorboard可正常使用
"""
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 记录想要可视化的数据
writer = SummaryWriter(comment='test_tensorboard')

for x in range(100):
    writer.add_scalar('y=2x', x * 2, x)
    writer.add_scalar('y=pow(2, x)', 2 ** x, x)

    writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x),
                                             "xcosx": x * np.cos(x),
                                             "arctanx": np.arctan(x)}, x)
# 记录完成之后会在当前目录下新建一个runs文件夹，会保存一个eventfile到硬盘当中
writer.close()

