# -*- coding: utf-8 -*-
"""
# @brief      : 各数据集的Dataset定义
"""

import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
object_label = {"brain": 0, "chair": 1}


# 数据集都继承自Dataset
class ObjectDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        object分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform, 数据预处理
        """
        self.label_name = {"brain": 0, "chair": 1}
        # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')  # 0~255

        if self.transform is not None:
            # 在这里做transform，转为tensor等等
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    # 定义静态方法，可以直接用类名调用
    def get_img_info(data_dir):
        data_info = list()

        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = object_label[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info


class AntsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.label_name = {"ants": 0, "bees": 1}
        self.data_info = self.get_img_info(data_dir)  # 是个list
        self.transform = transform

    def __getitem__(self, index):  # 拿到序号，读取数据，返回样本和标签
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data_info)

    def get_img_info(self, data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = self.label_name[sub_dir]
                    data_info.append((path_img, int(label)))

        if len(data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(data_info))
        return data_info
