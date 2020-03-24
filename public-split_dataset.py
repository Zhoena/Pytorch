# -*- coding: utf-8 -*-
"""
# @brief    :将数据集(101_object_categories数据集中的brain、chair两个类)
             划分为训练集(80%)，验证集(10%)，测试集(10%)
             数据集：.\data\object_orign\两个类
             划分之后的数据集：.\data\object_split\训练、验证、测试\两个类
"""

import os
import random
import shutil


# 创建文件夹
def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':
    random.seed(1)

    # os.path.join路径拼接
    dataset_dir = os.path.join("data", "object_orign")
    split_dir = os.path.join("data", "object_split")
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")

    # 训练集、验证集、测试集所占比例
    train_pct = 0.8
    valid_pct = 0.1
    test_pct = 0.1

    # os.walk() 在目录树中游走输出在目录中的文件名
    for root, dirs, files in os.walk(dataset_dir):
        # root 为根目录， dataset_dir目录
        # dirs 为根目录下的所有子目录list
        # files为根目录下文件
        # 第二次遍历dirs中的每一个目录
        for sub_dir in dirs:
            # os.listdir()返回指定的文件夹包含的文件或文件夹的名字的列表。
            imgs = os.listdir(os.path.join(root, sub_dir))  # 每个类中所有图片
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))  # endswith() 判断字符串是否以指定后缀结尾
            random.shuffle(imgs)  # 将序列的所有元素随机排序
            img_count = len(imgs)

            train_point = int(img_count * train_pct)  # 训练集截止点
            valid_point = int(img_count * (train_pct + valid_pct))  # 验证集截止点

            for i in range(img_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                makedir(out_dir)

                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])
                print(target_path)
                print(src_path)

                shutil.copy(src_path, target_path)  # 从src_path复制文件内容到target_path

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir, train_point, valid_point-train_point, img_count-valid_point))




