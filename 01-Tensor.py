import numpy as np
import torch

'''
运行时将对应模块flag设置为True
'''

# ============================= 直接创建 =======================
# flag = True
flag = False
if flag:
    arr = np.ones((3, 3))
    print("ndarray的数据类型：", arr.dtype)

    # t = torch.tensor(arr)
    # 将张量运行在gpu上
    t = torch.tensor(arr, device='cuda')

    print(t)

# ============================= 直接创建 =======================
# 通过 torch.from_numpy 创建
# 从torch.form_numpy创建的tensor与原ndarray共享内存，当修改其中一个数据，另外一个也将会被改动
# flag = True
flag = False
if flag:
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    t = torch.from_numpy(arr)
    print(
        "\nnumpy array1:", arr,
        "\ntensor1:", t,
    )

    arr[0, 0] = 0
    print(
        "\nnumpy array2:", arr,
        "\ntensor2:", t,
    )

    t[0, 0] = -1
    print(
        "\nnumpy array3:", arr,
        "\ntensor3:", t,
    )


# ==============================  依据数值创建  ===================================
# 通过torch.zeros创建张量
# flag = True
flag = False
if flag:
    out_t = torch.tensor([1])
    t = torch.zeros((3, 3), out=out_t)

    # t 与 out_t 其实是同一个数据，相当于复制，只不过命名不同。
    print(t, '\n', out_t)
    print(id(t), id(out_t), id(t) == id(out_t))


# ==============================  依据数值创建  ===================================
# 通过torch.full创建全10张量
# flag = True
flag = False
if flag:
    t = torch.full((3, 3), 10)
    print(t)


# ==============================  依据数值创建  ===================================
# 通过torch.arange创建等差数列张量
# flag = True
flag = False
if flag:
    t = torch.arange(2, 10, 2)
    print(t)


# ==============================  依据数值创建  ===================================
# 通过torch.linspace创建均分数列张量
# flag = True
flag = False
if flag:
    t1 = torch.linspace(2, 10, 5)
    t2 = torch.linspace(2, 10, 6)
    print(t1, '\n', t2)


# ==============================  依据数值创建  ===================================
# 通过torch.normal创建正态分布张量
# flag = True
flag = False
if flag:
    # mean张量 std张量
    mean = torch.arange(1, 5, dtype=torch.float)
    std = torch.arange(1, 5, dtype=torch.float)
    t_normal = torch.normal(mean, std)
    print("mean:{}\nstd:{}".format(mean, std))
    print(t_normal)

    # mean张量 std张量
    t_normal2 = torch.normal(0., 1., size=(4,))
    print('\n')
    print(t_normal2)


# ========================= 张量拼接 ================================
# 使用torch.cat()
# flag = True
flag = False
if flag:
    t = torch.ones((2, 3))
    t_0 = torch.cat([t, t], dim=0)  # dim=0 竖着拼
    t_1 = torch.cat([t, t], dim=1)  # dim=1 横着拼
    print("t_0:{} shape:{}\nt_1:{} shape:{}".format(t_0, t_0.shape, t_1, t_1.shape))


# ========================= 张量拼接 ================================
# 使用torch.stack
# flag = True
flag = False
if flag:
    t = torch.ones((2, 3))
    t_stack = torch.stack([t, t], dim=2)
    print("t_stack:{} shape:{}".format(t_stack, t_stack.shape))


# ========================= 张量切分 ================================
# 使用torch.chunk
# 若不能整除，最后一份张量小于其他张量  即向上取整
# flag = True
flag = False
if flag:
    a = torch.ones((2, 5))
    list_of_tensors = torch.chunk(a, dim=1, chunks=2)

    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量:{} shape is:{}".format(idx+1, t, t.shape))


# ========================= 张量切分 ================================
# 使用torch.split
# flag = True
flag = False
if flag:
    t = torch.ones((2, 5))
    list_of_tensors = torch.split(t, [2, 1, 2], dim=1)

    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量:{} shape is:{}".format(idx+1, t, t.shape))


# ========================= 张量索引 ================================
# 使用torch.index_select
# flag = True
flag = False
if flag:
    t = torch.randint(0, 9, size=(3, 3))
    idx = torch.tensor([0, 2], dtype=torch.long)
    t_select = torch.index_select(t, dim=0, index=idx)
    print("t:\n{}\nt_select:\n{}".format(t, t_select))


# ========================= 张量索引 ================================
# 使用torch.masked_select
# flag = True
flag = False
if flag:
    t = torch.randint(0, 9, size=(3, 3))
    mask = t.ge(5)  # 表示 >= 5 为True
    t_select = torch.masked_select(t, mask)
    print("t:\n{}\nmask:\n{}\nt_select:\n{}".format(t, mask, t_select))


# ========================= 张量变换 ================================
# 使用torch.reshape
# flag = True
flag = False
if flag:
    t = torch.randperm(8)
    t_reshape = torch.reshape(t, (2, 4))  # 也可以（-1， 2） -1时，会根据第二个维度计算
    print("t:\n{}\nt_reshape:\n{}".format(t, t_reshape))



# ========================= 张量变换 ================================
# 使用torch.transpose
# flag = True
flag = False
if flag:
    t = torch.rand(2, 3, 4)
    t_transpose = torch.transpose(t, dim0=1, dim1=2)  # 也可以（-1， 2） -1时，会根据第二个维度计算
    print("t shape:{}\nt_transpose shape: {}".format(t.shape, t_transpose.shape))



# ========================= 张量变换 ================================
# 使用torch.squeeze
# 若为None，移除所有长度为1的轴；若指定维度，当且仅当该轴长度为
# flag = True
flag = False
if flag:
    t = torch.rand(1, 2, 3, 1)
    t_sq = torch.squeeze(t)
    t_0 = torch.squeeze(t, dim=0)
    t_1 = torch.squeeze(t, dim=1)
    print(t.shape)
    print(t_sq.shape)
    print(t_0.shape)
    print(t_1.shape)


# ========================= 张量的数学运算 ================================
# 使用torch.squeeze
# 若为None，移除所有长度为1的轴；若指定维度，当且仅当该轴长度为
flag = True
# flag = False
if flag:
    t_0 = torch.randn((3, 3))
    t_1 = torch.ones_like(t_0)
    t_add = torch.add(t_0, 10, t_1)
    print("t_0:\n{}\nt_1:\n{}\nt_add_10:\n{}".format(t_0, t_1, t_add))
    print(t_1.shape)



