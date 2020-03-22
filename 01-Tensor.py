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
flag = True
# flag = False
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





