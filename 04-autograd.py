# -*- coding: utf-8 -*-
"""
@brife      ：autograd示例
"""

import torch
torch.manual_seed(10)  # 使每次初始化的数据一样

# =======================  retain_graph  ==============
# retain_graph  保存计算图
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    # y.backward() 实际直接调用了torch.autograd.backward()
    y.backward(retain_graph=True)  # retain_graph=True，使下面还可以使用计算图
    print(w.grad)
    y.backward()  # 这里如果想再次使用计算图，上面需要保存计算图 retain_graph=True，否则会报错
    print(w.grad)


# =======================  grad_tensors  ==============
# grad_tensors  多梯度的权重
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)

    y0 = torch.mul(a, b)  # y0 = (x+w) * (w+1)   dy0/dw = 5
    y1 = torch.add(a, b)  # y0 = (x+w) + (w+1)   dy1/dw = 2

    loss = torch.cat([y0, y1], dim=0)
    grad_tensors = torch.tensor([1., 2.])  # 设置梯度的权重

    loss.backward(gradient=grad_tensors)

    print(w.grad)  # (dy0/dw) * 权重 + （dy1/dw）* 权重


# =======================  autograd.grad  ==============
# autograd.grad  求取梯度
# flag = True
flag = False
if flag:

    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)  # y = x ** 2

    # create_graph=True 创建导数的计算图，用于高阶求导
    grad_1 = torch.autograd.grad(y, x, create_graph=True)  # grad_1 = dy/dx = 2x = 2*3 = 6
    print(grad_1)  #grad_1 为元组  结果为 (tensor([6.], grad_fn=<MulBackward0>),)

    grad_2 = torch.autograd.grad(grad_1[0], x)  # grad_2 = d(dy/dx)/dx = d(2x)/dx = 2
    print(grad_2)


# =======================  梯度叠加不自动清零  ==============
# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    for i in range(4):
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)

        y.backward()
        print(w.grad)

        # 清零用grad.zero_()  _表示原位操作
        w.grad.zero_()


# =======================  依赖于叶子结点的结点，requires_grad默认为True  ==============
# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    print(a.requires_grad, b.requires_grad, y.requires_grad)


# =======================  叶子结点不可执行in-place  ==============
# flag = True
flag = False
if flag:
    w = torch.tensor([1.,], requires_grad=True)
    x = torch.tensor([2.,], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    w.add_(1)  # 这里会报错

    y.backward()


flag = True
# flag = False
if flag:

    a = torch.ones((1, ))
    print(id(a), a)  # id为其内存

    a = a + torch.ones((1, ))  # 这里会开辟新的内存地址，即不是原位操作
    print(id(a), a)

    a += torch.ones((1, ))  #这里+=内存与a一样，是原位操作
    print(id(a), a)

