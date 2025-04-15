import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def create2DsobelFilter(device):
    # 定义 Sobel 过滤器
    sobel_x = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])  # 水平方向 (X) 梯度

    sobel_y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])  # 竖直方向 (Y) 梯度

    # 2D Sobel 核，大小 (2,1,3,3)，表示 2 个过滤器（X 和 Y 方向）
    sobelFilter = np.zeros((2, 1, 3, 3))

    # 赋值给两个 Sobel 方向
    sobelFilter[0, 0, :, :] = sobel_x  # X 方向 Sobel
    sobelFilter[1, 0, :, :] = sobel_y  # Y 方向 Sobel

    return torch.from_numpy(sobelFilter).float().to(device)  # 转换为 PyTorch Tensor 并放到 GPU


class SobelLayer(nn.Module):
    def __init__(self, device):
        super(SobelLayer, self).__init__()
        self.pad = nn.ReflectionPad2d(1)  # 反射填充，保持尺寸不变
        self.kernel = create2DsobelFilter(device)
        self.act = nn.Tanh()  # 归一化梯度

    def forward(self, x):
        paded = self.pad(x)  # 进行填充
        sobel = F.conv2d(paded, self.kernel, padding=0, groups=1) / 4  # 计算 Sobel 梯度
        n, c, h, w = sobel.size()  # 获取维度
        sobel_norm = torch.norm(sobel, dim=1, keepdim=True) / c * 3  # 计算梯度幅值
        out = self.act(sobel_norm) * 2 - 1  # 归一化到 [-1, 1]
        return out
