import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def create2DsobelFilter(device):
    sobel_x = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    sobel_y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    sobelFilter = np.zeros((2, 1, 3, 3))

    sobelFilter[0, 0, :, :] = sobel_x
    sobelFilter[1, 0, :, :] = sobel_y

    return torch.from_numpy(sobelFilter).float().to(device)


class SobelLayer(nn.Module):
    def __init__(self, device):
        super(SobelLayer, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.kernel = create2DsobelFilter(device)
        self.act = nn.Tanh()

    def forward(self, x):
        paded = self.pad(x)
        sobel = F.conv2d(paded, self.kernel, padding=0, groups=1) / 4
        n, c, h, w = sobel.size()
        sobel_norm = torch.norm(sobel, dim=1, keepdim=True) / c * 3
        out = self.act(sobel_norm) * 2 - 1
        return out
