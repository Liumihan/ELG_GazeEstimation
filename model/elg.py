from torch.nn import Module, Sequential
from torch.nn import Conv2d, BatchNorm2d, ReLU, Upsample
from torch import nn
import torch
import numpy as np


class ELGNetwork(Module):
    def __init__(self):
        pass

    def forward(self, input):
        pass

class HGNetwork(Module):
    def __init__(self):
        pass

    def forward(self, input):
        pass

class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        # 卷积模块
        self.convPart = Sequential(
            Conv2d(in_channels=in_channels, out_channels=int(out_channels/2), kernel_size=1),
            BatchNorm2d(int(out_channels/2)),
            ReLU(),

            Conv2d(in_channels=int(out_channels/2), out_channels=int(out_channels/2),
                   kernel_size=3, padding=1),
            BatchNorm2d(int(out_channels/2)),
            ReLU(),

            Conv2d(in_channels=int(out_channels/2), out_channels=out_channels, kernel_size=1),
            BatchNorm2d(out_channels),
            ReLU()
        )
        # short cut
        # 如果输入输出的层数不同的话那么就应该，通过一个1×1卷积让他们的通道数一样
        if in_channels != out_channels:
            self.skipConv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.in_channels, self.out_channels = in_channels, out_channels

    def forward(self, x):
        residual = x
        x = self.convPart(x)
        if self.in_channels != self.out_channels:
            residual = self.skipConv(residual)
        x += residual
        return x


def test_residual():
    dummy_input = np.random.randn(32, 1, 90, 150)
    dummy_input = torch.from_numpy(dummy_input)
    rs = ResidualBlock(in_channels=1, out_channels=32)
    output = rs.forward(dummy_input)

if __name__=="__main__":
    test_residual()