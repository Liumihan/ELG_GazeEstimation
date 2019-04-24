from torch.nn import Module, Sequential
from torch.nn import Conv2d, BatchNorm2d, ReLU, Upsample, MaxPool2d
from torch import nn
import torch
import numpy as np


class ELGNetwork(Module):
    def __init__(self, HGNet_num=3, input_shape=(1, 96, 160),  output_shape=(17, 96, 160), feature_channels=64):
        super(ELGNetwork, self).__init__()
        self.HGNet_num = HGNet_num
        self.res_before = ResidualBlock(in_channels=input_shape[0], out_channels=feature_channels)

        for i in range(HGNet_num):
            setattr(self, 'HG_layer_{}'.format(i), HGNetwork(feature_channels=feature_channels, layer_num=4))
        # 这一层是用来做1x1卷积，让他们的通道数等于我所要的通道数
        self.res_after = ResidualBlock(in_channels=feature_channels, out_channels=output_shape[0])

    def forward(self, x):
        x = self.res_before(x)
        for i in range(self.HGNet_num):
            x = getattr(self, 'HG_layer_{}'.format(i))(x)
        x = self.res_after(x)
        return x


class HGNetwork(Module): # 根据ELG的论文中的结构来的
    def __init__(self, feature_channels=64, layer_num=4):
        """
        :param feature_channels: 在HGNet里面传播的feature map的通道数
        :param layer_num:  每一个HGNet的一块的downsample的次数
        """
        super(HGNetwork, self).__init__()
        self.feature_channels = feature_channels
        self.layer_num = layer_num
        # down sample part
        for i in range(layer_num):
            setattr(self, 'down_res_{}'.format(i), ResidualBlock(feature_channels, feature_channels))
            setattr(self, 'down_pool_{}'.format(i), MaxPool2d(kernel_size=(2, 2)))
        # parallel part
        for i in range(3):
            setattr(self, 'paral_res_{}'.format(i), ResidualBlock(feature_channels, feature_channels))
        # up sample part
        for i in range(layer_num):
            setattr(self, 'up_res_{}'.format(i), ResidualBlock(feature_channels, feature_channels))
            setattr(self, 'up_sample_{}'.format(i), Upsample(scale_factor=2))
        # short cut part
        self.shortcut_conv = Conv2d(in_channels=feature_channels, out_channels=feature_channels, kernel_size=(1, 1))
        # 在forward 函数里面实现
    def forward(self, x):
        cache_stack = []  # 用来保存前面的值，short cut
        # down sample part
        for i in range(self.layer_num):
            x = getattr(self, 'down_res_{}'.format(i))(x)
            x = getattr(self, 'down_pool_{}'.format(i))(x)
            cache_stack.append(x)
        # parallel part
        for i in range(3):
            x = getattr(self, 'paral_res_{}'.format(i))(x)
        # up sample part
        for i in range(self.layer_num):
            residual = cache_stack.pop()
            residual = self.shortcut_conv(residual)
            x_in = residual + x
            x = getattr(self, 'up_res_{}'.format(i))(x_in)
            x = getattr(self, 'up_sample_{}'.format(i))(x)
        return x


class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        # 卷积模块
        half_out_channels = max(int(out_channels/2), 1)
        self.convPart = Sequential(
            Conv2d(in_channels=in_channels, out_channels=half_out_channels, kernel_size=1),
            BatchNorm2d(half_out_channels),
            ReLU(),

            Conv2d(in_channels=half_out_channels, out_channels=int(half_out_channels),
                   kernel_size=3, padding=1),
            BatchNorm2d(half_out_channels),
            ReLU(),

            Conv2d(in_channels=half_out_channels, out_channels=out_channels, kernel_size=1),
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


class GazeEstimator(Module):
    """
    直接根据HGnet 生成的feature map 来预测look vector
    """
    def __init__(self, feat_shape=(64, 96, 160)):
        self.feat_shape = feat_shape

        self.conv1 = self.feat_shape


    def forward(self, feature_maps, ldmks_tensor):



def test_residual():
    dummy_input = np.random.randn(32, 1, 96, 160).astype(np.float32)
    dummy_input = torch.from_numpy(dummy_input)
    rs = ResidualBlock(in_channels=1, out_channels=1)
    output = rs.forward(dummy_input)
    print(output.size())


def test_HGnet():
    dummy_input = np.random.randn(32, 64, 96, 160).astype(np.float32)
    dummy_input = torch.from_numpy(dummy_input)
    net = HGNetwork(feature_channels=64, layer_num=4)
    output = net.forward(dummy_input)
    print(output.size())

def test_ELGNetwork():
    device = 'cuda:0'
    dummy_input = np.random.randn(1, 1, 96, 160).astype(np.float32)
    dummy_inputs = [dummy_input for i in range(3)]
    net = ELGNetwork().to(device)
    for x in dummy_inputs:
        output = net(torch.from_numpy(x).to(device))
        print(output.size())

        del output


if __name__=="__main__":
    # test_residual()
    # test_HGnet()
    test_ELGNetwork(is_train=False)