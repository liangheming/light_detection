import torch
from copy import deepcopy
from torch import nn


class CBR(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 groups=1,
                 act_func=None,
                 ):
        super(CBR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        if act_func is None:
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            self.act = deepcopy(act_func)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            stride: int,
            act_func=None,
    ) -> None:
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.LeakyReLU(negative_slope=0.1, inplace=True) if act_func is None else deepcopy(act_func),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.LeakyReLU(negative_slope=0.1, inplace=True) if act_func is None else deepcopy(act_func),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.LeakyReLU(negative_slope=0.1, inplace=True) if act_func is None else deepcopy(act_func),
        )

    @staticmethod
    def depthwise_conv(
            i: int,
            o: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class DWCBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, bias=False, act_func=None):
        super(DWCBR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True) if act_func is None else deepcopy(act_func)
        )

    def forward(self, x):
        return self.dw(x)


class DWCBR2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, bias=False, act_func=None):
        super(DWCBR2, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True) if act_func is None else deepcopy(act_func),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True) if act_func is None else deepcopy(act_func)
        )

    def forward(self, x):
        return self.dw(x)
