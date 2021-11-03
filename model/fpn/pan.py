from torch import nn
from model.module.convs import CBR
import torch.nn.functional as f


class LightPAN(nn.Module):
    def __init__(self, in_channels_list, out_channels, act_func=None):
        super(LightPAN, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        lateral_convs = list()

        for in_channel in in_channels_list:
            lateral_convs.append(
                # CBR(in_channel, out_channels, 1, 1, act_func=act_func)
                nn.Conv2d(in_channel, out_channels, 1, 1)
            )
        self.lateral_convs = nn.ModuleList(lateral_convs)
        self.out_channels = [out_channels for _ in in_channels_list]

    def forward(self, xs):
        assert len(xs) == len(self.in_channels_list)
        laterals = [
            lateral_conv(xs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)
        down2up = list()
        down2up.append(laterals[-1])
        for i in range(used_backbone_levels - 1, 0, -1):
            down2up.append(laterals[i - 1] + f.interpolate(
                laterals[i], scale_factor=2, mode='nearest'))
        up2down = list()
        up2down.append(down2up[-1])
        for i in range(used_backbone_levels - 1, 0, -1):
            up2down.append(down2up[i - 1] + f.interpolate(
                down2up[i], scale_factor=0.5, mode='nearest', recompute_scale_factor=True))
        return up2down
