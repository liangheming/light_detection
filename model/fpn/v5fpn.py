import torch
from torch import nn
from model.module.convs import InvertedResidual, CBR, DWCBR


class V5FPN(nn.Module):
    def __init__(self, in_channels_list, act_func=None):
        super(V5FPN, self).__init__()
        assert len(in_channels_list) == 3, "only support 3 level now"
        c3, c4, c5 = in_channels_list
        self.latent_c5 = CBR(c5, c4)
        self.c4_fuse = nn.Sequential(
            CBR(c4 * 2, c4, act_func=act_func),
            InvertedResidual(c4, c4, 1, act_func=act_func))
        self.latent_c4 = CBR(c4, c3, act_func=act_func)
        self.c3_out = nn.Sequential(
            CBR(c3 * 2, c3, act_func=act_func),
            InvertedResidual(c3, c3, 1, act_func=act_func))
        self.c3_c4 = DWCBR(c3, c3, 3, 2, act_func=act_func)
        self.c4_out = nn.Sequential(
            CBR(c3 * 2, c4, 1, 1, act_func=act_func),
            InvertedResidual(c4, c4, 1, act_func=act_func))
        self.c4_c5 = DWCBR(c4, c4, 3, 2, act_func=act_func)
        self.c5_out = nn.Sequential(
            CBR(c4 * 2, c5, 1, 1, act_func=act_func),
            InvertedResidual(c5, c5, 1, act_func=act_func))
        self.out_channels = [c3, c4, c5]

    def forward(self, xs):
        c3, c4, c5 = xs
        latent_c5 = self.latent_c5(c5)
        f4 = torch.cat([nn.UpsamplingNearest2d(scale_factor=2)(latent_c5), c4], dim=1)
        c4_fuse = self.c4_fuse(f4)
        latent_c4 = self.latent_c4(c4_fuse)
        f3 = torch.cat([nn.UpsamplingNearest2d(scale_factor=2)(latent_c4), c3], dim=1)
        c3_out = self.c3_out(f3)
        c3_c4 = self.c3_c4(c3_out)
        c4_out = self.c4_out(torch.cat([c3_c4, latent_c4], dim=1))
        c4_c5 = self.c4_c5(c4_out)
        c5_out = self.c5_out(torch.cat([c4_c5, latent_c5], dim=1))
        return [c3_out, c4_out, c5_out]
