import torch

from torch import nn
from model.backbone.shufflenet import build_backbone
from model.fpn import build_fpn
from model.head.yolox_head import Xhead


class LightYOLOX(nn.Module):
    def __init__(self,
                 num_classes=80,
                 backbone="shufflenet_v2_x0_5",
                 neck="pan",
                 inner_channel=96,
                 pretrained=True,
                 stacks=1,
                 conf_thresh=0.1,
                 nms_thresh=0.6,
                 class_agnostic=False):
        super(LightYOLOX, self).__init__()
        act_func = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.backbone = build_backbone(backbone, pretrained=pretrained, act_func=act_func)
        self.fpn = build_fpn(neck, in_channels_list=self.backbone.out_channels, out_channels=inner_channel,
                             act_func=act_func)
        self.head = Xhead(in_channels_list=self.fpn.out_channels,
                          strides=[8., 16., 32.],
                          inner_channel=inner_channel,
                          num_classes=num_classes,
                          stacks=stacks,
                          act_func=act_func,
                          conf_thresh=conf_thresh,
                          nms_thresh=nms_thresh,
                          class_agnostic=class_agnostic,
                          )

    def feature_extra(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        return x

    def forward(self, x):
        x = self.feature_extra(x)
        x = self.head(x)
        return x

    @torch.no_grad()
    def predict(self, x):
        x = self.feature_extra(x)
        x = self.head.predict(x)
        return x


if __name__ == '__main__':
    from utils.flops import get_model_complexity_info
    net = LightYOLOX()
    inp = torch.randn(size=(4, 3, 416, 416))
    out = net(inp)
    flops, params = get_model_complexity_info(net, input_shape=(3, 320, 320))
    print(flops, params)
