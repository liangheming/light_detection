import torch

from torch import nn
from model.backbone.shufflenet import build_backbone
from model.fpn import build_fpn
from model.head.yolox_head import Xhead


class LightYOLOX(nn.Module):
    def __init__(self,
                 num_classes=80,
                 backbone=None,
                 neck=None,
                 head=None,
                 ):
        super(LightYOLOX, self).__init__()
        act_func = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        if backbone is None:
            backbone = {"name": "shufflenet_v2_x0_5", "pretrained": True}
        backbone.update({"act_func": act_func})

        self.backbone = build_backbone(**backbone)
        if neck is None:
            neck = {"name": "pan", "out_channels": 96}
        neck.update({"in_channels_list": self.backbone.out_channels, "act_func": act_func})
        self.fpn = build_fpn(**neck)

        if head is None:
            head = {
                "inner_channel": 96,
                "stacks": 1,
                "conf_thresh": 0.1,
                "nms_thresh": 0.6,
                "center_radius": 2.5,
                "reg_weights": 5.0,
                "iou_type": "iou",
                "class_agnostic": False
            }
        head.update({
            "in_channels_list": self.fpn.out_channels,
            "strides": [8., 16., 32.],
            "num_classes": num_classes,
            "act_func": act_func
        })
        self.head = Xhead(**head)

    def feature_extra(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        return x

    def forward(self, x, targets=None):
        if self.training:
            assert targets is not None
            x = self.feature_extra(x)
            x = self.head(x, targets)
        else:
            x = self.predict(x)
        return x

    @torch.no_grad()
    def predict(self, x):
        x = self.feature_extra(x)
        x = self.head.predict(x)
        return x


if __name__ == '__main__':
    from utils.flops import get_model_complexity_info
    import yaml

    with open("configs/shuffle_pan_yolox_s.yaml", "r") as rf:
        cfg = yaml.safe_load(rf)
    net = LightYOLOX(**cfg["model"]).eval()
    flops, params = get_model_complexity_info(net, input_shape=(3, cfg['data']['v_size'], cfg['data']['v_size']))
    print(flops, params)
