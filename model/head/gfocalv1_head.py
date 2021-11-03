import math
import torch
from torch import nn
from model.module.convs import DWCBR


class GFocalHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels_list,
                 strides,
                 cell_sizes,
                 inner_channel,
                 stacks=2,
                 reg_max=7,
                 share_cls_reg=True,
                 top_k=9,
                 qfl_weight=1.0,
                 dfl_weight=0.25,
                 iou_weight=2.0,
                 iou_type="giou",
                 conf_thresh=0.1,
                 nms_thresh=0.6,
                 class_agnostic=False,
                 act_func=None,
                 ):
        super(GFocalHead, self).__init__()
        self.in_channels_list = in_channels_list
        self.stacks = stacks
        self.inner_channel = inner_channel
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.share_cls_reg = share_cls_reg
        self.act_func = act_func
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.__init_layers()
        self.__init_weights()

    def __init_layers(self):
        for i in range(len(self.in_channels_list)):
            cls_sequence = list()
            reg_sequence = list()
            for j in range(self.stacks):
                cls_sequence.append(
                    DWCBR(self.in_channels_list[i] if j == 0 else self.inner_channel,
                          self.inner_channel, 3, 1,
                          act_func=self.act_func)
                )
                if not self.share_cls_reg:
                    reg_sequence.append(
                        DWCBR(self.in_channels_list[i] if j == 0 else self.inner_channel,
                              self.inner_channel, 3, 1,
                              act_func=self.act_func)
                    )
            self.cls_convs.append(nn.Sequential(*cls_sequence))
            if not self.share_cls_reg:
                self.reg_convs.append(nn.Sequential(*reg_sequence))
            self.cls_preds.append(
                nn.Conv2d(
                    self.inner_channel,
                    self.num_classes + 4 * (self.reg_max + 1) if self.share_cls_reg else self.num_classes,
                    1, 1
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    self.inner_channel,
                    4 * (self.reg_max + 1),
                    1, 1
                )
            )

    def __init_weights(self, prior_prob=1e-2):
        for conv in self.cls_preds:
            nn.init.normal_(conv.weight, 0, 0.01)
            nn.init.constant_(conv.bias, -math.log((1 - prior_prob) / prior_prob))
        for conv in self.reg_preds:
            nn.init.normal_(conv.weight, 0, 0.01)
            nn.init.constant_(conv.bias, 0)

    def forward_impl(self, xs):
        outputs = list()
        for k in range(len(xs)):
            x = xs[k]
            cls_feat = self.cls_convs[k](x)
            if self.share_cls_reg:
                feat = self.cls_preds[k](cls_feat)
                cls_out, reg_out = torch.split(feat, [self.num_classes, 4 * (self.reg_max + 1)], dim=1)
            else:
                reg_feat = self.reg_convs[k](x)
                cls_out = self.cls_preds[k](cls_feat)
                reg_out = self.reg_preds[k](reg_feat)
            outputs.append((cls_out, reg_out))
        return outputs

    def forward(self, xs, targets=None):
        if self.training:
            assert targets is not None
            predicts = self.forward_impl(xs)
            return predicts
        else:
            return self.predict(xs)

    def predict(self, xs):
        outputs = self.forward_impl(xs)
        return outputs


if __name__ == '__main__':
    GFocalHead([96, 128, 196], [8, 16, 32], None, 96, 80)
