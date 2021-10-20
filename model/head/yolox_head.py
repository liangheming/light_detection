import math
import torch
import torchvision

from torch import nn
from model.module.convs import CBR, InvertedResidual


class Xhead(nn.Module):
    def __init__(self,
                 in_channels_list,
                 strides,
                 inner_channel,
                 num_classes,
                 stacks=1,
                 conf_thresh=0.1,
                 nms_thresh=0.6,
                 class_agnostic=False,
                 act_func=None):
        super(Xhead, self).__init__()
        self.strides = strides
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.class_agnostic = class_agnostic
        self.nms_thresh = nms_thresh
        self.grids = [torch.zeros(1)] * len(in_channels_list)
        self.stems = nn.ModuleList()
        self.cls_convs = list()
        self.reg_convs = list()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        for i in range(len(in_channels_list)):
            self.stems.append(
                CBR(in_channels_list[i], inner_channel, 1, 1, act_func=act_func) if in_channels_list[i] != inner_channel
                else InvertedResidual(in_channels_list[i], inner_channel, 1, act_func=act_func)
            )
            # self.stems.append(
            #     CBR(in_channels_list[i], inner_channel, 1, 1, act_func=act_func)
            # )
            for j in range(stacks):
                self.cls_convs.append(
                    InvertedResidual(
                        inner_channel, inner_channel, 1, act_func=act_func)
                )
                self.reg_convs.append(
                    InvertedResidual(
                        inner_channel, inner_channel, 1, act_func=act_func
                    )
                )
            self.cls_preds.append(
                nn.Conv2d(inner_channel, num_classes, 1, 1, 0)
            )
            self.reg_preds.append(
                nn.Conv2d(
                    inner_channel, 4, 1, 1, 0
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    inner_channel, 1, 1, 1, 0
                )
            )
        self.cls_convs = nn.Sequential(*self.cls_convs)
        self.reg_convs = nn.Sequential(*self.reg_convs)
        self.initialize_biases(1e-2)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    @staticmethod
    def get_grid(h, w):
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
        grid = torch.stack((xv, yv), 0).unsqueeze(0)
        return grid

    def forward_impl(self, xs):
        outputs = list()
        for k in range(len(xs)):
            x = xs[k]
            x = self.stems[k](x)
            cls_output = self.cls_preds[k](self.cls_convs[k](x))
            reg_feat = self.reg_convs[k](x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            if self.grids[k].shape[2:4] != x.shape[2:4] or self.grids[k].device != x.device:
                self.grids[k] = self.get_grid(x.shape[2], x.shape[3]).type_as(x)
                self.grids[k].requires_grad_(False)
            outputs.append(torch.cat([(reg_output[:, :2, ...] + self.grids[k]) * self.strides[k],
                                      reg_output[:, 2:4, ...].exp() * self.strides[k],
                                      obj_output.sigmoid(), cls_output.sigmoid()], dim=1))
        return outputs

    def forward(self, xs):
        return self.forward_impl(xs)

    def predict(self, xs):
        box_list = list()
        outputs = self.forward_impl(xs)
        _, _, h, w = outputs[0].shape
        size = (w * self.strides[0], h * self.strides[0])
        for o in outputs:
            b, c, _, _ = o.shape
            box_list.append(o.view(b, c, -1).permute(0, 2, 1).contiguous())
        box_list = torch.cat(box_list, dim=1)
        bbox = self.postprocess(box_list, size)
        return bbox

    def postprocess(self, prediction, size):
        """
        :param prediction:
        :param size: (w, h)
        :return:
        """
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2).clamp(min=0.)
        box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2).clamp(min=0.)
        box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2).clamp(max=size[0])
        box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2).clamp(max=size[1])
        prediction[:, :, :4] = box_corner[:, :, :4]
        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            # [all,1]
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + self.num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= self.conf_thresh).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            # detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = torch.cat((image_pred[:, :4], class_conf * image_pred[:, 4:5], class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            if self.class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4],
                    self.nms_thresh,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4],
                    detections[:, 5],
                    self.nms_thresh,
                )

            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))
        return output
