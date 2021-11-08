import math
import torch
import torchvision
import torch.nn.functional as f
from torch import nn
from utils.box_utils import IOULoss, bboxes_iou
from model.module.convs import CBR, InvertedResidual


class Xhead(nn.Module):
    def __init__(self,
                 in_channels_list,
                 strides,
                 inner_channel,
                 num_classes,
                 stacks=1,
                 center_radius=2.5,
                 reg_weights=5.0,
                 iou_type="iou",
                 conf_thresh=0.1,
                 nms_thresh=0.6,
                 class_agnostic=False,
                 act_func=None):
        super(Xhead, self).__init__()
        self.center_radius = center_radius
        self.reg_weights = reg_weights
        self.grids_all = None
        self.expand_strides = None
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
        self.max_out = 100
        for i in range(len(in_channels_list)):
            self.stems.append(
                CBR(in_channels_list[i], inner_channel, 1, 1, act_func=act_func) if in_channels_list[i] != inner_channel
                else InvertedResidual(in_channels_list[i], inner_channel, 1, act_func=act_func)
            )
            cls_sequence = list()
            reg_sequence = list()
            for j in range(stacks):
                cls_sequence.append(
                    InvertedResidual(
                        inner_channel, inner_channel, 1, act_func=act_func)
                )
                reg_sequence.append(
                    InvertedResidual(
                        inner_channel, inner_channel, 1, act_func=act_func)
                )
            self.cls_convs.append(nn.Sequential(*cls_sequence))
            self.reg_convs.append(nn.Sequential(*reg_sequence))
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
        self.bce = nn.BCELoss(reduction="none")
        self.iou_loss = IOULoss(iou_type=iou_type, coord_type="xywh")
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

    def forward(self, xs, targets=None):
        if self.training:
            assert targets is not None
            predicts = self.forward_impl(xs)
            return self.compute_loss(predicts, targets)
        else:
            return self.predict(xs)

    def compute_loss(self, predicts, targets):
        self.build_grids_and_expand_strides(predicts)
        outputs_all = self.cat_layers_features(predicts)  # Tensor(size=(n, proposal_num,5+cls))
        labels_all = targets['label']

        label_list = labels_all.split(targets['batch_len'])
        cls_targets = list()
        reg_targets = list()
        fg_masks = list()
        for i in range(len(label_list)):
            label = label_list[i]
            if len(label) < 1:
                fg_mask = torch.zeros_like(self.expand_strides)
                fg_masks.append(fg_mask.bool())
                continue
            is_in_boxes_or_in_center, is_valid = self.get_in_boxes_info(label[:, 1:])
            fg_mask, matched_gt_inds, matched_ious = \
                self.get_assignments(outputs_all[i].detach(), label, is_in_boxes_or_in_center, is_valid)
            match_cls = label[:, 0][matched_gt_inds]
            cls_target = f.one_hot(
                match_cls.long(), self.num_classes
            ) * matched_ious[:, None]
            cls_targets.append(cls_target)
            reg_targets.append(label[:, 1:][matched_gt_inds])
            fg_masks.append(fg_mask)
        fg_masks = torch.stack(fg_masks, dim=0)
        gt_num = max(fg_masks.sum(), 1)
        fg_pred = outputs_all[fg_masks]
        cls_targets = torch.cat(cls_targets, dim=0)
        cls_pred = fg_pred[:, 5:]

        reg_targets = torch.cat(reg_targets, dim=0)
        reg_pred = fg_pred[:, :4]

        obj_pred = outputs_all[..., 4]
        obj_targets = fg_masks.float()

        loss_iou = self.reg_weights * (self.iou_loss(reg_pred, reg_targets).sum() / gt_num)
        loss_obj = self.bce(obj_pred, obj_targets).sum() / gt_num
        loss_cls = self.bce(cls_pred, cls_targets).sum() / gt_num
        loss = loss_iou + loss_obj + loss_cls
        return loss, loss_iou.detach(), loss_obj.detach(), loss_cls.detach(), gt_num

    def predict_bak(self, xs):
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

    def predict(self, xs):
        nms_pre = 1000
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
            box_predicts = torch.cat([(reg_output[:, :2, ...] + self.grids[k]) * self.strides[k],
                                      reg_output[:, 2:4, ...].exp() * self.strides[k]], dim=1)
            scores = obj_output.sigmoid() * cls_output.sigmoid()
            outputs.append((box_predicts, scores))
        _, _, h, w = outputs[0][0].shape
        inp_w, inp_h = (w * self.strides[0], h * self.strides[0])
        boxes = list()
        scores = list()
        dets = list()
        bs = xs[0].shape[0]
        for box, score in outputs:
            box = box.permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)
            score = score.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)
            box[..., :2] = box[..., :2] - box[..., 2:] * 0.5
            box[..., 2:] = box[..., :2] + box[..., 2:]
            box[..., [0, 2]] = box[..., [0, 2]].clamp(min=0, max=inp_w)
            box[..., [1, 3]] = box[..., [1, 3]].clamp(min=0, max=inp_h)
            if score.shape[1] > nms_pre:
                max_score, _ = score.max(dim=-1)
                _, topk_inds = max_score.topk(nms_pre, dim=-1)
                score = score[torch.arange(bs)[:, None], topk_inds, :]
                box = box[torch.arange(bs)[:, None], topk_inds, :]
            scores.append(score)
            boxes.append(box)
        scores = torch.cat(scores, dim=1)
        boxes = torch.cat(boxes, dim=1)
        for b in range(bs):
            instance_score = scores[b]
            instance_boxes = boxes[b]
            valid_mask = instance_score > self.conf_thresh
            if valid_mask.sum() < 1:
                dets.append(None)
                continue
            candidate_idx, class_idx = valid_mask.nonzero(as_tuple=True)
            valid_scores = instance_score[candidate_idx, class_idx]
            valid_boxes = instance_boxes[candidate_idx]
            idx = torchvision.ops.batched_nms(valid_boxes, valid_scores, class_idx.long(), self.nms_thresh)
            if len(idx) > self.max_out:
                idx = idx[:self.max_out]
            det = torch.cat([valid_boxes[idx], valid_scores[idx, None], class_idx[idx, None].type(valid_boxes.dtype)],
                            dim=-1)
            dets.append(det)
        return dets

    def postprocess(self, prediction, size):
        """
        :param prediction:
        :param size: (w, h)
        :return:(x1, y1, x2, y2, class_conf, class_pred)
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

    def build_grids_and_expand_strides(self, xs):
        proposal_num = sum([x.size(-1) * x.size(-2) for x in xs])
        if self.expand_strides is not None and len(
                self.expand_strides) == proposal_num and xs[0].device == self.expand_strides.device:
            return
        grids_all = list()
        expand_strides = list()
        for i in range(len(xs)):
            x = xs[i]
            b, c, h, w = x.shape
            grid = self.grids[i]
            grids_all.append(grid.permute(0, 2, 3, 1).contiguous().view(1, -1, 2))
            stride = self.strides[i]
            expand_strides.extend([stride] * (w * h))
        grids_all = torch.cat(grids_all, dim=1)
        expand_strides = torch.tensor(expand_strides).type_as(grids_all)
        self.grids_all = grids_all
        self.expand_strides = expand_strides

    @staticmethod
    def cat_layers_features(xs):
        outputs_all = list()
        for i in range(len(xs)):
            x = xs[i]
            b, c, h, w = x.shape
            outputs_all.append(x.permute(0, 2, 3, 1).contiguous().view(b, -1, c))
        outputs_all = torch.cat(outputs_all, dim=1)
        return outputs_all

    def get_in_boxes_info(self, gt_boxes):
        """
        :param gt_boxes: Tensor(size=(gt_num,4)) xywh no norm
        :return:
        """
        grid_center = (self.grids_all + 0.5) * self.expand_strides[None, :, None]
        # grid_center Tensor(size=(1,proposal,2))
        gt_box_lt = gt_boxes[:, :2] - 0.5 * gt_boxes[:, 2:]
        gt_box_rb = gt_boxes[:, :2] + 0.5 * gt_boxes[:, 2:]
        lt_delta = grid_center - gt_box_lt[:, None, :]
        rb_delta = gt_box_rb[:, None, :] - grid_center
        ltrb_delta = torch.cat([lt_delta, rb_delta], dim=-1)
        # ltrb_delta [gt_num,proposal_num,4]
        is_in_boxes = ltrb_delta.min(dim=-1).values > 0.0
        # is_in_boxes [gt_num,proposal_num]
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # is_in_boxes_all [proposal_num]

        shrink_gt_box_lt = gt_boxes[:, :2][:, None, :] - self.center_radius * self.expand_strides[None, :, None]
        shrink_gt_box_rb = gt_boxes[:, :2][:, None, :] + self.center_radius * self.expand_strides[None, :, None]

        shrink_lt_delta = grid_center - shrink_gt_box_lt
        shrink_rb_delta = shrink_gt_box_rb - grid_center

        shrink_ltrb_delta = torch.cat([shrink_lt_delta, shrink_rb_delta], dim=-1)
        is_in_centers = shrink_ltrb_delta.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        is_in_boxes_or_in_center = is_in_boxes_all | is_in_centers_all
        is_valid = is_in_boxes[:, is_in_boxes_or_in_center] & is_in_centers[:, is_in_boxes_or_in_center]
        return is_in_boxes_or_in_center, is_valid

    def get_assignments(self, proposal, labels, fg_mask, is_valid_mask):
        """
        :param proposal:
        :param labels:
        :param fg_mask:
        :param is_valid_mask:
        :return:
        """
        fg_num = fg_mask.sum()
        gt_num = len(labels)
        bbox_pred = proposal[:, :4][fg_mask]
        obj_pred = proposal[:, 4:5][fg_mask]
        cls_pred = proposal[:, 5:][fg_mask]
        pair_wise_ious = bboxes_iou(labels[:, 1:], bbox_pred, xyxy=False)
        gt_cls = f.one_hot(labels[:, 0].long(), self.num_classes).float()[:, None, :].repeat(1, fg_num, 1)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        combine_cls_pred = (cls_pred * obj_pred).sqrt_()[None, ...].repeat(gt_num, 1, 1)
        pair_wise_cls_loss = f.binary_cross_entropy(
            combine_cls_pred, gt_cls, reduction="none"
        ).sum(-1)
        cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_ious_loss
                + 100000.0 * (~is_valid_mask)
        )
        matching_matrix = self.dynamic_k_match(cost, pair_wise_ious)
        # gt_num, valid_anchor
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].float().argmax(0)
        matched_ious = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]

        return fg_mask, matched_gt_inds, matched_ious

    @staticmethod
    def dynamic_k_match(cost, pair_wise_ious):
        num_gt = cost.size(0)
        matching_matrix = torch.zeros_like(cost).bool()
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = topk_ious.sum(dim=1).long()
        dynamic_ks = dynamic_ks.clamp(min=1).tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = True
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] = False
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = True

        return matching_matrix
