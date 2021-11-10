import math
import torch
import torchvision
from torch import nn
from model.module.convs import DWCBR2
from utils.general import reduce_mean
from utils.box_utils import IOULoss, bbox_overlaps
import torch.nn.functional as f


class QFLoss(object):
    def __init__(self, weights=1.0, gama=2.0):
        super(QFLoss, self).__init__()
        self.weights = weights
        self.gama = gama

    def __call__(self, predicts, targets):
        pred_sigmoid = predicts.sigmoid()
        scale_factor = (pred_sigmoid - targets).abs().pow(self.gama)
        bce = f.binary_cross_entropy_with_logits(predicts, targets, reduction="none")
        return self.weights * scale_factor * bce


class DFLoss(object):
    def __init__(self, weights=1.0):
        super(DFLoss, self).__init__()
        self.weights = weights

    def __call__(self, predicts, targets, weights=1.0):
        reg_max = predicts.shape[-1] // 4
        predicts_flat = predicts.view(-1, reg_max)
        targets_flat = targets.view(-1)
        dis_left = targets_flat.long()
        dis_right = dis_left + 1
        weight_left = dis_right.float() - targets_flat
        weight_right = targets_flat - dis_left.float()
        loss = (
                f.cross_entropy(predicts_flat, dis_left, reduction="none") * weight_left
                + f.cross_entropy(predicts_flat, dis_right, reduction="none") * weight_right
        )
        return (loss.view(-1, 4).mean(-1) * weights) * self.weights


class Integral(nn.Module):
    def __init__(self, reg_max=7):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer(
            "project", torch.linspace(0, self.reg_max, self.reg_max + 1)
        )

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = f.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = f.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


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
                 max_out=100,
                 act_func=None,
                 ):
        super(GFocalHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels_list = in_channels_list
        self.strides = strides
        self.cell_sizes = cell_sizes
        self.stacks = stacks
        self.inner_channel = inner_channel
        self.top_k = top_k
        self.qfl_weight = qfl_weight
        self.dfl_weight = dfl_weight
        self.iou_weight = iou_weight
        self.iou_type = iou_type
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.max_out = max_out
        self.reg_max = reg_max
        self.share_cls_reg = share_cls_reg
        self.act_func = act_func
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.distribute_project = Integral(reg_max=reg_max)
        self.iou_loss = IOULoss(iou_type=iou_type, weights=iou_weight)
        self.qlf_loss = QFLoss(weights=qfl_weight)
        self.dlf_loss = DFLoss(weights=dfl_weight)
        self.__init_layers()
        self.__init_weights()

    def __init_layers(self):
        for i in range(len(self.in_channels_list)):
            cls_sequence = list()
            reg_sequence = list()
            for j in range(self.stacks):
                cls_sequence.append(
                    DWCBR2(self.in_channels_list[i] if j == 0 else self.inner_channel,
                           self.inner_channel, 3, 1,
                           act_func=self.act_func)
                )
                if not self.share_cls_reg:
                    reg_sequence.append(
                        DWCBR2(self.in_channels_list[i] if j == 0 else self.inner_channel,
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

    @staticmethod
    def get_assign_lazy(grid_cells, gt_boxes, cell_num_each_level, top_k=7, inf=1000000):
        num_gt, num_grid_cells = gt_boxes.size(0), sum(cell_num_each_level)
        overlaps = bbox_overlaps(grid_cells, gt_boxes)
        assigned_gt_inds = overlaps.new_full((num_grid_cells,), -1, dtype=torch.long)

        gt_center = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2.0

        gc_center = (grid_cells[:, :2] + grid_cells[:, 2:]) / 2.0

        distances = (
            (gc_center[:, None, :] - gt_center[None, :, :]).pow(2).sum(-1).sqrt()
        )
        candidate_idxs = []
        start_idx = 0
        for level, cell_num in enumerate(cell_num_each_level):
            end_idx = start_idx + cell_num
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_k = min(top_k, cell_num)
            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False
            )
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)
        gt_idxs = torch.arange(num_gt)[None, :].repeat((candidate_idxs.size(0), 1))
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_thr_per_gt = candidate_overlaps.mean(0) + candidate_overlaps.std(0)
        candidate_is_pos = candidate_overlaps >= overlaps_thr_per_gt
        candidate_center = gc_center[candidate_idxs, :]
        candidate_gts = gt_boxes[gt_idxs, :]
        lt = candidate_center - candidate_gts[..., :2]
        rb = candidate_gts[..., 2:] - candidate_center
        candidate_is_in_gts = torch.cat([lt, rb], dim=-1).min(dim=-1)[0] > 0.01
        candidate_is_pos = candidate_is_pos & candidate_is_in_gts
        overlaps_inf = torch.full_like(overlaps, -inf)
        pos_candidate_idx, pos_gt_idx = candidate_idxs[candidate_is_pos], gt_idxs[candidate_is_pos]
        overlaps_inf[pos_candidate_idx, pos_gt_idx] = overlaps[pos_candidate_idx, pos_gt_idx]
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[max_overlaps != -inf] = (
            argmax_overlaps[max_overlaps != -inf]
        )
        return assigned_gt_inds

    @staticmethod
    def build_grid_cells(grid_cell_sizes):
        grid_cells = list()
        for grid_w, grid_h, cell_w, cell_h, stride in grid_cell_sizes:
            yv, xv = torch.meshgrid(
                [torch.arange(grid_h, dtype=torch.float32), torch.arange(grid_w, dtype=torch.float32)])
            cx, cy = (xv + 0.5) * stride, (yv + 0.5) * stride
            x1, x2 = cx - 0.5 * cell_w, cx + 0.5 * cell_w
            y1, y2 = cy - 0.5 * cell_h, cy + 0.5 * cell_h
            grid_cells.append(torch.stack([x1, y1, x2, y2], dim=-1).view(-1, 4))
        grid_cells = torch.cat(grid_cells, dim=0)
        return grid_cells

    @staticmethod
    def cat_layer_output(predicts):
        cls_outputs = list()
        reg_outputs = list()
        for cls, reg in predicts:
            b, cc, rc = cls.shape[0], cls.shape[1], reg.shape[1]
            cls_outputs.append(cls.permute(0, 2, 3, 1).contiguous().view(b, -1, cc))
            reg_outputs.append(reg.permute(0, 2, 3, 1).contiguous().view(b, -1, rc))
        return torch.cat(cls_outputs, dim=1), torch.cat(reg_outputs, dim=1)

    @staticmethod
    def get_single_level_center_points(h, w, stride, tensor):
        yv, xv = torch.meshgrid(
            [torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32)])
        cx, cy = (xv + 0.5) * stride, (yv + 0.5) * stride
        return torch.stack([cx, cy], dim=-1).view(-1, 2).type_as(tensor)

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
            return self.compute_loss(predicts, targets)
        else:
            return self.predict(xs)

    def compute_loss(self, predicts, targets):
        expand_cls_predict, expand_reg_predicts = self.cat_layer_output(predicts)
        grid_sizes = [[cls.shape[3], cls.shape[2]] for cls, _ in predicts]
        grid_cell_sizes = [[*grid_sizes[i], *self.cell_sizes[i], self.strides[i]] for i in range(len(predicts))]
        grid_cells = self.build_grid_cells(grid_cell_sizes).type_as(expand_cls_predict)
        cell_num_each_level = [w * h for w, h in grid_sizes]
        expand_strides = torch.tensor(sum([[s] * n for n, s in zip(cell_num_each_level, self.strides)],
                                          [])).type_as(expand_cls_predict)
        labels_all = targets['label']
        label_list = labels_all.split(targets['batch_len'])

        cls_target_list = list()
        box_predict_list = list()
        box_targets_list = list()
        dfl_predict_list = list()
        dfl_targets_list = list()
        weights_targets_list = list()
        gt_num = 0
        for i in range(len(label_list)):
            label = label_list[i]
            cls_target = torch.zeros_like(expand_cls_predict[0])
            if len(label) < 1:
                cls_target_list.append(cls_target)
                continue
            assigned_gt_inds = self.get_assign_lazy(grid_cells, label[:, 1:], cell_num_each_level, self.top_k)
            gt_mask = assigned_gt_inds >= 0
            if gt_mask.sum() < 1:
                cls_target_list.append(cls_target)
                continue
            gt_num += gt_mask.sum()
            pos_bbox_pred = expand_reg_predicts[i][gt_mask]
            weights_targets = expand_cls_predict[i][gt_mask].detach().sigmoid().max(dim=1)[0]
            pos_cell_center = ((grid_cells[:, :2] + grid_cells[:, 2:]) / 2.0)[gt_mask]
            pos_bbox_targets = label[:, 1:][assigned_gt_inds[gt_mask]]
            pos_bbox_pred_corners = self.distribute_project(pos_bbox_pred) * expand_strides[gt_mask][:, None]
            x1y1 = pos_cell_center - pos_bbox_pred_corners[:, :2]
            x2y2 = pos_cell_center + pos_bbox_pred_corners[:, 2:]
            pos_decode_bbox_pred = torch.cat([x1y1, x2y2], dim=-1)
            cls_target[gt_mask, label[:, 0][assigned_gt_inds[gt_mask]].long()] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_bbox_targets, is_aligned=True)
            corner_lt = (pos_cell_center - pos_bbox_targets[:, :2]) / expand_strides[gt_mask][:, None]
            corner_rb = (pos_bbox_targets[:, 2:] - pos_cell_center) / expand_strides[gt_mask][:, None]
            corners_targets = torch.cat([corner_lt, corner_rb],
                                        dim=-1).clamp(min=0, max=self.reg_max - 0.1)
            cls_target_list.append(cls_target)
            box_predict_list.append(pos_decode_bbox_pred)
            box_targets_list.append(pos_bbox_targets)
            dfl_predict_list.append(pos_bbox_pred)
            dfl_targets_list.append(corners_targets)
            weights_targets_list.append(weights_targets)
        weights_targets_tensor = torch.cat(weights_targets_list, dim=0)
        dfl_predict_tensor = torch.cat(dfl_predict_list, dim=0)
        dfl_targets_tensor = torch.cat(dfl_targets_list, dim=0)
        qfl_predict_tensor = expand_cls_predict
        qfl_target_tensor = torch.stack(cls_target_list, dim=0)
        box_predict_tensor = torch.cat(box_predict_list, dim=0)
        box_targets_tensor = torch.cat(box_targets_list, dim=0)
        gt_num = reduce_mean(gt_num)
        avg_factor = weights_targets_tensor.sum()
        avg_factor = reduce_mean(avg_factor)
        qfl_loss = self.qlf_loss(qfl_predict_tensor, qfl_target_tensor).sum() / gt_num
        dfl_loss = self.dlf_loss(dfl_predict_tensor, dfl_targets_tensor, weights_targets_tensor).sum() / avg_factor
        iou_loss = self.iou_loss(box_predict_tensor, box_targets_tensor, weights_targets_tensor).sum() / avg_factor
        loss = qfl_loss + dfl_loss + iou_loss
        return loss, iou_loss.detach(), qfl_loss.detach(), dfl_loss.detach(), gt_num

    @torch.no_grad()
    def predict(self, xs):
        nms_pre = 1000
        outputs = self.forward_impl(xs)
        bs = xs[0].shape[0]
        inp_h, inp_w = xs[0].shape[2] * self.strides[0], xs[0].shape[3] * self.strides[0]
        boxes = list()
        scores = list()
        dets = list()
        for i, (cls, reg) in enumerate(outputs):
            score = cls.sigmoid().permute(0, 2, 3, 1).reshape(bs, -1, cls.shape[1])
            delta = reg.permute(0, 2, 3, 1).reshape(bs, -1, reg.shape[1])
            corner = self.distribute_project(delta).reshape(bs, -1, 4) * self.strides[i]
            center = self.get_single_level_center_points(cls.shape[2], cls.shape[3], self.strides[i], score)
            x1y1 = center - corner[..., :2]
            x2y2 = center + corner[..., 2:]
            box = torch.cat([x1y1, x2y2], dim=-1)
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
