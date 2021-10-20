import torch
import torch.nn.functional as f
from torch import nn
from utils.box_utils import bboxes_iou, IOULoss


class YOLOXLoss(nn.Module):
    def __init__(self, model,
                 center_radius=2.5,
                 reg_weights=5.0,
                 iou_type="iou"):
        super(YOLOXLoss, self).__init__()
        self.center_radius = center_radius
        self.model = model
        self.grids_all = None
        self.expand_strides = None
        self.reg_weights = reg_weights
        #  self.expand_strides Tensor(size=(proposal_num,))
        #  self.grids_all Tensor(size=(1,proposal_num,2))
        self.iou_loss = IOULoss(iou_type=iou_type, coord_type="xywh")
        self.bce = nn.BCELoss(reduction="none")

    def forward(self, predicts, targets):
        """
        :param predicts: list[Tensor(size=(n,5+cls,h,w))] len(predicts) = len(fpn_layers)
        :param targets: dict({"label": Tensor(size=(n,cls_id+4)),"batch_len":list[int,int,...]})
               coord form 'xywh' no norm
        :return:
        """
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
                match_cls.long(), self.model.head.num_classes
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
            grid = self.model.head.grids[i]
            grids_all.append(grid.permute(0, 2, 3, 1).contiguous().view(1, -1, 2))
            stride = self.model.head.strides[i]
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
        gt_cls = f.one_hot(labels[:, 0].long(), self.model.head.num_classes).float()[:, None, :].repeat(1, fg_num, 1)
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
