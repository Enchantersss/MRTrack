# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
DETR model and criterion classes.
"""
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List

from util import box_ops, checkpoint
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou

from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer_plus import build_deforamble_transformer, pos2posemb
from .qim import build as build_query_interaction_layer
from .deformable_detr import SetCriterion, MLP, sigmoid_focal_loss

class _LinearScheduler:
    def __init__(self, init_val: float, final_val: float, total_epochs: int):
        self.init = float(init_val); self.final = float(final_val)
        self.total = max(int(total_epochs), 1); self.epoch = 0
    def set_epoch(self, epoch: int): self.epoch = max(int(epoch), 0)
    def __call__(self) -> float:
        t = min(self.epoch, self.total)
        return self.init + (self.final - self.init) * (t / self.total)

class _ConstantScheduler:
    def __init__(self, val: float): self.val = float(val)
    def set_epoch(self, epoch: int): pass
    def __call__(self) -> float: return self.val

@torch.no_grad()
def _greedy_assignment(cost: torch.Tensor):
    Ns, Ng = cost.shape
    if Ns == 0 or Ng == 0:
        return (torch.as_tensor([], dtype=torch.long, device=cost.device),
                torch.as_tensor([], dtype=torch.long, device=cost.device))
    C = cost.clone()
    used_r = torch.zeros(Ns, dtype=torch.bool, device=cost.device)
    used_c = torch.zeros(Ng, dtype=torch.bool, device=cost.device)
    src_sel, tgt_sel = [], []
    for _ in range(min(Ns, Ng)):
        C_masked = C.masked_fill(used_r[:, None] | used_c[None, :], float('inf'))
        v, idx = torch.min(C_masked.view(-1), dim=0)
        if not torch.isfinite(v): break
        r = (idx // Ng).item(); c = (idx % Ng).item()
        src_sel.append(r); tgt_sel.append(c)
        used_r[r] = True; used_c[c] = True
    return (torch.as_tensor(src_sel, dtype=torch.long, device=cost.device),
            torch.as_tensor(tgt_sel, dtype=torch.long, device=cost.device))


def _get_dt_this(dts, frame_index: int) -> float:
    if dts is None or frame_index == 0:
        return 1.0
    idx = frame_index - 1
    if isinstance(dts, torch.Tensor):
        if dts.ndim == 0:
            return float(dts.item())
        if dts.ndim == 1:
            return float(dts[idx].item())
        return float(dts[0, idx].item())

    if isinstance(dts, (list, tuple)) and len(dts) > 0:
        first = dts[0]
        if isinstance(first, (list, tuple)) or (isinstance(first, torch.Tensor) and first.ndim > 0):
            dts = first
        x = dts[idx]
        if isinstance(x, torch.Tensor):
            if x.ndim == 0:
                return float(x.item())
            return float(x[0].item())
        if isinstance(x, (list, tuple)):
            return float(x[0])
        return float(x)

    return float(dts)

class ClipMatcher(SetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, losses, args=None):
        super().__init__(num_classes, matcher, weight_dict, losses)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0

        self.gm_enable = True if args is None else bool(getattr(args, 'gm_enable', True))
        self.total_epochs = 50 if args is None else int(getattr(args, 'epochs', 50))
        a_init = 2.0 if args is None else float(getattr(args, 'gm_alpha_init', 2.0))
        a_final = 0.5 if args is None else float(getattr(args, 'gm_alpha_final', 0.5))
        p_init = 0.9 if args is None else float(getattr(args, 'lock_p_init', 0.9))
        p_final = 0.3 if args is None else float(getattr(args, 'lock_p_final', 0.3))
        self.lock_sched_enable = True if args is None else bool(getattr(args, 'lock_sched_enable', True))
        self.alpha_scheduler = _LinearScheduler(a_init, a_final, self.total_epochs) if self.gm_enable else _ConstantScheduler(0.0)
        self.lock_scheduler  = _LinearScheduler(p_init, p_final, self.total_epochs) if self.lock_sched_enable else _ConstantScheduler(1.0)

        self.ids_margin_enable = True if args is None else bool(getattr(args, 'ids_margin_enable', True))
        self.ids_margin_m = 0.2 if args is None else float(getattr(args, 'ids_margin_m', 0.2))

        self._release_cnt = 0
        self._inherit_cnt = 0
        self._recover_cnt = 0
        self._rewrite_cnt = 0
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        self.alpha_scheduler.set_epoch(self._epoch)
        self.lock_scheduler.set_epoch(self._epoch)

    def initialize_for_single_clip(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}
        self._release_cnt = 0
        self._inherit_cnt = 0
        self._recover_cnt = 0
        self._rewrite_cnt = 0

    def _step(self): self._current_frame_idx += 1

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized(): torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes, **kwargs):
        loss_map = {'labels': self.loss_labels, 'cardinality': self.loss_cardinality, 'boxes': self.loss_boxes}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)

    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)
        mask = (target_obj_ids != -1)
        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))
        return {'loss_bbox': loss_bbox.sum() / num_boxes, 'loss_giou': loss_giou.sum() / num_boxes}

    def _compute_cost_matrix(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor, gt_instances: Instances,
                             use_focal: bool = True) -> torch.Tensor:
        device = pred_boxes.device
        Ns = pred_boxes.shape[0]; Ng = len(gt_instances)
        if Ns == 0 or Ng == 0: return torch.zeros((Ns, Ng), device=device)
        c_class = float(getattr(self.matcher, 'cost_class', 1.0))
        c_bbox  = float(getattr(self.matcher, 'cost_bbox', 5.0))
        c_giou  = float(getattr(self.matcher, 'cost_giou', 2.0))
        tgt_cls = gt_instances.labels.long()
        if use_focal:
            out_prob = pred_logits.sigmoid()
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_cls] - neg_cost_class[:, tgt_cls]
        elif pred_logits.shape[-1] == 1:
            prob_fg = pred_logits.sigmoid().squeeze(-1); cost_class = -prob_fg[:, None].expand(Ns, Ng)
        else:
            prob = pred_logits.sigmoid(); cost_class = -prob[:, tgt_cls]
        tgt_boxes = gt_instances.boxes
        cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)
        cost_giou = -box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(pred_boxes), box_ops.box_cxcywh_to_xyxy(tgt_boxes))
        return c_class * cost_class + c_bbox * cost_bbox + c_giou * cost_giou

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J)
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            labels.append(labels_per_img)
        target_classes_o = torch.cat(labels)
        target_classes[idx] = target_classes_o
        if self.focal_loss:
            gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[:, :, :-1].to(src_logits)
            loss_ce = sigmoid_focal_loss(src_logits.flatten(1), gt_labels_target.flatten(1),
                                         alpha=0.25, gamma=2, num_boxes=num_boxes, mean_in_dim1=False).sum()
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def match_for_single_frame(self, outputs: dict):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        gt_instances_i = self.gt_instances[self._current_frame_idx]
        track_instances: Instances = outputs_without_aux['track_instances']
        pred_logits_i = track_instances.pred_logits
        pred_boxes_i = track_instances.pred_boxes
        obj_idxes = gt_instances_i.obj_ids

        outputs_i = {'pred_logits': pred_logits_i.unsqueeze(0), 'pred_boxes': pred_boxes_i.unsqueeze(0)}

        num_disappear_track = 0
        track_instances.matched_gt_idxes[:] = -1
        i, j = torch.where(track_instances.obj_idxes[:, None] == obj_idxes)
        track_instances.matched_gt_idxes[i] = j
        device = pred_logits_i.device
        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long, device=device)
        matched_track_idxes = (track_instances.obj_idxes >= 0)
        prev_matched_indices = torch.stack([full_track_idxes[matched_track_idxes],
                                            track_instances.matched_gt_idxes[matched_track_idxes]], dim=1)
        if len(prev_matched_indices) > 0:
            valid_prev = prev_matched_indices[:, 1] >= 0
            prev_matched_indices = prev_matched_indices[valid_prev]

        guided_ok = self.gm_enable and hasattr(self.matcher, 'compute_cost') and hasattr(self.matcher, 'hungarian')

        if guided_ok:
            self._inherit_cnt += len(prev_matched_indices)
            if len(prev_matched_indices) > 0:
                p_lock = float(self.lock_scheduler()) if self.lock_sched_enable else 1.0
                randv = torch.rand(len(prev_matched_indices), device=device)
                locked_mask = randv < p_lock
                locked_pairs = prev_matched_indices[locked_mask]
                released_pairs = prev_matched_indices[~locked_mask]
                if len(released_pairs) > 0:
                    in_range = (released_pairs[:, 1] >= 0) & (released_pairs[:, 1] < len(gt_instances_i))
                    released_pairs = released_pairs[in_range]
            else:
                locked_pairs = prev_matched_indices.new_zeros((0, 2), dtype=torch.long)
                released_pairs = prev_matched_indices.new_zeros((0, 2), dtype=torch.long)
            self._release_cnt += len(released_pairs)

            unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]
            tgt_indexes = track_instances.matched_gt_idxes
            tgt_indexes = tgt_indexes[tgt_indexes != -1]
            tgt_state = torch.zeros(len(gt_instances_i), device=device)
            tgt_state[tgt_indexes] = 1
            untracked_tgt_indexes = torch.arange(len(gt_instances_i), device=device)[tgt_state == 0]

            released_track_idxes = released_pairs[:, 0] if len(released_pairs) > 0 else prev_matched_indices.new_zeros((0,), dtype=torch.long)
            released_prev_gt_idxes = released_pairs[:, 1] if len(released_pairs) > 0 else prev_matched_indices.new_zeros((0,), dtype=torch.long)
            slot_candidates = torch.cat([unmatched_track_idxes, released_track_idxes], dim=0)
            if len(gt_instances_i) == 0:
                gt_candidates = released_prev_gt_idxes.new_zeros((0,), dtype=torch.long)
            else:
                gt_candidates = torch.unique(torch.cat([untracked_tgt_indexes, released_prev_gt_idxes], dim=0), sorted=True)

            if len(slot_candidates) == 0 or len(gt_candidates) == 0:
                new_matched_indices = prev_matched_indices.new_zeros((0, 2), dtype=torch.long)
                matched_indices = torch.cat([new_matched_indices, locked_pairs], dim=0)
            else:
                subset_logits = track_instances.pred_logits[slot_candidates]
                subset_boxes  = track_instances.pred_boxes[slot_candidates]
                gt_subset = gt_instances_i[gt_candidates]

                if hasattr(self.matcher, 'compute_cost'):
                    C_base = self.matcher.compute_cost({'pred_logits': subset_logits.detach()[None],
                                                        'pred_boxes': subset_boxes.detach()[None]}, [gt_subset])[0]
                else:
                    C_base = self._compute_cost_matrix(subset_logits.detach(), subset_boxes.detach(), gt_subset)

                alpha = float(self.alpha_scheduler())
                C = C_base.clone()
                if len(released_track_idxes) > 0:
                    gt_abs_to_local = {int(g.item()): idx for idx, g in enumerate(gt_candidates)}
                    prev_gt_for_slot = {int(s.item()): int(g.item()) for s, g in released_pairs}
                    for local_i, slot in enumerate(slot_candidates.tolist()):
                        if slot in prev_gt_for_slot:
                            g_abs = prev_gt_for_slot[slot]
                            if g_abs in gt_abs_to_local:
                                j_local = gt_abs_to_local[g_abs]
                                C[local_i, j_local] = C[local_i, j_local] - alpha

                if self.ids_margin_enable and len(released_track_idxes) > 0:
                    gt_abs_to_local = {int(g.item()): idx for idx, g in enumerate(gt_candidates)}
                    slot_to_local = {int(s.item()): idx for idx, s in enumerate(slot_candidates)}
                    C_margin = self._compute_cost_matrix(subset_logits, subset_boxes, gt_subset)
                    ids_margin_loss = torch.tensor(0.0, device=device); cnt = 0
                    for s_abs, g_prev_abs in released_pairs.tolist():
                        if s_abs in slot_to_local and g_prev_abs in gt_abs_to_local:
                            i_local = slot_to_local[s_abs]; j_local = gt_abs_to_local[g_prev_abs]
                            row = C_margin[i_local]  # [Ng]
                            if row.numel() <= 1: continue
                            others = torch.cat([row[:j_local], row[j_local+1:]])
                            min_other = others.min()
                            ids_margin_loss = ids_margin_loss + F.relu(row[j_local] + self.ids_margin_m - min_other)
                            cnt += 1
                    self.losses_dict[f'frame_{self._current_frame_idx}_loss_ids_margin'] = ids_margin_loss

                if hasattr(self.matcher, 'hungarian'):
                    src_idx, tgt_idx = self.matcher.hungarian(C)
                else:
                    src_idx, tgt_idx = _greedy_assignment(C)

                new_matched_indices = torch.stack([slot_candidates[src_idx], gt_candidates[tgt_idx]], dim=1)
                matched_indices = torch.cat([new_matched_indices, locked_pairs], dim=0)

                if len(released_pairs) > 0 and len(new_matched_indices) > 0:
                    prev_map = {int(s.item()): int(g.item()) for s, g in released_pairs}
                    final_map = {int(s.item()): int(g.item()) for s, g in new_matched_indices}
                    for s_abs, g_prev_abs in prev_map.items():
                        if s_abs in final_map:
                            if final_map[s_abs] == g_prev_abs: self._recover_cnt += 1
                            else: self._rewrite_cnt += 1

        else:
            unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]
            tgt_indexes = track_instances.matched_gt_idxes
            tgt_indexes = tgt_indexes[tgt_indexes != -1]
            tgt_state = torch.zeros(len(gt_instances_i), device=pred_logits_i.device)
            tgt_state[tgt_indexes] = 1
            untracked_tgt_indexes = torch.arange(len(gt_instances_i), device=pred_logits_i.device)[tgt_state == 0]
            untracked_gt_instances = gt_instances_i[untracked_tgt_indexes]
            def match_for_single_decoder_layer(unmatched_outputs, matcher):
                new_track_indices = matcher(unmatched_outputs, [untracked_gt_instances])
                src_idx = new_track_indices[0][0]; tgt_idx = new_track_indices[0][1]
                return torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]], dim=1).to(pred_logits_i.device)
            unmatched_outputs = {'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
                                 'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0)}
            new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher)
            matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        if len(new_matched_indices) > 0:
            track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
            track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = pred_logits_i.device
        for loss in self.losses:
            ldict = self.get_loss(loss, outputs=outputs_i, gt_instances=[gt_instances_i],
                                  indices=[(matched_indices[:, 0], matched_indices[:, 1])], num_boxes=1)
            self.losses_dict.update({f'frame_{self._current_frame_idx}_{k}': v for k, v in ldict.items()})

        if guided_ok:
            denom = max(self._inherit_cnt, 1)
            rel = torch.tensor(self._release_cnt / denom, device=device)
            rec = torch.tensor(self._recover_cnt / max(self._release_cnt, 1), device=device) if self._release_cnt > 0 else torch.tensor(0.0, device=device)
            rew = torch.tensor(self._rewrite_cnt / max(self._release_cnt, 1), device=device) if self._release_cnt > 0 else torch.tensor(0.0, device=device)
            self.losses_dict[f'frame_{self._current_frame_idx}_release_rate'] = rel
            self.losses_dict[f'frame_{self._current_frame_idx}_recover_rate'] = rec
            self.losses_dict[f'frame_{self._current_frame_idx}_rewrite_rate'] = rew

        if 'aux_outputs' in outputs:
            for li, aux_outputs in enumerate(outputs['aux_outputs']):
                if guided_ok and 'slot_candidates' in locals() and 'gt_candidates' in locals():
                    Ns = len(slot_candidates); Ng = len(gt_candidates)
                    if Ns > 0 and Ng > 0:
                        subset_logits = aux_outputs['pred_logits'][0, slot_candidates]
                        subset_boxes  = aux_outputs['pred_boxes'][0, slot_candidates]
                        gt_subset = gt_instances_i[gt_candidates]
                        if hasattr(self.matcher, 'compute_cost'):
                            C_base = self.matcher.compute_cost({'pred_logits': subset_logits.detach()[None],
                                                                'pred_boxes': subset_boxes.detach()[None]}, [gt_subset])[0]
                        else:
                            C_base = self._compute_cost_matrix(subset_logits.detach(), subset_boxes.detach(), gt_subset)
                        alpha = float(self.alpha_scheduler()); C = C_base.clone()
                        if len(released_track_idxes) > 0:
                            gt_abs_to_local = {int(g.item()): idx for idx, g in enumerate(gt_candidates)}
                            prev_gt_for_slot = {int(s.item()): int(g.item()) for s, g in released_pairs}
                            for local_i, slot in enumerate(slot_candidates.tolist()):
                                if slot in prev_gt_for_slot:
                                    g_abs = prev_gt_for_slot[slot]
                                    if g_abs in gt_abs_to_local:
                                        j_local = gt_abs_to_local[g_abs]
                                        C[local_i, j_local] = C[local_i, j_local] - alpha
                        if hasattr(self.matcher, 'hungarian'):
                            src_idx, tgt_idx = self.matcher.hungarian(C)
                        else:
                            src_idx, tgt_idx = _greedy_assignment(C)
                        new_matched_indices_layer = torch.stack([slot_candidates[src_idx], gt_candidates[tgt_idx]], dim=1)
                        matched_indices_layer = torch.cat([new_matched_indices_layer, locked_pairs], dim=0)
                    else:
                        matched_indices_layer = matched_indices
                else:
                    unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]
                    tgt_indexes = track_instances.matched_gt_idxes
                    tgt_indexes = tgt_indexes[tgt_indexes != -1]
                    tgt_state = torch.zeros(len(gt_instances_i), device=device)
                    tgt_state[tgt_indexes] = 1
                    untracked_tgt_indexes = torch.arange(len(gt_instances_i), device=device)[tgt_state == 0]
                    def _m(unmatched_outputs, matcher):
                        new_track_indices = matcher(unmatched_outputs, [gt_instances_i[untracked_tgt_indexes]])
                        src_idx = new_track_indices[0][0]; tgt_idx = new_track_indices[0][1]
                        return torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]], dim=1).to(device)
                    unmatched_outputs_layer = {'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_idxes].unsqueeze(0),
                                               'pred_boxes': aux_outputs['pred_boxes'][0, unmatched_track_idxes].unsqueeze(0)}
                    new_matched_indices_layer = _m(unmatched_outputs_layer, self.matcher)
                    matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                for loss in self.losses:
                    if loss == 'masks': continue
                    l_dict = self.get_loss(loss, aux_outputs, gt_instances=[gt_instances_i],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])], num_boxes=1)
                    self.losses_dict.update({f'frame_{self._current_frame_idx}_aux{li}_{k}': v for k, v in l_dict.items()})

        if 'ps_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['ps_outputs']):
                ar = torch.arange(len(gt_instances_i), device=obj_idxes.device)
                l_dict = self.get_loss('boxes', aux_outputs, gt_instances=[gt_instances_i], indices=[(ar, ar)], num_boxes=1)
                self.losses_dict.update({f'frame_{self._current_frame_idx}_ps{i}_{k}': v for k, v in l_dict.items()})

        self._step()
        return track_instances

    def forward(self, outputs, input_data: dict):
        losses = outputs.pop("losses_dict")

        device = self.sample_device
        if device is None:
            if len(losses) > 0:
                device = next(iter(losses.values())).device
            else:
                device = torch.device("cpu")

        for k in self.weight_dict.keys():
            if k not in losses:
                losses[k] = torch.tensor(0.0, device=device)

        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in list(losses.items()):
            if ("release_rate" in loss_name) or ("recover_rate" in loss_name) or ("rewrite_rate" in loss_name):
                continue
            losses[loss_name] = loss / num_samples

        return losses

class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=10):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.score_thresh)
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_thresh)
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_idxes[to_del] = -1

class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes

        scores = out_logits[..., 0].sigmoid()

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        track_instances.boxes = boxes
        track_instances.scores = scores
        track_instances.labels = torch.full_like(scores, 0)
        return track_instances


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MOTR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, criterion, track_embed,
                 aux_loss=True, with_box_refine=False, two_stage=False, memory_bank=None, use_checkpoint=False,
                 query_denoise=0, ref_extrapolate_before_decoder=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.track_embed = track_embed
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.use_checkpoint = use_checkpoint
        self.query_denoise = query_denoise
        self.ref_extrapolate_before_decoder = ref_extrapolate_before_decoder
        self.position = nn.Embedding(num_queries, 4)
        self.yolox_embed = nn.Embedding(1, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if query_denoise:
            self.refine_embed = nn.Embedding(1, hidden_dim)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        nn.init.uniform_(self.position.weight.data, 0, 1)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.post_process = TrackerPostProcess()
        self.track_base = RuntimeTrackerBase()
        self.criterion = criterion
        self.memory_bank = memory_bank
        self.mem_bank_len = 0 if memory_bank is None else memory_bank.max_his_length

    def _generate_empty_tracks(self, proposals=None):
        track_instances = Instances((1, 1))
        num_queries, d_model = self.query_embed.weight.shape  # (300, 512)
        device = self.query_embed.weight.device
        if proposals is None:
            track_instances.ref_pts = self.position.weight # [num_queries, 4]
            track_instances.query_pos = self.query_embed.weight # [num_queries, d_model]
        else:
            track_instances.ref_pts = torch.cat([self.position.weight, proposals[:, :4]]) # learned ref_pts: [num_queries, 4] detection proposals: [N_prop, 4]
            track_instances.query_pos = torch.cat([self.query_embed.weight, pos2posemb(proposals[:, 4:], d_model) + self.yolox_embed.weight]) # learned embeddings: [num_queries, d_model]
        track_instances.output_embedding = torch.zeros((len(track_instances), d_model), device=device)
        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)
        track_instances.iou = torch.ones((len(track_instances),), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=device)
        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, d_model), dtype=torch.float32, device=device)
        track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros((len(track_instances), ), dtype=torch.float32, device=device)
        track_instances.vel = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        track_instances.prev_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        return track_instances.to(self.query_embed.weight.device)

    def clear(self):
        self.track_base.clear()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, }
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _forward_single_image(self, samples, track_instances: Instances, gtboxes=None,
                            n_new: int = 0, dt: float = 1.0):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        srcs, masks = [], []
        for l, feat in enumerate(features):
            s, m = feat.decompose()
            srcs.append(self.input_proj[l](s)); masks.append(m)
            assert m is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    s = self.input_proj[l](features[-1].tensors)
                else:
                    s = self.input_proj[l](srcs[-1])
                m = samples.mask
                m = F.interpolate(m[None].float(), size=s.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(s, m)).to(s.dtype)
                srcs.append(s); masks.append(m); pos.append(pos_l)

        ref_pts_base = track_instances.ref_pts  # [N_base, 4]
        N_base = ref_pts_base.shape[0]
        ref_pts_shifted = ref_pts_base

        if self.ref_extrapolate_before_decoder and hasattr(track_instances, "vel") and track_instances.vel.numel() > 0:
            vel_xy = track_instances.vel[..., :2]
            delta_xy = torch.zeros_like(ref_pts_base[..., :2])
            old_len = max(N_base - int(n_new), 0)
            if old_len > 0:
                delta_xy[int(n_new):, :] = vel_xy[int(n_new):, :] * float(dt)
            ref_pts_shifted = ref_pts_base.clone()
            ref_pts_shifted[..., :2] = (ref_pts_base[..., :2] + delta_xy).clamp(0, 1)

        if gtboxes is not None:
            n_dt = N_base
            ps_tgt = self.refine_embed.weight.expand(gtboxes.size(0), -1)
            query_embed = torch.cat([track_instances.query_pos, ps_tgt], dim=0)
            ref_pts = torch.cat([ref_pts_shifted, gtboxes], dim=0)
            attn_mask = torch.zeros((ref_pts.shape[0], ref_pts.shape[0]), dtype=torch.bool, device=ref_pts.device)
            attn_mask[:n_dt, n_dt:] = True
        else:
            query_embed = track_instances.query_pos
            ref_pts = ref_pts_shifted
            attn_mask = None

        trk_vel_xy = None
        if hasattr(track_instances, "vel") and track_instances.vel.numel() > 0:
            trk_vel_xy = track_instances.vel[..., :2]
            if gtboxes is not None:
                pad_vel = torch.zeros((gtboxes.shape[0], 2), dtype=trk_vel_xy.dtype, device=trk_vel_xy.device)
                trk_vel_xy = torch.cat([trk_vel_xy, pad_vel], dim=0)

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs, masks, pos, query_embed, ref_pts=ref_pts,
                            mem_bank=track_instances.mem_bank,
                            mem_bank_pad_mask=track_instances.mem_padding_mask,
                            attn_mask=attn_mask,
                            trk_vel_xy=trk_vel_xy,
                            dt_scalar=float(dt))

        outputs_classes, outputs_coords = [], []
        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4: tmp += reference
            else: tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class); outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        out['hs'] = hs[-1]
        return out



    def _post_process_single_image(self, frame_res, track_instances, is_last, dt: float = 1.0):
        if self.query_denoise > 0:
            n_ins = len(track_instances)
            ps_logits = frame_res['pred_logits'][:, n_ins:]
            ps_boxes = frame_res['pred_boxes'][:, n_ins:]
            frame_res['hs'] = frame_res['hs'][:, :n_ins]
            frame_res['pred_logits'] = frame_res['pred_logits'][:, :n_ins]
            frame_res['pred_boxes'] = frame_res['pred_boxes'][:, :n_ins]
            ps_outputs = [{'pred_logits': ps_logits, 'pred_boxes': ps_boxes}]
            for aux_outputs in frame_res.get('aux_outputs', []):
                ps_outputs.append({
                    'pred_logits': aux_outputs['pred_logits'][:, n_ins:],
                    'pred_boxes': aux_outputs['pred_boxes'][:, n_ins:],
                })
                aux_outputs['pred_logits'] = aux_outputs['pred_logits'][:, :n_ins]
                aux_outputs['pred_boxes'] = aux_outputs['pred_boxes'][:, :n_ins]
            frame_res['ps_outputs'] = ps_outputs

        with torch.no_grad():
            if self.training:
                track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values
            else:
                track_scores = frame_res['pred_logits'][0, :, 0].sigmoid()

        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res['pred_logits'][0]
        track_instances.pred_boxes = frame_res['pred_boxes'][0]
        track_instances.output_embedding = frame_res['hs'][0]

        if self.training:
            frame_res['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(frame_res)
        else:
            self.track_base.update(track_instances)
        with torch.no_grad():
            if self.training:
                valid_obs = (track_instances.matched_gt_idxes >= 0)
            else:
                valid_obs = (track_instances.scores >= self.track_base.score_thresh)

            dt_safe = float(dt) if (dt is not None and dt > 0) else 1.0
            beta = 0.8
            has_prev = (track_instances.prev_boxes.abs().sum(dim=-1) > 0)
            update_mask = valid_obs.to(dtype=torch.bool, device=track_instances.pred_boxes.device)
            if update_mask.any():
                delta = (track_instances.pred_boxes - track_instances.prev_boxes) / dt_safe
                updated_vel = torch.where(
                    has_prev.unsqueeze(-1),
                    beta * track_instances.vel + (1.0 - beta) * delta,
                    torch.zeros_like(track_instances.vel)
                )
                track_instances.vel = torch.where(
                    update_mask.unsqueeze(-1),
                    updated_vel,
                    track_instances.vel
                )
                track_instances.prev_boxes = torch.where(
                    update_mask.unsqueeze(-1),
                    track_instances.pred_boxes.detach(),
                    track_instances.prev_boxes
                )
        
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
        
        tmp = {}
        tmp['track_instances'] = track_instances
        if not is_last:
            out_track_instances = self.track_embed(tmp)
            frame_res['track_instances'] = out_track_instances
        else:
            frame_res['track_instances'] = None

        return frame_res

    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, track_instances=None, proposals=None):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        if track_instances is None:
            track_instances = self._generate_empty_tracks(proposals)
        else:
            track_instances = Instances.cat([
                self._generate_empty_tracks(proposals),
                track_instances])
        res = self._forward_single_image(img,
                                         track_instances=track_instances)
        res = self._post_process_single_image(res, track_instances, False)

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)
        ret = {'track_instances': track_instances}
        if 'ref_pts' in res:
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret

    def forward(self, data: dict):
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])

        frames = data['imgs']
        dts = data.get('dts', None)

        outputs = {'pred_logits': [], 'pred_boxes': []}
        track_instances = None
        keys = list(self._generate_empty_tracks()._fields.keys())

        for frame_index, (frame, gt, proposals) in enumerate(zip(frames, data['gt_instances'], data['proposals'])):
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1

            # denoising
            if self.query_denoise > 0:
                l_1 = l_2 = self.query_denoise
                gtboxes = gt.boxes.clone()
                _rs = torch.rand_like(gtboxes) * 2 - 1
                gtboxes[..., :2] += gtboxes[..., 2:] * _rs[..., :2] * l_1
                gtboxes[..., 2:] *= 1 + l_2 * _rs[..., 2:]
            else:
                gtboxes = None

            if track_instances is None:
                new_block = self._generate_empty_tracks(proposals)
                track_instances = new_block
            else:
                new_block = self._generate_empty_tracks(proposals)
                track_instances = Instances.cat([new_block, track_instances])

            n_new = len(new_block)
            dt_this = _get_dt_this(dts, frame_index)

            if self.use_checkpoint and frame_index < len(frames) - 1:
                def fn(frame_img, gtboxes_tensor, n_new_val, dt_val, *inst_fields):
                    frame_nt = nested_tensor_from_tensor_list([frame_img])
                    tmp_inst = Instances((1, 1), **dict(zip(keys, inst_fields)))
                    fr = self._forward_single_image(frame_nt, tmp_inst, gtboxes_tensor,
                                                    n_new=int(n_new_val), dt=float(dt_val))
                    aux_outputs = fr.get('aux_outputs', [])
                    return (
                        fr['pred_logits'], fr['pred_boxes'], fr['hs'],
                        *[aux['pred_logits'] for aux in aux_outputs],
                        *[aux['pred_boxes'] for aux in aux_outputs],
                    )

                args = [frame, gtboxes, float(n_new), float(dt_this)] + [track_instances.get(k) for k in keys]
                params = tuple((p for p in self.parameters() if p.requires_grad))
                tmp = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)
                aux_count = (len(tmp) - 3) // 2
                frame_res = {
                    'pred_logits': tmp[0], 'pred_boxes': tmp[1], 'hs': tmp[2],
                }
                if aux_count:
                    frame_res['aux_outputs'] = [
                        {'pred_logits': tmp[3+i], 'pred_boxes': tmp[3+aux_count+i]}
                        for i in range(aux_count)
                    ]
            else:
                frame_nt = nested_tensor_from_tensor_list([frame])
                frame_res = self._forward_single_image(frame_nt, track_instances, gtboxes,
                                                    n_new=int(n_new), dt=float(dt_this))

            frame_res = self._post_process_single_image(frame_res, track_instances, is_last, dt=dt_this)
            track_instances = frame_res['track_instances']

            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])

        if not self.training:
            outputs['track_instances'] = track_instances
        else:
            outputs['losses_dict'] = self.criterion.losses_dict
        return outputs




def build(args):
    dataset_to_num_classes = {
        'coco': 91,
        'coco_panoptic': 250,
        'e2e_mot': 1,
        'e2e_dance': 1,
        'e2e_joint': 1,
        'e2e_static_mot': 1,
    }
    assert args.dataset_file in dataset_to_num_classes
    num_classes = dataset_to_num_classes[args.dataset_file]
    device = torch.device(args.device)
    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    d_model = transformer.d_model
    hidden_dim = args.dim_feedforward
    query_interaction_layer = build_query_interaction_layer(args, args.query_interaction_layer, d_model, hidden_dim, d_model*2)

    img_matcher = build_matcher(args)
    num_frames_per_batch = max(args.sampler_lengths)
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_ce".format(i): args.cls_loss_coef,
                            'frame_{}_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_giou'.format(i): args.giou_loss_coef,
                            })

    # TODO this is a hack
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    })
            for j in range(args.dec_layers):
                weight_dict.update({"frame_{}_ps{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_ps{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_ps{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    })
    if args.memory_bank_type is not None and len(args.memory_bank_type) > 0:
        memory_bank = build_memory_bank(args, d_model, hidden_dim, d_model * 2)
        for i in range(num_frames_per_batch):
            weight_dict.update({"frame_{}_track_loss_ce".format(i): args.cls_loss_coef})
    else:
        memory_bank = None
    losses = ['labels', 'boxes']
    ids_margin_gamma = getattr(args, 'ids_margin_gamma', 0.1)
    for i in range(num_frames_per_batch):
        weight_dict.update({f"frame_{i}_loss_ids_margin": ids_margin_gamma})
    criterion = ClipMatcher(num_classes, matcher=img_matcher, weight_dict=weight_dict, losses=losses, args=args)
    criterion.to(device)
    postprocessors = {}
    model = MOTR(
        backbone,
        transformer,
        track_embed=query_interaction_layer,
        num_feature_levels=args.num_feature_levels,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        memory_bank=memory_bank,
        use_checkpoint=args.use_checkpoint,
        query_denoise=args.query_denoise,
        ref_extrapolate_before_decoder=getattr(args, 'ref_extrapolate_before_decoder', False),
    )
    return model, criterion, postprocessors
