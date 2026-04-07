"""
trainer.py
Hyperspectral Image Classification Trainer
Optimization includes gradient accumulation, mixed precision training, learning rate warmup, and early stopping.
CPU/GPU supported with flexible memory configuration.

start train with:
    trainer.cross_validate()
"""
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
import cv2, gc, json, os, re, time, \
        torch, traceback, warnings, copy, matplotlib

from tqdm import tqdm
from munch import Munch
from torch.nn import Module
from datetime import datetime
from typing import Tuple, Callable, Dict, Any, Iterable, Optional, List, Set
from contextlib import contextmanager
from concurrent.futures import as_completed
from pipeline.dataset import AbstractHSDataset
from torch.cuda.amp import GradScaler, autocast
from pipeline.monitor import tprint, _managed_pool
from scipy.ndimage import distance_transform_edt

from sklearn.base import BaseEstimator
from sklearn.metrics import (confusion_matrix as _cm, roc_curve,
                             precision_recall_fscore_support, auc)

warnings.filterwarnings("ignore")


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced classification.

    ```
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    ```
    
    Down-weights well-classified samples so training focuses on hard / rare ones.
    Supports segmentation outputs (N, C, H, W) with ignore_index masking.
    Optional label smoothing to prevent overconfident predictions.
    """

    def __init__(self, weight=None, gamma=2.0, ignore_index=-100,
                 reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.register_buffer('weight', weight)  # per-class weights (C,)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Flatten to 2D: (N*H*W, C) and (N*H*W,)
        if logits.dim() == 4:
            N, C, H, W = logits.shape
            logits = logits.permute(0, 2, 3, 1).reshape(-1, C)
            targets = targets.reshape(-1)
        elif logits.dim() == 3:
            logits = logits.reshape(-1, logits.shape[-1])
            targets = targets.reshape(-1)

        # Mask ignore pixels
        valid = (targets != self.ignore_index)
        logits = logits[valid]
        targets = targets[valid]

        if logits.numel() == 0:
            return logits.sum() * 0.0

        C = logits.shape[-1]
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()

        # Label smoothing: soft targets
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / C
            onehot = torch.zeros_like(log_p).scatter_(
                1, targets.unsqueeze(1), 1.0)
            soft_targets = onehot * (1.0 - self.label_smoothing) + smooth
            # Focal modulation on the true-class probability
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - p_t) ** self.gamma
            loss = -(soft_targets * log_p).sum(dim=-1) * focal_weight
        else:
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
            log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
            loss = -((1 - p_t) ** self.gamma) * log_p_t

        if self.weight is not None:
            loss = self.weight.to(logits.device)[targets] * loss

        return loss.mean() if self.reduction == 'mean' else loss.sum()


class HierarchicalSegLoss(nn.Module):
    """
    Hierarchical segmentation loss: BG/FG + foreground subclass loss.
    Expects model outputs with two-stage heads:
    
    - fg_bg_logits: binary logits for foreground/background separation.
    - fg_class_logits: multi-class logits for foreground subclass classification.
    
    Computes:
        total_loss = w_fg_loss * fg_class_loss + w_bgfg_loss * fg_bg_loss
    
    This encourages to first learn a coarse FG/BG distinction,
    then focus on fine-grained classification within the foreground.
    """

    def __init__(self,
                 num_classes: int,
                 class_weight=None,
                 fg_weight: float = 1.0,
                 bgfg_weight: float = 0.5,
                 bgfg_pos_weight: float = 2.0,
                 bgfg_hard_neg_weight: float = 2.0,
                 bgfg_boundary_weight: float = 0.0,
                 boundary_weight: float = 0.2,
                 aux_weight: float = 0.3,
                 fg_loss_type: str = 'focal',
                 cb_beta: float = 0.999,
                 logit_adjust_tau: float = 1.0,
                 bg_suppress_weight: float = 0.0,
                 bg_suppress_threshold: float = 0.2,
                 stage_b_bgfg_anchor_weight: float = 0.0,
                 stage_b_weak_cls_focus: float = 0.0,
                 stage_b_weak_cls_boost_max: float = 1.8,
                 stage_b_weak_cls_boost_momentum: float = 0.6,
                 stage_c_boundary_boost_max: float = 2.0,
                 stage_c_boundary_boost_min: float = 0.75,
                 gamma: float = 2.0,
                 ignore_index: int = 255,
                 label_smoothing: float = 0.0):
        
        super().__init__()
        self.num_classes = int(num_classes)
        self.fg_weight = float(fg_weight)
        self.bgfg_weight = float(bgfg_weight)
        self._base_fg_weight = float(fg_weight)
        self._base_bgfg_weight = float(bgfg_weight)
        self.bgfg_pos_weight = float(max(1.0, bgfg_pos_weight))
        self.bgfg_hard_neg_weight = float(max(0.0, bgfg_hard_neg_weight))
        self._base_bgfg_hard_neg_weight = float(max(0.0, bgfg_hard_neg_weight))
        self.bgfg_boundary_weight = float(max(0.0, bgfg_boundary_weight))
        self.boundary_weight = float(boundary_weight)
        self.aux_weight = float(aux_weight)
        self.fg_loss_type = str(fg_loss_type).lower()
        self.cb_beta = float(np.clip(cb_beta, 0.0, 0.999999))
        self.logit_adjust_tau = float(max(0.0, logit_adjust_tau))
        self.bg_suppress_weight = float(max(0.0, bg_suppress_weight))
        self.bg_suppress_threshold = float(np.clip(bg_suppress_threshold, 0.0, 1.0))
        self.stage_b_bgfg_anchor_weight = float(max(0.0, stage_b_bgfg_anchor_weight))
        self.stage_b_weak_cls_focus = float(np.clip(stage_b_weak_cls_focus, 0.0, 1.0))
        self.stage_b_weak_cls_boost_max = float(max(1.0, stage_b_weak_cls_boost_max))
        self.stage_b_weak_cls_boost_momentum = float(np.clip(stage_b_weak_cls_boost_momentum, 0.0, 0.99))
        self.stage_c_boundary_boost_max = float(max(1.0, stage_c_boundary_boost_max))
        self.stage_c_boundary_boost_min = float(np.clip(stage_c_boundary_boost_min, 0.1, 1.0))
        self.ignore_index = int(ignore_index)

        # Foreground subclass targets are remapped from [1..C-1] -> [0..C-2].
        # Use foreground-only class weights to avoid BG weight index misalignment.
        fg_class_weight = None
        if class_weight is not None:
            if torch.is_tensor(class_weight):
                cw = class_weight.detach().float().clone()
            else:
                cw = torch.as_tensor(class_weight, dtype=torch.float32)

            if cw.numel() == self.num_classes:
                fg_class_weight = cw[1:].clone()
            elif cw.numel() == (self.num_classes - 1):
                fg_class_weight = cw.clone()
            else:
                raise ValueError(
                    f"class_weight length mismatch: got {cw.numel()}, "
                    f"expected {self.num_classes} or {self.num_classes - 1}"
                )

            pos = fg_class_weight > 0
            if pos.any():
                fg_class_weight[pos] = fg_class_weight[pos] / fg_class_weight[pos].mean().clamp(min=1e-8)
                if self.fg_loss_type == 'cb_focal':
                    # Stronger long-tail emphasis for foreground subclasses.
                    fg_class_weight[pos] = fg_class_weight[pos].pow(1.25)
                    fg_class_weight[pos] = fg_class_weight[pos] / fg_class_weight[pos].mean().clamp(min=1e-8)

        self.fg_logit_adjust = None
        if fg_class_weight is not None:
            # Infer normalized foreground priors from inverse-frequency style weights.
            inv = 1.0 / fg_class_weight.clamp(min=1e-8)
            prior = inv / inv.sum().clamp(min=1e-8)
            self.fg_logit_adjust = prior.clamp(min=1e-8).log()

        self.multi_cls_loss = FocalLoss(
            weight=fg_class_weight,
            gamma=gamma,
            ignore_index=ignore_index,
            reduction='mean',
            label_smoothing=label_smoothing,
        )
        self._eps = 1e-8
        self.active_stage = 'C'
        self._fg_class_dynamic_boost = torch.ones(max(1, self.num_classes - 1), dtype=torch.float32)
        self._boundary_dynamic_scale = 1.0

    def set_active_stage(self, stage: str) -> None:
        stage = str(stage).upper()
        if stage not in {'A', 'B', 'C'}:
            stage = 'C'
        self.active_stage = stage

    def set_fg_class_recall_feedback(self, fg_class_recalls: Optional[np.ndarray],
                                     target_min_recall: float) -> None:
        """Update per-class weak boost based on latest foreground class recalls."""
        if fg_class_recalls is None:
            return
        recalls = np.asarray(fg_class_recalls, dtype=np.float32).reshape(-1)
        if recalls.size != max(1, self.num_classes - 1):
            return
        recalls = np.nan_to_num(recalls, nan=0.0, posinf=100.0, neginf=0.0)
        target = float(max(1e-6, target_min_recall))
        deficits = np.clip((target - recalls) / target, 0.0, 1.0)
        raw_boost = 1.0 + deficits * (self.stage_b_weak_cls_boost_max - 1.0)
        old = self._fg_class_dynamic_boost.detach().cpu().numpy()
        m = self.stage_b_weak_cls_boost_momentum
        smoothed = m * old + (1.0 - m) * raw_boost
        self._fg_class_dynamic_boost = torch.as_tensor(smoothed, dtype=torch.float32)

    def set_boundary_feedback(self, boundary_ba: float, target_ba: float) -> None:
        """Adapt boundary loss scale for stage C according to boundary-band BA."""
        ba = float(boundary_ba)
        if not np.isfinite(ba):
            return
        target = float(max(1e-6, target_ba))
        gap = (target - ba) / target
        if gap >= 0.0:
            scale = 1.0 + gap * (self.stage_c_boundary_boost_max - 1.0)
        else:
            scale = 1.0 + gap * (1.0 - self.stage_c_boundary_boost_min)
        self._boundary_dynamic_scale = float(np.clip(scale, self.stage_c_boundary_boost_min, self.stage_c_boundary_boost_max))

    @staticmethod
    def _soft_dice_binary(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        inter = (prob * target).sum()
        denom = prob.sum() + target.sum()
        return 1.0 - (2.0 * inter + 1e-8) / (denom + 1e-8)

    @staticmethod
    def _build_boundary_target(targets: torch.Tensor, ignore_index: int) -> torch.Tensor:
        """Build binary boundary target map from dense labels."""
        t = targets.clone()
        valid = (t != ignore_index)
        t[~valid] = 0

        edge = torch.zeros_like(t, dtype=torch.bool)
        # Horizontal and vertical label changes define boundaries.
        edge[:, :, 1:] |= (t[:, :, 1:] != t[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
        edge[:, 1:, :] |= (t[:, 1:, :] != t[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
        return edge.float()

    def set_curriculum(self, progress: float) -> None:
        """30/40/30 schedule: keep BG/FG supervision active through training."""
        progress = float(np.clip(progress, 0.0, 1.0))
        if progress < 0.3:
            self.bgfg_weight = 1.0
            self.fg_weight = 0.4
        elif progress < 0.7:
            self.bgfg_weight = 0.8
            self.fg_weight = 0.8
        else:
            self.bgfg_weight = 0.6
            self.fg_weight = 1.0

    def set_fg_bias_feedback(self, pred_fg_ratio: float, gt_fg_ratio: float) -> None:
        """Adapt hard-negative strength based on FG over/under prediction gap."""
        gap = float(pred_fg_ratio - gt_fg_ratio)
        if gap > 10.0:
            scale = 1.3
        elif gap < -10.0:
            scale = 0.9
        else:
            scale = 1.0
        self.bgfg_hard_neg_weight = float(np.clip(
            self._base_bgfg_hard_neg_weight * scale, 0.0, self._base_bgfg_hard_neg_weight * 2.5
        ))

    def _prepare_fg_logits(self, fg_class_logits: torch.Tensor) -> torch.Tensor:
        logits = fg_class_logits
        if self.fg_loss_type == 'logit_adjust' and self.fg_logit_adjust is not None and self.logit_adjust_tau > 0:
            bias = self.logit_adjust_tau * self.fg_logit_adjust.to(logits.device).view(1, -1, 1, 1)
            logits = logits + bias
        return logits

    def decompose(self, outputs, targets: torch.Tensor):
        if not isinstance(outputs, (dict, tuple, list)):
            raise TypeError("HierarchicalSegLoss expects model outputs with two-stage heads")

        if isinstance(outputs, dict):
            fg_bg_logits = outputs.get('fg_bg_logits')
            fg_class_logits = outputs.get('fg_class_logits')
            boundary_logits = outputs.get('boundary_logits')
            aux_fg_bg_logits = outputs.get('aux_fg_bg_logits')
            aux_fg_class_logits = outputs.get('aux_fg_class_logits')
        else:
            if len(outputs) < 3:
                raise ValueError("HierarchicalSegLoss expects (fused_logits, fg_bg_logits, fg_class_logits)")
            fg_bg_logits = outputs[1]
            fg_class_logits = outputs[2]
            boundary_logits = None
            aux_fg_bg_logits = None
            aux_fg_class_logits = None

        if fg_bg_logits is None or fg_class_logits is None:
            raise ValueError("Missing fg_bg_logits/fg_class_logits for direct two-stage supervision")

        # Foreground subclass loss on classes [1..C-1] with direct fg_class_logits.
        fg_targets = targets.clone()
        valid_fg = (fg_targets != self.ignore_index) & (fg_targets > 0)
        fg_targets[valid_fg] = fg_targets[valid_fg] - 1
        fg_targets[(fg_targets == 0) & (~valid_fg)] = self.ignore_index
        fg_class_logits_adj = self._prepare_fg_logits(fg_class_logits)
        stage = str(getattr(self, 'active_stage', 'C')).upper()
        fg_loss = self.multi_cls_loss(fg_class_logits_adj, fg_targets)

        if stage == 'B' and self.stage_b_weak_cls_focus > 0.0:
            valid_fg_mask = (fg_targets != self.ignore_index)
            if valid_fg_mask.any():
                ce_fg_map = F.cross_entropy(
                    fg_class_logits_adj,
                    fg_targets,
                    ignore_index=self.ignore_index,
                    reduction='none',
                )
                boost = self._fg_class_dynamic_boost.to(ce_fg_map.device, dtype=ce_fg_map.dtype)
                weight_map = torch.ones_like(ce_fg_map)
                for c in range(min(boost.numel(), max(1, self.num_classes - 1))):
                    bc = float(boost[c].item())
                    if bc > 1.0 + 1e-6:
                        weight_map = torch.where(fg_targets == c, weight_map * bc, weight_map)
                weak_ce = (ce_fg_map[valid_fg_mask] * weight_map[valid_fg_mask]).mean()
                alpha = self.stage_b_weak_cls_focus
                fg_loss = (1.0 - alpha) * fg_loss + alpha * weak_ce

        # Binary BG/FG loss with Dice + CE and hard-negative emphasis.
        bin_targets = targets.clone()
        valid = (bin_targets != self.ignore_index)
        bin_targets[valid] = (bin_targets[valid] > 0).long()

        ce_map = F.cross_entropy(fg_bg_logits, bin_targets, ignore_index=self.ignore_index, reduction='none')
        fg_prob = F.softmax(fg_bg_logits, dim=1)[:, 1, :, :]
        hard_neg = ((bin_targets == 0) & valid & (fg_prob > 0.5)).float()
        pos_mask = ((bin_targets == 1) & valid).float()
        boundary_target_bgfg = self._build_boundary_target(targets, self.ignore_index)
        ce_weight = (
            1.0
            + self.bgfg_hard_neg_weight * hard_neg
            + (self.bgfg_pos_weight - 1.0) * pos_mask
            + self.bgfg_boundary_weight * boundary_target_bgfg * valid.float()
        )
        ce_valid = ce_map[valid] * ce_weight[valid]
        ce_loss = ce_valid.mean() if ce_valid.numel() > 0 else ce_map.sum() * 0.0

        fg_gt = (bin_targets == 1).float()
        valid_f = valid.float()
        dice_loss = self._soft_dice_binary(fg_prob * valid_f, fg_gt * valid_f)
        bin_loss = 0.5 * ce_loss + 0.5 * dice_loss

        # Foreground class Dice to stabilize minority classes.
        fg_dice_loss = fg_loss * 0.0
        valid_fg_mask = (fg_targets != self.ignore_index)
        if valid_fg_mask.any():
            probs = F.softmax(fg_class_logits_adj, dim=1)
            fg_dice_terms = []
            for c in range(self.num_classes - 1):
                cls_t = (fg_targets == c).float()
                if cls_t.sum() <= 0:
                    continue
                cls_p = probs[:, c, :, :]
                cls_valid = valid_fg_mask.float()
                inter = (cls_p * cls_t * cls_valid).sum()
                denom = (cls_p * cls_valid).sum() + (cls_t * cls_valid).sum()
                fg_dice_terms.append(1.0 - (2.0 * inter + self._eps) / (denom + self._eps))
            if fg_dice_terms:
                fg_dice_loss = torch.stack(fg_dice_terms).mean()

        fg_loss = 0.7 * fg_loss + 0.3 * fg_dice_loss

        boundary_loss = fg_loss * 0.0
        if boundary_logits is not None:
            boundary_target = boundary_target_bgfg
            boundary_valid = ((targets != self.ignore_index) & (targets > 0)).float()
            bce = F.binary_cross_entropy_with_logits(
                boundary_logits.squeeze(1), boundary_target, reduction='none')
            denom = boundary_valid.sum().clamp(min=1.0)
            boundary_loss = (bce * boundary_valid).sum() / denom

        aux_bg_loss = fg_loss * 0.0
        aux_fg_loss = fg_loss * 0.0
        if aux_fg_bg_logits is not None and aux_fg_class_logits is not None:
            if aux_fg_bg_logits.shape[-2:] != bin_targets.shape[-2:]:
                aux_fg_bg_logits = F.interpolate(
                    aux_fg_bg_logits,
                    size=bin_targets.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            if aux_fg_class_logits.shape[-2:] != fg_targets.shape[-2:]:
                aux_fg_class_logits = F.interpolate(
                    aux_fg_class_logits,
                    size=fg_targets.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            aux_ce_map = F.cross_entropy(aux_fg_bg_logits, bin_targets, ignore_index=self.ignore_index, reduction='none')
            aux_bg_loss = aux_ce_map[valid].mean() if valid.any() else aux_ce_map.sum() * 0.0
            aux_fg_loss = self.multi_cls_loss(self._prepare_fg_logits(aux_fg_class_logits), fg_targets)

        bg_suppress_loss = fg_loss * 0.0
        if self.bg_suppress_weight > 0.0:
            bg_valid = (targets == 0) & valid
            if bg_valid.any():
                bg_prob_max = F.softmax(fg_class_logits_adj, dim=1).max(dim=1).values
                suppress = F.relu(bg_prob_max - self.bg_suppress_threshold)
                bg_suppress_loss = (suppress[bg_valid] ** 2).mean()

        boundary_weight = self.boundary_weight
        if stage == 'C':
            boundary_weight = boundary_weight * float(self._boundary_dynamic_scale)

        if stage == 'A':
            total = self.bgfg_weight * bin_loss + self.aux_weight * aux_bg_loss
        elif stage == 'B':
            total = (
                self.fg_weight * fg_loss
                + self.boundary_weight * boundary_loss
                + self.aux_weight * aux_fg_loss
                + self.bg_suppress_weight * bg_suppress_loss
                + self.stage_b_bgfg_anchor_weight * bin_loss
            )
        else:
            aux_loss = 0.5 * aux_bg_loss + 0.5 * aux_fg_loss
            total = (
                self.fg_weight * fg_loss
                + self.bgfg_weight * bin_loss
                + boundary_weight * boundary_loss
                + self.aux_weight * aux_loss
                + self.bg_suppress_weight * bg_suppress_loss
            )
        return total, bin_loss, fg_loss

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        total, _, _ = self.decompose(outputs, targets)
        return total


def _worker_plot_training_curves(train_losses, eval_losses,
                                 train_bgfg_losses, train_fg_class_losses,
                                 eval_bgfg_losses, eval_fg_class_losses,
                                 train_accs, eval_accs,
                                 stage_history,
                                 eval_interval, debug_mode,
                                 best_acc, model_name, output):
    """Worker: training loss/accuracy curves."""
    matplotlib.use('Agg')

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    total_epochs = len(train_losses)
    eval_epochs = []
    for e in range(total_epochs):
        should = ((e + 1) % eval_interval == 0) or (e + 1 == total_epochs)
        if should or debug_mode:
            eval_epochs.append(e)
    eval_epochs = eval_epochs[:len(eval_losses)]

    stage_colors = {
        'A': '#d8ecff',
        'B': '#ffe8cc',
        'C': '#e6f7e6',
    }
    if stage_history:
        start = 0
        curr = stage_history[0]
        for i in range(1, len(stage_history) + 1):
            boundary = (i == len(stage_history)) or (stage_history[i] != curr)
            if boundary:
                for ax in axes:
                    ax.axvspan(start, i - 1, color=stage_colors.get(curr, '#f2f2f2'), alpha=0.25)
                start = i
                if i < len(stage_history):
                    curr = stage_history[i]

    axes[0].plot(train_bgfg_losses, '--', color='#1f77b4', linewidth=1.8,
                 label='Train bgfg_loss')
    axes[0].plot(train_fg_class_losses, '--', color='#ff7f0e', linewidth=1.8,
                 label='Train fg_class_loss')
    if eval_bgfg_losses:
        axes[0].plot(eval_epochs, eval_bgfg_losses, '--', color='#2ca02c', linewidth=1.8,
                     label='Eval bgfg_loss')
    if eval_fg_class_losses:
        axes[0].plot(eval_epochs, eval_fg_class_losses, '--', color='#d62728', linewidth=1.8,
                     label='Eval fg_class_loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Sub-loss Curves'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_accs, label='Train Acc', marker='o', markersize=4)
    if eval_accs:
        axes[1].plot(eval_epochs, eval_accs, label='Eval Acc', marker='s', markersize=4)
    axes[1].axhline(y=best_acc, color='r', linestyle='--', label=f'Best: {best_acc:.2f}%')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy Curve'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output, 'training_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _worker_plot_stage_timeline(stage_history, transition_log,
                                model_name, output):
    """Worker: stage timeline for strict A/B/C training."""
    matplotlib.use('Agg')
    if not stage_history:
        return

    stage_to_id = {'A': 0, 'B': 1, 'C': 2}
    y = np.array([stage_to_id.get(str(s).upper(), 2) for s in stage_history], dtype=np.int64)
    x = np.arange(1, len(stage_history) + 1)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.step(x, y, where='post', linewidth=2.0, color='#2f5d8a')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['A: BG/FG', 'B: FG-subclass', 'C: Joint'])
    ax.set_xlabel('Epoch')
    ax.set_title(f'{model_name} - Training Stage Timeline', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)

    for entry in (transition_log or []):
        e = int(entry.get('epoch', 0))
        if e > 0:
            ax.axvline(e, linestyle='--', color='#c0392b', alpha=0.7, linewidth=1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output, 'stage_timeline.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _worker_write_stage_transition_summary(stage_transition_log, output):
    """Worker: write stage transition diagnostics for quick debugging."""
    path = os.path.join(output, 'stage_transition_summary.txt')
    lines = []
    for idx, item in enumerate(stage_transition_log or [], start=1):
        lines.append(
            f"{idx}. epoch={int(item.get('epoch', -1))}, "
            f"{item.get('from', '?')}->{item.get('to', '?')}, "
            f"metric={float(item.get('metric', 0.0)):.2f}, "
            f"reason={item.get('reason', 'n/a')}"
        )
    if not lines:
        lines = ['No stage transition recorded.']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def _worker_plot_confusion_matrix(cm, cm_norm, metrics_text,
                                  class_names, model_name, output,
                                  file_stem='confusion_matrix',
                                  title_prefix='Confusion Matrix',
                                  metrics_filename='classification_metrics.txt'):
    """worker: plot pre-computed confusion matrix.
    cm: shape (num_classes, num_classes) int — raw counts.
    cm_norm: shape (num_classes, num_classes) float — row-normalized.
    metrics_text: str — pre-formatted classification report to save.
    """
    matplotlib.use('Agg')

    with open(os.path.join(output, metrics_filename), 'w') as f:
        f.write(metrics_text)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title(f'{title_prefix} (Count)'); axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Greens', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title(f'{title_prefix} (Normalized)'); axes[1].set_ylabel('True'); axes[1].set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(output, f'{file_stem}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _worker_plot_per_class_metrics(p, r, f1, num_classes, class_names,
                                   model_name, output):
    """worker: plot pre-computed per-class precision/recall/f1.
    p, r, f1: each shape (num_classes,) float arrays.
    """
    matplotlib.use('Agg')

    x = np.arange(num_classes)
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, p, w, label='Precision', color='#2196F3', alpha=0.85)
    ax.bar(x, r, w, label='Recall', color='#FF5722', alpha=0.85)
    ax.bar(x + w, f1, w, label='F1-Score', color='#4CAF50', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names[:num_classes], rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
    ax.set_title(f'{model_name} — Per-Class Metrics', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _worker_plot_roc_curves(roc_data, num_classes, class_names,
                            model_name, output):
    """worker: plot pre-computed roc curves.
    roc_data: list of (fpr, tpr, auc_val) tuples, one per class.
              each fpr/tpr is a 1-d array (variable length).
    """
    matplotlib.use('Agg')

    if roc_data is None:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    for cls_i in range(num_classes):
        entry = roc_data[cls_i]
        if entry is None:
            continue   # class absent from sampled ROC set
        fpr, tpr, roc_auc = entry
        ax.plot(fpr, tpr, label=f'{class_names[cls_i]} (AUC={roc_auc:.3f})', linewidth=1.3)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{model_name} — Per-Class ROC Curves', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _worker_plot_grad_norm(grad_norms, model_name, output):
    """Worker: gradient L2 norm."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not grad_norms:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grad_norms, marker='o', markersize=3, linewidth=1.5, color='#FF5722')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Gradient L2 Norm')
    ax.set_title(f'{model_name} — Gradient Norm', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'gradient_norm.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _worker_plot_lr(lr_history, model_name, output):
    """Worker: LR schedule."""
    matplotlib.use('Agg')

    if not lr_history:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lr_history, linewidth=1.5, color='#2196F3')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate'); ax.set_yscale('log')
    ax.set_title(f'{model_name} — Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'lr_schedule.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _worker_plot_epoch_time(epoch_times, model_name, output):
    """Worker: epoch wall-clock time."""
    matplotlib.use('Agg')

    if not epoch_times:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(epoch_times)), epoch_times, color='#4CAF50', alpha=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Time (s)')
    ax.set_title(f'{model_name} — Per-Epoch Wall-Clock Time', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'epoch_times.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _worker_plot_fg_threshold_curve(scan_records, model_name, output):
    """Worker: plot validation FG-threshold calibration curves."""
    matplotlib.use('Agg')

    if not scan_records:
        return
    latest = scan_records[-1]
    scan = latest.get('scan', [])
    if not scan:
        return

    thr = np.asarray([x['thr'] for x in scan], dtype=np.float32)
    ba = np.asarray([x['ba_bgfg'] for x in scan], dtype=np.float32)
    dice = np.asarray([x['fg_dice'] for x in scan], dtype=np.float32)
    gap = np.asarray([abs(x['pred_fg_ratio'] - x['gt_fg_ratio']) for x in scan], dtype=np.float32)

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(thr, ba, color='#1f77b4', linewidth=1.8, label='BA(BG/FG)')
    ax1.plot(thr, dice, color='#ff7f0e', linewidth=1.8, label='FG Dice')
    ax1.set_xlabel('FG Threshold')
    ax1.set_ylabel('Score (%)')
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(thr, gap, color='#2ca02c', linestyle='--', linewidth=1.5, label='|PredFG-GtFG|')
    ax2.set_ylabel('FG Ratio Gap (%)')

    ax1.set_title(f"{model_name} — FG Threshold Scan (epoch {latest.get('epoch', '?')})")
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'bgfg_threshold_scan.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _worker_plot_fg_ratio_trajectory(pred_fg_hist, gt_fg_hist, model_name, output):
    """Worker: plot FG predicted/GT ratio trajectory across epochs."""
    matplotlib.use('Agg')
    if not pred_fg_hist or not gt_fg_hist:
        return

    n = min(len(pred_fg_hist), len(gt_fg_hist))
    x = np.arange(1, n + 1)
    pred = np.asarray(pred_fg_hist[:n], dtype=np.float32)
    gt = np.asarray(gt_fg_hist[:n], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, pred, marker='o', markersize=3, linewidth=1.5, label='Pred FG ratio (%)')
    ax.plot(x, gt, marker='s', markersize=3, linewidth=1.5, label='GT FG ratio (%)')
    ax.plot(x, pred - gt, linestyle='--', linewidth=1.2, label='Pred-GT gap (%)')
    ax.set_xlabel('Eval Epoch')
    ax.set_ylabel('Ratio / Gap (%)')
    ax.set_title(f'{model_name} — FG Ratio Trajectory', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'fg_ratio_trajectory.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _worker_write_pairwise_fg_metrics(pairwise_text: str, output: str):
    """Worker: write FG pairwise recall diagnostics to text file."""
    if not pairwise_text:
        return
    with open(os.path.join(output, 'classification_metrics_fg_pairwise.txt'), 'w') as f:
        f.write(pairwise_text)


class ModelEMA:
    """
    Exponential Moving Average of model weights for better generalization.
    
    Maintains a shadow copy of parameters updated as:
        shadow = decay * shadow + (1 - decay) * current_params
    
    Near-zero training overhead (one extra parameter copy + in-place lerp per step).
    At evaluation time, swap in shadow weights for improved test accuracy.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        # Deep copy the model parameters (detached, no grad)
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow weights with current model weights."""
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.lerp_(model_p.data, 1.0 - self.decay)
        # Also update buffers (e.g., BatchNorm running stats)
        for ema_b, model_b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)
    
    def state_dict(self):
        return self.shadow.state_dict()
    
    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)

    @torch.no_grad()
    def swap(self, model: nn.Module) -> None:
        """Zero-copy in-place swap of model parameters/buffers with EMA shadow.

        Exchanges the underlying storage pointers of every parameter and buffer
        between ``model`` and the shadow copy — no tensor allocation, no PCIe
        transfer.  Calling ``swap()`` a second time restores the original state,
        so no backup is needed.

        Handles ``nn.DataParallel`` by unwrapping to the inner module first.
        """
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        for p_m, p_s in zip(raw_model.parameters(), self.shadow.parameters()):
            # Swap storage pointers — no data copied, O(num_params) pointer assigns
            tmp        = p_m.data
            p_m.data   = p_s.data
            p_s.data   = tmp
        for b_m, b_s in zip(raw_model.buffers(), self.shadow.buffers()):
            tmp        = b_m.data
            b_m.data   = b_s.data
            b_s.data   = tmp


class _CudaBatchPrefetcher:
    """Prefetch DataLoader batches to device on a dedicated CUDA stream."""

    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.iter_loader = iter(loader)
        self.next_batch = None
        self._preload()

    def _to_device(self, obj: Any):
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, (tuple, list)):
            moved = [self._to_device(x) for x in obj]
            return type(obj)(moved)
        if isinstance(obj, dict):
            return {k: self._to_device(v) for k, v in obj.items()}
        return obj

    def _preload(self):
        try:
            batch = next(self.iter_loader)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = self._to_device(batch)

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration

        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch

class hsTrainer(BaseEstimator):
    """
    Sklearn-compatible trainer for hyperspectral classification.

    Follows sklearn API convention::

        trainer = hsTrainer(config=config, dataLoader=dataset, ...)
        trainer.fit()                    # train model, returns self
        predictions = trainer.predict()  # predict on test set
        accuracy = trainer.score()       # evaluate on test set
    """
    
    def __init__(
        self,
        config: Munch = None,
        dataLoader: AbstractHSDataset = None,
        epochs: int = 10,
        model: Callable[..., Module] = None,
        model_name: str = "model",
        debug_mode: bool = False,
        num_gpus: int = 1,
        train_loader=None,
        val_loader=None,   # Validation set for early stopping (not seen during final test)
        test_loader=None,  # Test set held out completely, evaluated only once at the end
        **kwargs
    ):
        
        self.config = config
        self.dataLoader = dataLoader
        self.epochs = epochs
        self.model = model
        self.model_name = model_name
        self.debug_mode = debug_mode
        self.num_gpus = num_gpus
        self.output = self.config.path.output + \
                        f'/{self.model_name}' + \
                        f'{datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}'
        self.kwargs = kwargs
        
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(os.path.join(self.output, 'CAM'), exist_ok=True)
        os.makedirs(os.path.join(self.output, 'models'), exist_ok=True)
        
        # training workflow
        self._setup_device()
        if train_loader is not None and test_loader is not None:
            self.train_loader = train_loader
            self.val_loader = val_loader  # may be None if legacy 2-loader mode
            self.test_loader = test_loader
            tprint(f"  Using injected data loaders "
                  f"(train: {len(train_loader)} batches, "
                  f"val: {len(val_loader) if val_loader else 'N/A'} batches, "
                  f"test: {len(test_loader)} batches)")
        else:
            self._load_data()
        self.stage_a_max_epochs = int(max(0, getattr(self.config.common, 'stage_a_max_epochs', 0)))
        self.stage_b_max_epochs = int(max(0, getattr(self.config.common, 'stage_b_max_epochs', 0)))
        self._create_model()
        self._setup_multi_gpu()
        self._setup_training()
        
        # training state
        self.train_losses = []
        self.train_accs = []
        self.eval_accs = []
        self.eval_losses = []
        self.train_bgfg_losses = []
        self.train_fg_class_losses = []
        self.eval_bgfg_losses = []
        self.eval_fg_class_losses = []
        self.epoch_metrics = []
        self.best_acc = 0.0
        self.best_epoch = 0
        self.best_model_state = None
        
        # tracking for visualizations
        self.lr_history = []          # lr per epoch
        self.grad_norms = []          # gradient L2 norm per epoch
        self.epoch_times = []         # wall-clock time per epoch
        self._eval_call_counter = 0
        self.stage_history = []
        self.stage_transition_log = []

        tprint(f"Trainer Initialized successfully with:")
        print(f"  model: {self.model_name}")
        print(f"  epoch: {self.epochs}")
        print(f"  device: {self.device}")
        print(f"  num_gpus: {self.num_gpus}")
        print(f"  output_dir: {self.output}")

    # public interface
    def summary(self) -> None:
        """Print a short summary of loaders/model for quick inspection."""
        train_batches = len(self.train_loader) if hasattr(self, 'train_loader') else 0
        test_batches = len(self.test_loader) if hasattr(self, 'test_loader') else 0
        print("[Trainer Summary]")
        print(f"  model: {self.model_name}, epochs: {self.epochs}, device: {self.device}")
        print(f"  loaders: train={train_batches} batches, test={test_batches} batches")
        if hasattr(self.dataLoader, 'describe'):
            self.dataLoader.describe(top_k=3)

    def fit(self, X=None, y=None, **fit_params):
        """
        Train the model. Follows sklearn ``fit`` convention.

        Args:
            X : ignored (data comes from internal DataLoaders).
            y : ignored.
            **fit_params:
                _cv_mode (bool): skip per-fold visualizations in CV.

        Returns:
            self
        """
        _cv_mode = fit_params.pop('_cv_mode', False)
        self.results_ = self.train(_cv_mode=_cv_mode)
        return self

    def predict(self, X=None):
        """
        Predict class labels using the trained model.

        Args:
            X : DataLoader to predict on, or None (uses test_loader).

        Returns:
            np.ndarray of predicted class labels.
        """
        _, _, _, predictions, _ = self.evaluate(collect_extra=True, loader=X)
        return predictions

    def score(self, X=None, y=None):
        """
        Evaluate and return balanced accuracy.

        Args:
            X : DataLoader to evaluate on, or None (uses val/test loader).
            y : ignored (labels come from DataLoader).

        Returns:
            float: balanced accuracy percentage.
        """
        _, acc, _, _, _ = self.evaluate(loader=X)
        return acc

    def fit_cv(self, n_folds=None, **kwargs):
        """
        Cross-validation fit. Follows sklearn convention, returns ``self``.

        Args:
            n_folds : int or None (defaults to config.common.cv_folds).

        Returns:
            self
        """
        if n_folds is None:
            n_folds = self.config.common.cv_folds
        self.cv_results_ = hsTrainer.cross_validate(
            self.config, self.dataLoader,
            n_folds, self.epochs, self.model,
            model_name=self.model_name,
            num_gpus=self.num_gpus,
            debug_mode=self.debug_mode)
        return self

    @staticmethod
    def _metric_labels_from_config(config: Munch, num_classes: int):
        """Return class indices used for metrics (optionally exclude BG=0)."""
        exclude_bg = bool(getattr(config.common, 'exclude_bg_in_metrics', True))
        if exclude_bg and num_classes > 1:
            labels = list(range(1, num_classes))
        else:
            labels = list(range(num_classes))
        return labels if labels else [0]

    def _metric_labels(self, num_classes: int):
        return hsTrainer._metric_labels_from_config(self.config, num_classes)

    def _select_eval_batch_indices(self, total_batches: int, eval_cap: int) -> Optional[List[int]]:
        """Random stratified selection of validation batches when capped.

        Splits [0, total_batches) into ``eval_cap`` contiguous strata and samples
        one index from each stratum to avoid front-slice bias.
        """
        if eval_cap <= 0 or total_batches <= eval_cap:
            return None

        base_seed = int(getattr(self.config.common, 'eval_sample_seed', 350234))
        call_id = int(getattr(self, '_eval_call_counter', 0))
        self._eval_call_counter = call_id + 1
        rng = np.random.RandomState(base_seed + call_id)

        all_idx = np.arange(total_batches, dtype=np.int64)
        strata = np.array_split(all_idx, eval_cap)
        picked = []
        for s in strata:
            if s.size == 0:
                continue
            sel = int(s[rng.randint(0, s.size)])
            picked.append(sel)

        picked = sorted(set(picked))
        return picked if picked else None

    def _build_capped_eval_loader(self, eval_loader, selected_eval_batches: Optional[List[int]]):
        """Build a compact DataLoader that contains only selected eval batches.

        This prevents iterating through all original validation batches just to
        skip most of them, which can dominate epoch time when val loaders are
        very large.
        """
        if not selected_eval_batches:
            return eval_loader

        dataset = getattr(eval_loader, 'dataset', None)
        batch_size = int(getattr(eval_loader, 'batch_size', 0) or 0)
        if dataset is None or batch_size <= 0:
            return eval_loader

        total_items = len(dataset)
        subset_indices: List[int] = []
        for b_idx in selected_eval_batches:
            start = int(b_idx) * batch_size
            if start >= total_items:
                continue
            end = min(start + batch_size, total_items)
            subset_indices.extend(range(start, end))

        if not subset_indices:
            return eval_loader

        subset = torch.utils.data.Subset(dataset, subset_indices)
        num_workers = int(getattr(eval_loader, 'num_workers', 0) or 0)
        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(getattr(eval_loader, 'pin_memory', False)),
            drop_last=False,
            collate_fn=getattr(eval_loader, 'collate_fn', None),
            timeout=int(getattr(eval_loader, 'timeout', 0) or 0),
            worker_init_fn=getattr(eval_loader, 'worker_init_fn', None),
            multiprocessing_context=getattr(eval_loader, 'multiprocessing_context', None),
            generator=getattr(eval_loader, 'generator', None),
            persistent_workers=bool(getattr(eval_loader, 'persistent_workers', False)) and (num_workers > 0),
        )

        prefetch_factor = getattr(eval_loader, 'prefetch_factor', None)
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = int(prefetch_factor)

        pin_memory_device = str(getattr(eval_loader, 'pin_memory_device', '') or '')
        if pin_memory_device:
            loader_kwargs['pin_memory_device'] = pin_memory_device

        return torch.utils.data.DataLoader(subset, **loader_kwargs)

    @staticmethod
    def _metrics_from_confusion(cm_np: np.ndarray, labels: list) -> Tuple[float, float, float]:
        """Compute BA/Kappa/mIoU from confusion matrix over selected labels."""
        row_sums = cm_np.sum(axis=1)
        col_sums = cm_np.sum(axis=0)
        diag = np.diag(cm_np)

        valid_cls = row_sums > 0
        eval_valid = np.zeros_like(valid_cls, dtype=bool)
        eval_valid[np.array(labels, dtype=np.int64)] = True
        eval_valid = eval_valid & valid_cls

        per_cls_recall = np.where(valid_cls, diag / np.maximum(row_sums, 1), 0.0)
        ba = float(per_cls_recall[eval_valid].mean()) * 100 if eval_valid.any() else 0.0

        labels_arr = np.array(labels, dtype=np.int64)
        num_all = cm_np.shape[0]
        remap = np.full(num_all, fill_value=len(labels_arr), dtype=np.int64)
        remap[labels_arr] = np.arange(len(labels_arr), dtype=np.int64)

        cm_kappa = np.zeros((len(labels_arr) + 1, len(labels_arr) + 1), dtype=np.int64)
        for i in range(num_all):
            ii = remap[i]
            row = cm_np[i]
            for j in range(num_all):
                jj = remap[j]
                cm_kappa[ii, jj] += int(row[j])

        row_k = cm_kappa.sum(axis=1)
        active = row_k > 0
        # Kappa is undefined with <2 active classes; keep 0.0 for stability.
        if int(active.sum()) < 2:
            kappa = 0.0
        else:
            total_k = int(cm_kappa.sum())
            expected = (
                row_k.astype(np.float64) * cm_kappa.sum(axis=0).astype(np.float64)
            ) / max(total_k, 1)
            p_o = float(np.diag(cm_kappa).sum()) / max(total_k, 1)
            p_e = float(expected.sum()) / max(total_k, 1)
            kappa = ((p_o - p_e) / max(1.0 - p_e, 1e-8)) * 100

        miou_vals = []
        for c in labels:
            tp = int(cm_np[c, c])
            fp = int(col_sums[c]) - tp
            fn = int(row_sums[c]) - tp
            denom = tp + fp + fn
            if denom > 0:
                miou_vals.append(tp / denom)
        miou = float(np.mean(miou_vals)) * 100 if miou_vals else 0.0
        return ba, kappa, miou

    def _select_early_stop_score(self) -> float:
        """Choose scalar score for early stopping according to config."""
        metric = getattr(self, 'early_stop_metric', 'composite')
        if metric == 'hybrid':
            ba_bgfg = float(getattr(self, '_last_ba_bgfg', 0.0))
            ba_fg = float(getattr(self, '_last_ba_fg', 0.0))
            fg_min_recall = float(getattr(self, '_last_fg_min_recall', 0.0))
            score = (
                self.metric_weight_bgfg * ba_bgfg
                + self.metric_weight_fg * ba_fg
                + self.metric_weight_fg_min_recall * fg_min_recall
            )
            if ba_bgfg < self.hybrid_min_ba_bgfg:
                score -= self.hybrid_low_metric_penalty
            if fg_min_recall < self.hybrid_min_fg_recall:
                score -= self.hybrid_low_metric_penalty
            return float(score)
        if metric == 'fg':
            return float(getattr(self, '_last_ba_fg', 0.0))
        if metric == 'all':
            return float(getattr(self, '_last_ba_all', 0.0))
        fg = float(getattr(self, '_last_ba_fg', 0.0))
        all_ = float(getattr(self, '_last_ba_all', 0.0))
        return self.metric_weight_fg * fg + self.metric_weight_all * all_

    def _current_stage_metric(self, stage: Optional[str] = None) -> float:
        stage = str(stage or getattr(self, 'current_stage', 'C')).upper()
        if stage == 'A':
            ba_bgfg = float(getattr(self, '_last_ba_bgfg', 0.0))
            ba_boundary = float(getattr(self, '_last_ba_bgfg_boundary', ba_bgfg))
            w = float(np.clip(getattr(self, 'stage_a_boundary_metric_weight', 0.0), 0.0, 1.0))
            return (1.0 - w) * ba_bgfg + w * ba_boundary
        if stage == 'B':
            ba_fg = float(getattr(self, '_last_ba_fg', 0.0))
            fg_min_recall = float(getattr(self, '_last_fg_min_recall', ba_fg))
            w = float(np.clip(getattr(self, 'stage_b_min_recall_metric_weight', 0.0), 0.0, 1.0))
            return (1.0 - w) * ba_fg + w * fg_min_recall
        return float(self._select_early_stop_score())

    def _record_stage_metric(self, metric: float) -> None:
        m = float(np.clip(getattr(self, 'stage_metric_ema_alpha', 0.6), 0.0, 0.99))
        if self._stage_metric_ema is None:
            self._stage_metric_ema = float(metric)
        else:
            self._stage_metric_ema = m * float(self._stage_metric_ema) + (1.0 - m) * float(metric)

        metric_ref = float(self._stage_metric_ema)
        if metric_ref > self._stage_best_metric + 1e-6:
            self._stage_best_metric = metric_ref
            self._stage_bad_epochs = 0
        else:
            self._stage_bad_epochs += 1

    def _transition_stage(self, next_stage: str, epoch: int, metric: float, reason: str) -> None:
        prev_stage = str(self.current_stage)
        if prev_stage == next_stage:
            return
        self.stage_transition_log.append({
            'epoch': int(epoch + 1),
            'from': prev_stage,
            'to': str(next_stage),
            'metric': float(metric),
            'reason': str(reason),
        })
        self.current_stage = str(next_stage)
        self.stage_start_epoch = int(epoch + 1)
        self._stage_best_metric = -1e18
        self._stage_bad_epochs = 0
        self._stage_metric_ema = None

        if prev_stage == 'A' and str(next_stage) == 'B':
            self._save_bgfg_split_snapshot(epoch=epoch, metric=metric, reason=reason)

    def _maybe_advance_stage(self, epoch: int) -> None:
        if self.stage_override in {'A', 'B', 'C'}:
            self.current_stage = self.stage_override
            return

        stage = str(self.current_stage)
        metric = self._current_stage_metric(stage)
        self._record_stage_metric(metric)
        elapsed = int(epoch + 1 - self.stage_start_epoch)

        if stage == 'A':
            a_boundary = float(getattr(self, '_last_ba_bgfg_boundary', 0.0))
            cond_min_epoch = elapsed >= self.stage_a_min_epochs
            cond_target = (
                metric >= self.stage_a_target_ba_bgfg
                and a_boundary >= self.stage_a_target_ba_bgfg_boundary
            )
            cond_plateau = (
                self._stage_bad_epochs >= self.stage_transition_patience
                and a_boundary >= self.stage_a_min_boundary_ba_for_transition
            )
            if cond_min_epoch and (cond_target or cond_plateau):
                reason = 'reach_target' if cond_target else 'plateau'
                self._transition_stage('B', epoch, metric, reason)
                return

            if self.stage_a_max_epochs > 0 and elapsed >= self.stage_a_max_epochs:
                self._transition_stage('B', epoch, metric, 'max_stage_a_epochs')
                return

        if stage == 'B':
            fg_min_recall = float(getattr(self, '_last_fg_min_recall', 0.0))
            cond_min_epoch = elapsed >= self.stage_b_min_epochs
            cond_target = (
                metric >= self.stage_b_target_ba_fg
                and fg_min_recall >= self.stage_b_target_fg_min_recall
            )
            cond_plateau = (
                self._stage_bad_epochs >= self.stage_transition_patience
                and fg_min_recall >= self.stage_b_min_fg_recall_for_c
            )
            if cond_min_epoch and (cond_target or cond_plateau):
                reason = 'reach_target' if cond_target else 'plateau'
                self._transition_stage('C', epoch, metric, reason)
                return

            remaining_epochs_b = int(self.epochs - (epoch + 1))
            if (
                self.stage_b_max_epochs > 0
                and elapsed >= self.stage_b_max_epochs
                and remaining_epochs_b >= self.stage_c_min_epochs
            ):
                self._transition_stage('C', epoch, metric, 'max_stage_b_epochs')
                return

        remaining_epochs = int(self.epochs - (epoch + 1))
        if stage == 'A':
            # Keep strict A->B->C semantics: never jump from A directly to C.
            if remaining_epochs <= (self.stage_b_min_epochs + self.stage_c_min_epochs):
                self._transition_stage('B', epoch, metric, 'force_stage_b_before_joint')
                return

        if stage == 'B':
            if remaining_epochs <= self.stage_c_min_epochs:
                fg_min_recall = float(getattr(self, '_last_fg_min_recall', 0.0))
                if (fg_min_recall >= self.stage_b_force_c_min_fg_recall) or (remaining_epochs <= 1):
                    self._transition_stage('C', epoch, metric, 'force_joint_finetune')

    def _parse_model_outputs(self, outputs):
        """Parse model outputs and return (fused_logits, fg_bg_logits, fg_class_logits)."""
        if isinstance(outputs, dict):
            fused = outputs.get('fused_logits')
            fg_bg = outputs.get('fg_bg_logits')
            fg_cls = outputs.get('fg_class_logits')
        elif isinstance(outputs, (tuple, list)):
            if len(outputs) < 3:
                raise ValueError("Model must return (fused_logits, fg_bg_logits, fg_class_logits)")
            fused, fg_bg, fg_cls = outputs[0], outputs[1], outputs[2]
        else:
            raise TypeError("Model output must be tuple/list/dict with two-stage heads")

        if fused is None or fg_bg is None or fg_cls is None:
            raise ValueError("Model output missing required two-stage logits")
        return fused, fg_bg, fg_cls

    @staticmethod
    def _decode_two_stage_predictions(fg_bg_logits: torch.Tensor,
                                      fg_class_logits: torch.Tensor,
                                      fg_threshold: float = -1.0,
                                      fg_class_min_conf: float = -1.0,
                                      fg_class_logit_bias: Optional[torch.Tensor] = None,
                                      fg_class_min_conf_per_class: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode final class labels via two-step decision.

        Step 1: FG/BG from ``fg_bg_logits`` (argmax over 2 channels).
        Step 2: For FG pixels only, argmax over ``fg_class_logits`` and shift by +1.
        """
        fg_bg_logits = torch.nan_to_num(fg_bg_logits, nan=0.0, posinf=50.0, neginf=-50.0)
        fg_class_logits = torch.nan_to_num(fg_class_logits, nan=0.0, posinf=50.0, neginf=-50.0)

        decode_fg_logits = fg_class_logits.float()
        if fg_class_logit_bias is not None:
            bias = fg_class_logit_bias.to(decode_fg_logits.device, dtype=decode_fg_logits.dtype).view(1, -1, 1, 1)
            if bias.shape[1] == decode_fg_logits.shape[1]:
                decode_fg_logits = decode_fg_logits + bias

        fg_bg_pred = torch.argmax(fg_bg_logits, dim=1)
        fg_cls_pred = torch.argmax(decode_fg_logits, dim=1) + 1
        fg_cls_probs = F.softmax(decode_fg_logits, dim=1)
        fg_cls_conf = fg_cls_probs.max(dim=1).values

        if float(fg_threshold) >= 0.0:
            # Use FP32 softmax for stable thresholding under AMP/FP16.
            p_bgfg = F.softmax(fg_bg_logits.float(), dim=1)
            fg_mask = (p_bgfg[:, 1, :, :] >= float(fg_threshold))
        else:
            fg_mask = (fg_bg_pred > 0)

        pred = torch.zeros_like(fg_bg_pred, dtype=torch.long)
        if float(fg_class_min_conf) >= 0.0:
            fg_mask = fg_mask & (fg_cls_conf >= float(fg_class_min_conf))
        if fg_class_min_conf_per_class is not None and fg_class_min_conf_per_class.numel() == fg_cls_probs.shape[1]:
            per_class_thr = fg_class_min_conf_per_class.to(fg_cls_probs.device, dtype=fg_cls_probs.dtype)
            pred_fg_idx = (fg_cls_pred - 1).clamp(min=0)
            pred_fg_thr = per_class_thr[pred_fg_idx]
            fg_mask = fg_mask & (fg_cls_conf >= pred_fg_thr)
        pred[fg_mask] = fg_cls_pred[fg_mask]
        return pred
    
    def _setup_device(self) -> None:
        """setup device and validate multi-GPU configuration"""
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.device = torch.device(device_type)
            if device_type == 'cuda':
                self.available_gpus = torch.cuda.device_count()
                if self.num_gpus > self.available_gpus:
                    raise RuntimeError(
                        f"Requested {self.num_gpus} GPUs but only {self.available_gpus} available.")
                if self.num_gpus < 1:
                    raise RuntimeError(f"num_gpus must be >= 1, got {self.num_gpus}")
                self.gpu_ids = list(range(self.num_gpus))
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"  Device: {self.num_gpus}x {gpu_name} ({gpu_mem:.1f}GB), CUDA {torch.version.cuda}")
                tprint('cuda.empty_cache() performing...')
                torch.cuda.empty_cache()
                tprint('cuda.empty_cache() performed!')
        except Exception as e:
            print(f"Initialization on CPU due to {e}")
            self.device = torch.device('cpu')
            self.num_gpus = 0
            self.gpu_ids = []
            self.available_gpus = 0
        
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'

    @contextmanager
    def _dataset_import_manager(self):
        """Context manager for DataLoader import/build lifecycle."""
        tic = time.perf_counter()
        tprint("[dataset_import_manager] start")
        try:
            yield
        finally:
            toc = time.perf_counter()
            train_batches = len(self.train_loader) if hasattr(self, 'train_loader') else 0
            val_batches = len(self.val_loader) if hasattr(self, 'val_loader') and self.val_loader else 0
            test_batches = len(self.test_loader) if hasattr(self, 'test_loader') else 0
            tprint("[dataset_import_manager] done in "
                   f"{toc - tic:.2f}s, batches(train/val/test): "
                   f"{train_batches}/{val_batches}/{test_batches}")

    @contextmanager
    def _training_manager(self):
        """Context manager for training lifecycle and post-train cleanup."""
        tic = time.perf_counter()
        tprint(f"[training_manager] start: {self.model_name}")
        try:
            yield
        finally:
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            toc = time.perf_counter()
            tprint(f"[training_manager] finished in {toc - tic:.2f}s")

    @contextmanager
    def _batch_stream_manager(self, loader):
        """Context manager yielding a stream-ready batch iterator.

        On CUDA, prefetches upcoming batch to overlap host->device copies with
        current-step compute. On CPU, falls back to the original loader.
        """
        use_stream_prefetch = (
            self.device.type == 'cuda'
            and loader is not None
        )

        if use_stream_prefetch:
            batch_iter = _CudaBatchPrefetcher(loader, self.device)
        else:
            batch_iter = loader

        try:
            yield batch_iter
        finally:
            batch_iter = None
    
    def _load_data(self) -> None:
        """load & create data loaders with optimized performance settings"""
        tprint("Loading data and creating data loaders with:")
        try:
            # Optimize data loading for multi-GPU
            num_workers = self.config.memory.num_workers
            batch_size = self.config.split.batch_size
            prefetch_factor = 2
            persistent_workers = False
            
            if self.num_gpus > 1:
                # define num_workers with flexible way:
                # min(CPU core/2, 8, GPU*2)
                # Too mang workers on CPU causes imbalance.
                available_cpus = os.cpu_count() or 4
                num_workers = min(available_cpus // 2, 8, self.num_gpus * 2)
                
                # Increase batch size for multi-GPU to improve GPU utilization
                batch_size = self.config.split.batch_size * self.num_gpus
                
                # prefetch_factor=2: preload 2 batches per worker.
                # persistent_workers=True: avoid frequent worker del.
                prefetch_factor = self.config.memory.prefetch_factor
                persistent_workers = True
                
                print(f"  [Multi-GPU Optimization] Scaling for {self.num_gpus} GPUs:")
                print(f"    num_workers: {self.config.memory.num_workers} -> {num_workers}")
                print(f"    batch_size: {self.config.split.batch_size} -> {batch_size} "
                      f"({batch_size // self.num_gpus} per GPU)")
                print(f"    prefetch_factor: {prefetch_factor}, persistent_workers: {persistent_workers}")

            # Cached split pipeline loads patch chunks from large samples;
            # aggressive worker/prefetch scaling can exhaust host memory.
            if bool(getattr(self.dataLoader, '_use_cached_split_pipeline', False)):
                cached_workers = int(getattr(self.config.memory, 'cached_loader_num_workers', 2))
                cached_prefetch = int(getattr(self.config.memory, 'cached_loader_prefetch_factor', 1))
                cached_persistent = bool(getattr(self.config.memory, 'cached_loader_persistent_workers', False))
                num_workers = max(0, cached_workers)
                prefetch_factor = max(1, cached_prefetch)
                persistent_workers = cached_persistent and (num_workers > 0)
                print("  [Cached Split Optimization] Override DataLoader strategy:")
                print(
                    f"    num_workers -> {num_workers}, prefetch_factor -> {prefetch_factor}, "
                    f"persistent_workers -> {persistent_workers}"
                )
            
            loader_fn = getattr(self.dataLoader, 'get_loaders', None) \
                        or getattr(self.dataLoader, 'create_data_loader', None)
            if loader_fn is None:
                raise RuntimeError("Dataset object does not expose get_loaders/create_data_loader")

            # New API returns (train, val, test); supports legacy 2-tuple fallback
            with self._dataset_import_manager():
                loaders = loader_fn(
                    num_workers=num_workers,
                    batch_size=batch_size,
                    pin_memory=self.config.memory.pin_memory,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=persistent_workers
                )
                if len(loaders) == 3:
                    self.train_loader, self.val_loader, self.test_loader = loaders
                else:
                    # Legacy 2-loader mode: use test as val for backward compatibility
                    self.train_loader, self.test_loader = loaders
                    self.val_loader = None
                tprint(f"Data loaders created!")
        except Exception as e:
            raise RuntimeError(f"Error during data loading: {e}")
    
    def _create_model(self) -> None:
        """create model and print parameter count"""
        tprint("Creating model with:")
        try:
            if self.model is None:
                raise RuntimeError("model is None; please pass a model constructor or nn.Module when initializing hsTrainer")

            # self.model could be callable or nn.Module
            if isinstance(self.model, nn.Module):
                pass
            elif callable(self.model):
                self.model = self.model(**self.kwargs) if self.kwargs else self.model()
                if not isinstance(self.model, nn.Module):
                    raise TypeError(
                        "model constructor must return torch.nn.Module, "
                        f"got {type(self.model)}"
                    )
            else:
                raise TypeError(
                    "model must be either a callable constructor or torch.nn.Module, "
                    f"got {type(self.model)}"
                )

            self.model.to(self.device)
            
            # stats for model parameter count
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"  model: {self.model_name}")
            print(f"  total parameters: {total_params:,}")
            print(f"  trainable parameters: {trainable_params:,}")
        except Exception as e:
            raise RuntimeError(f"Error during model creation: {e}")
    
    def _setup_multi_gpu(self) -> None:
        """Setup DataParallel for multi-GPU training if num_gpus > 1"""
        if self.num_gpus > 1 and self.device.type == 'cuda':
            try:
                self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids, output_device=self.gpu_ids[0])
                print(f"  DataParallel: {self.num_gpus} GPUs")
            except Exception as e:
                print(f"  Warning: DataParallel failed ({e}), using single GPU")
    
    def _setup_training(self) -> None:
        """Initialize training components: optimizer, scheduler, loss, precision"""
        print("Training components:")
        
        torch.backends.cudnn.benchmark = self.config.memory.benchmark
        torch.backends.cuda.matmul.allow_tf32 = self.config.memory.benchmark
        torch.backends.cudnn.allow_tf32 = self.config.memory.benchmark
        print("\ttorch.backends.cudnn.benchmark={},\n\t"
               "torch.backends.cuda.matmul.allow_tf32={},\n\t "
               "torch.backends.cudnn.allow_tf32={}\n\t".format(
                   torch.backends.cudnn.benchmark,
                   torch.backends.cuda.matmul.allow_tf32,
                   torch.backends.cudnn.allow_tf32
               ))
        
        # Optimizer with decoupled learning rates: backbone lower, heads higher.
        lr = self.config.common.lr
        weight_decay = self.config.common.weight_decay
        head_keys = ('fg_bg_head', 'fg_class_head', 'boundary_head', 'seg_decoder', 'aux_')
        backbone_params, head_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in head_keys):
                head_params.append(p)
            else:
                backbone_params.append(p)

        param_groups = []
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': lr * 0.5})
        if head_params:
            param_groups.append({'params': head_params, 'lr': lr})
        if not param_groups:
            param_groups.append({'params': [p for p in self.model.parameters() if p.requires_grad], 'lr': lr})

        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        self.base_lrs = [pg['lr'] for pg in self.optimizer.param_groups]

        # Stable cosine decay without warm restarts.
        sc = self.config.common.scheduler
        eta_min = getattr(sc, 'eta_min', 1e-6)
        full_train_batches = len(self.train_loader) if self.train_loader is not None else 1
        sched_batches = max(1, min(full_train_batches, int(getattr(self.config.common, 'max_train_batches_per_epoch', 0) or full_train_batches)))
        self.total_sched_steps = max(1, self.epochs * sched_batches)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.total_sched_steps, eta_min=eta_min)

        print(f"\tAdamW(backbone_lr={lr*0.5:.2e}, head_lr={lr:.2e}, wd={weight_decay}) + CosineDecay")
        
        # Class-weighted loss from dense pixel labels (ignore padding=255)
        focal_gamma = self.config.common.focal_gamma
        class_weights = None
        
        tprint(f"  Computing class weights with focal_gamma={focal_gamma}...")

        try:
            num_cls = self.config.clsf.num
            class_counts = None

            used_cached_manifest = False
            if hasattr(self.dataLoader, 'get_train_class_counts'):
                try:
                    cc = self.dataLoader.get_train_class_counts()
                    cc = np.asarray(cc, dtype=np.float64)
                    if cc.shape[0] != num_cls:
                        raise RuntimeError(
                            f"train class count size mismatch: expected {num_cls}, got {cc.shape[0]}"
                        )
                    class_counts = cc
                    sampled_patch_count = int(getattr(self.dataLoader, '_cached_train_patches', 0))
                    sampled_pixel_count = int(class_counts.sum())
                    used_cached_manifest = True
                    tprint(
                        f"    class-weight stats from cached train manifest: "
                        f"pixels={sampled_pixel_count:,}"
                    )
                except Exception as e:
                    tprint(f"    fallback to sampled class weights (cached manifest unavailable: {e})")

            if not used_cached_manifest:
                class_counts = np.zeros(num_cls, dtype=np.float64)
                rng = np.random.RandomState(int(getattr(self.config.common, 'class_weight_sample_seed', 350234)))

                train_indices = getattr(self.train_loader.dataset, 'indices', None)
                if train_indices is None:
                    raise RuntimeError("train_loader.dataset must expose indices for pixel-level class weighting")

                train_indices = np.asarray(train_indices, dtype=np.int64)
                total_train_patches = int(train_indices.shape[0])
                if total_train_patches <= 0:
                    raise RuntimeError("empty train indices for class weighting")

                max_sample_patches = int(getattr(self.config.common, 'class_weight_max_sample_patches', 20000))
                max_sample_pixels = int(getattr(self.config.common, 'class_weight_max_sample_pixels', 2_000_000))

                if max_sample_patches > 0 and total_train_patches > max_sample_patches:
                    sampled_pos = rng.choice(total_train_patches, size=max_sample_patches, replace=False)
                    sampled_indices = train_indices[sampled_pos]
                else:
                    sampled_indices = train_indices

                sampled_patch_count = int(sampled_indices.shape[0])
                patch_area = int(self.config.split.patch_size) * int(self.config.split.patch_size)
                est_valid_pixels = sampled_patch_count * patch_area
                pixel_keep_prob = 1.0
                if max_sample_pixels > 0 and est_valid_pixels > max_sample_pixels:
                    pixel_keep_prob = max_sample_pixels / float(est_valid_pixels)
                    pixel_keep_prob = max(1e-6, min(1.0, pixel_keep_prob))

                sampled_pixel_count = 0
                for idx in sampled_indices:
                    label_patch = self.dataLoader._get_label_patch_(int(idx))
                    valid = label_patch != 255
                    if valid.any():
                        valid_labels = label_patch[valid].reshape(-1)
                        if pixel_keep_prob < 1.0:
                            keep_mask = rng.rand(valid_labels.shape[0]) < pixel_keep_prob
                            if not keep_mask.any():
                                continue
                            valid_labels = valid_labels[keep_mask]

                        cnt = np.bincount(valid_labels, minlength=num_cls).astype(np.float64)
                        class_counts += cnt
                        sampled_pixel_count += int(valid_labels.shape[0])

            total_samples = class_counts.sum()
            if total_samples <= 0:
                raise RuntimeError("no valid training pixels found for class weighting")

            # Effective number weighting (Cui et al., CVPR 2019)
            beta = (total_samples - 1) / total_samples
            effective_num = 1.0 - np.power(beta, class_counts)
            raw_weights = 1.0 / (effective_num + 1e-8)

            # Avoid exploding/invalid weights for unseen classes.
            raw_weights[class_counts <= 0] = 0.0
            raw_weights = raw_weights / raw_weights.mean()
            class_weights = torch.FloatTensor(raw_weights).to(self.device)
            
            imbalance = class_counts.max() / (class_counts.min() + 1)
            print(f"  Imbalance report (pixel-level): imbalance ratio {imbalance:.1f}x, {num_cls} classes")
            print(f"  Class-weight stats: sampled_patches={sampled_patch_count:,}, sampled_pixels={sampled_pixel_count:,}")
            print(f"  Class weights: {dict(zip(self.config.clsf.targets, class_weights.cpu().numpy()))}")
        except Exception as e:
            print(f"  Warning: uniform class weights ({e})")

        self.early_stop_metric = str(getattr(self.config.common, 'early_stop_metric', 'composite')).lower()
        self.metric_weight_fg = float(getattr(self.config.common, 'metric_weight_fg', 0.5))
        self.metric_weight_all = float(getattr(self.config.common, 'metric_weight_all', 0.5))
        self.metric_weight_bgfg = float(getattr(self.config.common, 'metric_weight_bgfg', 0.4))
        self.metric_weight_fg_min_recall = float(getattr(self.config.common, 'metric_weight_fg_min_recall', 0.2))
        self.hybrid_min_ba_bgfg = float(getattr(self.config.common, 'hybrid_min_ba_bgfg', 52.0))
        self.hybrid_min_fg_recall = float(getattr(self.config.common, 'hybrid_min_fg_recall', 10.0))
        self.hybrid_low_metric_penalty = float(getattr(self.config.common, 'hybrid_low_metric_penalty', 8.0))
        self.save_topk = int(max(1, getattr(self.config.common, 'save_topk_models', 3)))
        self.topk_models = []

        if self.config.clsf.num <= 1:
            raise ValueError("Two-stage supervision requires clsf.num > 1")

        fg_w = float(getattr(self.config.common, 'fg_loss_weight', 1.0))
        bgfg_w = float(getattr(self.config.common, 'bgfg_loss_weight', 0.5))
        
        # HierarchicalSegLoss combines:
        # fg_bg_seghead: binary loss for fg/bg separation.
        # fg_class_seghead: focal loss for fg inner classification.
        self.criterion = HierarchicalSegLoss(
            num_classes=self.config.clsf.num,
            class_weight=class_weights,
            fg_weight=fg_w,
            bgfg_weight=bgfg_w,
            bgfg_pos_weight=float(getattr(self.config.common, 'bgfg_pos_weight', 2.0)),
            bgfg_hard_neg_weight=float(getattr(self.config.common, 'bgfg_hard_neg_weight', 2.0)),
            bgfg_boundary_weight=float(getattr(self.config.common, 'bgfg_boundary_weight', 0.0)),
            boundary_weight=float(getattr(self.config.common, 'boundary_loss_weight', 0.2)),
            aux_weight=float(getattr(self.config.common, 'aux_loss_weight', 0.3)),
            fg_loss_type=str(getattr(self.config.common, 'fg_loss_type', 'focal')),
            cb_beta=float(getattr(self.config.common, 'cb_beta', 0.999)),
            logit_adjust_tau=float(getattr(self.config.common, 'logit_adjust_tau', 1.0)),
            bg_suppress_weight=float(getattr(self.config.common, 'bg_suppress_weight', 0.05)),
            bg_suppress_threshold=float(getattr(self.config.common, 'bg_suppress_threshold', 0.2)),
            stage_b_bgfg_anchor_weight=float(getattr(self.config.common, 'stage_b_bgfg_anchor_weight', 0.0)),
            stage_b_weak_cls_focus=float(getattr(self.config.common, 'stage_b_weak_cls_focus', 0.0)),
            stage_b_weak_cls_boost_max=float(getattr(self.config.common, 'stage_b_weak_cls_boost_max', 1.8)),
            stage_b_weak_cls_boost_momentum=float(getattr(self.config.common, 'stage_b_weak_cls_boost_momentum', 0.6)),
            stage_c_boundary_boost_max=float(getattr(self.config.common, 'stage_c_boundary_boost_max', 2.0)),
            stage_c_boundary_boost_min=float(getattr(self.config.common, 'stage_c_boundary_boost_min', 0.75)),
            gamma=focal_gamma,
            ignore_index=255,
            label_smoothing=getattr(self.config.common, 'label_smoothing', 0.0),
        )
        print(f"  HierarchicalSegLoss(gamma={focal_gamma}, fg_w={fg_w}, bgfg_w={bgfg_w})")
        
        # Mixed precision
        use_amp = self.config.common.use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler() if use_amp else None
        self.use_amp = use_amp
        
        # Hyperparameters
        self.grad_clip = getattr(self.config.common, 'grad_clip', 1.0)
        self.warmup_epochs = getattr(self.config.common, 'warmup_epochs', 5)
        self.patience = getattr(self.config.common, 'patience', 20)
        self.eval_interval = getattr(self.config.common, 'eval_interval', 1)
        self.max_train_batches_per_epoch = int(getattr(self.config.common, 'max_train_batches_per_epoch', 0))
        self.max_val_batches_per_epoch = int(
            getattr(self.config.common, 'max_val_batches_per_epoch', 0)
        )
        decode_thr_cfg = getattr(self.config.common, 'eval_decode_fg_threshold', None)
        if decode_thr_cfg is None:
            decode_thr_cfg = getattr(self.config.common, 'eval_fg_gate_threshold', -1.0)
        self.eval_decode_fg_threshold = float(decode_thr_cfg)
        self.auto_calibrate_fg_threshold = bool(getattr(self.config.common, 'auto_calibrate_fg_threshold', True))
        self.fg_threshold_metric = str(getattr(self.config.common, 'fg_threshold_metric', 'ba_bgfg')).lower()
        self.best_fg_threshold = 0.55 if self.eval_decode_fg_threshold < 0.0 else self.eval_decode_fg_threshold
        self.eval_fg_class_min_conf = float(getattr(self.config.common, 'eval_fg_class_min_conf', -1.0))
        fg_bias_cfg = list(getattr(self.config.common, 'eval_fg_class_logit_bias', []))
        fg_dim = max(0, int(self.config.clsf.num) - 1)
        if len(fg_bias_cfg) == fg_dim and fg_dim > 0:
            self.eval_fg_class_logit_bias = torch.as_tensor(fg_bias_cfg, dtype=torch.float32)
        else:
            self.eval_fg_class_logit_bias = torch.zeros(fg_dim, dtype=torch.float32)
        fg_min_conf_pc_cfg = list(getattr(self.config.common, 'eval_fg_class_min_conf_per_class', []))
        if len(fg_min_conf_pc_cfg) == fg_dim and fg_dim > 0:
            self.eval_fg_class_min_conf_per_class = torch.as_tensor(fg_min_conf_pc_cfg, dtype=torch.float32)
        else:
            self.eval_fg_class_min_conf_per_class = torch.full((fg_dim,), -1.0, dtype=torch.float32)
        self._last_threshold_scan = []
        self.threshold_scan_history = []
        self._last_selected_fg_threshold = float(self.best_fg_threshold)
        self.stage_rescue_ba_bgfg_threshold = float(getattr(self.config.common, 'stage_rescue_ba_bgfg_threshold', 40.0))
        self.stage_rescue_epoch_limit = int(getattr(self.config.common, 'stage_rescue_epoch_limit', 3))
        self.stage_override = None
        self.stage_transition_patience = int(getattr(self.config.common, 'stage_transition_patience', 3))
        self.stage_a_min_epochs = int(getattr(self.config.common, 'stage_a_min_epochs', max(1, int(round(self.epochs * 0.15)))))
        self.stage_b_min_epochs = int(getattr(self.config.common, 'stage_b_min_epochs', max(1, int(round(self.epochs * 0.35)))))
        self.stage_c_min_epochs = int(getattr(self.config.common, 'stage_c_min_epochs', 3))
        self.stage_a_target_ba_bgfg = float(getattr(self.config.common, 'stage_a_target_ba_bgfg', 80.0))
        self.stage_b_target_ba_fg = float(getattr(self.config.common, 'stage_b_target_ba_fg', 70.0))
        self.stage_a_target_ba_bgfg_boundary = float(
            getattr(self.config.common, 'stage_a_target_ba_bgfg_boundary', 55.0)
        )
        self.stage_a_min_boundary_ba_for_transition = float(
            getattr(self.config.common, 'stage_a_min_boundary_ba_for_transition', 50.0)
        )
        self.stage_a_boundary_metric_weight = float(
            np.clip(getattr(self.config.common, 'stage_a_boundary_metric_weight', 0.45), 0.0, 1.0)
        )
        self.stage_b_target_fg_min_recall = float(
            getattr(self.config.common, 'stage_b_target_fg_min_recall', 20.0)
        )
        self.stage_b_weak_cls_target_recall = float(
            getattr(self.config.common, 'stage_b_weak_cls_target_recall', self.stage_b_target_fg_min_recall)
        )
        self.stage_b_min_fg_recall_for_c = float(
            getattr(self.config.common, 'stage_b_min_fg_recall_for_c', 12.0)
        )
        self.stage_b_force_c_min_fg_recall = float(
            getattr(self.config.common, 'stage_b_force_c_min_fg_recall', self.stage_b_min_fg_recall_for_c)
        )
        self.stage_b_min_recall_metric_weight = float(
            np.clip(getattr(self.config.common, 'stage_b_min_recall_metric_weight', 0.35), 0.0, 1.0)
        )
        self.eval_boundary_band_dilation = int(max(0, getattr(self.config.common, 'eval_boundary_band_dilation', 2)))
        self.stage_c_boundary_target_ba = float(
            getattr(self.config.common, 'stage_c_boundary_target_ba', self.stage_a_target_ba_bgfg_boundary)
        )
        self.stage_eval_uncapped = bool(getattr(self.config.common, 'stage_eval_uncapped', True))
        self.calibrate_threshold_stage_c_only = bool(getattr(self.config.common, 'calibrate_threshold_stage_c_only', True))
        self.stage_metric_ema_alpha = float(np.clip(getattr(self.config.common, 'stage_metric_ema_alpha', 0.6), 0.0, 0.99))
        self._stage_metric_ema = None
        self.stage_b_keep_bgfg_head = bool(getattr(self.config.common, 'stage_b_keep_bgfg_head', False))
        extra_patterns = list(getattr(self.config.common, 'stage_b_extra_patterns', []))
        if not extra_patterns:
            extra_patterns = ['seg_decoder', 'transformer_levels', 'downsample_', 'patch_embed', 'spectral_conv']
        self.stage_b_extra_patterns = [str(x) for x in extra_patterns]
        self.current_stage = str(getattr(self.config.common, 'stage_start', 'A')).upper()
        if self.current_stage not in {'A', 'B', 'C'}:
            self.current_stage = 'A'
        self.stage_start_epoch = 0
        self._stage_best_metric = -1e18
        self._stage_bad_epochs = 0
        if self.max_val_batches_per_epoch <= 0 and self.max_train_batches_per_epoch > 0:
            # Keep validation affordable by default; can be overridden explicitly.
            self.max_val_batches_per_epoch = max(1, self.max_train_batches_per_epoch // 2)

        # EMA
        ema_decay = getattr(self.config.common, 'ema_decay', 0.999)
        self.ema = ModelEMA(self.model, decay=ema_decay)
        
        if use_amp:
            print(f"  AMP enabled, EMA(decay={ema_decay}), patience={self.patience}")
        else:
            print(f"  EMA(decay={ema_decay}), patience={self.patience}")
        if self.max_train_batches_per_epoch > 0:
            print(f"  Train epoch cap: {self.max_train_batches_per_epoch} batches/epoch")
        if self.max_val_batches_per_epoch > 0:
            print(f"  Val eval cap: {self.max_val_batches_per_epoch} batches/eval")

        # Backward-compatible fields for external code paths that still inspect ratio schedule.
        self.stage_a_epochs = self.stage_a_min_epochs
        self.stage_b_epochs = self.stage_b_min_epochs
        self.stage_c_start = min(self.epochs - 1, self.stage_a_epochs + self.stage_b_epochs)
        self._last_fg_class_recalls = np.zeros(max(1, int(self.config.clsf.num) - 1), dtype=np.float32)
    
    def train_epoch(self, epoch: int) -> Tuple[float, float, float, float]:
        """
        Train for a single epoch.
        This method uses indexed batch unpacking and non-blocking transfers,
            which are crucial for memory and time efficiency.
            
        NOTE: A POSSIBLE **MEMORY FATAL ERROR**
        hsi.permute() in the method may cause incontinuous memory,
        try hsi.permute().contiguous() if goes wrong.
        see https://blog.csdn.net/weixin_42046845/article/details/134667338.
        
        Return: Tuple (``loss``, ``accuracy``)
        """
        self.model.train()
        total_loss = 0.0
        total_bgfg_loss = 0.0
        total_fg_class_loss = 0.0
        correct = 0
        total = 0

        stage = self._apply_training_stage(epoch)
        if hasattr(self.criterion, 'set_active_stage'):
            self.criterion.set_active_stage(stage)

        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * warmup_factor

        full_train_batches = len(self.train_loader)
        effective_train_batches = full_train_batches
        if self.max_train_batches_per_epoch > 0:
            effective_train_batches = min(full_train_batches, self.max_train_batches_per_epoch)

        with self._batch_stream_manager(self.train_loader) as batch_iter:
            pbar = tqdm(batch_iter, total=effective_train_batches,
                        desc=f'Epoch {epoch+1}/{self.epochs}', leave=False)

            for batch_idx, batch_data in enumerate(pbar):
                if batch_idx >= effective_train_batches:
                    break
                hsi, labels, _ = self._unpack_batch(batch_data)

                if hsi.device.type != self.device.type:
                    hsi = hsi.to(self.device, non_blocking=True)
                if labels.device.type != self.device.type:
                    labels = labels.to(self.device, non_blocking=True,
                                       memory_format=torch.contiguous_format)

                if hsi.dim() == 4 and hsi.shape[-1] <= 16:
                    hsi = hsi.permute(0, 3, 1, 2)

                self.optimizer.zero_grad()
                if self.use_amp:
                    with autocast():
                        outputs = self.model(hsi)
                        _, fg_bg_logits, fg_class_logits = self._parse_model_outputs(outputs)
                        loss, bgfg_loss, fg_class_loss = self.criterion.decompose(outputs, labels)
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(hsi)
                    _, fg_bg_logits, fg_class_logits = self._parse_model_outputs(outputs)
                    loss, bgfg_loss, fg_class_loss = self.criterion.decompose(outputs, labels)
                    loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                self.ema.update(self.model)
                if epoch >= self.warmup_epochs:
                    self.scheduler.step()

                total_loss += float(loss.item())
                total_bgfg_loss += float(bgfg_loss.item())
                total_fg_class_loss += float(fg_class_loss.item())
                with torch.no_grad():
                    predicted = self._decode_two_stage_predictions(
                        fg_bg_logits.detach(), fg_class_logits.detach(), fg_threshold=-1.0)
                    valid_mask = (labels != 255)
                    total += int(valid_mask.sum().item())
                    correct += int(((predicted == labels) & valid_mask).sum().item())

                acc = 100.0 * correct / total if total > 0 else 0.0
                pbar.set_postfix({
                    'bgfg_loss': f'{bgfg_loss.item():.4f}',
                    'fg_class_loss': f'{fg_class_loss.item():.4f}',
                    'acc': f'{acc:.2f}%',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })

        processed_batches = max(batch_idx + 1 if 'batch_idx' in locals() else 0, 1)
        epoch_loss = total_loss / processed_batches
        epoch_bgfg_loss = total_bgfg_loss / processed_batches
        epoch_fg_class_loss = total_fg_class_loss / processed_batches
        epoch_acc = 100.0 * correct / total if total > 0 else 0.0
        
        # record lr for this epoch
        self.lr_history.append(self.optimizer.param_groups[0]['lr'])
        
        # record gradient L2 norm
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        self.grad_norms.append(total_norm ** 0.5)
        
        return epoch_loss, epoch_acc, epoch_bgfg_loss, epoch_fg_class_loss

    def _apply_training_stage(self, epoch: int) -> str:
        """Apply 3-stage training policy without changing external API."""
        raw_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        if self.stage_override in {'A', 'B', 'C'}:
            stage = self.stage_override
        else:
            stage = str(getattr(self, 'current_stage', 'A')).upper()

        def _match_any(name: str, patterns: List[str]) -> bool:
            return any(pat in name for pat in patterns)

        # Stage A: focus on FG/BG + decoder stability.
        if stage == 'A':
            # Prefer coarse FG/BG learning first; keep a light feature path trainable.
            stage_a_patterns = [
                'fg_bg_head',
                'aux_fg_bg_head',
                'seg_decoder',
                'spectral_conv',
                'in_conv',
                'decoder_conv',
                'stem',
                'patch_embed',
                'blocks.0',
                'norm',
            ]
            for name, p in raw_model.named_parameters():
                trainable = _match_any(name, stage_a_patterns)
                p.requires_grad_(trainable)
        # Stage B: focus on foreground subclass discrimination.
        elif stage == 'B':
            # Freeze BG/FG branch and train only FG-subclass heads.
            stage_b_patterns = [
                'fg_class_head',
                'aux_fg_class_head',
                'boundary_head',
                'norm',
            ]
            if bool(getattr(self, 'stage_b_keep_bgfg_head', False)):
                stage_b_patterns.extend(['fg_bg_head', 'aux_fg_bg_head'])
            stage_b_patterns.extend(list(getattr(self, 'stage_b_extra_patterns', [])))
            for name, p in raw_model.named_parameters():
                trainable = _match_any(name, stage_b_patterns)
                p.requires_grad_(trainable)
        # Stage C: joint finetune.
        else:
            for p in raw_model.parameters():
                p.requires_grad_(True)
        return stage
    
    @torch.no_grad()
    def evaluate(self, collect_extra: bool = False, use_ema: bool = False,
                 loader=None, collect_reconstruction: bool = False,
                 uncapped: bool = False) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Evaluate on a given loader with optimized performance.

        Optimization:
        1) GPU-side confusion matrix (int64) accumulated via scatter_add_
              each batch; only 64 numbers transferred to CPU after the loop,
              instead of ~1.5 GB of per-pixel predictions.
        2) All metrics (OA, BA, Kappa, mIoU) derived analytically from the
              single confusion matrix; confusion_matrix() is called only once,
              eliminating 19 redundant full-array scans.

        Args:
            collect_extra: if True, also collect per-pixel predictions/targets
                           (needed for plots) and softmax probabilities (ROC).
                           Only set True for the final evaluation — not every epoch.
            collect_reconstruction: if True, collect reconstruction maps only
                           (no ROC/vis arrays), used by A->B BGFG snapshot.
            uncapped: if True, bypass eval batch cap and run full loader.
            use_ema: if True, temporarily swap in EMA shadow weights for
                     evaluation, then restore original weights.
            loader: DataLoader to evaluate on. If None, uses val_loader (for epoch
                    eval/early stopping) or test_loader if val_loader is unavailable.

        Return:
            Tuple (``loss``, ``accuracy``, ``kappa``, ``predictions``, ``labels``)
            predictions/labels are full arrays when collect_extra=True,
            empty arrays otherwise (sufficient for epoch-level logging).
            When collect_extra is True, also stores self._last_probas.
        """
        if use_ema:
            self.ema.swap(self.model)

        self.model.eval()
        model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        num_classes = int(model_ref.num_classes)
        metric_labels_fg = self._metric_labels(num_classes)
        metric_labels_all = list(range(num_classes))

        total_loss = 0.0
        total_bgfg_loss = 0.0
        total_fg_class_loss = 0.0
        cm_gpu = torch.zeros(num_classes, num_classes,
                             dtype=torch.long, device=self.device)
        pred_fg_pixels = 0
        gt_fg_pixels = 0
        valid_pixels = 0
        sum_fg_prob = 0.0
        thr_grid = np.arange(0.30, 0.80 + 1e-9, 0.02, dtype=np.float32)
        thr_tp = np.zeros_like(thr_grid, dtype=np.float64)
        thr_fp = np.zeros_like(thr_grid, dtype=np.float64)
        thr_tn = np.zeros_like(thr_grid, dtype=np.float64)
        thr_fn = np.zeros_like(thr_grid, dtype=np.float64)
        boundary_tp = 0.0
        boundary_fp = 0.0
        boundary_tn = 0.0
        boundary_fn = 0.0

        using_val_loader = (loader is None and self.val_loader is not None) or (loader is self.val_loader)
        decode_threshold = float(self.best_fg_threshold)
        if self.eval_decode_fg_threshold >= 0:
            decode_threshold = float(self.eval_decode_fg_threshold)
        self._last_selected_fg_threshold = decode_threshold

        predictions_list = [] if collect_extra else None
        targets_list = [] if collect_extra else None
        probas_list = [] if collect_extra else None
        roc_targets_list = [] if collect_extra else None
        vis_pairs = [] if collect_extra else None
        need_reconstruction = bool(collect_extra or collect_reconstruction)

        eval_loader = self.val_loader if loader is None and self.val_loader else (loader or self.test_loader)

        # Speed-up for epoch-level validation only; keep final collect_extra eval full-pass.
        eval_cap = 0
        if not collect_extra:
            using_val_loader = (eval_loader is self.val_loader)
            if using_val_loader:
                eval_cap = self.max_val_batches_per_epoch
        if uncapped:
            eval_cap = 0
        if collect_reconstruction and bool(getattr(self.config.common, 'bgfgsplit_collect_uncapped', False)):
            eval_cap = 0

        total_eval_batches = len(eval_loader)
        selected_eval_batches = self._select_eval_batch_indices(total_eval_batches, eval_cap)
        if selected_eval_batches is None:
            eval_iter_loader = eval_loader
            effective_eval_batches = total_eval_batches
        else:
            eval_iter_loader = self._build_capped_eval_loader(eval_loader, selected_eval_batches)
            effective_eval_batches = len(eval_iter_loader)
            tprint(
                f"  Eval cap active: randomly stratified {effective_eval_batches}/{total_eval_batches} batches"
            )

        recon_state = self._init_reconstruction_state(eval_iter_loader) if need_reconstruction else None
        roc_per_batch = max(256, 2_000_000 // max(len(eval_iter_loader), 1)) if collect_extra else 0
        plot_per_batch = max(256, 1_000_000 // max(len(eval_iter_loader), 1)) if collect_extra else 0
        max_vis_samples = self.config.common.vis_samples

        processed_eval_batches = 0
        with self._batch_stream_manager(eval_iter_loader) as batch_iter:
            pbar = tqdm(total=effective_eval_batches, desc='Evaluating', leave=False)
            for eval_batch_idx, batch_data in enumerate(batch_iter):
                hsi, labels, batch_meta = self._unpack_batch(batch_data)
                if hsi.device.type != self.device.type:
                    hsi = hsi.to(self.device, non_blocking=True)
                if labels.device.type != self.device.type:
                    labels = labels.to(self.device, non_blocking=True)

                if hsi.dim() == 4 and hsi.shape[-1] <= 16:
                    hsi = hsi.permute(0, 3, 1, 2)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(hsi)
                        _, fg_bg_logits, fg_class_logits = self._parse_model_outputs(outputs)
                        loss, bgfg_loss, fg_class_loss = self.criterion.decompose(outputs, labels)
                else:
                    outputs = self.model(hsi)
                    _, fg_bg_logits, fg_class_logits = self._parse_model_outputs(outputs)
                    loss, bgfg_loss, fg_class_loss = self.criterion.decompose(outputs, labels)

                total_loss += float(loss.item())
                total_bgfg_loss += float(bgfg_loss.item())
                total_fg_class_loss += float(fg_class_loss.item())
                predicted = self._decode_two_stage_predictions(
                    fg_bg_logits, fg_class_logits,
                    fg_threshold=decode_threshold,
                    fg_class_min_conf=self.eval_fg_class_min_conf,
                    fg_class_logit_bias=self.eval_fg_class_logit_bias,
                    fg_class_min_conf_per_class=self.eval_fg_class_min_conf_per_class)

                valid_mask = (labels != 255)
                pred_valid = predicted[valid_mask]
                tgt_valid = labels[valid_mask]
                v_count = int(valid_mask.sum().item())
                if v_count > 0:
                    valid_pixels += v_count
                    pred_fg_pixels += int((pred_valid > 0).sum().item())
                    gt_fg_pixels += int((tgt_valid > 0).sum().item())

                    # Boundary-band FG/BG quality: evaluate on a dilated GT boundary mask.
                    boundary_mask = (HierarchicalSegLoss._build_boundary_target(labels, 255) > 0.5)
                    if self.eval_boundary_band_dilation > 0:
                        k = 2 * self.eval_boundary_band_dilation + 1
                        boundary_mask = F.max_pool2d(
                            boundary_mask.float().unsqueeze(1),
                            kernel_size=k,
                            stride=1,
                            padding=self.eval_boundary_band_dilation,
                        ).squeeze(1) > 0.5
                    boundary_valid = boundary_mask & valid_mask
                    if boundary_valid.any():
                        pred_fg_b = (predicted > 0)[boundary_valid]
                        tgt_fg_b = (labels > 0)[boundary_valid]
                        boundary_tp += float((pred_fg_b & tgt_fg_b).sum().item())
                        boundary_fp += float((pred_fg_b & (~tgt_fg_b)).sum().item())
                        boundary_fn += float(((~pred_fg_b) & tgt_fg_b).sum().item())
                        boundary_tn += float(((~pred_fg_b) & (~tgt_fg_b)).sum().item())

                    # Aggregate FG probabilities in FP32 to avoid FP16 overflow.
                    p_bgfg = F.softmax(fg_bg_logits.float(), dim=1)
                    fg_prob_valid = p_bgfg[:, 1, :, :][valid_mask]
                    fg_prob_valid = torch.nan_to_num(fg_prob_valid, nan=0.0, posinf=1.0, neginf=0.0)
                    sum_fg_prob += float(fg_prob_valid.sum(dtype=torch.float32).item())

                    # Threshold scan for BG/FG calibration on validation set.
                    tgt_fg_valid = (tgt_valid > 0)
                    for i_thr, thr in enumerate(thr_grid):
                        pred_fg_thr = (fg_prob_valid >= float(thr))
                        tp = (pred_fg_thr & tgt_fg_valid).sum().item()
                        fp = (pred_fg_thr & (~tgt_fg_valid)).sum().item()
                        fn = ((~pred_fg_thr) & tgt_fg_valid).sum().item()
                        tn = ((~pred_fg_thr) & (~tgt_fg_valid)).sum().item()
                        thr_tp[i_thr] += tp
                        thr_fp[i_thr] += fp
                        thr_fn[i_thr] += fn
                        thr_tn[i_thr] += tn

                linear_idx = tgt_valid * num_classes + pred_valid
                cm_gpu.view(-1).scatter_add_(
                    0, linear_idx,
                    torch.ones_like(linear_idx, dtype=torch.long)
                )

                if need_reconstruction:
                    if collect_reconstruction and not collect_extra:
                        # For A->B snapshot, use direct BG/FG head output to avoid FG-subclass decoding artifacts.
                        pred_bin = torch.argmax(fg_bg_logits.float(), dim=1).long()
                        label_bin = labels.clone().long()
                        valid_bin = (label_bin != 255)
                        label_bin[valid_bin] = (label_bin[valid_bin] > 0).long()
                        label_bin[~valid_bin] = 255
                        self._accumulate_reconstruction_batch(recon_state, pred_bin, label_bin, batch_meta=batch_meta)
                    else:
                        self._accumulate_reconstruction_batch(recon_state, predicted, labels, batch_meta=batch_meta)

                if collect_extra:
                    # Keep full metrics from cm_gpu; store only sampled pixels for plotting.
                    if pred_valid.numel() > plot_per_batch:
                        sel = torch.randperm(pred_valid.numel(), device=self.device)[:plot_per_batch]
                        predictions_list.append(pred_valid[sel].cpu())
                        targets_list.append(tgt_valid[sel].cpu())
                    else:
                        predictions_list.append(pred_valid.cpu())
                        targets_list.append(tgt_valid.cpu())
                    if len(vis_pairs) < max_vis_samples and predicted.dim() == 3 and labels.dim() == 3:
                        remaining = max_vis_samples - len(vis_pairs)
                        take_n = min(predicted.shape[0], remaining)
                        for b_idx in range(take_n):
                            vis_pairs.append((
                                predicted[b_idx].detach().cpu().numpy(),
                                labels[b_idx].detach().cpu().numpy(),
                            ))

                    p_bgfg = F.softmax(fg_bg_logits.float(), dim=1)
                    p_fg_cond = F.softmax(fg_class_logits.float(), dim=1)
                    proba = torch.cat([p_bgfg[:, 0:1, :, :], p_fg_cond * p_bgfg[:, 1:2, :, :]], dim=1)
                    proba = torch.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
                    proba_valid = proba.permute(0, 2, 3, 1)[valid_mask]
                    n_v = proba_valid.shape[0]
                    if n_v > roc_per_batch:
                        sel = torch.randperm(n_v, device=self.device)[:roc_per_batch]
                        probas_list.append(proba_valid[sel].cpu().numpy())
                        roc_targets_list.append(tgt_valid[sel].cpu().numpy())
                    else:
                        probas_list.append(proba_valid.cpu().numpy())
                        roc_targets_list.append(tgt_valid.cpu().numpy())

                processed_eval_batches += 1
                pbar.update(1)
                if processed_eval_batches >= effective_eval_batches:
                    break
            pbar.close()

        cm_np = cm_gpu.cpu().numpy().astype(np.int64)
        total = int(cm_np.sum())
        oa = float(np.diag(cm_np).sum()) / max(total, 1) * 100

        ba_all, kappa_all, miou_all = self._metrics_from_confusion(cm_np, metric_labels_all)
        ba_fg, kappa_fg, miou_fg = self._metrics_from_confusion(cm_np, metric_labels_fg)

        # FG/BG binary metrics derived from full confusion matrix (class 0 = BG).
        tp_fg = float(cm_np[1:, 1:].sum()) if cm_np.shape[0] > 1 else 0.0
        fn_fg = float(cm_np[1:, 0].sum()) if cm_np.shape[0] > 1 else 0.0
        fp_fg = float(cm_np[0, 1:].sum()) if cm_np.shape[0] > 1 else 0.0
        fg_precision = 100.0 * tp_fg / max(tp_fg + fp_fg, 1.0)
        fg_recall = 100.0 * tp_fg / max(tp_fg + fn_fg, 1.0)
        fg_dice = 100.0 * (2.0 * tp_fg) / max(2.0 * tp_fg + fp_fg + fn_fg, 1.0)

        if self.early_stop_metric == 'fg':
            acc = ba_fg
            kappa = kappa_fg
            miou = miou_fg
        elif self.early_stop_metric == 'all':
            acc = ba_all
            kappa = kappa_all
            miou = miou_all
        else:
            acc = self.metric_weight_fg * ba_fg + self.metric_weight_all * ba_all
            kappa = self.metric_weight_fg * kappa_fg + self.metric_weight_all * kappa_all
            miou = self.metric_weight_fg * miou_fg + self.metric_weight_all * miou_all

        processed_eval_batches = max(processed_eval_batches, 1)
        loss = total_loss / processed_eval_batches
        self._last_eval_bgfg_loss = total_bgfg_loss / processed_eval_batches
        self._last_eval_fg_class_loss = total_fg_class_loss / processed_eval_batches
        self._last_oa = oa
        self._last_ba_all = ba_all
        self._last_ba_fg = ba_fg
        self._last_kappa_all = kappa_all
        self._last_kappa_fg = kappa_fg
        self._last_miou_all = miou_all
        self._last_miou_fg = miou_fg
        self._last_miou = miou
        self._last_fg_precision = fg_precision
        self._last_fg_recall = fg_recall
        self._last_fg_dice = fg_dice
        self._last_pred_fg_ratio = (100.0 * pred_fg_pixels / max(valid_pixels, 1))
        self._last_gt_fg_ratio = (100.0 * gt_fg_pixels / max(valid_pixels, 1))
        self._last_ba_bgfg = 50.0 * (
            tp_fg / max(tp_fg + fn_fg, 1.0) +
            float(cm_np[0, 0]) / max(float(cm_np[0, :].sum()), 1.0)
        ) if cm_np.shape[0] > 1 else 0.0
        if (boundary_tp + boundary_fp + boundary_fn + boundary_tn) > 0:
            self._last_ba_bgfg_boundary = 50.0 * (
                boundary_tp / max(boundary_tp + boundary_fn, 1.0)
                + boundary_tn / max(boundary_tn + boundary_fp, 1.0)
            )
        else:
            self._last_ba_bgfg_boundary = float(self._last_ba_bgfg)

        if cm_np.shape[0] > 1:
            fg_rows = cm_np[1:, :]
            fg_row_sums = fg_rows.sum(axis=1).astype(np.float64)
            fg_diag = np.diag(cm_np)[1:].astype(np.float64)
            fg_valid_cls = fg_row_sums > 0
            if fg_valid_cls.any():
                fg_recalls = np.zeros_like(fg_row_sums, dtype=np.float64)
                fg_recalls[fg_valid_cls] = fg_diag[fg_valid_cls] / np.maximum(fg_row_sums[fg_valid_cls], 1.0)
                self._last_fg_min_recall = float(np.min(fg_recalls[fg_valid_cls]) * 100.0)
                self._last_fg_mean_recall = float(np.mean(fg_recalls[fg_valid_cls]) * 100.0)
                self._last_fg_class_recalls = (fg_recalls * 100.0).astype(np.float32)
            else:
                self._last_fg_min_recall = 0.0
                self._last_fg_mean_recall = 0.0
                self._last_fg_class_recalls = np.zeros_like(fg_row_sums, dtype=np.float32)
        else:
            self._last_fg_min_recall = 0.0
            self._last_fg_mean_recall = 0.0
            self._last_fg_class_recalls = np.zeros(max(1, int(self.config.clsf.num) - 1), dtype=np.float32)
        mean_fg_prob = (100.0 * sum_fg_prob / max(valid_pixels, 1))
        if not np.isfinite(mean_fg_prob):
            tprint(f"  WARNING: non-finite mean FG prob detected ({mean_fg_prob}); clamped to 0.0")
            mean_fg_prob = 0.0
        self._last_mean_fg_prob = mean_fg_prob

        self._last_threshold_scan = []
        calibrate_allowed = bool(self.auto_calibrate_fg_threshold)
        if bool(getattr(self, 'calibrate_threshold_stage_c_only', False)):
            calibrate_allowed = calibrate_allowed and (str(getattr(self, 'current_stage', 'C')) == 'C')

        if using_val_loader and calibrate_allowed and thr_grid.size > 0:
            best_idx = 0
            best_tuple = (-1e9, -1e9, 1e9)  # primary score, secondary, fg-ratio-gap
            gt_fg_ratio = 100.0 * (thr_tp[0] + thr_fn[0]) / max(thr_tp[0] + thr_fn[0] + thr_tn[0] + thr_fp[0], 1.0)
            for i_thr, thr in enumerate(thr_grid):
                tp = thr_tp[i_thr]
                fp = thr_fp[i_thr]
                fn = thr_fn[i_thr]
                tn = thr_tn[i_thr]
                tpr = tp / max(tp + fn, 1.0)
                tnr = tn / max(tn + fp, 1.0)
                ba_bgfg = 50.0 * (tpr + tnr)
                fg_dice = 100.0 * (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0)
                pred_fg_ratio = 100.0 * (tp + fp) / max(tp + fp + tn + fn, 1.0)
                fg_gap = abs(pred_fg_ratio - gt_fg_ratio)
                self._last_threshold_scan.append({
                    'thr': float(thr),
                    'ba_bgfg': float(ba_bgfg),
                    'fg_dice': float(fg_dice),
                    'pred_fg_ratio': float(pred_fg_ratio),
                    'gt_fg_ratio': float(gt_fg_ratio),
                })
                if self.fg_threshold_metric == 'fg_dice':
                    candidate = (fg_dice, ba_bgfg, -fg_gap)
                else:
                    candidate = (ba_bgfg, fg_dice, -fg_gap)
                if candidate > best_tuple:
                    best_tuple = candidate
                    best_idx = i_thr

            self.best_fg_threshold = float(thr_grid[best_idx])
            self._last_selected_fg_threshold = self.best_fg_threshold
        self._metric_labels_used = metric_labels_fg

        # final-eval: full arrays for plots / ROC (single GPU -> CPU transfer)
        if collect_extra:
            predictions = torch.cat(predictions_list, dim=0).numpy() if predictions_list else np.empty(0, dtype=np.int64)
            targets     = torch.cat(targets_list,     dim=0).numpy() if targets_list else np.empty(0, dtype=np.int64)
            self._last_vis_pairs = vis_pairs
            self._last_reconstruction = recon_state
            self._last_transition_reconstruction = recon_state
            if probas_list:
                self._last_probas         = np.concatenate(probas_list, axis=0)
                self._last_probas_targets = np.concatenate(roc_targets_list, axis=0)
            else:
                self._last_probas         = None
                self._last_probas_targets = None
        else:
            # epoch-level eval: callers only use scalar metrics
            predictions = np.empty(0, dtype=np.int64)
            targets     = np.empty(0, dtype=np.int64)
            self._last_vis_pairs = None
            self._last_reconstruction = None
            self._last_transition_reconstruction = recon_state if collect_reconstruction else None

        # restore original weights via second swap
        # e.g. if EMA was used, swap back to original model weights
        if use_ema:
            self.ema.swap(self.model)

        return loss, acc, kappa, predictions, targets

    def _flush_epoch_metrics(self) -> None:
        """Persist epoch-level metrics to JSON and CSV for downstream analysis."""
        if not self.epoch_metrics:
            return

        json_path = os.path.join(self.output, 'epoch_metrics.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.epoch_metrics, f, indent=2)

        csv_path = os.path.join(self.output, 'epoch_metrics.csv')
        fieldnames = [
            'epoch',
            'train_loss',
            'train_bgfg_loss',
            'train_fg_class_loss',
            'train_acc',
            'eval_loss',
            'eval_bgfg_loss',
            'eval_fg_class_loss',
            'eval_score',
            'eval_ba_all',
            'eval_ba_fg',
            'eval_kappa_all',
            'eval_kappa_fg',
            'eval_miou_all',
            'eval_miou_fg',
            'eval_fg_precision',
            'eval_fg_recall',
            'eval_fg_dice',
            'eval_ba_bgfg',
            'eval_ba_bgfg_boundary',
            'eval_fg_min_recall',
            'eval_fg_mean_recall',
            'eval_pred_fg_ratio',
            'eval_gt_fg_ratio',
            'eval_mean_fg_prob',
            'eval_selected_fg_threshold',
            'stage',
            'stage_eval_metric',
        ]

        # Keep CSV writer robust when new metric keys are added in the future.
        if self.epoch_metrics:
            known = set(fieldnames)
            for row in self.epoch_metrics:
                for key in row.keys():
                    if key not in known:
                        fieldnames.append(key)
                        known.add(key)

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.epoch_metrics:
                writer.writerow(row)

    def _register_topk_model(self, epoch: int, score: float) -> None:
        """Store top-k EMA snapshots for robust post-hoc model selection."""
        state = self.ema.state_dict()
        state_cpu = {k: v.detach().cpu().clone() for k, v in state.items()}
        entry = {
            'epoch': int(epoch),
            'score': float(score),
            'threshold': float(getattr(self, '_last_selected_fg_threshold', self.best_fg_threshold)),
            'model_state': state_cpu,
        }
        self.topk_models.append(entry)
        self.topk_models.sort(key=lambda x: x['score'], reverse=True)
        if len(self.topk_models) > self.save_topk:
            self.topk_models = self.topk_models[:self.save_topk]

    def _reselect_best_from_topk(self) -> None:
        """Re-evaluate top-k snapshots on validation set and keep the most stable one."""
        if not self.topk_models:
            return
        if self.val_loader is None:
            return

        prev_auto = self.auto_calibrate_fg_threshold
        prev_thr = self.best_fg_threshold
        self.auto_calibrate_fg_threshold = False

        best_entry = None
        best_eval_score = -1e18
        try:
            for entry in self.topk_models:
                self.model.load_state_dict(entry['model_state'])
                self.best_fg_threshold = float(entry.get('threshold', prev_thr))
                self._last_selected_fg_threshold = self.best_fg_threshold
                _ = self.evaluate(collect_extra=False, use_ema=False, loader=self.val_loader)
                eval_score = self._select_early_stop_score()
                if eval_score > best_eval_score:
                    best_eval_score = float(eval_score)
                    best_entry = entry
        finally:
            self.auto_calibrate_fg_threshold = prev_auto

        if best_entry is None:
            self.best_fg_threshold = prev_thr
            return

        self.best_epoch = int(best_entry['epoch'])
        self.best_acc = float(best_entry['score'])
        self.best_fg_threshold = float(best_entry.get('threshold', prev_thr))
        self.best_model_state = {
            'epoch': int(best_entry['epoch']),
            'model_state': best_entry['model_state'],
            'acc': float(best_entry['score']),
            'kappa': float(getattr(self, '_last_kappa_fg', 0.0)),
        }
        self._last_selected_fg_threshold = self.best_fg_threshold
        self._save_model()
    
    def train(self, _cv_mode: bool = False) -> Dict[str, float]:
        """
        Training loop with optional data validation.
        
        Args:
            _cv_mode: When True (called from cross_validate), skip per-fold
                      visualizations; aggregated CV plots are generated later.
        
        Returns:
            Dict[``str``, ``float``] of final results.
        """
        print("\n")
        if self.debug_mode:
            tprint(f"Debug mode enabled: CAM and visualizations enabled every epoch.")
        print(f"Training ({self.model_name})")
        print("\n")
        
        with self._training_manager():
            tic = time.perf_counter()

            for epoch in range(self.epochs):
                stage_this_epoch = str(getattr(self, 'current_stage', 'A')).upper()
                self.stage_history.append(stage_this_epoch)
                # training
                epoch_tic = time.perf_counter()
                train_loss, train_acc, train_bgfg_loss, train_fg_class_loss = self.train_epoch(epoch)
                self.epoch_times.append(time.perf_counter() - epoch_tic)
                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)
                self.train_bgfg_losses.append(train_bgfg_loss)
                self.train_fg_class_losses.append(train_fg_class_loss)

                # validating
                should_eval = ((epoch + 1) % self.eval_interval == 0) or (epoch + 1 == self.epochs)

                if should_eval or self.debug_mode:
                    collect_recon_now = (
                        (stage_this_epoch == 'A')
                        and bool(getattr(self.config.common, 'bgfgsplit_collect_from_eval', True))
                        and ((epoch + 1) >= max(1, int(getattr(self, 'stage_a_min_epochs', 1))))
                    )
                    # Evaluate on val set for early stopping (not test set)
                    eval_loss, eval_acc, kappa, pred, target = self.evaluate(
                        use_ema=True,
                        collect_reconstruction=collect_recon_now,
                    )

                    if stage_this_epoch in {'A', 'B'} and self.stage_eval_uncapped:
                        elapsed = int(epoch + 1 - self.stage_start_epoch)
                        min_need = self.stage_a_min_epochs if stage_this_epoch == 'A' else self.stage_b_min_epochs
                        if elapsed >= max(0, min_need - 1):
                            _ = self.evaluate(
                                use_ema=True,
                                collect_reconstruction=False,
                                uncapped=True,
                            )

                    self.eval_losses.append(eval_loss)
                    self.eval_accs.append(eval_acc)
                    self.eval_bgfg_losses.append(float(getattr(self, '_last_eval_bgfg_loss', 0.0)))
                    self.eval_fg_class_losses.append(float(getattr(self, '_last_eval_fg_class_loss', 0.0)))
                    eval_score = self._current_stage_metric(stage_this_epoch)

                    if hasattr(self.criterion, 'set_fg_bias_feedback'):
                        self.criterion.set_fg_bias_feedback(
                            float(getattr(self, '_last_pred_fg_ratio', 0.0)),
                            float(getattr(self, '_last_gt_fg_ratio', 0.0)),
                        )
                    if hasattr(self.criterion, 'set_fg_class_recall_feedback'):
                        self.criterion.set_fg_class_recall_feedback(
                            getattr(self, '_last_fg_class_recalls', None),
                            float(getattr(self, 'stage_b_weak_cls_target_recall', self.stage_b_target_fg_min_recall)),
                        )
                    if hasattr(self.criterion, 'set_boundary_feedback'):
                        self.criterion.set_boundary_feedback(
                            float(getattr(self, '_last_ba_bgfg_boundary', 0.0)),
                            float(getattr(self, 'stage_c_boundary_target_ba', self.stage_a_target_ba_bgfg_boundary)),
                        )

                    if (epoch + 1) <= max(1, self.stage_rescue_epoch_limit):
                        ba_bgfg = float(getattr(self, '_last_ba_bgfg', 0.0))
                        if ba_bgfg < self.stage_rescue_ba_bgfg_threshold:
                            self.stage_override = 'B' if (epoch + 1) < self.stage_rescue_epoch_limit else 'C'

                    self._maybe_advance_stage(epoch)

                    self.epoch_metrics.append({
                        'epoch': epoch + 1,
                        'train_loss': float(train_loss),
                        'train_bgfg_loss': float(train_bgfg_loss),
                        'train_fg_class_loss': float(train_fg_class_loss),
                        'train_acc': float(train_acc),
                        'eval_loss': float(eval_loss),
                        'eval_bgfg_loss': float(getattr(self, '_last_eval_bgfg_loss', 0.0)),
                        'eval_fg_class_loss': float(getattr(self, '_last_eval_fg_class_loss', 0.0)),
                        'eval_score': float(eval_score),
                        'eval_ba_all': float(getattr(self, '_last_ba_all', eval_acc)),
                        'eval_ba_fg': float(getattr(self, '_last_ba_fg', eval_acc)),
                        'eval_kappa_all': float(getattr(self, '_last_kappa_all', kappa)),
                        'eval_kappa_fg': float(getattr(self, '_last_kappa_fg', kappa)),
                        'eval_miou_all': float(getattr(self, '_last_miou_all', 0.0)),
                        'eval_miou_fg': float(getattr(self, '_last_miou_fg', 0.0)),
                        'eval_fg_precision': float(getattr(self, '_last_fg_precision', 0.0)),
                        'eval_fg_recall': float(getattr(self, '_last_fg_recall', 0.0)),
                        'eval_fg_dice': float(getattr(self, '_last_fg_dice', 0.0)),
                        'eval_ba_bgfg': float(getattr(self, '_last_ba_bgfg', 0.0)),
                        'eval_ba_bgfg_boundary': float(getattr(self, '_last_ba_bgfg_boundary', 0.0)),
                        'eval_fg_min_recall': float(getattr(self, '_last_fg_min_recall', 0.0)),
                        'eval_fg_mean_recall': float(getattr(self, '_last_fg_mean_recall', 0.0)),
                        'eval_pred_fg_ratio': float(getattr(self, '_last_pred_fg_ratio', 0.0)),
                        'eval_gt_fg_ratio': float(getattr(self, '_last_gt_fg_ratio', 0.0)),
                        'eval_mean_fg_prob': float(getattr(self, '_last_mean_fg_prob', 0.0)),
                        'eval_selected_fg_threshold': float(getattr(self, '_last_selected_fg_threshold', self.best_fg_threshold)),
                        'stage': stage_this_epoch,
                        'stage_eval_metric': float(eval_score),
                    })
                    if getattr(self, '_last_threshold_scan', None):
                        self.threshold_scan_history.append({
                            'epoch': int(epoch + 1),
                            'scan': copy.deepcopy(self._last_threshold_scan),
                        })

                    miou_str = ''
                    if hasattr(self, '_last_miou'):
                        miou_str = (
                            f" mIoU(all/fg): "
                            f"{getattr(self, '_last_miou_all', self._last_miou):6.2f}%/"
                            f"{getattr(self, '_last_miou_fg', self._last_miou):6.2f}%"
                        )

                    oa_str = f' OA: {self._last_oa:6.2f}%' if hasattr(self, '_last_oa') else ''
                    eval_set_name = 'Val' if self.val_loader else 'Test'
                    tprint(f"\n[Epoch {epoch+1:3d}] "
                          f"Train(bgfg/fg_cls): {train_bgfg_loss:.4f}/{train_fg_class_loss:.4f} Acc: {train_acc:6.2f}% | "
                          f"{eval_set_name}(bgfg/fg_cls): {getattr(self, '_last_eval_bgfg_loss', 0.0):.4f}/{getattr(self, '_last_eval_fg_class_loss', 0.0):.4f} "
                          f"BA(all/fg): {getattr(self, '_last_ba_all', eval_acc):6.2f}%/"
                          f"{getattr(self, '_last_ba_fg', eval_acc):6.2f}%{oa_str} "
                          f"Kappa(all/fg): {getattr(self, '_last_kappa_all', kappa):6.2f}%/"
                          f"{getattr(self, '_last_kappa_fg', kappa):6.2f}% "
                          f"FG(P/R/D): {getattr(self, '_last_fg_precision', 0.0):6.2f}%/"
                          f"{getattr(self, '_last_fg_recall', 0.0):6.2f}%/"
                          f"{getattr(self, '_last_fg_dice', 0.0):6.2f}% "
                          f"BA(bgfg): {getattr(self, '_last_ba_bgfg', 0.0):6.2f}% "
                          f"BA(boundary): {getattr(self, '_last_ba_bgfg_boundary', 0.0):6.2f}% "
                          f"FGminR: {getattr(self, '_last_fg_min_recall', 0.0):6.2f}% "
                          f"FGDiag(pred/gt/prob): {getattr(self, '_last_pred_fg_ratio', 0.0):6.2f}%/"
                          f"{getattr(self, '_last_gt_fg_ratio', 0.0):6.2f}%/"
                          f"{getattr(self, '_last_mean_fg_prob', 0.0):6.2f}% "
                          f"Stage: {stage_this_epoch} "
                          f"thr={getattr(self, '_last_selected_fg_threshold', self.best_fg_threshold):.2f} "
                          f"Score: {eval_score:6.2f}%{miou_str}")

                    if stage_this_epoch == 'C':
                        self._register_topk_model(epoch=epoch, score=eval_score)

                    # save the best model
                    if stage_this_epoch == 'C' and eval_score > self.best_acc:
                        self.best_acc = eval_score
                        self.best_epoch = epoch
                        self.best_model_state = {
                            'epoch': epoch,
                            'model_state': self.ema.state_dict(),
                            'acc': eval_score,
                            'kappa': kappa
                        }
                        self._save_model()
                        tprint(f"  Best model saved (score: {eval_score:.2f}%) at {os.path.join(self.output, 'models', f'{self.model_name}_best.onnx')}")
                    else:
                        # early stopping check
                        if stage_this_epoch == 'C' and epoch - self.best_epoch > self.patience:
                            tprint(f"\n  Early stopping: {epoch - self.best_epoch} epochs without improvement")
                            break
                else:
                    tprint(f"[Epoch {epoch+1:3d}] Train(bgfg/fg_cls): {train_bgfg_loss:.4f}/{train_fg_class_loss:.4f} Acc: {train_acc:6.2f}%", end='')
                    self.epoch_metrics.append({
                        'epoch': epoch + 1,
                        'train_loss': float(train_loss),
                        'train_bgfg_loss': float(train_bgfg_loss),
                        'train_fg_class_loss': float(train_fg_class_loss),
                        'train_acc': float(train_acc),
                        'eval_loss': None,
                        'eval_bgfg_loss': None,
                        'eval_fg_class_loss': None,
                        'eval_score': None,
                        'eval_ba_all': None,
                        'eval_ba_fg': None,
                        'eval_kappa_all': None,
                        'eval_kappa_fg': None,
                        'eval_miou_all': None,
                        'eval_miou_fg': None,
                        'eval_fg_precision': None,
                        'eval_fg_recall': None,
                        'eval_fg_dice': None,
                        'eval_ba_bgfg': None,
                        'eval_ba_bgfg_boundary': None,
                        'eval_fg_min_recall': None,
                        'eval_fg_mean_recall': None,
                        'eval_pred_fg_ratio': None,
                        'eval_gt_fg_ratio': None,
                        'eval_mean_fg_prob': None,
                        'eval_selected_fg_threshold': None,
                        'stage': stage_this_epoch,
                        'stage_eval_metric': None,
                    })

                self._flush_epoch_metrics()

                # Generate visualizations
                if self.debug_mode:
                    try:
                        self._generate_cam(epoch)
                    except Exception as e:
                        tprint(f"  CAM error: {e}")

            toc = time.perf_counter()
            training_time = toc - tic
        
        print("\n")
        tprint(f"Training completed with:")
        print(f"  Best epoch: {self.best_epoch + 1} (BA: {self.best_acc:.2f}%)")
        print(f"  Total time: {training_time:.2f}s")
        print("\n")
        
        # load best model for final evaluation
        if self.best_model_state is not None:
            self._reselect_best_from_topk()
            tprint(f"Loading model from epoch {self.best_epoch + 1} for final eval...")
            self.model.load_state_dict(self.best_model_state['model_state'])
        else:
            # Fallback when stage-C did not run long enough to produce a checkpoint.
            self.best_epoch = max(0, len(self.train_losses) - 1)
            self.best_acc = float(self._current_stage_metric('C'))
            self.best_model_state = {
                'epoch': self.best_epoch,
                'model_state': self.ema.state_dict(),
                'acc': self.best_acc,
                'kappa': float(getattr(self, '_last_kappa_fg', 0.0)),
            }
            self._save_model()
        
        # Final evaluation on held-out TEST set (not val set) with full data collection
        # This is the only time test set is used, ensuring unbiased generalization estimate
        final_loss, final_acc, final_kappa, final_pred, final_target = self.evaluate(
            collect_extra=True, loader=self.test_loader)
        self._last_ba = final_acc
        tprint(f"Final eval on TEST set completed (BA: {final_acc:.2f}%)")
        
        # Store for CV aggregation (accessed before trainer is deleted)
        self._final_pred = final_pred
        self._final_target = final_target
        
        # single-model visualizations, skip in CV mode for aggregated plots
        if not _cv_mode:
            self._save_full_reconstruction_maps()
            self._save_prediction_label_comparison()
            self._save_all_plots(final_target, final_pred)
        
        results = {
            'best_epoch': self.best_epoch + 1,
            'best_accuracy': self.best_acc,
            'final_accuracy': final_acc,
            'final_kappa': final_kappa,
            'training_time': training_time,
            'model_path': os.path.join(self.output, 'models', f'{self.model_name}_best.onnx'),
            'selected_fg_threshold': float(getattr(self, '_last_selected_fg_threshold', self.best_fg_threshold)),
        }
        results['final_accuracy_all'] = float(getattr(self, '_last_ba_all', final_acc))
        results['final_accuracy_fg'] = float(getattr(self, '_last_ba_fg', final_acc))
        results['final_kappa_all'] = float(getattr(self, '_last_kappa_all', final_kappa))
        results['final_kappa_fg'] = float(getattr(self, '_last_kappa_fg', final_kappa))
        results['final_miou_all'] = float(getattr(self, '_last_miou_all', 0.0))
        results['final_miou_fg'] = float(getattr(self, '_last_miou_fg', 0.0))
        # Always include mIoU
        if hasattr(self, '_last_miou'):
            results['final_miou'] = self._last_miou
        return results

    #  Cross-Validation
    @staticmethod
    def cross_validate(
        config: Munch,
        dataLoader: AbstractHSDataset,
        n_folds: int,
        epochs: int,
        model: Callable[..., Module],
        model_name: str = "model",
        num_gpus: int = 1,
        debug_mode: bool = False,
        _fold_loaders_override=None,
    ) -> Dict[str, float]:
        """
        Grouped K-Fold cross-validation with patient-level grouping.

        Data is loaded and preprocessed ONCE by ``dataLoader``; only the
        train/test index split changes per fold.  Each fold trains a fresh
        model from scratch (new weights, optimizer, EMA, etc.).

        The return dict is **compatible** with the single-train ``train()``
        return format so that ``AblationRunner`` can consume either one
        without changes to its result-gathering logic.

        Args:
            config: Global Munch config.
            dataLoader: An already-loaded ``NpyHSDataset``.
            n_folds: Number of CV folds (typically 5).
            epochs: Training epochs per fold.
            model: **Callable** that returns a fresh ``nn.Module``.
            model_name: Name prefix for output dirs.
            num_gpus: GPU count.
            debug_mode: Enable per-epoch CAM / viz.

        Returns:
            Dict with aggregated mean/std results plus per-fold details.
        """
        cv_output = os.path.join(config.path.output, f'{model_name}')
        os.makedirs(cv_output, exist_ok=True)

        num_workers = config.memory.num_workers
        batch_size = config.split.batch_size
        prefetch_factor = 2
        persistent_workers = False

        if num_gpus > 1:
            available_cpus = os.cpu_count() or 4
            num_workers = min(available_cpus // 2, 8, num_gpus * 2)
            batch_size = config.split.batch_size * num_gpus
            prefetch_factor = config.memory.prefetch_factor
            persistent_workers = True

        print("\n")
        tprint(f"Initializing {n_folds}-Fold Grouped Cross-Validation: {model_name}")
        if _fold_loaders_override is None:
            print(f"  Total patches: {len(dataLoader)}, "
                  f"Unique patients: {dataLoader._num_patients}")
        print("\n")

        if _fold_loaders_override is not None:
            fold_loaders = _fold_loaders_override
            n_folds = len(fold_loaders)
            print(f"  Using injected fold loaders ({n_folds} folds)")
        else:
            cv_loader_fn = getattr(dataLoader, 'get_cv_loaders', None) \
                            or getattr(dataLoader, 'create_cv_data_loaders', None)
            if cv_loader_fn is None:
                raise RuntimeError("Dataset object does not expose get_cv_loaders/create_cv_data_loaders")
            fold_loaders = cv_loader_fn(
                n_folds=n_folds,
                num_workers=num_workers,
                batch_size=batch_size,
                pin_memory=config.memory.pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
            )

        num_classes = config.clsf.num
        metric_labels = hsTrainer._metric_labels_from_config(config, num_classes)
        eval_num_classes = len(metric_labels)
        class_names = [config.clsf.targets[i] for i in metric_labels if i < len(config.clsf.targets)]
        common_fpr = np.linspace(0, 1, 200)

        fold_results = []
        fold_curves = []     # lightweight: only epoch-level scalars per fold
        all_accs = []
        all_kappas = []
        all_mious = []
        all_train_accs = []
        total_time = 0.0

        # Incremental accumulators for mean/std of confusion matrix, precision, recall, F1, ROC curves
        sum_cm_norm = np.zeros((eval_num_classes, eval_num_classes), dtype=np.float64)
        sum_cm_norm_sq = np.zeros((eval_num_classes, eval_num_classes), dtype=np.float64)
        fold_precisions = []   # list of (eval_num_classes,) arrays
        fold_recalls = []
        fold_f1s = []
        roc_tprs = [[] for _ in range(eval_num_classes)]  # interpolated TPR per class
        roc_aucs = [[] for _ in range(eval_num_classes)]

        for fold_idx, (f_train_loader, f_test_loader) in enumerate(fold_loaders):
            print("\n")
            tprint(f"Performing Fold {fold_idx + 1}/{n_folds}")
            print("\n")

            fold_name = f"{model_name}_fold{fold_idx + 1}"

            # Per-fold config: redirect output into the CV directory
            fold_config = copy.deepcopy(config)
            fold_config.path.output = cv_output

            # Create a fresh trainer with injected loaders (skips _load_data)
            trainer = hsTrainer(
                config=fold_config,
                dataLoader=dataLoader,
                epochs=epochs,
                model=model,            # callable -> _create_model calls it
                model_name=fold_name,
                debug_mode=debug_mode,
                num_gpus=num_gpus,
                train_loader=f_train_loader,
                test_loader=f_test_loader,
            )

            trainer.fit(_cv_mode=True)
            result = trainer.results_

            # Scalar metrics
            fold_results.append(result)
            all_accs.append(result['final_accuracy'])
            all_kappas.append(result['final_kappa'])
            if 'final_miou' in result:
                all_mious.append(result['final_miou'])

            best_idx = trainer.best_epoch
            if best_idx < len(trainer.train_accs):
                all_train_accs.append(trainer.train_accs[best_idx])
            else:
                all_train_accs.append(
                    trainer.train_accs[-1] if trainer.train_accs else 0.0)

            total_time += result['training_time']

            # curve data
            fold_curves.append({
                'train_losses': list(trainer.train_losses),
                'train_accs': list(trainer.train_accs),
                'eval_losses': list(trainer.eval_losses),
                'eval_accs': list(trainer.eval_accs),
                'grad_norms': list(trainer.grad_norms),
                'eval_interval': trainer.eval_interval,
            })

            # classification summaries
            pred = trainer._final_pred
            target = trainer._final_target

            cm = _cm(target, pred, labels=metric_labels)
            cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-8)
            sum_cm_norm += cm_norm
            sum_cm_norm_sq += cm_norm ** 2

            p, r, f1, _ = precision_recall_fscore_support(
                target, pred, labels=metric_labels, zero_division=0)
            fold_precisions.append(p)
            fold_recalls.append(r)
            fold_f1s.append(f1)

            # release output label memory
            if hasattr(trainer, '_last_probas') and trainer._last_probas is not None:
                probas = trainer._last_probas
                for local_idx, cls_i in enumerate(metric_labels):
                    y_bin = (target == cls_i).astype(int)
                    if y_bin.sum() > 0:
                        fpr_arr, tpr_arr, _ = roc_curve(y_bin, probas[:, cls_i])
                        roc_tprs[local_idx].append(
                            np.interp(common_fpr, fpr_arr, tpr_arr))
                        roc_aucs[local_idx].append(auc(fpr_arr, tpr_arr))
                del probas

            del pred, target

            # release trainer (model, EMA, optimizer, heavy arrays) memory
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                tprint('cuda.empty_cache() performing...')
                torch.cuda.empty_cache()
                tprint('cuda.empty_cache() performed!')

        accs = np.array(all_accs)
        kappas = np.array(all_kappas)
        train_accs_arr = np.array(all_train_accs)
        overfit_gaps = np.maximum(train_accs_arr - accs, 0.0)

        best_fold_idx = int(np.argmax(accs))

        summary = {
            # Compatible with single-train return dict
            'best_epoch': int(np.mean([r['best_epoch'] for r in fold_results])),
            'best_accuracy': float(np.max(accs)),
            'final_accuracy': float(np.mean(accs)),
            'final_kappa': float(np.mean(kappas)),
            'training_time': total_time,
            'model_path': fold_results[best_fold_idx].get('model_path', ''),
            'output_dir': cv_output,
            # Cross-validation specifics
            'cv_folds': n_folds,
            'cv_accuracy_mean': float(np.mean(accs)),
            'cv_accuracy_std': float(np.std(accs)),
            'cv_kappa_mean': float(np.mean(kappas)),
            'cv_kappa_std': float(np.std(kappas)),
            'cv_overfit_gap_mean': float(np.mean(overfit_gaps)),
            'cv_train_acc_mean': float(np.mean(train_accs_arr)),
            'cv_fold_accuracies': accs.tolist(),
            'cv_fold_kappas': kappas.tolist(),
        }

        if all_mious:
            mious = np.array(all_mious)
            summary['final_miou'] = float(np.mean(mious))
            summary['cv_miou_mean'] = float(np.mean(mious))
            summary['cv_miou_std'] = float(np.std(mious))

        # summary
        print(f"\n")
        print(f"  Cross-Validation Summary ({n_folds} folds): {model_name}")
        print(f"  Accuracy: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
        print(f"  Kappa:    {np.mean(kappas):.2f}% ± {np.std(kappas):.2f}%")
        if all_mious:
            print(f"  mIoU:     {np.mean(all_mious):.2f}% ± {np.std(all_mious):.2f}%")
        print(f"  Gap:      {np.mean(overfit_gaps):.2f}%")
        for i, r in enumerate(fold_results):
            miou_s = (f" mIoU:{r.get('final_miou', 0):.1f}%"
                      if 'final_miou' in r else "")
            print(f"    Fold {i+1}: Acc={r['final_accuracy']:.2f}% "
                  f"Kappa={r['final_kappa']:.2f}%{miou_s} "
                  f"({r['training_time']:.0f}s)")
        print(f"  Total time: {total_time:.0f}s")
        print(f"\n")

        # save summary JSON
        summary_path = os.path.join(cv_output, f'{model_name}_cv_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        tprint(f"  Cross validation summary saved to {summary_path}")

        # generate all plots in parallel, fast
        mious_arr = np.array(all_mious) if all_mious else None

        # Derive mean/std confusion matrix from running sums
        mean_cm = sum_cm_norm / n_folds
        std_cm = np.sqrt(np.maximum(
            sum_cm_norm_sq / n_folds - mean_cm ** 2, 0.0))

        plot_tasks = [
            (hsTrainer._plot_summary,
             (accs, kappas, mious_arr, model_name, cv_output)),
            (hsTrainer._plot_training_curves,
             (fold_curves, model_name, cv_output)),
            (hsTrainer._plot_confusion_matrix,
                         (mean_cm, std_cm, eval_num_classes, class_names,
              model_name, cv_output)),
            (hsTrainer._plot_per_class_metrics,
             (fold_precisions, fold_recalls, fold_f1s,
                            eval_num_classes, class_names, model_name, cv_output)),
            (hsTrainer._plot_roc_curves,
             (roc_tprs, roc_aucs, common_fpr,
                            eval_num_classes, class_names, model_name, cv_output)),
            (hsTrainer._plot_grad_norms,
             (fold_curves, model_name, cv_output)),
        ]

        t0 = time.perf_counter()
        max_workers = min(len(plot_tasks), (os.cpu_count() or 4))
        with _managed_pool(max_workers, "CV plot workers") as pool:
            futures = {pool.submit(fn, *args): fn.__name__
                       for fn, args in plot_tasks}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    print(f"  Warning: {futures[fut]} failed: {exc}")
        tprint(f"  CV plots generated in {time.perf_counter() - t0:.1f}s "
              f"({max_workers} workers)")

        return summary

    @staticmethod
    def _plot_summary(accs, kappas, mious, model_name, output_dir):
        """Bar chart of per-fold Accuracy and Kappa with mean±std lines."""
        n = len(accs)
        x = np.arange(n)

        ncols = 3 if mious is not None else 2
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

        # Accuracy
        axes[0].bar(x, accs, color='#2196F3', alpha=0.85,
                    edgecolor='k', linewidth=0.5)
        axes[0].axhline(y=np.mean(accs), color='r', linestyle='--',
                        label=f'Mean: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%')
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title(f'{model_name} — CV Accuracy')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f'F{i+1}' for i in range(n)])
        axes[0].legend(fontsize=9)
        axes[0].grid(axis='y', alpha=0.3)

        # Kappa
        axes[1].bar(x, kappas, color='#4CAF50', alpha=0.85,
                    edgecolor='k', linewidth=0.5)
        axes[1].axhline(y=np.mean(kappas), color='r', linestyle='--',
                        label=f'Mean: {np.mean(kappas):.2f}% ± {np.std(kappas):.2f}%')
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('Kappa (%)')
        axes[1].set_title(f'{model_name} — CV Kappa')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'F{i+1}' for i in range(n)])
        axes[1].legend(fontsize=9)
        axes[1].grid(axis='y', alpha=0.3)

        # mIoU (optional)
        if mious is not None:
            axes[2].bar(x, mious, color='#FF5722', alpha=0.85,
                        edgecolor='k', linewidth=0.5)
            axes[2].axhline(y=np.mean(mious), color='r', linestyle='--',
                            label=f'Mean: {np.mean(mious):.2f}% ± {np.std(mious):.2f}%')
            axes[2].set_xlabel('Fold')
            axes[2].set_ylabel('mIoU (%)')
            axes[2].set_title(f'{model_name} — CV mIoU')
            axes[2].set_xticks(x)
            axes[2].set_xticklabels([f'F{i+1}' for i in range(n)])
            axes[2].legend(fontsize=9)
            axes[2].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, f'{model_name}_cv_metrics.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        tprint(f"  CV metrics plot saved to {path}")

    # Aggregated CV Visualizations (mean ± bounds)
    @staticmethod
    def _plot_training_curves(fold_curves, model_name, output_dir):
        """Training loss/accuracy averaged across CV folds with min-max shading.

        Handles folds of different lengths (due to early stopping) by
        padding shorter sequences with NaN and using ``np.nanmean``, etc.
        """
        max_epochs = max(len(fd['train_losses']) for fd in fold_curves)

        def _pad(lists, length):
            out = np.full((len(lists), length), np.nan)
            for i, lst in enumerate(lists):
                out[i, :len(lst)] = lst
            return out

        train_losses = _pad([fd['train_losses'] for fd in fold_curves], max_epochs)
        train_accs = _pad([fd['train_accs'] for fd in fold_curves], max_epochs)
        max_eval_pts = max(len(fd['eval_losses']) for fd in fold_curves)
        eval_losses = _pad([fd['eval_losses'] for fd in fold_curves], max_eval_pts)
        eval_accs = _pad([fd['eval_accs'] for fd in fold_curves], max_eval_pts)

        epochs = np.arange(max_epochs)
        eval_interval = fold_curves[0].get('eval_interval', 4)
        eval_epochs = [e for e in range(max_epochs)
                       if (e + 1) % eval_interval == 0 or e + 1 == max_epochs]
        eval_epochs = eval_epochs[:max_eval_pts]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # loss
        mean_tl = np.nanmean(train_losses, axis=0)
        axes[0].plot(epochs, mean_tl, color='#2196F3', lw=2, label='Train (mean)')
        axes[0].fill_between(epochs, np.nanmin(train_losses, 0),
                             np.nanmax(train_losses, 0),
                             color='#2196F3', alpha=0.15, label='Train (min–max)')
        if eval_epochs:
            mean_el = np.nanmean(eval_losses, axis=0)[:len(eval_epochs)]
            axes[0].plot(eval_epochs, mean_el, color='#FF5722', lw=2,
                         label='Test (mean)')
            axes[0].fill_between(eval_epochs,
                                 np.nanmin(eval_losses, 0)[:len(eval_epochs)],
                                 np.nanmax(eval_losses, 0)[:len(eval_epochs)],
                                 color='#FF5722', alpha=0.15,
                                 label='Test (min-max)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{model_name} — CV Loss (mean ± range)')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # acc
        mean_ta = np.nanmean(train_accs, axis=0)
        axes[1].plot(epochs, mean_ta, color='#2196F3', lw=2, label='Train (mean)')
        axes[1].fill_between(epochs, np.nanmin(train_accs, 0),
                             np.nanmax(train_accs, 0),
                             color='#2196F3', alpha=0.15, label='Train (min-max)')
        if eval_epochs:
            mean_ea = np.nanmean(eval_accs, axis=0)[:len(eval_epochs)]
            axes[1].plot(eval_epochs, mean_ea, color='#FF5722', lw=2,
                         label='Test (mean)')
            axes[1].fill_between(eval_epochs,
                                 np.nanmin(eval_accs, 0)[:len(eval_epochs)],
                                 np.nanmax(eval_accs, 0)[:len(eval_epochs)],
                                 color='#FF5722', alpha=0.15,
                                 label='Test (min-max)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title(f'{model_name} — CV Accuracy (mean ± range)')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, f'{model_name}_cv_training_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        tprint(f"  CV training curves saved to {path}")

    @staticmethod
    def _plot_confusion_matrix(mean_cm, std_cm, num_classes, class_names,
                                  model_name, output_dir):
        """Plot averaged normalized confusion matrix from pre-computed mean/std."""
        annot = np.empty_like(mean_cm, dtype=object)
        for i in range(num_classes):
            for j in range(num_classes):
                annot[i, j] = f'{mean_cm[i, j]:.1%}\n±{std_cm[i, j]:.1%}'

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(mean_cm, annot=annot, fmt='', cmap='Greens', ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0, vmax=1)
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
        ax.set_title(f'{model_name} — CV Confusion Matrix (mean ± std)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir,
                            f'{model_name}_cv_confusion_matrix.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        tprint(f"  CV confusion matrix saved to {path}")

    @staticmethod
    def _plot_per_class_metrics(fold_precisions, fold_recalls, fold_f1s,
                                   num_classes, class_names, model_name,
                                   output_dir):
        """Per-class P/R/F1 from pre-computed per-fold arrays with min-max error bars."""
        all_p = np.array(fold_precisions)   # (K, num_classes)
        all_r = np.array(fold_recalls)
        all_f1 = np.array(fold_f1s)
        mean_p, mean_r, mean_f1 = all_p.mean(0), all_r.mean(0), all_f1.mean(0)

        x = np.arange(num_classes)
        w = 0.25
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - w, mean_p, w, label='Precision', color='#2196F3', alpha=0.85,
               yerr=[mean_p - all_p.min(0), all_p.max(0) - mean_p],
               capsize=3, error_kw={'linewidth': 1})
        ax.bar(x, mean_r, w, label='Recall', color='#FF5722', alpha=0.85,
               yerr=[mean_r - all_r.min(0), all_r.max(0) - mean_r],
               capsize=3, error_kw={'linewidth': 1})
        ax.bar(x + w, mean_f1, w, label='F1-Score', color='#4CAF50', alpha=0.85,
               yerr=[mean_f1 - all_f1.min(0), all_f1.max(0) - mean_f1],
               capsize=3, error_kw={'linewidth': 1})

        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Score')
        ax.set_title(f'{model_name} — CV Per-Class Metrics '
                     f'(mean, error bars = range)',
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir,
                            f'{model_name}_cv_per_class_metrics.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        tprint(f"  CV per-class metrics saved to {path}")

    @staticmethod
    def _plot_roc_curves(roc_tprs, roc_aucs, common_fpr,
                            num_classes, class_names, model_name, output_dir):
        """Per-class ROC from pre-interpolated TPR arrays with min-max bands."""
        has_any = any(len(tprs) > 0 for tprs in roc_tprs)
        if not has_any:
            return

        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
        fig, ax = plt.subplots(figsize=(8, 7))

        for cls_i in range(num_classes):
            if not roc_tprs[cls_i]:
                continue
            tprs = np.array(roc_tprs[cls_i])
            mean_auc_val = np.mean(roc_aucs[cls_i])
            ax.plot(common_fpr, tprs.mean(0), color=colors[cls_i], lw=1.5,
                    label=f'{class_names[cls_i]} '
                          f'(AUC={mean_auc_val:.3f}±{np.std(roc_aucs[cls_i]):.3f})')
            ax.fill_between(common_fpr, tprs.min(0), tprs.max(0),
                            color=colors[cls_i], alpha=0.1)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} — CV ROC Curves (mean ± range)',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f'{model_name}_cv_roc_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        tprint(f"  CV ROC curves saved to {path}")

    @staticmethod
    def _plot_grad_norms(fold_curves, model_name, output_dir):
        """Gradient L2 norms averaged across CV folds with min-max band."""
        max_len = max(len(fd['grad_norms']) for fd in fold_curves)
        padded = np.full((len(fold_curves), max_len), np.nan)
        for i, fd in enumerate(fold_curves):
            padded[i, :len(fd['grad_norms'])] = fd['grad_norms']

        epochs = np.arange(max_len)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, np.nanmean(padded, 0), color='#FF5722', lw=2,
                label='Mean')
        ax.fill_between(epochs, np.nanmin(padded, 0), np.nanmax(padded, 0),
                        color='#FF5722', alpha=0.15, label='Min-Max')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient L2 Norm')
        ax.set_title(f'{model_name} — CV Gradient Norm (mean ± range)',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir,
                            f'{model_name}_cv_gradient_norm.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        tprint(f"  CV gradient norm saved to {path}")

    # for visual reconstructions.
    def _save_prediction_label_comparison(self) -> None:
        """Save side-by-side prediction/label map images from final eval."""
        vis_pairs = getattr(self, '_last_vis_pairs', None)
        if not vis_pairs:
            return

        num_classes = int(self.config.clsf.num)
        max_items = int(getattr(self.config.common, 'vis_samples', 8))
        num_items = min(len(vis_pairs), max_items)
        if num_items <= 0:
            return

        out_dir = os.path.join(self.output, 'reconstruction')
        os.makedirs(out_dir, exist_ok=True)

        fig, axes = plt.subplots(num_items, 2, figsize=(8, 3.5 * num_items))
        if num_items == 1:
            axes = np.expand_dims(axes, axis=0)

        cmap = plt.cm.get_cmap('tab20', num_classes)
        for idx in range(num_items):
            pred_map, label_map = vis_pairs[idx]
            pred_show = pred_map.astype(np.float32).copy()
            label_show = label_map.astype(np.float32).copy()
            ignore_mask = (label_show == 255)
            pred_show[ignore_mask] = np.nan
            label_show[ignore_mask] = np.nan

            axes[idx, 0].imshow(pred_show, cmap=cmap, vmin=0, vmax=max(num_classes - 1, 0))
            axes[idx, 0].set_title(f'Predicted #{idx + 1}')
            axes[idx, 0].axis('off')

            axes[idx, 1].imshow(label_show, cmap=cmap, vmin=0, vmax=max(num_classes - 1, 0))
            axes[idx, 1].set_title(f'Label #{idx + 1}')
            axes[idx, 1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(out_dir, 'cmp_pred_label.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        tprint(f"  Prediction-vs-label plot saved to {save_path}")

    def _resolve_dataset_and_indices(self, loader):
        """Resolve base dataset and subset indices from a possibly wrapped loader dataset."""
        if loader is None or not hasattr(loader, 'dataset'):
            return None, None

        ds = loader.dataset
        indices = None
        visited = set()
        while ds is not None and id(ds) not in visited:
            visited.add(id(ds))
            if hasattr(ds, 'indices') and indices is None:
                try:
                    indices = np.asarray(ds.indices)
                except Exception:
                    indices = None
            if hasattr(ds, 'dataset'):
                ds = ds.dataset
            else:
                break
        return ds, indices

    def _init_reconstruction_state(self, loader):
        """Initialize full-image reconstruction state from loader metadata."""
        base_ds, subset_indices = self._resolve_dataset_and_indices(loader)
        if base_ds is None:
            return None

        # Cached chunk dataset: reconstruct by chunk metadata directly.
        if hasattr(base_ds, 'chunks') and hasattr(base_ds, 'samples'):
            samples = list(getattr(base_ds, 'samples', []))
            if not samples:
                return None

            pred_maps = []
            label_maps = []
            patient_names = []
            for i, sample in enumerate(samples):
                h = int(sample.get('height', 0))
                w = int(sample.get('width', 0))
                if h <= 0 or w <= 0:
                    continue
                pred_maps.append(np.full((h, w), 255, dtype=np.int32))
                label_maps.append(np.full((h, w), 255, dtype=np.int32))
                patient_names.append(str(sample.get('name', f'sample_{i:03d}')))

            if not pred_maps:
                return None

            return {
                'enabled': True,
                'mode': 'cached_chunk',
                'base_ds': base_ds,
                'subset_indices': subset_indices,
                'sample_ptr': 0,
                'margin': int(getattr(base_ds, 'margin', 0)),
                'pred_maps': pred_maps,
                'label_maps': label_maps,
                'patient_names': patient_names,
                'patch_counts': np.zeros(len(pred_maps), dtype=np.int64),
            }

        if not hasattr(base_ds, 'patch_indices'):
            return None

        margin = int(getattr(base_ds, 'margin', 0))
        patch_indices = base_ds.patch_indices
        if len(patch_indices) == 0:
            return None

        # Multi-patient NpyHSDataset: patch_indices is (N, 3) => (patient, r, c)
        if hasattr(base_ds, '_patient_padded_labels') and patch_indices.shape[1] == 3:
            pred_maps = []
            label_maps = []
            patient_names = []
            for p_idx, padded_lbl in enumerate(base_ds._patient_padded_labels):
                if margin > 0:
                    h = padded_lbl.shape[0] - 2 * margin
                    w = padded_lbl.shape[1] - 2 * margin
                else:
                    h, w = padded_lbl.shape
                pred_maps.append(np.full((h, w), 255, dtype=np.int32))
                label_maps.append(np.full((h, w), 255, dtype=np.int32))
                if hasattr(base_ds, '_patient_names'):
                    patient_names.append(str(base_ds._patient_names.get(p_idx, f'patient_{p_idx:02d}')))
                else:
                    patient_names.append(f'patient_{p_idx:02d}')

            return {
                'enabled': True,
                'mode': 'multi_patient',
                'base_ds': base_ds,
                'subset_indices': subset_indices,
                'sample_ptr': 0,
                'margin': margin,
                'pred_maps': pred_maps,
                'label_maps': label_maps,
                'patient_names': patient_names,
                'patch_counts': np.zeros(len(pred_maps), dtype=np.int64),
            }

        # Single-image dataset: patch_indices is (N, 2) => (r, c)
        if hasattr(base_ds, 'raw_labels') and patch_indices.shape[1] == 2:
            h, w = base_ds.raw_labels.shape
            return {
                'enabled': True,
                'mode': 'single_image',
                'base_ds': base_ds,
                'subset_indices': subset_indices,
                'sample_ptr': 0,
                'margin': margin,
                'pred_maps': [np.full((h, w), 255, dtype=np.int32)],
                'label_maps': [np.full((h, w), 255, dtype=np.int32)],
                'patient_names': ['full_image'],
                'patch_counts': np.zeros(1, dtype=np.int64),
            }

        return None

    def _accumulate_reconstruction_batch(self, recon_state, predicted, labels, batch_meta=None) -> None:
        """Accumulate center-pixel predictions into full reconstructed maps."""
        if not recon_state or not recon_state.get('enabled', False):
            return

        if recon_state.get('mode') == 'cached_chunk':
            if not isinstance(batch_meta, (list, tuple)) or len(batch_meta) == 0:
                return

            center_r = int(predicted.shape[1] // 2)
            center_c = int(predicted.shape[2] // 2)
            pred_center = predicted[:, center_r, center_c].detach().cpu().numpy().astype(np.int32, copy=False)
            label_center = labels[:, center_r, center_c].detach().cpu().numpy().astype(np.int32, copy=False)

            ptr = 0
            for meta in batch_meta:
                if not isinstance(meta, dict):
                    continue
                n = int(meta.get('count', 0))
                if n <= 0:
                    continue

                stop = min(ptr + n, pred_center.shape[0])
                n_eff = stop - ptr
                if n_eff <= 0:
                    break

                sample_idx = int(meta.get('sample_idx', -1))
                start = int(meta.get('start', 0))
                width = int(meta.get('width', 0))
                if sample_idx < 0 or sample_idx >= len(recon_state['pred_maps']) or width <= 0:
                    ptr = stop
                    continue

                centers = np.arange(start, start + n_eff, dtype=np.int64)
                rr = centers // width
                cc = centers % width

                pred_chunk = pred_center[ptr:stop]
                label_chunk = label_center[ptr:stop]
                h, w = recon_state['pred_maps'][sample_idx].shape
                inside = (rr >= 0) & (cc >= 0) & (rr < h) & (cc < w)
                if np.any(inside):
                    r_in = rr[inside]
                    c_in = cc[inside]
                    recon_state['pred_maps'][sample_idx][r_in, c_in] = pred_chunk[inside]
                    recon_state['label_maps'][sample_idx][r_in, c_in] = label_chunk[inside]
                    recon_state['patch_counts'][sample_idx] += int(inside.sum())

                ptr = stop
            return

        base_ds = recon_state['base_ds']
        patch_indices = base_ds.patch_indices
        subset_indices = recon_state['subset_indices']
        margin = recon_state['margin']

        batch_size = int(predicted.shape[0])
        start = int(recon_state['sample_ptr'])
        stop = start + batch_size

        if subset_indices is not None:
            if start >= len(subset_indices):
                return
            stop = min(stop, len(subset_indices))
            original_idx = np.asarray(subset_indices[start:stop], dtype=np.int64)
        else:
            if start >= len(patch_indices):
                return
            stop = min(stop, len(patch_indices))
            original_idx = np.arange(start, stop, dtype=np.int64)

        n = int(original_idx.shape[0])
        if n <= 0:
            return

        patch_meta = patch_indices[original_idx]
        if patch_indices.shape[1] == 3:
            p_idx = patch_meta[:, 0].astype(np.int64, copy=False)
            r_pad = patch_meta[:, 1].astype(np.int64, copy=False)
            c_pad = patch_meta[:, 2].astype(np.int64, copy=False)
        else:
            p_idx = np.zeros(n, dtype=np.int64)
            r_pad = patch_meta[:, 0].astype(np.int64, copy=False)
            c_pad = patch_meta[:, 1].astype(np.int64, copy=False)

        rr = r_pad - margin
        cc = c_pad - margin
        pred_center = predicted[:n, margin, margin].detach().cpu().numpy().astype(np.int32, copy=False)
        label_center = labels[:n, margin, margin].detach().cpu().numpy().astype(np.int32, copy=False)

        valid = (rr >= 0) & (cc >= 0)
        if not np.any(valid):
            recon_state['sample_ptr'] = stop
            return

        p_idx = p_idx[valid]
        rr = rr[valid]
        cc = cc[valid]
        pred_center = pred_center[valid]
        label_center = label_center[valid]

        for pid in np.unique(p_idx):
            m = (p_idx == pid)
            r_sel = rr[m]
            c_sel = cc[m]
            h, w = recon_state['pred_maps'][pid].shape
            inside = (r_sel < h) & (c_sel < w)
            if not np.any(inside):
                continue
            r_in = r_sel[inside]
            c_in = c_sel[inside]
            recon_state['pred_maps'][pid][r_in, c_in] = pred_center[m][inside]
            recon_state['label_maps'][pid][r_in, c_in] = label_center[m][inside]
            recon_state['patch_counts'][pid] += int(inside.sum())

        recon_state['sample_ptr'] = stop

    def _save_full_reconstruction_maps(self) -> None:
        """Save reconstructed full-image prediction-vs-label maps with rich metadata."""
        recon = getattr(self, '_last_reconstruction', None)
        if not recon or not recon.get('enabled', False):
            tprint("  Full reconstruction skipped: no reconstruction state available")
            return

        out_dir = os.path.join(self.output, 'reconstruction')
        os.makedirs(out_dir, exist_ok=True)

        num_classes = int(self.config.clsf.num)
        class_names = list(self.config.clsf.targets[:num_classes])
        oa = float(getattr(self, '_last_oa', 0.0))
        score = float(getattr(self, '_last_ba', self.best_acc))
        ba_all = float(getattr(self, '_last_ba_all', score))
        ba_fg = float(getattr(self, '_last_ba_fg', score))
        miou = float(getattr(self, '_last_miou', 0.0))
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        cmap = plt.cm.get_cmap('tab20', num_classes)
        total_saved = 0
        summary_lines = []

        for p_idx, (pred_map, label_map) in enumerate(zip(recon['pred_maps'], recon['label_maps'])):
            patient_name = recon['patient_names'][p_idx]
            safe_patient = re.sub(r'[^A-Za-z0-9_.-]+', '_', patient_name)

            valid_mask = (label_map != 255)
            covered = int(valid_mask.sum())
            if covered == 0:
                continue
            correct = int(((pred_map == label_map) & valid_mask).sum())
            pixel_acc = 100.0 * correct / covered
            coverage_ratio = covered / float(pred_map.size)

            # NaN denotes uncovered pixels for clearer rendering.
            diff_map = np.full(pred_map.shape, np.nan, dtype=np.float32)
            diff_map[valid_mask] = (pred_map[valid_mask] != label_map[valid_mask]).astype(np.float32)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # For sparse-center reconstruction, fill uncovered display pixels by nearest valid neighbor
            # so structure remains readable. Metrics still use original covered pixels only.
            pred_show = self._dense_fill_for_display(pred_map, invalid_value=255)
            label_show = self._dense_fill_for_display(label_map, invalid_value=255)

            im0 = axes[0].imshow(pred_show, cmap=cmap, vmin=0, vmax=max(num_classes - 1, 0))
            axes[0].set_title('Prediction')
            axes[0].axis('off')

            axes[1].imshow(label_show, cmap=cmap, vmin=0, vmax=max(num_classes - 1, 0))
            axes[1].set_title('Label')
            axes[1].axis('off')

            diff_cmap = plt.cm.get_cmap('coolwarm').copy()
            diff_cmap.set_bad(color='lightgrey')
            axes[2].imshow(diff_map, cmap=diff_cmap, vmin=0, vmax=1)
            axes[2].set_title('Mismatch Map')
            axes[2].legend(handles=[
                plt.Line2D([0], [0], marker='s', color='w', label='covered-correct', markerfacecolor='grey', markersize=10),
                plt.Line2D([0], [0], marker='s', color='w', label='covered-wrong', markerfacecolor='red', markersize=10),
                plt.Line2D([0], [0], marker='s', color='w', label='uncovered', markerfacecolor='lightgrey', markersize=10)
            ], loc='upper right')
            axes[2].axis('off')

            cbar = fig.colorbar(im0, ax=axes[:2], fraction=0.028, pad=0.02)
            cbar.set_label('Class Index')

            title = (
                f"{self.model_name} | TEST Reconstruction | patient={patient_name} | "
                f"patches={int(recon['patch_counts'][p_idx])} | covered_px={covered} | "
                f"cover_ratio={coverage_ratio:.2%} | "
                f"pixel_acc={pixel_acc:.2f}% | OA={oa:.2f}% | "
                f"BA(all/fg)={ba_all:.2f}%/{ba_fg:.2f}% | "
                f"Score={score:.2f}% | mIoU={miou:.2f}%"
            )
            fig.suptitle(title, fontsize=11, fontweight='bold')
            plt.tight_layout()

            save_name = (
                f"recon_{safe_patient}_pxacc{pixel_acc:.2f}_oa{oa:.2f}_"
                f"miou{miou:.2f}_{timestamp}.png"
            )
            save_path = os.path.join(out_dir, save_name)
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            plt.close()

            summary_lines.append(
                f"patient={patient_name}, patches={int(recon['patch_counts'][p_idx])}, "
                f"covered_px={covered}, correct_px={correct}, pixel_acc={pixel_acc:.4f}, "
                f"OA={oa:.4f}, BA_all={ba_all:.4f}, BA_fg={ba_fg:.4f}, "
                f"Score={score:.4f}, mIoU={miou:.4f}, file={save_name}"
            )
            total_saved += 1

        if total_saved == 0:
            tprint("  Full reconstruction skipped: no valid labeled pixels found")
            return

        summary_path = os.path.join(out_dir, f'reconstruction_summary_{timestamp}.txt')
        with open(summary_path, 'w') as f:
            f.write(f"model={self.model_name}\n")
            f.write(f"timestamp={timestamp}\n")
            f.write(f"classes={class_names}\n")
            f.write(
                f"OA={oa:.4f}, BA_all={ba_all:.4f}, BA_fg={ba_fg:.4f}, "
                f"Score={score:.4f}, mIoU={miou:.4f}\n"
            )
            f.write("\n".join(summary_lines))

        tprint(f"  Full reconstruction plots saved: {total_saved} file(s) -> {out_dir}")

    @torch.no_grad()
    def _save_bgfg_split_snapshot(self, epoch: int, metric: float, reason: str) -> None:
        """Save random whole-image FG/BG split maps when stage changes A -> B."""
        recon_state = getattr(self, '_last_transition_reconstruction', None)

        if not recon_state or not recon_state.get('enabled', False):
            # Fallback path: capped and optional to avoid long pause during stage transition.
            if not bool(getattr(self.config.common, 'bgfgsplit_allow_fallback_forward', False)):
                tprint("  BGFGsplit skipped: no cached reconstruction from eval")
                return

            loader = self.val_loader if self.val_loader is not None else self.test_loader
            if loader is None:
                tprint("  BGFGsplit skipped: no val/test loader available")
                return

            recon_state = self._init_reconstruction_state(loader)
            if not recon_state or not recon_state.get('enabled', False):
                tprint("  BGFGsplit skipped: reconstruction state unavailable")
                return

            was_training = self.model.training
            use_ema = bool(getattr(self.config.common, 'bgfgsplit_use_ema', True))
            max_batches = int(getattr(self.config.common, 'bgfgsplit_max_batches', 32))

            if use_ema:
                self.ema.swap(self.model)
            self.model.eval()

            try:
                batch_count = 0
                with self._batch_stream_manager(loader) as batch_iter:
                    for batch_data in batch_iter:
                        if max_batches > 0 and batch_count >= max_batches:
                            break

                        hsi, labels, batch_meta = self._unpack_batch(batch_data)
                        if hsi.device.type != self.device.type:
                            hsi = hsi.to(self.device, non_blocking=True)
                        if labels.device.type != self.device.type:
                            labels = labels.to(self.device, non_blocking=True, memory_format=torch.contiguous_format)
                        if hsi.dim() == 4 and hsi.shape[-1] <= 16:
                            hsi = hsi.permute(0, 3, 1, 2)

                        outputs = self.model(hsi)
                        _, fg_bg_logits, _ = self._parse_model_outputs(outputs)
                        fg_bg_logits = torch.nan_to_num(fg_bg_logits, nan=0.0, posinf=50.0, neginf=-50.0)
                        pred_bin = torch.argmax(fg_bg_logits, dim=1).long()

                        label_bin = labels.clone().long()
                        valid = (label_bin != 255)
                        label_bin[valid] = (label_bin[valid] > 0).long()
                        label_bin[~valid] = 255
                        self._accumulate_reconstruction_batch(recon_state, pred_bin, label_bin, batch_meta=batch_meta)
                        batch_count += 1
            finally:
                if use_ema:
                    self.ema.swap(self.model)
                if was_training:
                    self.model.train()

        out_dir = os.path.join(self.output, 'BGFGsplit')
        os.makedirs(out_dir, exist_ok=True)
        num_samples = int(getattr(self.config.common, 'bgfgsplit_num_samples', 6))
        num_samples = max(1, num_samples)
        min_cover_ratio = float(getattr(self.config.common, 'bgfgsplit_min_cover_ratio', 0.15))

        valid_ids = []
        for i, label_map in enumerate(recon_state['label_maps']):
            if np.any(label_map != 255):
                valid_ids.append(i)
        if not valid_ids:
            tprint("  BGFGsplit skipped: no covered pixels after reconstruction")
            return

        seed = int(getattr(self.config.split, 'split_seed', 350234)) + int(epoch + 1)
        rng = np.random.RandomState(seed)
        pick_count = min(num_samples, len(valid_ids))
        chosen = rng.choice(np.array(valid_ids, dtype=np.int64), size=pick_count, replace=False)

        cmap = plt.cm.get_cmap('tab10', 2)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        summary_lines = [
            f"epoch={int(epoch + 1)}",
            f"transition=A->B",
            f"metric={float(metric):.4f}",
            f"reason={reason}",
            f"sample_count={int(pick_count)}",
            f"seed={seed}",
        ]

        for idx in chosen.tolist():
            pred_map_raw = recon_state['pred_maps'][idx]
            label_map_raw = recon_state['label_maps'][idx]
            pred_map = pred_map_raw.copy()
            label_map = label_map_raw.copy()
            pred_valid = (pred_map != 255)
            label_valid = (label_map != 255)
            pred_map[pred_valid] = (pred_map[pred_valid] > 0).astype(np.int32)
            label_map[label_valid] = (label_map[label_valid] > 0).astype(np.int32)
            patient_name = recon_state['patient_names'][idx]
            safe_patient = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(patient_name))

            valid_mask = (label_map != 255)
            covered = int(valid_mask.sum())
            if covered <= 0:
                continue
            correct = int(((pred_map == label_map) & valid_mask).sum())
            pixel_acc = 100.0 * correct / covered
            cover_ratio = covered / float(label_map.size)

            diff_map = np.full(pred_map.shape, np.nan, dtype=np.float32)
            diff_map[valid_mask] = (pred_map[valid_mask] != label_map[valid_mask]).astype(np.float32)
            pred_show = pred_map.astype(np.float32).copy()
            label_show = label_map.astype(np.float32).copy()
            pred_show[~valid_mask] = np.nan
            label_show[~valid_mask] = np.nan

            fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
            base_cmap = cmap.copy()
            base_cmap.set_bad(color='lightgrey')
            axes[0].imshow(pred_show, cmap=base_cmap, vmin=0, vmax=1)
            axes[0].set_title('Pred FG/BG')
            axes[0].axis('off')

            axes[1].imshow(label_show, cmap=base_cmap, vmin=0, vmax=1)
            axes[1].set_title('Label FG/BG')
            axes[1].axis('off')

            diff_cmap = plt.cm.get_cmap('coolwarm').copy()
            diff_cmap.set_bad(color='lightgrey')
            axes[2].imshow(diff_map, cmap=diff_cmap, vmin=0, vmax=1)
            axes[2].set_title('Mismatch Map')
            axes[2].axis('off')

            fig.suptitle(
                f"A->B BGFG split | epoch={epoch + 1} | patient={patient_name} | "
                f"covered={covered} ({cover_ratio:.1%}) | px_acc={pixel_acc:.2f}%",
                fontsize=11,
                fontweight='bold',
            )
            plt.tight_layout()

            save_name = f"bgfgsplit_e{epoch + 1:03d}_{safe_patient}_{timestamp}.png"
            save_path = os.path.join(out_dir, save_name)
            plt.savefig(save_path, dpi=170, bbox_inches='tight')
            plt.close()

            summary_lines.append(
                f"patient={patient_name}, covered_px={covered}, cover_ratio={cover_ratio:.4f}, "
                f"pixel_acc={pixel_acc:.4f}, low_coverage={cover_ratio < min_cover_ratio}, file={save_name}"
            )

        summary_path = os.path.join(out_dir, f'bgfgsplit_summary_e{epoch + 1:03d}_{timestamp}.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines) + '\n')

        tprint(f"  BGFGsplit snapshots saved to {out_dir} ({pick_count} sample(s))")

    @staticmethod
    def _dense_fill_for_display(x: np.ndarray, invalid_value: int = 255) -> np.ndarray:
        """Fill sparse/invalid pixels by nearest valid neighbor for visualization only."""
        out = x.astype(np.float32, copy=True)
        valid = (x != invalid_value)
        if not np.any(valid):
            return np.full_like(out, np.nan, dtype=np.float32)
        if np.all(valid):
            return out

        invalid = ~valid
        _, nn_idx = distance_transform_edt(invalid, return_indices=True)
        out[invalid] = out[nn_idx[0][invalid], nn_idx[1][invalid]]
        return out

    def _save_all_plots(self, final_target, final_pred):
        """generate all single-model plots in parallel.

        pre-computes all heavy metrics (cm, p/r/f1, roc) in the main process
        so that only small summary arrays (~kb) are pickled to workers,
        instead of raw predictions (~gb). this reduces plot time from
        ~10 min to ~seconds.
        """
        num_classes = self.config.clsf.num
        all_labels = list(range(num_classes))
        metric_labels = self._metric_labels(num_classes)
        eval_num_classes = len(metric_labels)
        all_class_names = [
            self.config.clsf.targets[i] if i < len(self.config.clsf.targets) else f'Class_{i}'
            for i in all_labels
        ]
        class_names = [
            self.config.clsf.targets[i] if i < len(self.config.clsf.targets) else f'Class_{i}'
            for i in metric_labels
        ]
        probas = getattr(self, '_last_probas', None)

        # pre-compute heavy metrics in main process (avoids pickle of huge arrays)
        tprint("  pre-computing plot metrics in main process...")
        t_pre = time.perf_counter()

        # confusion matrices: full classes + metric-selected classes.
        cm_all = _cm(final_target, final_pred, labels=all_labels)
        cm_all_norm = cm_all.astype('float') / (cm_all.sum(axis=1, keepdims=True) + 1e-8)
        y_true_present = set(np.unique(final_target).tolist())
        metric_has_support = any(lbl in y_true_present for lbl in metric_labels)
        if metric_has_support:
            cm = _cm(final_target, final_pred, labels=metric_labels)
        else:
            cm = np.zeros((len(metric_labels), len(metric_labels)), dtype=np.int64)
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

        precision_all, recall_all, f1_all, _ = precision_recall_fscore_support(
            final_target, final_pred, labels=all_labels, zero_division=0)
        report_lines_all = []
        for i, cls_i in enumerate(all_labels):
            name = all_class_names[i] if i < len(all_class_names) else f'Class_{cls_i}'
            report_lines_all.append(
                f"{name}: P={precision_all[i]:.4f}, R={recall_all[i]:.4f}, F1={f1_all[i]:.4f}"
            )
        metrics_text_all = '\n'.join(report_lines_all) + '\n'

        # per-class precision, recall, f1 — each shape (num_classes,)
        if metric_has_support:
            precision, recall, f1, _ = precision_recall_fscore_support(
                final_target, final_pred, labels=metric_labels, zero_division=0)
        else:
            precision = np.zeros(len(metric_labels), dtype=np.float64)
            recall = np.zeros(len(metric_labels), dtype=np.float64)
            f1 = np.zeros(len(metric_labels), dtype=np.float64)

        # format metrics text for saving
        report_lines = []
        for i in range(eval_num_classes):
            name = class_names[i] if i < len(class_names) else f'Class_{i}'
            report_lines.append(
                f"{name}: P={precision[i]:.4f}, R={recall[i]:.4f}, F1={f1[i]:.4f}")
        if not metric_has_support:
            report_lines.append('WARNING: no fg-only labels present in y_true for this evaluation split.')
        metrics_text = '\n'.join(report_lines) + '\n'

        pairwise_lines = []
        if len(metric_labels) >= 2 and metric_has_support:
            for i in range(len(metric_labels)):
                for j in range(i + 1, len(metric_labels)):
                    ci = metric_labels[i]
                    cj = metric_labels[j]
                    name_i = class_names[i]
                    name_j = class_names[j]
                    denom_i = float(cm[ i, i] + cm[i, j]) if i < cm.shape[0] and j < cm.shape[1] else 0.0
                    denom_j = float(cm[j, j] + cm[j, i]) if j < cm.shape[0] and i < cm.shape[1] else 0.0
                    rec_i_vs_j = (100.0 * float(cm[i, i]) / max(denom_i, 1.0)) if denom_i > 0 else 0.0
                    rec_j_vs_i = (100.0 * float(cm[j, j]) / max(denom_j, 1.0)) if denom_j > 0 else 0.0
                    pairwise_lines.append(
                        f"{name_i}<->{name_j}: recall_{name_i}_vs_{name_j}={rec_i_vs_j:.2f}%, "
                        f"recall_{name_j}_vs_{name_i}={rec_j_vs_i:.2f}%"
                    )
        pairwise_text = '\n'.join(pairwise_lines) + ('\n' if pairwise_lines else '')

        pred_fg_hist = [
            float(x.get('eval_pred_fg_ratio', np.nan))
            for x in self.epoch_metrics if x.get('eval_pred_fg_ratio') is not None
        ]
        gt_fg_hist = [
            float(x.get('eval_gt_fg_ratio', np.nan))
            for x in self.epoch_metrics if x.get('eval_gt_fg_ratio') is not None
        ]
        threshold_scan_records = list(getattr(self, 'threshold_scan_history', []))

        # roc curves — list of (fpr, tpr, auc_val) per class, ~200 points each.
        # probas is already sub-sampled to ~2M pixels (done in evaluate()).
        # Per-class binary label avoids:
        #   (a) np.zeros((4.4B, 8)) — 14 GB allocation
        #   (b) 4.4B-iteration Python for-loop
        #   (c) roc_curve sorting pixels * classes ~ 8 * 4.4B*log(4.4B) ops
        roc_data = None
        probas_targets = getattr(self, '_last_probas_targets', None)
        if probas is not None and probas_targets is not None:
            roc_data = []
            for cls_i in metric_labels:
                y_bin = (probas_targets == cls_i).astype(np.uint8)  # (n_sampled,)
                if y_bin.sum() == 0:
                    roc_data.append(None)   # absent class — worker will skip
                    continue
                fpr, tpr, _ = roc_curve(y_bin, probas[:, cls_i])
                roc_auc = auc(fpr, tpr)
                # downsample roc to ~200 points to keep pickle tiny
                if len(fpr) > 200:
                    idx = np.linspace(0, len(fpr) - 1, 200, dtype=int)
                    fpr, tpr = fpr[idx], tpr[idx]
                roc_data.append((fpr.astype(np.float32),
                                 tpr.astype(np.float32),
                                 float(roc_auc)))

        # free large arrays before spawning workers
        del final_target, final_pred, probas, probas_targets
        self._last_probas         = None
        self._last_probas_targets = None

        tprint(f"  metrics pre-computed in {time.perf_counter() - t_pre:.1f}s")

        # dispatch lightweight plot tasks to workers
        tasks = [
            (_worker_plot_training_curves,
             (list(self.train_losses), list(self.eval_losses),
              list(self.train_bgfg_losses), list(self.train_fg_class_losses),
              list(self.eval_bgfg_losses), list(self.eval_fg_class_losses),
              list(self.train_accs), list(self.eval_accs),
                            list(self.stage_history),
              self.eval_interval, self.debug_mode,
              self.best_acc, self.model_name, self.output)),
            (_worker_plot_confusion_matrix,
                         (cm_all, cm_all_norm, metrics_text_all,
                            all_class_names, self.model_name, self.output,
                            'confusion_matrix_all', 'All-Class Confusion Matrix',
                            'classification_metrics_all.txt')),
                        (_worker_plot_confusion_matrix,
                         (cm, cm_norm, metrics_text,
                            class_names, self.model_name, self.output,
                            'confusion_matrix_fg_only', 'FG-Only Confusion Matrix',
                            'classification_metrics_fg_only.txt')),
            (_worker_plot_per_class_metrics,
                         (precision, recall, f1, eval_num_classes, class_names,
              self.model_name, self.output)),
            (_worker_plot_roc_curves,
                         (roc_data, eval_num_classes, class_names,
              self.model_name, self.output)),
            (_worker_plot_grad_norm,
             (list(self.grad_norms), self.model_name, self.output)),
            (_worker_plot_lr,
             (list(self.lr_history), self.model_name, self.output)),
            (_worker_plot_epoch_time,
             (list(self.epoch_times), self.model_name, self.output)),
              (_worker_plot_fg_threshold_curve,
               (threshold_scan_records, self.model_name, self.output)),
              (_worker_plot_fg_ratio_trajectory,
               (pred_fg_hist, gt_fg_hist, self.model_name, self.output)),
              (_worker_write_pairwise_fg_metrics,
               (pairwise_text, self.output)),
            (_worker_plot_stage_timeline,
             (list(self.stage_history), list(self.stage_transition_log), self.model_name, self.output)),
            (_worker_write_stage_transition_summary,
             (list(self.stage_transition_log), self.output)),
        ]

        t0 = time.perf_counter()
        max_workers = min(len(tasks), (os.cpu_count() or 4))
        with _managed_pool(max_workers, "single-run plot workers") as pool:
            futures = {pool.submit(fn, *args): fn.__name__
                       for fn, args in tasks}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    print(f"  Warning: {futures[fut]} failed: {exc}")
        tprint(f"  All plots generated in {time.perf_counter() - t0:.1f}s "
              f"({max_workers} workers)")

    def _generate_cam(self, epoch: int) -> None:
        """
        Generate full-image Grad-CAM heatmap overlays.
        
        For each sample, shows:
          1) Pseudo-RGB image
          2) Grad-CAM heatmap (jet colormap)
          3) Heatmap overlaid on RGB
        """
        self.model.eval()
        
        try:
            # get a batch of test data
            test_iter = iter(self.test_loader)
            batch_data = next(test_iter)
            hsi, labels, _ = self._unpack_batch(batch_data)
            
            hsi = hsi.to(self.device)
            num_samples = min(4, hsi.shape[0])
            
            fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            
            col_titles = ['Input', 'Grad-CAM', 'Overlay']
            
            with torch.no_grad():
                outputs = self.model(hsi[:num_samples])
                fused_logits, _, _ = self._parse_model_outputs(outputs)
                preds = torch.argmax(fused_logits, dim=1)
            
            for i in range(num_samples):
                # pseudo-RGB visualization
                sample = hsi[i].cpu().numpy()
                if sample.ndim == 4:
                    sample = sample[0]
                sample_norm = (sample - sample.min()) / (sample.max() - sample.min() + 1e-8)
                
                if sample_norm.shape[0] >= 3:
                    rgb = sample_norm[:3].transpose(1, 2, 0)  # (H, W, 3)
                else:
                    rgb = np.repeat(sample_norm[0:1].transpose(1, 2, 0), 3, axis=2)
                
                # Grad-CAM for the predicted class (full spatial map)
                pred_class = preds[i]
                # use the most frequent predicted class across the patch
                if pred_class.dim() > 0:
                    pred_cls_idx = int(torch.mode(pred_class.flatten()).values.item())
                else:
                    pred_cls_idx = int(pred_class.item())
                
                cam = self._compute_gradcam(
                    hsi[i:i+1], class_idx=pred_cls_idx
                )  # (H, W), values in [0, 1]
                
                # resize CAM to match spatial dims
                h_img, w_img = rgb.shape[0], rgb.shape[1]
                if cam.shape[0] != h_img or cam.shape[1] != w_img:
                    cam = cv2.resize(cam, (w_img, h_img))
                
                # heatmap overlay
                heatmap_color = plt.cm.jet(cam)[..., :3]  # (H, W, 3) float
                overlay = 0.5 * rgb + 0.5 * heatmap_color
                overlay = np.clip(overlay, 0, 1)
                
                axes[i, 0].imshow(rgb)
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(cam, cmap='jet', vmin=0, vmax=1)
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(overlay)
                axes[i, 2].axis('off')
            
            # column titles on top row only
            for j, title in enumerate(col_titles):
                axes[0, j].set_title(title, fontsize=11)
            
            plt.suptitle(f'Epoch {epoch+1} - Grad-CAM', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            cam_path = os.path.join(self.output, 'CAM', f'epoch_{epoch+1:03d}.png')
            plt.savefig(cam_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            tprint(f"Error during generating CAM: {e}")
            traceback.print_exc()

    def _unpack_batch(self, batch_data) -> Tuple[torch.Tensor, torch.Tensor, Optional[Any]]:
        """
        Unpack batch data from DataLoader.
        
        Returns:
            Tuple[tensor, tensor, Any|None]: (``hsi``, ``labels``, ``batch_meta``)
        """
        if isinstance(batch_data, (list, tuple)):
            if len(batch_data) == 3:
                hsi = batch_data[0]
                # Support both legacy (hsi, indices, labels) and
                # cached-chunk (hsi, labels, chunk_meta) formats.
                if torch.is_tensor(batch_data[2]):
                    batch_meta = batch_data[1]
                    labels = batch_data[2]
                else:
                    labels = batch_data[1]
                    batch_meta = batch_data[2]
            elif len(batch_data) == 2:
                hsi, labels = batch_data
                batch_meta = None
            else:
                raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
        else:
            raise TypeError(f"batch must be list or tuple, got: {type(batch_data)}")
        
        return hsi, labels, batch_meta
    
    def _compute_miou(self, y_true: np.ndarray, y_pred: np.ndarray,
                      num_classes: int) -> float:
        """
        Compute mean Intersection-over-Union (mIoU) for segmentation evaluation.
        Ignores classes not present in both y_true and y_pred.
        
        Returns:
            mIoU percentage (0-100).
        """
        ious = []
        for cls in range(num_classes):
            intersection = ((y_pred == cls) & (y_true == cls)).sum()
            union = ((y_pred == cls) | (y_true == cls)).sum()
            if union > 0:
                ious.append(float(intersection) / float(union))
        return float(np.mean(ious)) * 100.0 if ious else 0.0
    
    def _save_model(self) -> None:
        if self.best_model_state is None:
            return

        # Load best EMA weights into the model for export
        raw_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        # Save as ONNX
        onnx_path = os.path.join(self.output, 'models', f'{self.model_name}_best.onnx')
        dummy_input = torch.randn(
            1, self.config.preprocess.pca_components,
            self.config.split.patch_size,
            self.config.split.patch_size
        ).to(self.device)

        raw_model.eval()
        try:
            class _FusedOnlyWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    out = self.model(x)
                    if isinstance(out, dict):
                        return out['fused_logits']
                    if isinstance(out, (tuple, list)):
                        return out[0]
                    return out

            export_model = _FusedOnlyWrapper(raw_model)
            torch.onnx.export(
                export_model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            tprint(f"  ONNX model saved to: {onnx_path}")
        except Exception as e:
            # Fallback: save as .pth if ONNX export fails
            pth_path = os.path.join(self.output, 'models', f'{self.model_name}_best.pth')
            torch.save(self.best_model_state, pth_path)
            tprint(f"  ONNX export failed ({e}), saved .pth to: {pth_path}")
    
    def _compute_gradcam(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """Compute Grad-CAM heatmap for model interpretability.
        
        Handles DataParallel by unwrapping to the underlying module,
        and ensures hooks are always cleaned up via try/finally.
        """
        features = None
        gradients = None
        
        def forward_hook(module, input, output):
            nonlocal features
            features = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0].detach()
        
        # Unwrap DataParallel to access real modules (avoids "dead Module" error)
        base_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        # Find last Conv2d layer on the unwrapped model
        target_layer = None
        for module in base_model.modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        
        if target_layer is None:
            return np.ones((x.shape[-2], x.shape[-1])) * 0.5
        
        # Register hooks on the unwrapped model's layer
        h_f = target_layer.register_forward_hook(forward_hook)
        h_b = target_layer.register_full_backward_hook(backward_hook)
        
        try:
            # Forward pass through the ORIGINAL self.model (may be DataParallel)
            self.model.zero_grad()
            output = self.model(x)
            fused_logits, _, _ = self._parse_model_outputs(output)
            # For segmentation output [B, K, H, W], sum spatial dims to get scalar
            if fused_logits.dim() == 4:
                loss = fused_logits[0, class_idx].sum()
            else:
                loss = fused_logits[0, class_idx]
            loss.backward(retain_graph=False)
            
            # Compute CAM
            if features is not None and gradients is not None:
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                cam = (features * weights).sum(dim=1).squeeze(0)
                cam = torch.relu(cam).cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                cam = cv2.resize(cam, (x.shape[-1], x.shape[-2])) if cam.size > 0 else np.ones((x.shape[-2], x.shape[-1])) * 0.5
            else:
                cam = np.ones((x.shape[-2], x.shape[-1])) * 0.5
        finally:
            # Always remove hooks to prevent corruption of subsequent backward passes
            h_f.remove()
            h_b.remove()
        
        return cam
