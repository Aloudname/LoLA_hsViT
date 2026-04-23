# trainer.py
from tqdm import tqdm
from munch import Munch
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import copy, json, math, time
import torch, torch.nn as nn, torch.nn.functional as F

from pipeline.monitor import tprint


def _cfg_get(config, path, default=None):
    current = config
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, Munch):
            current = getattr(current, part, None)
        elif isinstance(current, dict):
            current = current.get(part, None)
        else:
            current = getattr(current, part, None)
    return default if current is None else current


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        for name, ema_param in self.ema.state_dict().items():
            model_param = model_state[name].detach()
            if not torch.is_floating_point(ema_param):
                ema_param.copy_(model_param)
            else:
                ema_param.mul_(self.decay).add_(model_param, alpha=1.0 - self.decay)


@dataclass
class TrainerResult:
    """Training summary for downstream pipeline."""

    history: Dict[str, Any]
    best_epoch: int
    best_metric: float


class CompositeSegLoss(nn.Module):
    def __init__(self, config: Munch):
        super().__init__()
        self.config = config
        self.num_classes = int(_cfg_get(config, "data.num_classes", 2))
        self.include_background = bool(_cfg_get(config, "loss.include_background", False))
        self.smooth = float(_cfg_get(config, "loss.smooth", 1e-6))
        self.ce_weight = float(_cfg_get(config, "loss.ce_weight", 1.0))
        self.focal_weight = float(_cfg_get(config, "loss.focal_weight", 0.5))
        self.dice_weight = float(_cfg_get(config, "loss.dice_weight", 1.0))
        self.tversky_weight = float(_cfg_get(config, "loss.tversky_weight", 0.0))
        self.focal_gamma = float(_cfg_get(config, "loss.focal_gamma", 2.0))
        self.tversky_alpha = float(_cfg_get(config, "loss.tversky_alpha", 0.3))
        self.tversky_beta = float(_cfg_get(config, "loss.tversky_beta", 0.7))
        self.label_smoothing = float(_cfg_get(config, "loss.label_smoothing", 0.0))
        self.class_weights = self._build_class_weights()

    def _build_class_weights(self) -> torch.Tensor:
        weights = torch.ones(self.num_classes, dtype=torch.float32)
        configured = _cfg_get(self.config, "loss.class_weights", None)
        if configured is not None:
            configured = list(configured)
            limit = min(len(configured), self.num_classes)
            weights[:limit] = torch.tensor(configured[:limit], dtype=torch.float32)
        rare_boost = float(_cfg_get(self.config, "loss.rare_class_boost", 0.0) or 0.0)
        rare_indices = _cfg_get(self.config, "loss.rare_class_indices", None)
        if rare_boost > 0 and rare_indices is not None:
            for idx in rare_indices:
                if 0 <= int(idx) < self.num_classes:
                    weights[int(idx)] *= rare_boost

        bg_weight = _cfg_get(self.config, "loss.background_weight", None)
        if bg_weight is not None and self.num_classes > 0:
            weights[0] = float(bg_weight)

        weights = torch.clamp(weights, min=1e-6)
        weights = weights / weights.mean().clamp_min(1e-6)
        return weights

    def _get_weights(self, device: torch.device) -> torch.Tensor:
        return self.class_weights.to(device)

    def _foreground_indices(self, device: torch.device) -> torch.Tensor:
        start = 0 if self.include_background else 1
        if start >= self.num_classes:
            start = 0
        return torch.arange(start, self.num_classes, device=device)

    def _standard_focal_loss(self, logits, targets,
                              class_weights: torch.Tensor = None,
                              device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = logits.device
        if class_weights is None:
            class_weights = self.class_weights.to(device)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha = class_weights.gather(0, targets.view(-1)).view_as(target_probs)
        focal = -alpha * torch.pow(1.0 - target_probs.clamp(0.0, 1.0), self.focal_gamma) * target_log_probs
        return focal.mean()

    def _masked_class_mean(self, values, mask,
                            weights: torch.Tensor = None,
                            device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = values.device
        if weights is not None:
            weights = weights.to(device) * mask.float()
            denom = weights.sum()
            if denom <= 0:
                return values.new_tensor(0.0)
            return (values * weights).sum() / denom
        mask_values = values[mask]
        if mask_values.numel() == 0:
            return values.new_tensor(0.0)
        return mask_values.mean()

    def _dice_loss(self, probs, targets_one_hot, class_weights: torch.Tensor = None,
                    device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = probs.device
        if class_weights is None:
            class_weights = self.class_weights.to(device)
        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        fg_idx = self._foreground_indices(probs.device)
        fg_dice = dice[fg_idx]
        fg_target = targets_one_hot.sum(dim=dims)[fg_idx]
        fg_mask = fg_target > 0
        if not fg_mask.any():
            return probs.new_tensor(0.0)
        fg_weights = class_weights[fg_idx]
        return 1.0 - self._masked_class_mean(fg_dice, fg_mask, fg_weights)

    def _tversky_loss(self, probs, targets_one_hot, class_weights: torch.Tensor = None,
                    device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = probs.device
        if class_weights is None:
            class_weights = self.class_weights.to(device)
        dims = (0, 2, 3)
        tp = (probs * targets_one_hot).sum(dim=dims)
        fp = (probs * (1.0 - targets_one_hot)).sum(dim=dims)
        fn = ((1.0 - probs) * targets_one_hot).sum(dim=dims)
        tversky = (tp + self.smooth) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + self.smooth)

        fg_idx = self._foreground_indices(probs.device)
        fg_tversky = tversky[fg_idx]
        fg_target = targets_one_hot.sum(dim=dims)[fg_idx]
        fg_mask = fg_target > 0
        if not fg_mask.any():
            return probs.new_tensor(0.0)
        fg_weights = class_weights[fg_idx]
        return 1.0 - self._masked_class_mean(fg_tversky, fg_mask, fg_weights)

    def compute(self, logits, targets) -> Tuple[torch.Tensor, Dict[str, float]]:
        if logits.ndim != 4:
            raise ValueError(f"Expected logits with shape [B, C, H, W], got {tuple(logits.shape)}")
        if targets.ndim != 3:
            raise ValueError(f"Expected targets with shape [B, H, W], got {tuple(targets.shape)}")

        class_weights = self._get_weights(logits.device)
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()

        ce_loss = F.cross_entropy(
            logits,
            targets.long(),
            weight=class_weights,
            reduction="mean",
            label_smoothing=self.label_smoothing,
        )
        focal_loss = self._standard_focal_loss(logits, targets.long(), class_weights)
        dice_loss = self._dice_loss(probs, targets_one_hot, class_weights)
        tversky_loss = self._tversky_loss(probs, targets_one_hot, class_weights)

        total = (
            self.ce_weight * ce_loss
            + self.focal_weight * focal_loss
            + self.dice_weight * dice_loss
            + self.tversky_weight * tversky_loss
        )

        loss_dict = {
            "ce": float(ce_loss.detach().cpu().item()),
            "focal": float(focal_loss.detach().cpu().item()),
            "dice": float(dice_loss.detach().cpu().item()),
            "tversky": float(tversky_loss.detach().cpu().item()),
            "total": float(total.detach().cpu().item()),
        }
        return total, loss_dict

    def forward(self, logits, targets):
        total, _ = self.compute(logits, targets)
        return total


class SpectralDiscriminativeLoss(nn.Module):
    """
    Stage-1 spectral discriminative loss.
    This loss is used to train the spectral encoder,
        to encourage more discriminative features.

    It's a weighted combination of:
        - intra-class compactness,
        - inter-class separation, 
        - direct pixel classification (Focal).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = int(_cfg_get(config, "data.num_classes", 2))
        self.lambda_intra = float(_cfg_get(config, "loss.stage1.lambda_intra", 1.0))
        self.lambda_inter = float(_cfg_get(config, "loss.stage1.lambda_inter", 0.5))
        self.lambda_focal = float(_cfg_get(config, "loss.stage1.lambda_focal", 1.0))
        self.focal_gamma = float(_cfg_get(config, "loss.stage1.focal_gamma", 2.0))
        self.center_beta = float(_cfg_get(config, "loss.stage1.center_beta", 0.999))
        self.margin = float(_cfg_get(config, "loss.stage1.margin", 1.0))
        self.max_pixels = max(256, int(_cfg_get(config, "loss.stage1.max_pixels", 8192)))
        class_weights = _cfg_get(config, "loss.class_weights", [1.0] * self.num_classes)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def _sample_pixels(self, features, targets) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = features.shape
        feats = features.permute(0, 2, 3, 1).reshape(-1, c)
        labels = targets.reshape(-1).long()
        if feats.shape[0] <= self.max_pixels:
            return feats, labels
        idx = torch.randperm(feats.shape[0], device=feats.device)[: self.max_pixels]
        return feats[idx], labels[idx]

    def _focal_loss(self, logits, targets) -> torch.Tensor:
        if logits.ndim == 4:
            # (B, C, H, W) -> (B, H, W, C) -> (B * H * W, C)
            logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
        targets = targets.reshape(-1).long()
        class_weights = self.class_weights.to(logits.device)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha = class_weights.gather(0, targets)
        focal = -alpha * torch.pow(1.0 - target_probs.clamp(0.0, 1.0), self.focal_gamma) * target_log_probs
        return focal.mean()

    def _compute_centers_and_counts(self, feats, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        centers: List[torch.Tensor] = []
        counts: List[torch.Tensor] = []
        for cls_idx in range(self.num_classes):
            mask = labels == cls_idx
            if mask.any():
                centers.append(feats[mask].mean(dim=0))
                counts.append(mask.sum().float())
            else:
                centers.append(feats.new_zeros((feats.shape[1],)))
                counts.append(feats.new_zeros(()))
        return torch.stack(centers, dim=0), torch.stack(counts, dim=0)

    def _compute_intra_loss(self, feats, labels, centers_t, counts_t) -> torch.Tensor:
        total = feats.new_tensor(0.0)
        valid = 0
        effective = self.class_weights.to(feats.device)
        effective = effective / effective.mean().clamp_min(1e-6)
        for cls_idx in range(self.num_classes):
            if counts_t[cls_idx] <= 0:
                continue
            mask = labels == cls_idx
            d = (feats[mask] - centers_t[cls_idx]).pow(2).sum(dim=1).mean()
            total = total + effective[cls_idx] * d
            valid += 1
        return total / max(1, valid)

    def _compute_inter_loss(self, centers_t, counts_t) -> torch.Tensor:
        inter_terms = []
        for i in range(self.num_classes):
            if counts_t[i] <= 0:
                continue
            for j in range(i + 1, self.num_classes):
                if counts_t[j] <= 0:
                    continue
                dist = torch.norm(centers_t[i] - centers_t[j], p=2)
                inter_terms.append(torch.relu(self.margin - dist).pow(2))
        if not inter_terms:
            return centers_t.new_tensor(0.0)
        return torch.stack(inter_terms).mean()

    def compute(self, spectral_features, spectral_logits, targets) -> Tuple[torch.Tensor, Dict[str, float]]:
        if spectral_features is None or spectral_logits is None:
            return targets.new_tensor(0.0, dtype=torch.float32), {"stage1_intra": 0.0, "stage1_inter": 0.0, "stage1_focal": 0.0, "stage1_total": 0.0}
        if spectral_features.shape[-2:] != targets.shape[-2:]:
            spectral_features = F.interpolate(spectral_features, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        if spectral_logits.shape[-2:] != targets.shape[-2:]:
            spectral_logits = F.interpolate(spectral_logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        feats, labels = self._sample_pixels(spectral_features, targets)
        centers_t, counts_t = self._compute_centers_and_counts(feats, labels)
        intra = self._compute_intra_loss(feats, labels, centers_t, counts_t)
        inter = self._compute_inter_loss(centers_t, counts_t)
        focal = self._focal_loss(spectral_logits, targets.long())
        total = self.lambda_intra * intra + self.lambda_inter * inter + self.lambda_focal * focal
        return total, {"stage1_intra": float(intra.detach().cpu().item()), "stage1_inter": float(inter.detach().cpu().item()), "stage1_focal": float(focal.detach().cpu().item()), "stage1_total": float(total.detach().cpu().item())}

class Trainer:
    def __init__(self, model, config, output_dir):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "artifacts" / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        requested_device = _cfg_get(config, "runtime.device", None)
        if requested_device is not None:
            self.device = torch.device(requested_device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.channels_last = bool(config.train.memory.channels_last or False)
        if self.channels_last and self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)

        self.num_classes = int(_cfg_get(config, "model.num_classes", _cfg_get(config, "data.num_classes", 2)))
        self.class_names = list(_cfg_get(config, "data.class_names", []))
        if not self.class_names:
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]
        elif len(self.class_names) < self.num_classes:
            self.class_names = self.class_names + [f"class_{i}" for i in range(len(self.class_names), self.num_classes)]

        self.rare_class_indices = _cfg_get(config, "loss.rare_class_indices", None)
        if self.rare_class_indices is None:
            self.rare_class_indices = [idx for idx, name in enumerate(self.class_names) if str(name).lower() == "pg"]
        self.rare_class_indices = [int(idx) for idx in self.rare_class_indices if 0 <= int(idx) < self.num_classes]

        self.criterion = CompositeSegLoss(config)
        self.stage1_criterion = SpectralDiscriminativeLoss(config)
        self.optimizer = self._build_optimizer()
        self.scheduler = None
        self.scheduler_mode = "none"
        self.scheduler_step_on = "epoch"

        ema_decay = float(_cfg_get(config, "train.ema_decay", 0.0))
        self.use_ema = bool(_cfg_get(config, "train.use_ema", ema_decay > 0 and bool(_cfg_get(config, "train.ema_decay", None) is not None)))
        self.ema = ModelEMA(self.model, decay=ema_decay) if self.use_ema and ema_decay > 0 else None

        self.use_amp = bool(
            _cfg_get(
                config,
                "train.amp",
                _cfg_get(config, "runtime.use_amp", _cfg_get(config, "train.use_amp", self.device.type == "cuda")),
            )
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.device.type == "cuda")
        self.grad_accum_steps = max(1, int(_cfg_get(config, "train.grad_accum_steps", 1)))
        self.max_grad_norm = float(_cfg_get(config, "train.grad_clip", _cfg_get(config, "train.max_grad_norm", 0.0)))
        self.aux_loss_weight = float(_cfg_get(config, "train.aux_loss_weight", _cfg_get(config, "loss.aux_weight", 0.0)))
        self.max_train_steps_per_epoch = max(0, int(_cfg_get(config, "train.max_train_steps_per_epoch", 0)))
        self.max_eval_steps = max(0, int(_cfg_get(config, "train.max_eval_steps", 0)))
        self.oom_skip_batch = bool(_cfg_get(config, "train.memory.oom_skip_batch", True))
        self.empty_cache_steps = max(0, int(_cfg_get(config, "train.memory.empty_cache_steps", 0)))
        self.staged_cfg = _cfg_get(config, "train.staged", {})
        self.staged_enabled = bool(_cfg_get(config, "train.staged.enabled", False))
        self.stage_definitions = list(_cfg_get(config, "train.staged.stages", [])) if self.staged_enabled else []
        self.early_stop_cfg = _cfg_get(config, "train.early_stop", {})

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "eval_loss": [],
            "train_dice": [],
            "val_dice": [],
            "eval_dice": [],
            "lr": [],
            "learning_rate": [],
            "train_macro_dice": [],
            "val_macro_dice": [],
            "train_min_fg_dice": [],
            "val_min_fg_dice": [],
            "train_presence_recall": [],
            "val_presence_recall": [],
            "train_rare_dice": [],
            "val_rare_dice": [],
            "train_fg_distribution_gap": [],
            "val_fg_distribution_gap": [],
            "train_fg_dominant_ratio": [],
            "val_fg_dominant_ratio": [],
            "train_per_class_dice": [],
            "val_per_class_dice": [],
            "train_loss_components": [],
            "val_loss_components": [],
            "stage_name": [],
            "stage_spans": [],
            "epochs": [],
        }
        self.best_val_dice = -float("inf")

    def _build_optimizer(self):
        tprint(f"building optimizer...")
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer_cfg = _cfg_get(self.config, "train.optimizer", "adamw")
        if isinstance(optimizer_cfg, str):
            optimizer_name = optimizer_cfg.lower()
        else:
            optimizer_name = str(_cfg_get(self.config, "train.optimizer.name", "adamw")).lower()

        lr = float(_cfg_get(self.config, "train.lr", _cfg_get(self.config, "train.optimizer.lr", 3e-4)))
        weight_decay = float(_cfg_get(self.config, "train.weight_decay", _cfg_get(self.config, "train.optimizer.weight_decay", 1e-4)))

        if optimizer_name == "sgd":
            momentum = float(_cfg_get(self.config, "train.optimizer.momentum", 0.9))
            nesterov = bool(_cfg_get(self.config, "train.optimizer.nesterov", True))
            return torch.optim.SGD(
                trainable_params,
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay,
            )

        betas = tuple(_cfg_get(self.config, "train.optimizer.betas", [0.9, 0.999]))
        eps = float(_cfg_get(self.config, "train.optimizer.eps", 1e-8))
        tprint(f"optimizer built.")
        return torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    def _current_lr(self):
        if not self.optimizer.param_groups:
            return 0.0
        return float(self.optimizer.param_groups[0]["lr"])

    def _autocast_context(self):
        return torch.cuda.amp.autocast(enabled=self.use_amp and self.device.type == "cuda")

    def _resolve_stage_scheduler_cfg(self, stage_cfg: Optional[Mapping[str, Any]] = None):
        if stage_cfg is not None and isinstance(stage_cfg, Mapping) and "scheduler" in stage_cfg:
            return stage_cfg.get("scheduler")
        return _cfg_get(self.config, "train.scheduler", "none")

    def _set_stage_trainability(self, stage_cfg: Mapping[str, Any]) -> None:
        tprint(f"setting params grad...")
        freeze_cfg = stage_cfg.get("freeze", {}) if isinstance(stage_cfg, Mapping) else {}
        freeze_backbone = bool(freeze_cfg.get("backbone", False))
        freeze_decoder = bool(freeze_cfg.get("decoder", False))
        freeze_spectral = bool(freeze_cfg.get("spectral_encoder", False))
        for name, param in self.model.named_parameters():
            lower = name.lower()
            req_grad = True
            if freeze_backbone and "backbone" in lower:
                req_grad = False
            if freeze_decoder and ("decoder" in lower or "classifier" in lower or "seg_head" in lower):
                req_grad = False
            if freeze_spectral and ("spectral_encoder" in lower or "spectral_aux_head" in lower):
                req_grad = False
            param.requires_grad_(req_grad)
        tprint(f"params grad initialized.")
        self.optimizer = self._build_optimizer()

    def _build_scheduler(self, train_loader, num_epochs, scheduler_cfg_override=None):
        tprint(f"building scheduler...")
        scheduler_cfg = scheduler_cfg_override if scheduler_cfg_override is not None else _cfg_get(self.config, "train.scheduler", "none")
        if isinstance(scheduler_cfg, Mapping):
            scheduler_name = str(scheduler_cfg.get("name", "none")).lower()
            scheduler_ns = scheduler_cfg
        elif isinstance(scheduler_cfg, str):
            scheduler_name = scheduler_cfg.lower()
            scheduler_ns = {}
        else:
            scheduler_name = "none"
            scheduler_ns = {}

        self.scheduler_mode = scheduler_name
        self.scheduler_step_on = "epoch"
        if scheduler_name in {"none", "", "null"}:
            self.scheduler = None
            tprint(f"scheduler None built.")
            return
        
        steps_per_epoch = max(1, math.ceil(len(train_loader) / self.grad_accum_steps))
        total_steps = max(1, steps_per_epoch * int(max(1, num_epochs)))
        warmup_epochs = float(scheduler_ns.get("warmup_epochs", _cfg_get(self.config, "train.scheduler.warmup_epochs", 0.0)))
        warmup_steps = int(scheduler_ns.get("warmup_steps", round(warmup_epochs * steps_per_epoch)))
        min_lr_ratio = float(scheduler_ns.get("min_lr_ratio", _cfg_get(self.config, "train.scheduler.min_lr_ratio", 0.01)))
        power = float(scheduler_ns.get("power", _cfg_get(self.config, "train.scheduler.power", 0.9)))

        if scheduler_name in {"warmup_cosine", "cosine", "cosine_warmup"}:
            self.scheduler_step_on = "step"

            def lr_lambda(step):
                if warmup_steps > 0 and step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            tprint(f"scheduler built.")
            return

        if scheduler_name == "poly":
            tprint(f"scheduler built.")
            self.scheduler_step_on = "step"

            def lr_lambda(step):
                if warmup_steps > 0 and step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                return max(min_lr_ratio, (1.0 - progress) ** power)

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            return

        if scheduler_name == "step":
            step_size = int(scheduler_ns.get("step_size", _cfg_get(self.config, "train.scheduler.step_size", max(1, int(num_epochs) // 3))))
            gamma = float(scheduler_ns.get("gamma", _cfg_get(self.config, "train.scheduler.gamma", 0.1)))
            self.scheduler_step_on = "epoch"
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
            tprint(f"scheduler built.")
            return

        self.scheduler = None
        self.scheduler_mode = "none"

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            images = batch.get("image", batch.get("images", batch.get("x", batch.get("inputs"))))
            masks = batch.get("mask", batch.get("masks", batch.get("y", batch.get("target", batch.get("targets")))))
            meta = {k: v for k, v in batch.items() if k not in {"image", "images", "x", "inputs", "mask", "masks", "y", "target", "targets"}}
            return images, masks, meta
        if isinstance(batch, (list, tuple)):
            if len(batch) == 0:
                raise ValueError("Received empty batch.")
            images = batch[0]
            masks = batch[1] if len(batch) > 1 else None
            meta = batch[2:] if len(batch) > 2 else None
            return images, masks, meta
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _normalize_aux_outputs(self, aux_outputs):
        if aux_outputs is None:
            return {}
        if torch.is_tensor(aux_outputs):
            return {"aux_0": aux_outputs}
        if isinstance(aux_outputs, dict):
            normalized = {}
            for key, value in aux_outputs.items():
                if torch.is_tensor(value):
                    normalized[key] = value
                elif isinstance(value, (list, tuple)):
                    for idx, tensor in enumerate(value):
                        if torch.is_tensor(tensor):
                            normalized[f"{key}_{idx}"] = tensor
            return normalized
        if isinstance(aux_outputs, (list, tuple)):
            return {f"aux_{idx}": value for idx, value in enumerate(aux_outputs) if torch.is_tensor(value)}
        return {}

    def _forward_with_aux(self, images):
        aux_outputs = {}
        if hasattr(self.model, "forward_with_aux"):
            result = self.model.forward_with_aux(images)
            if isinstance(result, tuple):
                logits = result[0]
                aux_outputs = self._normalize_aux_outputs(result[1] if len(result) > 1 else None)
            elif isinstance(result, dict):
                logits = result.get("logits", result.get("main"))
                aux_outputs = self._normalize_aux_outputs({k: v for k, v in result.items() if k not in {"logits", "main"}})
            else:
                logits = result
        else:
            logits = self.model(images)
            if hasattr(self.model, "get_aux_outputs"):
                aux_outputs = self._normalize_aux_outputs(self.model.get_aux_outputs())
            elif hasattr(self.model, "get_spectral_supervision_tensors"):
                aux_outputs = self._normalize_aux_outputs(self.model.get_spectral_supervision_tensors())
        return logits, aux_outputs

    def _compute_aux_loss(self, aux_outputs, targets):
        if self.aux_loss_weight <= 0 or not aux_outputs:
            return targets.new_tensor(0.0, dtype=torch.float32), {}

        total_aux = None
        used = 0
        details = {}
        for name, aux_tensor in aux_outputs.items():
            if not torch.is_tensor(aux_tensor) or aux_tensor.ndim != 4:
                continue
            if aux_tensor.shape[1] != self.num_classes:
                continue
            if aux_tensor.shape[-2:] != targets.shape[-2:]:
                aux_tensor = F.interpolate(aux_tensor, size=targets.shape[-2:], mode="bilinear", align_corners=False)
            aux_loss, aux_loss_dict = self.criterion.compute(aux_tensor, targets)
            total_aux = aux_loss if total_aux is None else total_aux + aux_loss
            used += 1
            details[f"{name}_total"] = float(aux_loss_dict["total"])

        if used == 0:
            return targets.new_tensor(0.0, dtype=torch.float32), {}
        total_aux = total_aux / used
        details["aux_total"] = float(total_aux.detach().cpu().item())
        return total_aux, details

    def _confusion_matrix(self, preds, targets):
        preds = preds.view(-1).to(torch.int64)
        targets = targets.view(-1).to(torch.int64)
        valid = (targets >= 0) & (targets < self.num_classes)
        preds = preds[valid]
        targets = targets[valid]
        index = targets * self.num_classes + preds
        conf = torch.bincount(index, minlength=self.num_classes * self.num_classes)
        return conf.view(self.num_classes, self.num_classes).cpu()

    def _metrics_from_confusion(self, confusion):
        confusion = confusion.to(torch.float64)
        tp = torch.diag(confusion)
        fp = confusion.sum(dim=0) - tp
        fn = confusion.sum(dim=1) - tp
        support = confusion.sum(dim=1)
        pred_support = confusion.sum(dim=0)
        total = confusion.sum().clamp_min(1.0)

        dice = (2.0 * tp + 1e-8) / (2.0 * tp + fp + fn + 1e-8)
        iou = (tp + 1e-8) / (tp + fp + fn + 1e-8)
        precision = (tp + 1e-8) / (tp + fp + 1e-8)
        recall = (tp + 1e-8) / (tp + fn + 1e-8)

        fg_start = 0 if _cfg_get(self.config, "loss.include_background", False) else 1
        if fg_start >= self.num_classes:
            fg_start = 0
        fg_slice = slice(fg_start, self.num_classes)

        fg_support = support[fg_slice]
        fg_pred_support = pred_support[fg_slice]
        fg_dice = dice[fg_slice]
        fg_recall = recall[fg_slice]
        fg_present = fg_support > 0

        if fg_present.any():
            fg_macro_dice = float(fg_dice[fg_present].mean().item())
            fg_macro_recall = float(fg_recall[fg_present].mean().item())
            fg_min_dice = float(fg_dice[fg_present].min().item())
            fg_presence_recall = float((torch.diag(confusion)[fg_slice][fg_present] > 0).double().mean().item())
        else:
            fg_macro_dice = float(fg_dice.mean().item()) if fg_dice.numel() > 0 else float(dice.mean().item())
            fg_macro_recall = float(fg_recall.mean().item()) if fg_recall.numel() > 0 else float(recall.mean().item())
            fg_min_dice = float(fg_dice.min().item()) if fg_dice.numel() > 0 else float(dice.min().item())
            fg_presence_recall = 0.0

        pred_ratio = (pred_support / total).cpu().numpy()
        true_ratio = (support / total).cpu().numpy()
        if len(pred_ratio[fg_start:]) > 0:
            fg_pred = pred_ratio[fg_start:]
            fg_true = true_ratio[fg_start:]
            fg_mass = float(np.clip(fg_pred.sum(), 1e-8, None))
            fg_dominant_ratio = float(np.max(fg_pred) / fg_mass) if fg_pred.size > 0 else 0.0
            fg_distribution_gap = float(np.mean(np.abs(fg_pred - fg_true))) if fg_pred.size > 0 else 0.0
        else:
            fg_dominant_ratio = 0.0
            fg_distribution_gap = 0.0

        per_class = {}
        for idx in range(self.num_classes):
            per_class[self.class_names[idx]] = {
                "dice": float(dice[idx].item()),
                "iou": float(iou[idx].item()),
                "precision": float(precision[idx].item()),
                "recall": float(recall[idx].item()),
                "support": float(support[idx].item()),
                "pred_support": float(pred_support[idx].item()),
            }

        rare_values = [float(dice[idx].item()) for idx in self.rare_class_indices if support[idx] > 0]
        rare_macro_dice = float(np.mean(rare_values)) if rare_values else fg_macro_dice

        summary = {
            "dice": fg_macro_dice,
            "macro_dice": fg_macro_dice,
            "macro_recall": fg_macro_recall,
            "min_fg_dice": fg_min_dice,
            "presence_recall": fg_presence_recall,
            "rare_dice": rare_macro_dice,
            "fg_distribution_gap": fg_distribution_gap,
            "fg_dominant_ratio": fg_dominant_ratio,
            "pixel_accuracy": float(tp.sum().item() / total.item()),
        }
        return {"summary": summary, "per_class": per_class, "pred_ratio": pred_ratio.tolist(), "true_ratio": true_ratio.tolist()}

    def _batch_dice(self, outputs, targets):
        preds = torch.argmax(outputs, dim=1)
        confusion = self._confusion_matrix(preds, targets)
        metrics = self._metrics_from_confusion(confusion)
        return float(metrics["summary"]["dice"])

    def train_one_epoch(self, train_loader, stage_cfg=None):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        epoch_confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
        total_loss = 0.0
        num_batches = 0
        running_loss_components = {}
        optimizer_steps = 0

        loss_weights = (stage_cfg or {}).get("loss_weights", {}) if isinstance(stage_cfg, Mapping) else {}
        w_stage1 = float(loss_weights.get("stage1", 0.0))
        w_comp = float(loss_weights.get("composite", 1.0))

        an_epoch = tqdm(train_loader, desc=f"Training", total=len(train_loader))
        
        for step, batch in enumerate(an_epoch):
            if self.max_train_steps_per_epoch > 0 and step >= self.max_train_steps_per_epoch:
                an_epoch.close()
                break

            images, targets, _ = self._unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)

            if self.channels_last and images.ndim == 4 and self.device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            
            targets = targets.to(self.device, non_blocking=True).long()

            try:
                with self._autocast_context():
                    # ⭐ forward pass
                    logits, aux_outputs = self._forward_with_aux(images)
                    # ⭐ compute losses
                    main_loss, loss_dict = self.criterion.compute(logits, targets)
                    aux_loss, aux_loss_dict = self._compute_aux_loss(aux_outputs, targets)
                    composite_total = main_loss + self.aux_loss_weight * aux_loss
                    spectral_features = aux_outputs.get("spectral_features", None)
                    spectral_logits = aux_outputs.get("spectral_logits", None)
                    stage1_total, stage1_loss_dict = self.stage1_criterion.compute(spectral_features, spectral_logits, targets)
                    total_batch_loss = w_comp * composite_total + w_stage1 * stage1_total
            except RuntimeError as exc:
                if self.device.type == "cuda" and self.oom_skip_batch and "out of memory" in str(exc).lower():
                    tprint(f"[warn] OOM at train step {step}, skip batch")
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                raise

            # ⭐ scale loss
            scaled_loss = total_batch_loss / self.grad_accum_steps
            # ⭐ backward pass
            self.scaler.scale(scaled_loss).backward()

            should_step = ((step + 1) % self.grad_accum_steps == 0) or ((step + 1) == len(train_loader))
            if should_step:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                prev_scale = float(self.scaler.get_scale()) if (self.use_amp and self.device.type == "cuda") else 1.0
                self.scaler.step(self.optimizer)
                self.scaler.update()
                new_scale = float(self.scaler.get_scale()) if (self.use_amp and self.device.type == "cuda") else 1.0
                optimizer_stepped = not (self.use_amp and self.device.type == "cuda" and new_scale < prev_scale)
                self.optimizer.zero_grad(set_to_none=True)
                if optimizer_stepped:
                    optimizer_steps += 1

                    if self.scheduler is not None and self.scheduler_step_on == "step":
                        self.scheduler.step()
                    if self.ema is not None:
                        self.ema.update(self.model)
                    if self.device.type == "cuda" and self.empty_cache_steps > 0 and (optimizer_steps % self.empty_cache_steps == 0):
                        torch.cuda.empty_cache()

            preds = torch.argmax(logits.detach(), dim=1)
            epoch_confusion += self._confusion_matrix(preds.cpu(), targets.detach().cpu())
            total_loss += float(total_batch_loss.detach().cpu().item())
            num_batches += 1

            merged_loss_dict = dict(loss_dict)
            if aux_loss_dict:
                merged_loss_dict.update(aux_loss_dict)
            merged_loss_dict.update(stage1_loss_dict)
            merged_loss_dict["composite_total"] = float(composite_total.detach().cpu().item())
            merged_loss_dict["total"] = float(total_batch_loss.detach().cpu().item())
            for key, value in merged_loss_dict.items():
                running_loss_components[key] = running_loss_components.get(key, 0.0) + float(value)

        if self.scheduler is not None and self.scheduler_step_on == "epoch" and optimizer_steps > 0:
            self.scheduler.step()

        avg_loss = total_loss / max(1, num_batches)
        metrics = self._metrics_from_confusion(epoch_confusion)
        avg_loss_components = {k: v / max(1, num_batches) for k, v in running_loss_components.items()}
        metrics["loss"] = avg_loss
        metrics["loss_components"] = avg_loss_components
        metrics["optimizer_steps"] = optimizer_steps
        return metrics

    @torch.no_grad()
    def validate(self, val_loader, stage_cfg=None):
        model = self.ema.ema if self.ema is not None else self.model
        model.eval()

        epoch_confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
        total_loss = 0.0
        num_batches = 0
        running_loss_components = {}

        loss_weights = (stage_cfg or {}).get("loss_weights", {}) if isinstance(stage_cfg, Mapping) else {}
        w_stage1 = float(loss_weights.get("stage1", 0.0))
        w_comp = float(loss_weights.get("composite", 1.0))

        for step, batch in enumerate(val_loader):
            if self.max_eval_steps > 0 and step >= self.max_eval_steps:
                break
            images, targets, _ = self._unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            if self.channels_last and images.ndim == 4 and self.device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            targets = targets.to(self.device, non_blocking=True).long()

            with self._autocast_context():
                if hasattr(model, "forward_with_aux"):
                    result = model.forward_with_aux(images)
                    if isinstance(result, tuple):
                        logits = result[0]
                        aux_outputs = self._normalize_aux_outputs(result[1] if len(result) > 1 else None)
                    elif isinstance(result, dict):
                        logits = result.get("logits", result.get("main"))
                        aux_outputs = self._normalize_aux_outputs({k: v for k, v in result.items() if k not in {"logits", "main"}})
                    else:
                        logits = result
                        aux_outputs = {}
                else:
                    logits = model(images)
                    aux_outputs = {}
                    if hasattr(model, "get_aux_outputs"):
                        aux_outputs = self._normalize_aux_outputs(model.get_aux_outputs())
                    elif hasattr(model, "get_spectral_supervision_tensors"):
                        aux_outputs = self._normalize_aux_outputs(model.get_spectral_supervision_tensors())

                main_loss, loss_dict = self.criterion.compute(logits, targets)
                aux_loss, aux_loss_dict = self._compute_aux_loss(aux_outputs, targets)
                composite_total = main_loss + self.aux_loss_weight * aux_loss
                spectral_features = aux_outputs.get("spectral_features", None)
                spectral_logits = aux_outputs.get("spectral_logits", None)
                stage1_total, stage1_loss_dict = self.stage1_criterion.compute(spectral_features, spectral_logits, targets)
                total_batch_loss = w_comp * composite_total + w_stage1 * stage1_total

            preds = torch.argmax(logits, dim=1)
            epoch_confusion += self._confusion_matrix(preds.cpu(), targets.cpu())
            total_loss += float(total_batch_loss.detach().cpu().item())
            num_batches += 1

            merged_loss_dict = dict(loss_dict)
            if aux_loss_dict:
                merged_loss_dict.update(aux_loss_dict)
            merged_loss_dict.update(stage1_loss_dict)
            merged_loss_dict["composite_total"] = float(composite_total.detach().cpu().item())
            merged_loss_dict["total"] = float(total_batch_loss.detach().cpu().item())
            for key, value in merged_loss_dict.items():
                running_loss_components[key] = running_loss_components.get(key, 0.0) + float(value)

        avg_loss = total_loss / max(1, num_batches)
        metrics = self._metrics_from_confusion(epoch_confusion)
        avg_loss_components = {k: v / max(1, num_batches) for k, v in running_loss_components.items()}
        metrics["loss"] = avg_loss
        metrics["loss_components"] = avg_loss_components
        return metrics

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "best_val_dice": self.best_val_dice,
            "config": _to_serializable(self.config),
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.ema.state_dict()

        last_path = self.checkpoint_dir / "last_checkpoint.pt"
        torch.save(checkpoint, last_path)
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_checkpoint.pt")
            torch.save(self.model.state_dict(), self.checkpoint_dir / "best_model.pt")
            if self.ema is not None:
                torch.save(self.ema.ema.state_dict(), self.checkpoint_dir / "best_model_ema.pt")

    def cleanup_checkpoints(self) -> None:
        keep = {"last_checkpoint.pt", "best_checkpoint.pt", "best_model.pt", "best_model_ema.pt"}
        for pt_path in self.checkpoint_dir.glob("*.pt"):
            if pt_path.name not in keep:
                pt_path.unlink(missing_ok=True)

    def _save_history(self):
        with open(self.output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(_to_serializable(self.history), f, indent=2)

    def fit(self, train_loader, val_loader=None, num_epochs=None):
        if num_epochs is None:
            num_epochs = int(_cfg_get(self.config, "train.epochs", 1))

        if self.staged_enabled and self.stage_definitions:
            stages = self.stage_definitions
        else:
            stages = [
                {
                    "name": "single_stage",
                    "freeze": {"backbone": False, "decoder": False, "spectral_encoder": False},
                    "loss_weights": {"stage1": 0.0, "composite": 1.0},
                    "transition": {"metric": "eval_dice", "mode": "max", "patience": int(num_epochs), "min_delta": 0.0},
                    "scheduler": _cfg_get(self.config, "train.scheduler", "none"),
                    "max_epochs_guard": int(num_epochs),
                }
            ]

        global_epoch = 0
        best_epoch = 0
        early_enabled = bool(_cfg_get(self.config, "train.early_stop.enabled", False))
        early_metric = str(_cfg_get(self.config, "train.early_stop.metric", "eval_dice"))
        early_mode = str(_cfg_get(self.config, "train.early_stop.mode", "max")).lower()
        early_patience = int(_cfg_get(self.config, "train.early_stop.patience", 10))
        early_min_delta = float(_cfg_get(self.config, "train.early_stop.min_delta", 0.0))
        early_best = -float("inf") if early_mode == "max" else float("inf")
        early_bad = 0

        stage_spans: List[Dict[str, Any]] = []
        for stage_idx, stage_cfg in enumerate(stages):
            if global_epoch >= int(num_epochs):
                break
            stage_name = str(stage_cfg.get("name", f"stage_{stage_idx + 1}"))
            self._set_stage_trainability(stage_cfg)

            tprint(f"stage[{stage_name}]: train/eval loop")

            self._build_scheduler(
                train_loader,
                int(stage_cfg.get("max_epochs_guard", num_epochs)),
                scheduler_cfg_override=self._resolve_stage_scheduler_cfg(stage_cfg),
            )

            transition = stage_cfg.get("transition", {})
            metric_name = str(transition.get("metric", "eval_dice"))
            mode = str(transition.get("mode", "max")).lower()
            patience = int(transition.get("patience", 5))
            min_delta = float(transition.get("min_delta", 0.0))
            guard_epochs = int(stage_cfg.get("max_epochs_guard", num_epochs))
            metric_best = -float("inf") if mode == "max" else float("inf")
            bad_epochs = 0
            stage_start = global_epoch + 1

            stage_epoch_budget = min(max(1, guard_epochs), max(0, int(num_epochs) - global_epoch))
            if stage_epoch_budget <= 0:
                break
            
            for _ in range(stage_epoch_budget):
                epoch_start = time.time()
                global_epoch += 1
                train_metrics = self.train_one_epoch(train_loader, stage_cfg=stage_cfg)
                val_metrics = self.validate(val_loader, stage_cfg=stage_cfg) if val_loader is not None else train_metrics
                train_summary = train_metrics["summary"]
                val_summary = val_metrics["summary"]

                self.history["epochs"].append(global_epoch)
                self.history["stage_name"].append(stage_name)
                self.history["lr"].append(self._current_lr())
                self.history["learning_rate"].append(self._current_lr())
                self.history["train_loss"].append(float(train_metrics["loss"]))
                self.history["val_loss"].append(float(val_metrics["loss"]))
                self.history["eval_loss"].append(float(val_metrics["loss"]))
                self.history["train_dice"].append(float(train_summary["dice"]))
                self.history["val_dice"].append(float(val_summary["dice"]))
                self.history["eval_dice"].append(float(val_summary["dice"]))
                self.history["train_macro_dice"].append(float(train_summary["macro_dice"]))
                self.history["val_macro_dice"].append(float(val_summary["macro_dice"]))
                self.history["train_min_fg_dice"].append(float(train_summary["min_fg_dice"]))
                self.history["val_min_fg_dice"].append(float(val_summary["min_fg_dice"]))
                self.history["train_presence_recall"].append(float(train_summary["presence_recall"]))
                self.history["val_presence_recall"].append(float(val_summary["presence_recall"]))
                self.history["train_rare_dice"].append(float(train_summary["rare_dice"]))
                self.history["val_rare_dice"].append(float(val_summary["rare_dice"]))
                self.history["train_fg_distribution_gap"].append(float(train_summary["fg_distribution_gap"]))
                self.history["val_fg_distribution_gap"].append(float(val_summary["fg_distribution_gap"]))
                self.history["train_fg_dominant_ratio"].append(float(train_summary["fg_dominant_ratio"]))
                self.history["val_fg_dominant_ratio"].append(float(val_summary["fg_dominant_ratio"]))
                self.history["train_per_class_dice"].append({k: float(v["dice"]) for k, v in train_metrics["per_class"].items()})
                self.history["val_per_class_dice"].append({k: float(v["dice"]) for k, v in val_metrics["per_class"].items()})
                self.history["train_loss_components"].append(train_metrics.get("loss_components", {}))
                self.history["val_loss_components"].append(val_metrics.get("loss_components", {}))

                is_best = float(val_summary["dice"]) >= float(self.best_val_dice)
                if is_best:
                    self.best_val_dice = float(val_summary["dice"])
                    best_epoch = global_epoch
                self._save_checkpoint(global_epoch, is_best=is_best)
                self._save_history()

                metric_value = None
                if metric_name in {"eval_dice", "val_dice"}:
                    metric_value = float(val_summary["dice"])
                elif metric_name in {"train_stage1_total", "stage1_total"}:
                    metric_value = float(train_metrics.get("loss_components", {}).get("stage1_total", float("inf")))
                elif metric_name in {"val_loss", "eval_loss"}:
                    metric_value = float(val_metrics["loss"])
                else:
                    metric_value = float(val_summary.get("dice", 0.0))

                improved = False
                if mode == "max":
                    improved = metric_value > (metric_best + min_delta)
                else:
                    improved = metric_value < (metric_best - min_delta)
                if improved:
                    metric_best = metric_value
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                if early_enabled:
                    early_target = float(val_summary["dice"]) if early_metric in {"eval_dice", "val_dice"} else float(val_metrics["loss"])
                    if early_mode == "max":
                        early_improved = early_target > (early_best + early_min_delta)
                    else:
                        early_improved = early_target < (early_best - early_min_delta)
                    if early_improved:
                        early_best = early_target
                        early_bad = 0
                    else:
                        early_bad += 1
                    if early_bad >= early_patience:
                        stage_spans.append({"name": stage_name, "start": stage_start, "end": global_epoch})
                        self.history["stage_spans"] = stage_spans
                        return TrainerResult(history=self.history, best_epoch=best_epoch or global_epoch, best_metric=float(self.best_val_dice))

                epoch_time = time.time() - epoch_start
                print(
                    f"Epoch {global_epoch}\n"
                    f"\t- stage: {stage_name}\n"
                    f"\t- lr: {self._current_lr():.6e}\n"
                    f"\t- train_loss: {train_metrics['loss']:.4f}\n"
                    f"\t- val_loss: {val_metrics['loss']:.4f}\n"
                    f"\t- train_dice: {train_summary['dice']:.4f}\n"
                    f"\t- val_dice: {val_summary['dice']:.4f}\n"
                    f"\t- time: {epoch_time:.1f}s\n"
                )

                if bad_epochs >= patience:
                    break
            stage_spans.append({"name": stage_name, "start": stage_start, "end": global_epoch})

        self.history["stage_spans"] = stage_spans
        return TrainerResult(history=self.history, best_epoch=best_epoch or global_epoch, best_metric=float(self.best_val_dice))

    @torch.no_grad()
    def predict(self, data_loader=None, dataloader=None, keep_images: int = 0, keep_features: int = 0):
        if data_loader is None and dataloader is None:
            raise ValueError("predict requires `data_loader` or `dataloader`")
        data_loader = data_loader if data_loader is not None else dataloader
        model = self.ema.ema if self.ema is not None else self.model
        model.eval()

        probabilities = []
        predictions = []
        targets_all = []
        image_samples = []

        for batch in data_loader:
            images, targets, _ = self._unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)

            with self._autocast_context():
                logits = model(images)
                probs = F.softmax(logits, dim=1)

            probabilities.append(probs.detach().cpu().numpy())
            predictions.append(torch.argmax(probs, dim=1).detach().cpu().numpy())
            if targets is not None:
                targets_all.append(targets.detach().cpu().numpy())
            if keep_images > 0 and len(image_samples) < keep_images:
                image_samples.extend(images.detach().cpu().numpy())

        result = {
            "probabilities": np.concatenate(probabilities, axis=0) if probabilities else np.empty((0, self.num_classes, 0, 0)),
            "predictions": np.concatenate(predictions, axis=0) if predictions else np.empty((0, 0, 0), dtype=np.int64),
        }
        if targets_all:
            result["targets"] = np.concatenate(targets_all, axis=0)
            confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
            preds = result["predictions"]
            targets = result["targets"]
            for pred_map, target_map in zip(preds, targets):
                pred_t = torch.from_numpy(pred_map)
                target_t = torch.from_numpy(target_map)
                confusion += self._confusion_matrix(pred_t, target_t).numpy()
            result["metrics"] = self._metrics_from_confusion(torch.from_numpy(confusion))
        result["prob_maps"] = result["probabilities"]
        result["pred_masks"] = result["predictions"]
        result["gt_masks"] = result.get("targets", np.empty((0, 0, 0), dtype=np.int64))
        result["image_samples"] = image_samples[:keep_images] if keep_images > 0 else []
        result["features"] = np.empty((0, 0), dtype=np.float32)
        result["feature_labels"] = np.empty((0,), dtype=np.int64)
        return result

    @torch.no_grad()
    def extract_features(self, data_loader, max_points: int = 0):
        model = self.ema.ema if self.ema is not None else self.model
        model.eval()

        features = []
        targets_all = []

        for batch in data_loader:
            images, targets, _ = self._unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)

            if not hasattr(model, "forward_features"):
                raise AttributeError("Model does not implement forward_features().")

            # Feature probing for visualization prefers stability over speed.
            # Keep this path in fp32 to avoid CUDA SDP kernel issues under AMP.
            with torch.cuda.amp.autocast(enabled=False):
                feats = model.forward_features(images.float())

            features.append(feats.detach().cpu().numpy())
            if targets is not None:
                targets_all.append(targets.detach().cpu().numpy())

        result = {
            "features": np.concatenate(features, axis=0) if features else np.empty((0, 0, 0, 0)),
        }
        if targets_all:
            result["targets"] = np.concatenate(targets_all, axis=0)
        if max_points > 0 and result["features"].size > 0:
            feats = result["features"].reshape(result["features"].shape[0], -1)
            labels = result.get("targets", np.empty((feats.shape[0],), dtype=np.int64))
            if labels.ndim > 1:
                labels = labels.reshape(labels.shape[0], -1)[:, 0]
            take = min(max_points, feats.shape[0])
            idx = np.random.choice(feats.shape[0], size=take, replace=False)
            return feats[idx], labels[idx]
        return result