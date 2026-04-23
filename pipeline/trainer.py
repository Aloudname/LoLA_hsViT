import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(config, path, default=None):
    current = config
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(part, None)
        else:
            current = getattr(current, part, None)
    return default if current is None else current


def _to_serializable(value):
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


class CompositeSegLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = int(_cfg_get(config, "model.num_classes", _cfg_get(config, "data.num_classes", 2)))
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

    def _build_class_weights(self):
        weights = torch.ones(self.num_classes, dtype=torch.float32)
        configured = _cfg_get(self.config, "loss.class_weights", None)
        if configured is not None:
            configured = list(configured)
            limit = min(len(configured), self.num_classes)
            weights[:limit] = torch.tensor(configured[:limit], dtype=torch.float32)

        rare_indices = _cfg_get(self.config, "loss.rare_class_indices", None)
        if rare_indices is None:
            class_names = _cfg_get(self.config, "data.class_names", None) or []
            rare_indices = [idx for idx, name in enumerate(class_names) if str(name).lower() == "pg"]
        rare_boost = float(_cfg_get(self.config, "loss.rare_class_boost", 1.0))
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

    def _get_weights(self, device):
        return self.class_weights.to(device)

    def _foreground_indices(self, device):
        start = 0 if self.include_background else 1
        if start >= self.num_classes:
            start = 0
        return torch.arange(start, self.num_classes, device=device)

    def _standard_focal_loss(self, logits, targets, class_weights):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha = class_weights.gather(0, targets.view(-1)).view_as(target_probs)
        focal = -alpha * torch.pow(1.0 - target_probs.clamp(0.0, 1.0), self.focal_gamma) * target_log_probs
        return focal.mean()

    def _masked_class_mean(self, values, mask, weights=None):
        if weights is not None:
            weights = weights * mask.float()
            denom = weights.sum()
            if denom <= 0:
                return values.new_tensor(0.0)
            return (values * weights).sum() / denom
        mask_values = values[mask]
        if mask_values.numel() == 0:
            return values.new_tensor(0.0)
        return mask_values.mean()

    def _dice_loss(self, probs, targets_one_hot, class_weights):
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

    def _tversky_loss(self, probs, targets_one_hot, class_weights):
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

    def compute(self, logits, targets):
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


class Trainer:
    def __init__(self, model, config, output_dir):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        requested_device = _cfg_get(config, "runtime.device", None)
        if requested_device is not None:
            self.device = torch.device(requested_device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

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
        self.optimizer = self._build_optimizer()
        self.scheduler = None
        self.scheduler_mode = "none"
        self.scheduler_step_on = "epoch"

        ema_decay = float(_cfg_get(config, "train.ema_decay", 0.0))
        self.use_ema = bool(_cfg_get(config, "train.use_ema", ema_decay > 0 and bool(_cfg_get(config, "train.ema_decay", None) is not None)))
        self.ema = ModelEMA(self.model, decay=ema_decay) if self.use_ema and ema_decay > 0 else None

        self.use_amp = bool(_cfg_get(config, "runtime.use_amp", _cfg_get(config, "train.use_amp", self.device.type == "cuda")))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.device.type == "cuda")
        self.grad_accum_steps = max(1, int(_cfg_get(config, "train.grad_accum_steps", 1)))
        self.max_grad_norm = float(_cfg_get(config, "train.max_grad_norm", 0.0))
        self.aux_loss_weight = float(_cfg_get(config, "train.aux_loss_weight", _cfg_get(config, "loss.aux_weight", 0.0)))

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_dice": [],
            "val_dice": [],
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
            "epochs": [],
        }
        self.best_val_dice = -float("inf")

    def _build_optimizer(self):
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
        return torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    def _build_scheduler(self, train_loader, num_epochs):
        scheduler_cfg = _cfg_get(self.config, "train.scheduler", "none")
        if isinstance(scheduler_cfg, str):
            scheduler_name = scheduler_cfg.lower()
        else:
            scheduler_name = str(_cfg_get(self.config, "train.scheduler.name", "none")).lower()

        self.scheduler_mode = scheduler_name
        self.scheduler_step_on = "epoch"
        if scheduler_name in {"none", "", "null"}:
            self.scheduler = None
            return

        steps_per_epoch = max(1, math.ceil(len(train_loader) / self.grad_accum_steps))
        total_steps = max(1, steps_per_epoch * int(num_epochs))
        warmup_epochs = float(_cfg_get(self.config, "train.scheduler.warmup_epochs", 0.0))
        warmup_steps = int(_cfg_get(self.config, "train.scheduler.warmup_steps", round(warmup_epochs * steps_per_epoch)))
        min_lr_ratio = float(_cfg_get(self.config, "train.scheduler.min_lr_ratio", 0.01))
        power = float(_cfg_get(self.config, "train.scheduler.power", 0.9))

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
            return

        if scheduler_name == "poly":
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
            step_size = int(_cfg_get(self.config, "train.scheduler.step_size", max(1, int(num_epochs) // 3)))
            gamma = float(_cfg_get(self.config, "train.scheduler.gamma", 0.1))
            self.scheduler_step_on = "epoch"
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
            return

        self.scheduler = None
        self.scheduler_mode = "none"

    def _current_lr(self):
        if not self.optimizer.param_groups:
            return 0.0
        return float(self.optimizer.param_groups[0]["lr"])

    def _autocast_context(self):
        return torch.cuda.amp.autocast(enabled=self.use_amp and self.device.type == "cuda")

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

    def train_one_epoch(self, train_loader):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        epoch_confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
        total_loss = 0.0
        num_batches = 0
        running_loss_components = {}
        optimizer_steps = 0

        for step, batch in enumerate(train_loader):
            images, targets, _ = self._unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True).long()

            with self._autocast_context():
                logits, aux_outputs = self._forward_with_aux(images)
                main_loss, loss_dict = self.criterion.compute(logits, targets)
                aux_loss, aux_loss_dict = self._compute_aux_loss(aux_outputs, targets)
                total_batch_loss = main_loss + self.aux_loss_weight * aux_loss

            scaled_loss = total_batch_loss / self.grad_accum_steps
            self.scaler.scale(scaled_loss).backward()

            should_step = ((step + 1) % self.grad_accum_steps == 0) or ((step + 1) == len(train_loader))
            if should_step:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

                if self.scheduler is not None and self.scheduler_step_on == "step":
                    self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.model)

            preds = torch.argmax(logits.detach(), dim=1)
            epoch_confusion += self._confusion_matrix(preds.cpu(), targets.detach().cpu())
            total_loss += float(total_batch_loss.detach().cpu().item())
            num_batches += 1

            merged_loss_dict = dict(loss_dict)
            if aux_loss_dict:
                merged_loss_dict.update(aux_loss_dict)
            merged_loss_dict["total"] = float(total_batch_loss.detach().cpu().item())
            for key, value in merged_loss_dict.items():
                running_loss_components[key] = running_loss_components.get(key, 0.0) + float(value)

        if self.scheduler is not None and self.scheduler_step_on == "epoch":
            self.scheduler.step()

        avg_loss = total_loss / max(1, num_batches)
        metrics = self._metrics_from_confusion(epoch_confusion)
        avg_loss_components = {k: v / max(1, num_batches) for k, v in running_loss_components.items()}
        metrics["loss"] = avg_loss
        metrics["loss_components"] = avg_loss_components
        metrics["optimizer_steps"] = optimizer_steps
        return metrics

    @torch.no_grad()
    def validate(self, val_loader):
        model = self.ema.ema if self.ema is not None else self.model
        model.eval()

        epoch_confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
        total_loss = 0.0
        num_batches = 0
        running_loss_components = {}

        for batch in val_loader:
            images, targets, _ = self._unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
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
                total_batch_loss = main_loss + self.aux_loss_weight * aux_loss

            preds = torch.argmax(logits, dim=1)
            epoch_confusion += self._confusion_matrix(preds.cpu(), targets.cpu())
            total_loss += float(total_batch_loss.detach().cpu().item())
            num_batches += 1

            merged_loss_dict = dict(loss_dict)
            if aux_loss_dict:
                merged_loss_dict.update(aux_loss_dict)
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

        last_path = self.output_dir / "last_checkpoint.pt"
        torch.save(checkpoint, last_path)
        if is_best:
            torch.save(checkpoint, self.output_dir / "best_checkpoint.pt")
            torch.save(self.model.state_dict(), self.output_dir / "best_model.pt")
            if self.ema is not None:
                torch.save(self.ema.ema.state_dict(), self.output_dir / "best_model_ema.pt")

    def _save_history(self):
        with open(self.output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(_to_serializable(self.history), f, indent=2)

    def fit(self, train_loader, val_loader=None, num_epochs=None):
        if num_epochs is None:
            num_epochs = int(_cfg_get(self.config, "train.epochs", 1))
        self._build_scheduler(train_loader, num_epochs)

        start_epoch = len(self.history["epochs"])
        for epoch in range(start_epoch, start_epoch + int(num_epochs)):
            epoch_start = time.time()
            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.validate(val_loader) if val_loader is not None else train_metrics

            train_summary = train_metrics["summary"]
            val_summary = val_metrics["summary"]

            self.history["epochs"].append(epoch + 1)
            self.history["lr"].append(self._current_lr())
            self.history["learning_rate"].append(self._current_lr())
            self.history["train_loss"].append(float(train_metrics["loss"]))
            self.history["val_loss"].append(float(val_metrics["loss"]))
            self.history["train_dice"].append(float(train_summary["dice"]))
            self.history["val_dice"].append(float(val_summary["dice"]))
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

            self._save_checkpoint(epoch + 1, is_best=is_best)
            self._save_history()

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1}/{start_epoch + int(num_epochs)} "
                f"- lr: {self._current_lr():.6e} "
                f"- train_loss: {train_metrics['loss']:.4f} "
                f"- val_loss: {val_metrics['loss']:.4f} "
                f"- train_dice: {train_summary['dice']:.4f} "
                f"- val_dice: {val_summary['dice']:.4f} "
                f"- val_min_fg_dice: {val_summary['min_fg_dice']:.4f} "
                f"- val_presence_recall: {val_summary['presence_recall']:.4f} "
                f"- val_rare_dice: {val_summary['rare_dice']:.4f} "
                f"- val_fg_gap: {val_summary['fg_distribution_gap']:.4f} "
                f"- val_fg_dom: {val_summary['fg_dominant_ratio']:.4f} "
                f"- time: {epoch_time:.1f}s"
            )

        return self.history

    @torch.no_grad()
    def predict(self, data_loader):
        model = self.ema.ema if self.ema is not None else self.model
        model.eval()

        probabilities = []
        predictions = []
        targets_all = []

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
        return result

    @torch.no_grad()
    def extract_features(self, data_loader):
        model = self.ema.ema if self.ema is not None else self.model
        model.eval()

        features = []
        targets_all = []

        for batch in data_loader:
            images, targets, _ = self._unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)

            if not hasattr(model, "forward_features"):
                raise AttributeError("Model does not implement forward_features().")

            with self._autocast_context():
                feats = model.forward_features(images)

            features.append(feats.detach().cpu().numpy())
            if targets is not None:
                targets_all.append(targets.detach().cpu().numpy())

        result = {
            "features": np.concatenate(features, axis=0) if features else np.empty((0, 0, 0, 0)),
        }
        if targets_all:
            result["targets"] = np.concatenate(targets_all, axis=0)
        return result