from __future__ import annotations

import contextlib
# trainer for segmentation models with dice/focal/tversky losses.
import copy
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from munch import Munch
from tqdm import tqdm
from tqdm import TqdmExperimentalWarning

from pipeline.monitor import tprint


os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def set_trainable_layers(model: nn.Module, config: Munch) -> None:
    """Optionally unfreeze last N backbone blocks from trainer side."""
    cfg = config
    n = int(getattr(cfg.model, "unfreeze_last_n", 0))
    if n <= 0:
        return

    backbone = getattr(model, "backbone", None)
    if backbone is None or not hasattr(backbone, "blocks"):
        return

    blocks = list(backbone.blocks)
    if not blocks:
        return

    n = min(n, len(blocks))
    for blk in blocks[-n:]:
        for p in blk.parameters():
            p.requires_grad_(True)

    if hasattr(backbone, "norm"):
        for p in backbone.norm.parameters():
            p.requires_grad_(True)

class CompositeSegLoss(nn.Module):
    """composite segmentation loss = dice + focal + tversky."""

    def __init__(self, config: Munch, num_classes: int) -> None:
        super().__init__()
        cfg = config
        self.num_classes = num_classes

        self.dice_weight = float(cfg.loss.dice_weight)
        self.focal_weight = float(cfg.loss.focal_weight)
        self.tversky_weight = float(cfg.loss.tversky_weight)

        self.focal_gamma = float(cfg.loss.focal_gamma)
        self.tversky_alpha = float(cfg.loss.tversky_alpha)
        self.tversky_beta = float(cfg.loss.tversky_beta)

        class_weights = np.asarray(cfg.loss.class_weights, dtype=np.float32)
        if class_weights.size != self.num_classes:
            class_weights = np.ones(self.num_classes, dtype=np.float32)
        self.register_buffer("class_weights", torch.from_numpy(class_weights))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self._dice_loss(logits, targets)
        focal = self._focal_loss(logits, targets)
        tversky = self._tversky_loss(logits, targets)
        return self.dice_weight * dice + self.focal_weight * focal + self.tversky_weight * tversky

    def _dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        reduce_dims = (0, 2, 3)
        inter = torch.sum(probs * one_hot, dim=reduce_dims)
        den = torch.sum(probs, dim=reduce_dims) + torch.sum(one_hot, dim=reduce_dims)
        dice = (2.0 * inter + 1e-6) / (den + 1e-6)

        # foreground mean dice for better small-target emphasis.
        fg = dice[1:] if dice.numel() > 1 else dice
        return 1.0 - fg.mean()

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.class_weights)
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.focal_gamma) * ce
        return focal.mean()

    def _tversky_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        reduce_dims = (0, 2, 3)
        tp = torch.sum(probs * one_hot, dim=reduce_dims)
        fp = torch.sum(probs * (1.0 - one_hot), dim=reduce_dims)
        fn = torch.sum((1.0 - probs) * one_hot, dim=reduce_dims)

        score = (tp + 1e-6) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + 1e-6)
        fg = score[1:] if score.numel() > 1 else score
        return 1.0 - fg.mean()


class ModelEMA:
    """exponential moving average model wrapper."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.ema_model = self._clone_model(model)

    def _clone_model(self, model: nn.Module) -> nn.Module:
        clone = copy.deepcopy(model)
        clone.to(next(model.parameters()).device)
        clone.eval()
        for p in clone.parameters():
            p.requires_grad_(False)
        return clone

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()
        for key in ema_state.keys():
            if torch.is_floating_point(ema_state[key]):
                ema_state[key].mul_(self.decay).add_(model_state[key], alpha=1.0 - self.decay)
            else:
                ema_state[key].copy_(model_state[key])


@dataclass
class TrainerResult:
    """trainer output summary."""

    best_metric: float
    best_epoch: int
    best_ckpt_path: str
    history: Dict[str, List[float]]


class Trainer:
    """training loop manager with grad accumulation and ema."""

    def __init__(self, model: nn.Module, config: Munch, output_dir: str) -> None:
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.show_progress = bool(getattr(self.config.runtime, "progress_bar", True))

        device_name = str(self.config.runtime.device).lower()
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)

        self.model.to(self.device)
        set_trainable_layers(self.model, self.config)
        self.loss_fn = CompositeSegLoss(self.config, num_classes=int(self.config.data.num_classes)).to(self.device)

        params = [p for p in self.model.parameters() if p.requires_grad] or list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=float(self.config.train.lr),
            weight_decay=float(self.config.train.weight_decay),
        )

        self.use_amp = bool(self.config.train.amp) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.grad_clip = float(self.config.train.grad_clip)
        self.grad_accum_steps = max(1, int(self.config.train.grad_accum_steps))

        ema_decay = float(self.config.train.ema_decay)
        self.ema = ModelEMA(self.model, decay=ema_decay) if ema_decay > 0 else None

        self.max_train_steps = int(self.config.train.max_train_steps_per_epoch)
        self.max_eval_steps = int(self.config.train.max_eval_steps)

        total_params = int(sum(p.numel() for p in self.model.parameters()))
        trainable_params = int(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        tprint(
            "trainer init:\n"
            f"\tdevice={self.device} amp={self.use_amp} ema_decay={ema_decay}\n"
            f"\ttrainable_params={trainable_params} total_params={total_params}\n"
        )
        tprint(
            "optimizer:\n"
            f"\tAdamW(lr={float(self.config.train.lr):.6g},\n"
            f"\tweight_decay={float(self.config.train.weight_decay):.6g},\n"
            f"\tgrad_clip={self.grad_clip}, grad_accum_steps={self.grad_accum_steps})"
        )

    def fit(self, train_loader, eval_loader, epochs: int) -> TrainerResult:
        """train model and keep best checkpoint by eval dice."""
        history = {
            "train_loss": [],
            "eval_loss": [],
            "train_dice": [],
            "eval_dice": [],
        }

        best_metric = -1.0
        best_epoch = 0
        # Keep .pt checkpoint as a temporary training artifact only.
        # Final deployment artifact is exported as ONNX by pipeline.core.
        best_ckpt = self.output_dir / ".cache" / "best_model.pt"
        best_ckpt.parent.mkdir(parents=True, exist_ok=True)

        # # debug
        # an iterable may cause unexpectedly long time cost.
        # sample_batch = next(iter(train_loader))
        # img, masks = sample_batch[0]
        # shapes = (img.shape, masks.shape)

        tprint(
            "fit start:\n"
            f"\tepochs = {epochs}\n"
            f"\ttrain batches = {len(train_loader)}, eval batches = {len(eval_loader)} \n"
            # f"\tbatch shape = {shapes}\n"
        )

        for epoch in range(1, epochs + 1):
            tprint(f"epoch {epoch:03d}/{epochs:03d} start")
            epoch_t0 = time.perf_counter()
            train_loss, train_dice, train_t = self.train_one_epoch(train_loader, epoch=epoch, total_epochs=epochs)
            eval_loss, eval_dice, eval_t = self.validate(eval_loader, phase="eval", epoch=epoch, total_epochs=epochs)
            epoch_total = time.perf_counter() - epoch_t0

            history["train_loss"].append(train_loss)
            history["eval_loss"].append(eval_loss)
            history["train_dice"].append(train_dice)
            history["eval_dice"].append(eval_dice)

            tprint(
                f"epoch {epoch:03d}:\n"
                f"\ttrain_loss = {train_loss:.4f}, eval_loss = {eval_loss:.4f}\n"
                f"\ttrain_dice = {train_dice:.4f}, eval_dice = {eval_dice:.4f}\n"
            )
            tprint(
                "epoch timing:\n"
                f"\ttrain_total={train_t['total_s']:.2f}s, train_first_batch={train_t['first_batch_s']:.2f}s\n"
                f"\teval_total={eval_t['total_s']:.2f}s, eval_first_batch={eval_t['first_batch_s']:.2f}s\n"
                f"\tepoch_total={epoch_total:.2f}s"
            )

            if eval_dice > best_metric:
                best_metric = eval_dice
                best_epoch = epoch
                self.save_checkpoint(str(best_ckpt), epoch=best_epoch, metric=best_metric)
        self.load_checkpoint(str(best_ckpt))

        # Clean temporary checkpoint so the run output is ONNX-oriented.
        if best_ckpt.exists():
            with contextlib.suppress(OSError):
                best_ckpt.unlink()
        if best_ckpt.parent.exists():
            try:
                next(best_ckpt.parent.iterdir())
            except StopIteration:
                with contextlib.suppress(OSError):
                    best_ckpt.parent.rmdir()
        return TrainerResult(
            best_metric=best_metric,
            best_epoch=best_epoch,
            best_ckpt_path=str(best_ckpt),
            history=history,
        )

    @staticmethod
    def _planned_steps(dataloader, max_steps: int) -> Optional[int]:
        total = len(dataloader)
        return min(total, max_steps) if max_steps > 0 else total

    def train_one_epoch(self, dataloader, epoch: int, total_epochs: int) -> Tuple[float, float, Dict[str, float]]:
        """run one train epoch and return loss/dice."""
        self.model.train()

        losses: List[float] = []
        dice_scores: List[float] = []

        self.optimizer.zero_grad(set_to_none=True)
        epoch_t0 = time.perf_counter()
        first_batch_s: Optional[float] = None

        train_iter = tqdm(
            enumerate(dataloader, start=1),
            total=self._planned_steps(dataloader, self.max_train_steps),
            desc=f"train {epoch:03d}/{total_epochs:03d}",
            dynamic_ncols=True,
            leave=False,
            disable=not self.show_progress,
            mininterval=0.2,
        )

        for step, (images, masks) in train_iter:
            if self.max_train_steps > 0 and step > self.max_train_steps:
                break

            if first_batch_s is None:
                first_batch_s = time.perf_counter() - epoch_t0

            # # debug
            # print("\n>>> BEFORE MODEL:", images.shape)
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, masks) / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if step % self.grad_accum_steps == 0:
                self._unscale_optimizer()
            losses.append(float(loss.item() * self.grad_accum_steps))
            dice_scores.append(float(self._batch_dice(logits, masks)))

            train_iter.set_postfix(
                loss=f"{losses[-1]:.4f}",
                dice=f"{dice_scores[-1]:.4f}",
                refresh=False,
            )

        train_iter.close()
        total_s = time.perf_counter() - epoch_t0

        if not losses:
            return 0.0, 0.0, {
                "total_s": float(total_s),
                "first_batch_s": float(first_batch_s if first_batch_s is not None else total_s),
            }

        return float(np.mean(losses)), float(np.mean(dice_scores)), {
            "total_s": float(total_s),
            "first_batch_s": float(first_batch_s if first_batch_s is not None else total_s),
        }

    # TODO Rename this here and in `train_one_epoch`
    def _unscale_optimizer(self):
        """unscale gradients, optionally clip, step optimizer, and update ema."""
        self.scaler.unscale_(self.optimizer)
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if self.ema is not None:
            self.ema.update(self.model)

    @torch.no_grad()
    def validate(self, dataloader, phase: str, epoch: int, total_epochs: int) -> Tuple[float, float, Dict[str, float]]:
        """run one validation epoch and return loss/dice."""
        model = self.ema.ema_model if self.ema is not None else self.model
        model.eval()

        losses: List[float] = []
        dice_scores: List[float] = []
        eval_t0 = time.perf_counter()
        first_batch_s: Optional[float] = None

        eval_iter = tqdm(
            enumerate(dataloader, start=1),
            total=self._planned_steps(dataloader, self.max_eval_steps),
            desc=f"{phase}  {epoch:03d}/{total_epochs:03d}",
            dynamic_ncols=True,
            leave=False,
            disable=not self.show_progress,
            mininterval=0.2,
        )

        for step, (images, masks) in eval_iter:
            if self.max_eval_steps > 0 and step > self.max_eval_steps:
                break

            if first_batch_s is None:
                first_batch_s = time.perf_counter() - eval_t0

            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = model(images)
            loss = self.loss_fn(logits, masks)

            losses.append(float(loss.item()))
            dice_scores.append(float(self._batch_dice(logits, masks)))

            eval_iter.set_postfix(
                loss=f"{losses[-1]:.4f}",
                dice=f"{dice_scores[-1]:.4f}",
                refresh=False,
            )

        eval_iter.close()
        total_s = time.perf_counter() - eval_t0

        if not losses:
            return 0.0, 0.0, {
                "total_s": float(total_s),
                "first_batch_s": float(first_batch_s if first_batch_s is not None else total_s),
            }

        return float(np.mean(losses)), float(np.mean(dice_scores)), {
            "total_s": float(total_s),
            "first_batch_s": float(first_batch_s if first_batch_s is not None else total_s),
        }

    @torch.no_grad()
    def predict(
        self,
        dataloader,
        keep_images: int = 3,
        keep_features: int = 3000,
    ) -> Dict[str, Any]:
        """
        predict masks and collect arrays for analysis/visualization.
        
        Return Dict keys:
        - `pred_masks`: `list` of predicted mask `np.ndarray`.
        - `gt_masks`: `list` of ground truth mask `np.ndarray`.
        - `prob_maps`: `list` of predicted probability map `np.ndarray`.
        - `image_samples`: `list` of input image `np.ndarray` (up to keep_images).
        - `features`: `np.ndarray` of collected feature vectors (up to keep_features).
        - `feature_labels`: `np.ndarray` of corresponding feature labels.
        - `attention_map`: `np.ndarray` of optional attention map array if model provides it.
        """
        model = self.ema.ema_model if self.ema is not None else self.model
        model.eval()

        pred_masks: List[np.ndarray] = []
        gt_masks: List[np.ndarray] = []
        prob_maps: List[np.ndarray] = []
        image_samples: List[np.ndarray] = []
        feature_bank: List[np.ndarray] = []
        label_bank: List[np.ndarray] = []
        attention_map: Optional[np.ndarray] = None

        collected_features = 0

        test_iter = tqdm(
            enumerate(dataloader, start=1),
            total=self._planned_steps(dataloader, self.max_eval_steps),
            desc="test  predict",
            dynamic_ncols=True,
            leave=False,
            disable=not self.show_progress,
            mininterval=0.2,
        )

        for step, (images, masks) in test_iter:
            if self.max_eval_steps > 0 and step > self.max_eval_steps:
                break

            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            if attention_map is None and hasattr(model, "get_attention_map"):
                attn = model.get_attention_map()
                if attn is not None:
                    attention_map = attn.detach().cpu().numpy()

            pred_np = preds.detach().cpu().numpy()
            mask_np = masks.detach().cpu().numpy()
            prob_np = probs.detach().cpu().numpy()

            pred_masks.extend(list(pred_np))
            gt_masks.extend(list(mask_np))
            prob_maps.extend(list(prob_np))

            if len(image_samples) < keep_images:
                remain = keep_images - len(image_samples)
                image_samples.extend(list(images.detach().cpu().numpy()[:remain]))

            # collect feature vectors for tsne.
            if hasattr(model, "forward_features") and collected_features < keep_features:
                feat = model(images)
                feat_np = feat.detach().cpu().numpy()
                b, c, h, w = feat_np.shape
                feat_flat = np.moveaxis(feat_np, 1, -1).reshape(-1, c)

                mask_small = F.interpolate(masks.unsqueeze(1).float(), size=(h, w), mode="nearest")
                mask_flat = mask_small.squeeze(1).long().detach().cpu().numpy().reshape(-1)

                need = max(0, keep_features - collected_features)
                if feat_flat.shape[0] > need:
                    idx = np.random.choice(feat_flat.shape[0], size=need, replace=False)
                    feat_flat = feat_flat[idx]
                    mask_flat = mask_flat[idx]

                feature_bank.append(feat_flat)
                label_bank.append(mask_flat)
                collected_features += feat_flat.shape[0]

            test_iter.set_postfix(
                kept_imgs=f"{len(image_samples)}/{keep_images}",
                feat_pts=collected_features,
                refresh=False,
            )

        test_iter.close()

        return {
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
            "prob_maps": prob_maps,
            "image_samples": image_samples,
            "features": np.concatenate(feature_bank, axis=0) if feature_bank else np.empty((0, 0), dtype=np.float32),
            "feature_labels": np.concatenate(label_bank, axis=0) if label_bank else np.empty((0,), dtype=np.int64),
            "attention_map": attention_map,
        }

    def save_checkpoint(self, path: str, epoch: int, metric: float) -> None:
        """save checkpoint with model and optimizer state."""
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch": epoch,
            "metric": metric,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.ema is not None:
            payload["ema_model"] = self.ema.ema_model.state_dict()
        torch.save(payload, str(ckpt_path))

    def load_checkpoint(self, path: str) -> None:
        """load checkpoint into model and ema."""
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        if self.ema is not None and "ema_model" in payload:
            self.ema.ema_model.load_state_dict(payload["ema_model"])

    @staticmethod
    def _batch_dice(logits: torch.Tensor, targets: torch.Tensor) -> float:
        """compute foreground mean dice for one batch."""
        num_classes = logits.shape[1]
        preds = torch.argmax(logits, dim=1)

        dices: List[torch.Tensor] = []
        for cls_idx in range(1, num_classes):
            pred_c = (preds == cls_idx)
            tgt_c = (targets == cls_idx)
            inter = torch.sum(pred_c & tgt_c).float()
            den = torch.sum(pred_c).float() + torch.sum(tgt_c).float()
            dice = (2.0 * inter + 1e-6) / (den + 1e-6)
            dices.append(dice)

        return float(torch.mean(torch.stack(dices)).item()) if dices else 0.0
