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
from typing import Tuple, Callable, Dict, Any, Optional, List
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


class SegmentationLoss(nn.Module):
    """HSI-aware composite segmentation loss on dense logits.

    Total loss:
        L = L_focal + lambda_dice * L_dice + lambda_boundary * L_boundary
    """

    def __init__(self,
                 class_weight=None,
                 gamma: float = 2.0,
                 ignore_index: int = 255,
                 label_smoothing: float = 0.0,
                 dice_weight: float = 0.35,
                 boundary_weight: float = 0.20,
                 boundary_dilation: int = 1):
        super().__init__()
        self.class_weight = class_weight
        self.ignore_index = int(ignore_index)
        self.dice_weight = float(dice_weight)
        self.boundary_weight = float(boundary_weight)
        self.boundary_dilation = int(max(0, boundary_dilation))
        self.loss = FocalLoss(
            weight=class_weight,
            gamma=gamma,
            ignore_index=ignore_index,
            reduction='mean',
            label_smoothing=label_smoothing,
        )

    def _soft_dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        valid = (targets != self.ignore_index)
        if not valid.any():
            return logits.sum() * 0.0

        safe_targets = targets.clone()
        safe_targets[~valid] = 0
        one_hot = F.one_hot(safe_targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()

        valid_f = valid.unsqueeze(1).float()
        probs = probs * valid_f
        one_hot = one_hot * valid_f

        reduce_dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dim=reduce_dims)
        denom = probs.sum(dim=reduce_dims) + one_hot.sum(dim=reduce_dims)
        dice = (2.0 * inter + 1e-6) / (denom + 1e-6)

        # Only classes present in current batch contribute.
        present = one_hot.sum(dim=reduce_dims) > 0
        if present.any():
            return 1.0 - dice[present].mean()
        return logits.sum() * 0.0

    def _boundary_mask(self, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        boundary = torch.zeros_like(valid)
        boundary[:, :, 1:] |= (targets[:, :, 1:] != targets[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
        boundary[:, 1:, :] |= (targets[:, 1:, :] != targets[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
        if self.boundary_dilation > 0:
            k = 2 * self.boundary_dilation + 1
            boundary = F.max_pool2d(
                boundary.float().unsqueeze(1),
                kernel_size=k,
                stride=1,
                padding=self.boundary_dilation,
            ).squeeze(1) > 0.5
        return boundary & valid

    def _boundary_ce_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = (targets != self.ignore_index)
        if not valid.any():
            return logits.sum() * 0.0

        boundary = self._boundary_mask(targets, valid)
        if not boundary.any():
            return logits.sum() * 0.0

        ce_map = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weight,
            ignore_index=self.ignore_index,
            reduction='none',
        )
        return (ce_map * boundary.float()).sum() / boundary.float().sum().clamp_min(1.0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal = self.loss(logits, targets)
        dice = self._soft_dice_loss(logits, targets)
        boundary = self._boundary_ce_loss(logits, targets)
        return focal + self.dice_weight * dice + self.boundary_weight * boundary


def _worker_plot_training_curves(train_losses, eval_losses,
                                 train_accs, eval_accs,
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

    axes[0].plot(train_losses, '--', color='#1f77b4', linewidth=1.8,
                 label='Train loss')
    if eval_losses:
        axes[0].plot(eval_epochs, eval_losses, '--', color='#2ca02c', linewidth=1.8,
                     label='Eval loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Curves'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_accs, label='Train Acc', marker='o', markersize=4)
    if eval_accs:
        axes[1].plot(eval_epochs, eval_accs, label='Eval Acc', marker='s', markersize=4)
    axes[1].axhline(y=best_acc, color='r', linestyle='--', label=f'Best: {best_acc:.2f}%')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy Curve'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output, 'training_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()



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



def _worker_write_pairwise_class_metrics(pairwise_text: str, output: str):
    """Worker: write pairwise class recall diagnostics to text file."""
    if not pairwise_text:
        return
    with open(os.path.join(output, 'classification_metrics_pairwise.txt'), 'w') as f:
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
        self._create_model()
        self._setup_multi_gpu()
        self._setup_training()
        
        # training state
        self.train_losses = []
        self.train_accs = []
        self.eval_accs = []
        self.eval_losses = []
        self.epoch_metrics = []
        self.best_acc = 0.0
        self.best_epoch = 0
        self.best_model_state = None
        
        # tracking for visualizations
        self.lr_history = []          # lr per epoch
        self.grad_norms = []          # gradient L2 norm per epoch
        self.epoch_times = []         # wall-clock time per epoch
        self._eval_call_counter = 0

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
        """Return class indices used for metrics (optionally exclude class-0)."""
        exclude_class0 = bool(getattr(config.common, 'exclude_class0_in_metrics', True))
        if exclude_class0 and num_classes > 1:
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
        """Choose scalar score for early stopping."""
        return float(getattr(self, '_last_ba', 0.0))

    def _parse_model_outputs(self, outputs):
        """Parse model outputs and return dense class logits [B, C, H, W]."""
        if isinstance(outputs, dict):
            logits = outputs.get('logits')
        elif isinstance(outputs, (tuple, list)):
            if len(outputs) < 1:
                raise ValueError("Model output tuple/list is empty")
            logits = outputs[0]
        elif torch.is_tensor(outputs):
            logits = outputs
        else:
            raise TypeError("Model output must be tensor/tuple/list/dict")

        if logits is None:
            raise ValueError("Model output missing logits")
        return logits
    
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
        head_keys = ('seg_decoder', 'seg_head')
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

        self.early_stop_metric = str(getattr(self.config.common, 'early_stop_metric', 'eval')).lower()
        self.save_topk = int(max(1, getattr(self.config.common, 'save_topk_models', 3)))
        self.topk_models = []

        if self.config.clsf.num <= 1:
            raise ValueError("Segmentation supervision requires clsf.num > 1")

        self.criterion = SegmentationLoss(
            class_weight=class_weights,
            gamma=focal_gamma,
            ignore_index=255,
            label_smoothing=getattr(self.config.common, 'label_smoothing', 0.0),
            dice_weight=float(getattr(self.config.common, 'loss_dice_weight', 0.35)),
            boundary_weight=float(getattr(self.config.common, 'loss_boundary_weight', 0.20)),
            boundary_dilation=int(getattr(self.config.common, 'loss_boundary_dilation', 1)),
        )
        print(
            f"  SegmentationLoss(gamma={focal_gamma}, "
            f"dice_w={getattr(self.config.common, 'loss_dice_weight', 0.35)}, "
            f"boundary_w={getattr(self.config.common, 'loss_boundary_weight', 0.20)})"
        )
        
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
        self.eval_boundary_band_dilation = int(max(0, getattr(self.config.common, 'eval_boundary_band_dilation', 2)))
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

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
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
        correct = 0
        total = 0

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
                        logits = self._parse_model_outputs(outputs)
                        loss = self.criterion(logits, labels)
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(hsi)
                    logits = self._parse_model_outputs(outputs)
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                self.ema.update(self.model)
                if epoch >= self.warmup_epochs:
                    self.scheduler.step()

                total_loss += float(loss.item())
                with torch.no_grad():
                    predicted = torch.argmax(logits.detach(), dim=1)
                    valid_mask = (labels != 255)
                    total += int(valid_mask.sum().item())
                    correct += int(((predicted == labels) & valid_mask).sum().item())

                acc = 100.0 * correct / total if total > 0 else 0.0
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc:.2f}%',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })

        processed_batches = max(batch_idx + 1 if 'batch_idx' in locals() else 0, 1)
        epoch_loss = total_loss / processed_batches
        epoch_acc = 100.0 * correct / total if total > 0 else 0.0
        
        # record lr for this epoch
        self.lr_history.append(self.optimizer.param_groups[0]['lr'])
        
        # record gradient L2 norm
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        self.grad_norms.append(total_norm ** 0.5)
        
        return epoch_loss, epoch_acc

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
                           (no ROC/vis arrays).
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
        metric_labels_eval = self._metric_labels(num_classes)

        total_loss = 0.0
        cm_gpu = torch.zeros(num_classes, num_classes,
                             dtype=torch.long, device=self.device)

        using_val_loader = (loader is None and self.val_loader is not None) or (loader is self.val_loader)

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
                        logits = self._parse_model_outputs(outputs)
                        loss = self.criterion(logits, labels)
                else:
                    outputs = self.model(hsi)
                    logits = self._parse_model_outputs(outputs)
                    loss = self.criterion(logits, labels)

                total_loss += float(loss.item())
                predicted = torch.argmax(logits, dim=1)

                valid_mask = (labels != 255)
                pred_valid = predicted[valid_mask]
                tgt_valid = labels[valid_mask]

                linear_idx = tgt_valid * num_classes + pred_valid
                cm_gpu.view(-1).scatter_add_(
                    0, linear_idx,
                    torch.ones_like(linear_idx, dtype=torch.long)
                )

                if need_reconstruction:
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

                    proba = F.softmax(logits.float(), dim=1)
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
        ba_metric, kappa_metric, miou_metric = self._metrics_from_confusion(cm_np, metric_labels_eval)
        acc = ba_metric
        kappa = kappa_metric
        miou = miou_metric

        processed_eval_batches = max(processed_eval_batches, 1)
        loss = total_loss / processed_eval_batches
        self._last_eval_loss = loss
        self._last_ba = ba_metric
        self._last_kappa = kappa_metric
        self._last_miou_metric = miou_metric
        self._last_miou = miou
        if cm_np.shape[0] > 1:
            eval_rows = cm_np[1:, :]
            eval_row_sums = eval_rows.sum(axis=1).astype(np.float64)
            eval_diag = np.diag(cm_np)[1:].astype(np.float64)
            eval_valid_cls = eval_row_sums > 0
            if eval_valid_cls.any():
                eval_recalls = np.zeros_like(eval_row_sums, dtype=np.float64)
                eval_recalls[eval_valid_cls] = eval_diag[eval_valid_cls] / np.maximum(eval_row_sums[eval_valid_cls], 1.0)
                self._last_min_recall = float(np.min(eval_recalls[eval_valid_cls]) * 100.0)
                self._last_mean_recall = float(np.mean(eval_recalls[eval_valid_cls]) * 100.0)
            else:
                self._last_min_recall = 0.0
                self._last_mean_recall = 0.0
        else:
            self._last_min_recall = 0.0
            self._last_mean_recall = 0.0

        self._metric_labels_used = metric_labels_eval

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
            'train_acc',
            'eval_loss',
            'eval_score',
            'eval_ba',
            'eval_kappa',
            'eval_miou',
            'eval_min_recall',
            'eval_mean_recall',
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

        best_entry = None
        best_eval_score = -1e18
        try:
            for entry in self.topk_models:
                self.model.load_state_dict(entry['model_state'])
                _ = self.evaluate(collect_extra=False, use_ema=False, loader=self.val_loader)
                eval_score = self._select_early_stop_score()
                if eval_score > best_eval_score:
                    best_eval_score = float(eval_score)
                    best_entry = entry
        finally:
            pass

        if best_entry is None:
            return

        self.best_epoch = int(best_entry['epoch'])
        self.best_acc = float(best_entry['score'])
        self.best_model_state = {
            'epoch': int(best_entry['epoch']),
            'model_state': best_entry['model_state'],
            'acc': float(best_entry['score']),
            'kappa': float(getattr(self, '_last_kappa', 0.0)),
        }
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
                # training
                epoch_tic = time.perf_counter()
                train_loss, train_acc = self.train_epoch(epoch)
                self.epoch_times.append(time.perf_counter() - epoch_tic)
                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)

                # validating
                should_eval = ((epoch + 1) % self.eval_interval == 0) or (epoch + 1 == self.epochs)

                if should_eval or self.debug_mode:
                    # Evaluate on val set for early stopping (not test set)
                    eval_loss, eval_acc, kappa, pred, target = self.evaluate(
                        use_ema=True,
                        collect_reconstruction=False,
                    )

                    self.eval_losses.append(eval_loss)
                    self.eval_accs.append(eval_acc)
                    eval_score = float(self._select_early_stop_score())

                    self.epoch_metrics.append({
                        'epoch': epoch + 1,
                        'train_loss': float(train_loss),
                        'train_acc': float(train_acc),
                        'eval_loss': float(eval_loss),
                        'eval_score': float(eval_score),
                        'eval_ba': float(getattr(self, '_last_ba', eval_acc)),
                        'eval_kappa': float(getattr(self, '_last_kappa', kappa)),
                        'eval_miou': float(getattr(self, '_last_miou_metric', 0.0)),
                        'eval_min_recall': float(getattr(self, '_last_min_recall', 0.0)),
                        'eval_mean_recall': float(getattr(self, '_last_mean_recall', 0.0)),
                    })

                    miou_str = ''
                    if hasattr(self, '_last_miou'):
                        miou_str = f" mIoU(eval): {getattr(self, '_last_miou_metric', self._last_miou):6.2f}%"

                    eval_set_name = 'Val' if self.val_loader else 'Test'
                    tprint(f"\n[Epoch {epoch+1:3d}] "
                          f"Train(loss): {train_loss:.4f} Acc: {train_acc:6.2f}% | "
                          f"{eval_set_name}(loss): {eval_loss:.4f} "
                          f"BA(eval): {getattr(self, '_last_ba', eval_acc):6.2f}% "
                          f"Kappa(eval): {getattr(self, '_last_kappa', kappa):6.2f}% "
                          f"MinR: {getattr(self, '_last_min_recall', 0.0):6.2f}% "
                          f"Score: {eval_score:6.2f}%{miou_str}")

                    self._register_topk_model(epoch=epoch, score=eval_score)

                    # save the best model
                    if eval_score > self.best_acc:
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
                        if epoch - self.best_epoch > self.patience:
                            tprint(f"\n  Early stopping: {epoch - self.best_epoch} epochs without improvement")
                            break
                else:
                    tprint(f"[Epoch {epoch+1:3d}] Train(loss): {train_loss:.4f} Acc: {train_acc:6.2f}%", end='')
                    self.epoch_metrics.append({
                        'epoch': epoch + 1,
                        'train_loss': float(train_loss),
                        'train_acc': float(train_acc),
                        'eval_loss': None,
                        'eval_score': None,
                        'eval_ba': None,
                        'eval_kappa': None,
                        'eval_miou': None,
                        'eval_min_recall': None,
                        'eval_mean_recall': None,
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
            self.best_acc = float(self._select_early_stop_score())
            self.best_model_state = {
                'epoch': self.best_epoch,
                'model_state': self.ema.state_dict(),
                'acc': self.best_acc,
                'kappa': float(getattr(self, '_last_kappa', 0.0)),
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
        }
        results['final_accuracy_eval'] = float(getattr(self, '_last_ba', final_acc))
        results['final_kappa_eval'] = float(getattr(self, '_last_kappa', final_kappa))
        results['final_miou_eval'] = float(getattr(self, '_last_miou_metric', 0.0))
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
        index_chain: List[np.ndarray] = []
        visited = set()
        while ds is not None and id(ds) not in visited:
            visited.add(id(ds))
            if hasattr(ds, 'indices'):
                try:
                    idx = np.asarray(ds.indices, dtype=np.int64)
                    index_chain.append(idx)
                except Exception:
                    pass
            if hasattr(ds, 'dataset'):
                ds = ds.dataset
            else:
                break

        # Compose indices from outer wrapper to inner wrapper so reconstruction
        # always receives base-dataset absolute indices.
        indices = None
        for idx in index_chain:
            if indices is None:
                indices = idx
            else:
                try:
                    indices = idx[indices]
                except Exception:
                    indices = None
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
        score = float(getattr(self, '_last_ba', self.best_acc))
        ba_eval = float(getattr(self, '_last_ba', score))
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
                f"pixel_acc={pixel_acc:.2f}% | "
                f"BA(eval)={ba_eval:.2f}% | "
                f"Score={score:.2f}% | mIoU={miou:.2f}%"
            )
            fig.suptitle(title, fontsize=11, fontweight='bold')
            plt.tight_layout()

            save_name = (
                f"recon_{safe_patient}_pxacc{pixel_acc:.2f}_"
                f"miou{miou:.2f}_{timestamp}.png"
            )
            save_path = os.path.join(out_dir, save_name)
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            plt.close()

            summary_lines.append(
                f"patient={patient_name}, patches={int(recon['patch_counts'][p_idx])}, "
                f"covered_px={covered}, correct_px={correct}, pixel_acc={pixel_acc:.4f}, "
                f"BA_eval={ba_eval:.4f}, "
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
                f"BA_eval={ba_eval:.4f}, "
                f"Score={score:.4f}, mIoU={miou:.4f}\n"
            )
            f.write("\n".join(summary_lines))

        tprint(f"  Full reconstruction plots saved: {total_saved} file(s) -> {out_dir}")

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
        metric_labels = self._metric_labels(num_classes)
        eval_num_classes = len(metric_labels)
        class_names = [
            self.config.clsf.targets[i] if i < len(self.config.clsf.targets) else f'Class_{i}'
            for i in metric_labels
        ]
        probas = getattr(self, '_last_probas', None)

        # pre-compute heavy metrics in main process (avoids pickle of huge arrays)
        tprint("  pre-computing plot metrics in main process...")
        t_pre = time.perf_counter()

        # confusion matrix: foreground-only labels.
        y_true_present = set(np.unique(final_target).tolist())
        metric_has_support = any(lbl in y_true_present for lbl in metric_labels)
        if metric_has_support:
            cm = _cm(final_target, final_pred, labels=metric_labels)
        else:
            cm = np.zeros((len(metric_labels), len(metric_labels)), dtype=np.int64)
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

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
            report_lines.append('WARNING: no valid evaluation labels present in y_true for this evaluation split.')
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
              list(self.train_accs), list(self.eval_accs),
              self.eval_interval, self.debug_mode,
              self.best_acc, self.model_name, self.output)),
                        (_worker_plot_confusion_matrix,
                         (cm, cm_norm, metrics_text,
                            class_names, self.model_name, self.output,
                                             'confusion_matrix_eval_classes', 'Evaluation-Class Confusion Matrix',
                                             'classification_metrics_eval_classes.txt')),
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
            (_worker_write_pairwise_class_metrics,
               (pairwise_text, self.output)),
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
                logits = self._parse_model_outputs(outputs)
                preds = torch.argmax(logits, dim=1)
            
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
    
    def _save_model(self) -> None:
        if self.best_model_state is None:
            return

        # Load best EMA weights into the model for export
        raw_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        # Save as ONNX
        onnx_path = os.path.join(self.output, 'models', f'{self.model_name}_best.onnx')
        input_channels = int(getattr(raw_model, 'in_channels', self.config.preprocess.pca_components))
        input_patch = int(getattr(raw_model, 'patch_size', self.config.split.patch_size))
        dummy_input = torch.randn(
            1, input_channels,
            input_patch,
            input_patch
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
                        return out.get('logits', out.get('fused_logits'))
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
            logits = self._parse_model_outputs(output)
            # For segmentation output [B, K, H, W], sum spatial dims to get scalar
            if logits.dim() == 4:
                loss = logits[0, class_idx].sum()
            else:
                loss = logits[0, class_idx]
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
