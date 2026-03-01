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
import matplotlib.pyplot as plt
import cv2, gc, json, os, time, torch, traceback, warnings, copy, matplotlib

from tqdm import tqdm
from munch import Munch
from torch.nn import Module
from pipeline.monitor import tprint
from pipeline.dataset import AbstractHSDataset
from torch.cuda.amp import GradScaler, autocast
from dataclasses import dataclass, field
from typing import Tuple, Callable, Dict, Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score,
                             confusion_matrix as _cm, roc_curve,
                             precision_recall_fscore_support, auc)
import torch.nn.functional as F

warnings.filterwarnings("ignore")


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced classification.

    ```
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    ```
    Down-weights well-classified samples so training focuses on hard / rare ones.
    Supports segmentation outputs (N, C, H, W) with ignore_index masking.
    """

    def __init__(self, weight=None, gamma=2.0, ignore_index=-100,
                 reduction='mean'):
        super().__init__()
        self.register_buffer('weight', weight)  # per-class weights (C,)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

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

        log_p = F.log_softmax(logits, dim=-1)
        p_t = log_p.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)

        loss = -((1 - p_t) ** self.gamma) * log_p_t

        if self.weight is not None:
            loss = self.weight.to(logits.device)[targets] * loss

        return loss.mean() if self.reduction == 'mean' else loss.sum()


def _worker_plot_training_curves(train_losses, eval_losses, train_accs,
                                 eval_accs, eval_interval, debug_mode,
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

    axes[0].plot(train_losses, label='Train Loss', marker='o', markersize=4)
    if eval_losses:
        axes[0].plot(eval_epochs, eval_losses, label='Test Loss', marker='s', markersize=4)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Curve'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_accs, label='Train Acc', marker='o', markersize=4)
    if eval_accs:
        axes[1].plot(eval_epochs, eval_accs, label='Test Acc', marker='s', markersize=4)
    axes[1].axhline(y=best_acc, color='r', linestyle='--', label=f'Best: {best_acc:.2f}%')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy Curve'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output, 'training_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _worker_plot_confusion_matrix(cm, cm_norm, metrics_text,
                                  class_names, model_name, output):
    """worker: plot pre-computed confusion matrix.
    cm: shape (num_classes, num_classes) int — raw counts.
    cm_norm: shape (num_classes, num_classes) float — row-normalized.
    metrics_text: str — pre-formatted classification report to save.
    """
    matplotlib.use('Agg')

    with open(os.path.join(output, 'classification_metrics.txt'), 'w') as f:
        f.write(metrics_text)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title('Confusion Matrix (Count)'); axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Greens', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title('Confusion Matrix (Normalized)'); axes[1].set_ylabel('True'); axes[1].set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
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

class hsTrainer:
    """
    Trainer for hyperspectral dataset.
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
        test_loader=None,
        **kwargs
    ):
        
        self.config = config
        self.dataLoader = dataLoader
        self.epochs = epochs
        self.model = model
        self.model_name = model_name
        self.debug_mode = debug_mode
        self.num_gpus = num_gpus
        self.output = self.config.path.output + f'/{self.model_name}'
        self.kwargs = kwargs
        
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(os.path.join(self.output, 'CAM'), exist_ok=True)
        os.makedirs(os.path.join(self.output, 'models'), exist_ok=True)
        
        # training workflow
        self._setup_device()
        if train_loader is not None and test_loader is not None:
            self.train_loader = train_loader
            self.test_loader = test_loader
            tprint(f"  Using injected data loaders "
                  f"(train: {len(train_loader)} batches, "
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
        self.best_acc = 0.0
        self.best_epoch = 0
        self.best_model_state = None
        
        # memory-efficient tracking for visualizations
        self.lr_history = []          # lr per epoch
        self.grad_norms = []          # gradient L2 norm per epoch
        self.epoch_times = []         # wall-clock time per epoch

        tprint(f"Trainer Initialized successfully with:")
        print(f"  model: {self.model_name}")
        print(f"  epoch: {self.epochs}")
        print(f"  device: {self.device}")
        print(f"  num_gpus: {self.num_gpus}")
        print(f"  output_dir: {self.output}")
    
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
            
            self.train_loader, self.test_loader = self.dataLoader.create_data_loader(
                num_workers=num_workers,
                batch_size=batch_size,
                pin_memory=self.config.memory.pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers
            )
            print(f"  training set: {len(self.train_loader)} batches")
            print(f"  test set: {len(self.test_loader)} batches")
        except Exception as e:
            raise RuntimeError(f"Error during data loading: {e}")
    
    def _create_model(self) -> None:
        """create model and print parameter count"""
        tprint("Creating model with:")
        try:
            self.model = self.model(**self.kwargs)
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
        
        # Optimizer
        lr = self.config.common.lr
        weight_decay = self.config.common.weight_decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr,
            weight_decay=weight_decay, betas=(0.9, 0.999))
        
        # LR scheduler
        sc = self.config.common.scheduler
        T_0 = getattr(sc, 'T_0', 10)
        T_mult = getattr(sc, 'T_mult', 2)
        eta_min = getattr(sc, 'eta_min', 1e-6)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
        
        print(f"  AdamW(lr={lr}, wd={weight_decay}) + CosineWR(T0={T_0}, Tm={T_mult})")
        
        # Class-weighted Focal Loss (handles severe class imbalance better than CE)
        focal_gamma = getattr(self.config.common, 'focal_gamma', 2.0)
        class_weights = None
        try:
            train_labels = self.dataLoader.patch_labels[
                self.train_loader.dataset.indices]
            num_cls = self.config.clsf.num
            class_counts = np.bincount(train_labels, minlength=num_cls).astype(np.float64)
            total_samples = class_counts.sum()
            # Effective number weighting (Cui et al., CVPR 2019)
            beta = (total_samples - 1) / total_samples
            effective_num = 1.0 - np.power(beta, class_counts)
            raw_weights = 1.0 / (effective_num + 1e-8)
            raw_weights = raw_weights / raw_weights.mean()
            class_weights = torch.FloatTensor(raw_weights).to(self.device)
            imbalance = class_counts.max() / (class_counts.min() + 1)
            print(f"  Class weights (effective-number): imbalance ratio {imbalance:.1f}x, {num_cls} classes")
        except Exception as e:
            print(f"  Warning: uniform class weights ({e})")
        
        self.criterion = FocalLoss(
            weight=class_weights, gamma=focal_gamma, ignore_index=255)
        print(f"  FocalLoss(gamma={focal_gamma})")
        
        # Mixed precision
        use_amp = self.config.common.use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler() if use_amp else None
        self.use_amp = use_amp
        
        # Hyperparameters
        self.grad_clip = getattr(self.config.common, 'grad_clip', 1.0)
        self.warmup_epochs = getattr(self.config.common, 'warmup_epochs', 5)
        self.patience = getattr(self.config.common, 'patience', 20)
        self.eval_interval = getattr(self.config.common, 'eval_interval', 1)

        # EMA
        ema_decay = getattr(self.config.common, 'ema_decay', 0.999)
        self.ema = ModelEMA(self.model, decay=ema_decay)
        
        if use_amp:
            print(f"  AMP enabled, EMA(decay={ema_decay}), patience={self.patience}")
        else:
            print(f"  EMA(decay={ema_decay}), patience={self.patience}")
    
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
        
        # lr warm-up
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            lr = self.config.common.lr * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}', leave=False)
        
        # use idx to unpack batch-by-batch.
        for batch_idx, batch_data in enumerate(pbar):
            hsi, labels = self._unpack_batch(batch_data)
            # non_blocking=True to:
            # transfer new batch from memory to GPU parallel with last GPU conduct.
            hsi = hsi.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # NOTE: method permute() may cause incontinuous memory, FATAL ERROR.
            # try permute().contiguous() if goes wrong.
            # see https://blog.csdn.net/weixin_42046845/article/details/134667338.
            if hsi.dim() == 4 and hsi.shape[-1] <= 16:  # [B, H, W, C]
                hsi = hsi.permute(0, 3, 1, 2)  # [B, C, H, W]
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(hsi)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(hsi)
                loss = self.criterion(outputs, labels)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            # update EMA shadow weights (negligible overhead)
            self.ema.update(self.model)
            
            # update lr after warmup
            if epoch >= self.warmup_epochs:
                self.scheduler.step(epoch + batch_idx / len(self.train_loader))
            
            total_loss += loss.item()
            with torch.no_grad():
                _, predicted = torch.max(outputs.detach(), 1)
                # Pixel-wise accuracy excluding ignore pixels (label==255)
                valid_mask = (labels != 255)
                total += valid_mask.sum().item()
                correct += ((predicted == labels) & valid_mask).sum().item()
            
            # tqdm
            acc = 100.0 * correct / total if total > 0 else 0.0
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # NOTE:
            # torch.cuda.empty_cache() removed here.
            # it forces a device sync every 5 steps,
            # resulting in a significant shutdown for GPU units,
            # which is time counter-productive.
        
        epoch_loss = total_loss / len(self.train_loader)
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
    def evaluate(self, collect_extra: bool = False, use_ema: bool = False
                 ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Evaluate on the test set with optimized performance.

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
            use_ema: if True, temporarily swap in EMA shadow weights for
                     evaluation, then restore original weights.

        Return:
            Tuple (``loss``, ``accuracy``, ``kappa``, ``predictions``, ``labels``)
            predictions/labels are full arrays when collect_extra=True,
            empty arrays otherwise (sufficient for epoch-level logging).
            When collect_extra is True, also stores self._last_probas.
        """
        # Swap in EMA shadow weights (zero-copy pointer exchange; call again to restore)
        if use_ema:
            self.ema.swap(self.model)

        self.model.eval()
        model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        num_classes = model_ref.num_classes

        total_loss = 0.0

        # GPU-side confusion matrix — shape (C, C), int64
        # Updated in-loop via scatter_add_; only C² numbers transferred to CPU at end.
        cm_gpu = torch.zeros(num_classes, num_classes,
                             dtype=torch.long, device=self.device)

        # collect_extra: per-pixel arrays needed for final plots & ROC curves only
        predictions_list = [] if collect_extra else None
        targets_list     = [] if collect_extra else None
        probas_list      = [] if collect_extra else None   # sub-sampled; see roc_per_batch
        roc_targets_list = [] if collect_extra else None   # targets paired with sub-sampled probas

        eval_loader = self.test_loader
        # ROC sub-sampling budget: keep ≤2M pixels total (~1350 per batch for typical test set).
        # ROC AUC is statistically stable with ≥1M samples; this avoids the 14 GB y_onehot
        # allocation and the 4.4B-iteration Python loop in _save_all_plots.
        roc_per_batch = max(256, 2_000_000 // max(len(eval_loader), 1)) if collect_extra else 0
        pbar = tqdm(eval_loader, desc='Evaluating', leave=False)

        with torch.no_grad():
            for batch_data in pbar:
                hsi, labels = self._unpack_batch(batch_data)
                hsi    = hsi.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # format check
                if hsi.dim() == 4 and hsi.shape[-1] <= 16:
                    hsi = hsi.permute(0, 3, 1, 2)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(hsi)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(hsi)
                    loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                # filter padding/background (ignore_index = 255)
                valid_mask  = (labels != 255)
                pred_valid  = predicted[valid_mask]   # shape: (n_valid,)
                tgt_valid   = labels[valid_mask]      # shape: (n_valid,)

                # linearise (true_cls, pred_cls) -> flat index and accumulate
                # cm_gpu[t, p] += 1  equivalent, without any Python loop
                linear_idx = tgt_valid * num_classes + pred_valid
                cm_gpu.view(-1).scatter_add_(
                    0, linear_idx,
                    torch.ones_like(linear_idx, dtype=torch.long))

                # collect full arrays only for final-eval plots / ROC
                if collect_extra:
                    predictions_list.append(pred_valid)   # stays on GPU
                    targets_list.append(tgt_valid)        # stays on GPU
                    # Sub-sample probas: keep at most roc_per_batch valid pixels per batch.
                    # Avoids storing all 4.4B×8-class float32 (~14 GB) in host RAM.
                    proba = outputs.softmax(dim=1)                       # [B, K, H, W]
                    proba_valid = proba.permute(0, 2, 3, 1)[valid_mask]  # [n_v, K]
                    n_v = proba_valid.shape[0]
                    if n_v > roc_per_batch:
                        sel = torch.randperm(n_v, device=self.device)[:roc_per_batch]
                        probas_list.append(proba_valid[sel].cpu().numpy())
                        roc_targets_list.append(tgt_valid[sel].cpu().numpy())
                    else:
                        probas_list.append(proba_valid.cpu().numpy())
                        roc_targets_list.append(tgt_valid.cpu().numpy())

        cm_np    = cm_gpu.cpu().numpy().astype(np.int64)   # shape (C, C)
        row_sums = cm_np.sum(axis=1)                       # actual count per class
        col_sums = cm_np.sum(axis=0)                       # predicted count per class
        diag     = np.diag(cm_np)
        total    = int(cm_np.sum())

        # overall Accuracy
        oa = float(diag.sum()) / total * 100 if total > 0 else 0.0

        # balanced Accuracy (mean per-class recall, ignoring absent classes)
        valid_cls     = row_sums > 0
        per_cls_recall = np.where(valid_cls, diag / np.maximum(row_sums, 1), 0.0)
        acc = float(per_cls_recall[valid_cls].mean()) * 100 if valid_cls.any() else 0.0

        # Cohen's Kappa (analytical formula, no sklearn)
        expected = row_sums.astype(np.float64) * col_sums.astype(np.float64) / max(total, 1)
        p_o = float(diag.sum()) / max(total, 1)
        p_e = float(expected.sum()) / max(total, 1)
        kappa = ((p_o - p_e) / max(1.0 - p_e, 1e-8)) * 100

        # mIoU (tp / (tp + fp + fn) per class, from confusion matrix columns/rows)
        miou_vals = []
        for c in range(num_classes):
            tp    = int(cm_np[c, c])
            fp    = int(col_sums[c]) - tp
            fn    = int(row_sums[c]) - tp
            denom = tp + fp + fn
            if denom > 0:
                miou_vals.append(tp / denom)
        miou = float(np.mean(miou_vals)) * 100 if miou_vals else 0.0

        loss = total_loss / len(eval_loader)
        self._last_oa   = oa
        self._last_miou = miou

        # final-eval: full arrays for plots / ROC (single GPU -> CPU transfer)
        if collect_extra:
            predictions = torch.cat(predictions_list, dim=0).cpu().numpy()
            targets     = torch.cat(targets_list,     dim=0).cpu().numpy()
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

        # restore original weights via second swap
        # e.g. if EMA was used, swap back to original model weights
        if use_ema:
            self.ema.swap(self.model)

        return loss, acc, kappa, predictions, targets
    
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
                eval_loss, eval_acc, kappa, pred, target = self.evaluate(use_ema=True)
                self.eval_losses.append(eval_loss)
                self.eval_accs.append(eval_acc)
                
                miou_str = ''
                if hasattr(self, '_last_miou'):
                    miou_str = f' mIoU: {self._last_miou:6.2f}%'
                
                oa_str = f' OA: {self._last_oa:6.2f}%' if hasattr(self, '_last_oa') else ''
                tprint(f"\n[Epoch {epoch+1:3d}] "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:6.2f}% | "
                      f"Test Loss: {eval_loss:.4f} BA: {eval_acc:6.2f}%{oa_str} "
                      f"Kappa: {kappa:6.2f}%{miou_str}")
                
                # save the best model
                if eval_acc > self.best_acc:
                    self.best_acc = eval_acc
                    self.best_epoch = epoch
                    self.best_model_state = {
                        'epoch': epoch,
                        'model_state': self.ema.state_dict(),
                        'acc': eval_acc,
                        'kappa': kappa
                    }
                    self._save_model()
                    tprint(f"  Best model saved (BA: {eval_acc:.2f}%) at {os.path.join(self.output, 'models', f'{self.model_name}_best.pth')}")
                else:
                    # early stopping check
                    if epoch - self.best_epoch > self.patience:
                        tprint(f"\n  Early stopping: {epoch - self.best_epoch} epochs without improvement")
                        break
            else:
                tprint(f"[Epoch {epoch+1:3d}] Train Loss: {train_loss:.4f} Acc: {train_acc:6.2f}%", end='')
            
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
            tprint(f"Loading model from epoch {self.best_epoch + 1} for final eval...")
            self.model.load_state_dict(self.best_model_state['model_state'])
        
        # final evaluation with extra data collection (probabilities, features)
        final_loss, final_acc, final_kappa, final_pred, final_target = self.evaluate(
            collect_extra=True)
        tprint(f"Final eval completed!")
        
        # Store for CV aggregation (accessed before trainer is deleted)
        self._final_pred = final_pred
        self._final_target = final_target
        
        # single-model visualizations, skip in CV mode for aggregated plots
        if not _cv_mode:
            self._save_all_plots(final_target, final_pred)
        
        results = {
            'best_epoch': self.best_epoch + 1,
            'best_accuracy': self.best_acc,
            'final_accuracy': final_acc,
            'final_kappa': final_kappa,
            'training_time': training_time,
            'model_path': os.path.join(self.output, 'models', f'{self.model_name}_best.pth')
        }
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
            fold_loaders = dataLoader.create_cv_data_loaders(
                n_folds=n_folds,
                num_workers=num_workers,
                batch_size=batch_size,
                pin_memory=config.memory.pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
            )

        num_classes = config.clsf.num
        common_fpr = np.linspace(0, 1, 200)

        fold_results = []
        fold_curves = []     # lightweight: only epoch-level scalars per fold
        all_accs = []
        all_kappas = []
        all_mious = []
        all_train_accs = []
        total_time = 0.0

        # Incremental accumulators for mean/std of confusion matrix, precision, recall, F1, ROC curves
        sum_cm_norm = np.zeros((num_classes, num_classes), dtype=np.float64)
        sum_cm_norm_sq = np.zeros((num_classes, num_classes), dtype=np.float64)
        fold_precisions = []   # list of (num_classes,) arrays
        fold_recalls = []
        fold_f1s = []
        roc_tprs = [[] for _ in range(num_classes)]  # interpolated TPR per class
        roc_aucs = [[] for _ in range(num_classes)]

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

            result = trainer.train(_cv_mode=True)

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

            cm = _cm(target, pred, labels=range(num_classes))
            cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-8)
            sum_cm_norm += cm_norm
            sum_cm_norm_sq += cm_norm ** 2

            p, r, f1, _ = precision_recall_fscore_support(
                target, pred, labels=range(num_classes), zero_division=0)
            fold_precisions.append(p)
            fold_recalls.append(r)
            fold_f1s.append(f1)

            # release output label memory
            if hasattr(trainer, '_last_probas') and trainer._last_probas is not None:
                probas = trainer._last_probas
                for cls_i in range(num_classes):
                    y_bin = (target == cls_i).astype(int)
                    if y_bin.sum() > 0:
                        fpr_arr, tpr_arr, _ = roc_curve(y_bin, probas[:, cls_i])
                        roc_tprs[cls_i].append(
                            np.interp(common_fpr, fpr_arr, tpr_arr))
                        roc_aucs[cls_i].append(auc(fpr_arr, tpr_arr))
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
        class_names = list(config.clsf.targets[:num_classes])
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
             (mean_cm, std_cm, num_classes, class_names,
              model_name, cv_output)),
            (hsTrainer._plot_per_class_metrics,
             (fold_precisions, fold_recalls, fold_f1s,
              num_classes, class_names, model_name, cv_output)),
            (hsTrainer._plot_roc_curves,
             (roc_tprs, roc_aucs, common_fpr,
              num_classes, class_names, model_name, cv_output)),
            (hsTrainer._plot_grad_norms,
             (fold_curves, model_name, cv_output)),
        ]

        t0 = time.perf_counter()
        max_workers = min(len(plot_tasks), (os.cpu_count() or 4))
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
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

    def _save_all_plots(self, final_target, final_pred):
        """generate all single-model plots in parallel.

        pre-computes all heavy metrics (cm, p/r/f1, roc) in the main process
        so that only small summary arrays (~kb) are pickled to workers,
        instead of raw predictions (~gb). this reduces plot time from
        ~10 min to ~seconds.
        """
        num_classes = self.config.clsf.num
        class_names = list(self.config.clsf.targets[:num_classes])
        probas = getattr(self, '_last_probas', None)

        # pre-compute heavy metrics in main process (avoids pickle of huge arrays)
        tprint("  pre-computing plot metrics in main process...")
        t_pre = time.perf_counter()

        # confusion matrix — cm: shape (num_classes, num_classes) int
        cm = _cm(final_target, final_pred, labels=range(num_classes))
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

        # per-class precision, recall, f1 — each shape (num_classes,)
        precision, recall, f1, _ = precision_recall_fscore_support(
            final_target, final_pred, labels=range(num_classes), zero_division=0)

        # format metrics text for saving
        report_lines = []
        for i in range(num_classes):
            name = class_names[i] if i < len(class_names) else f'Class_{i}'
            report_lines.append(
                f"{name}: P={precision[i]:.4f}, R={recall[i]:.4f}, F1={f1[i]:.4f}")
        metrics_text = '\n'.join(report_lines) + '\n'

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
            for cls_i in range(num_classes):
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
              class_names, self.model_name, self.output)),
            (_worker_plot_per_class_metrics,
             (precision, recall, f1, num_classes, class_names,
              self.model_name, self.output)),
            (_worker_plot_roc_curves,
             (roc_data, num_classes, class_names,
              self.model_name, self.output)),
            (_worker_plot_grad_norm,
             (list(self.grad_norms), self.model_name, self.output)),
            (_worker_plot_lr,
             (list(self.lr_history), self.model_name, self.output)),
            (_worker_plot_epoch_time,
             (list(self.epoch_times), self.model_name, self.output)),
        ]

        t0 = time.perf_counter()
        max_workers = min(len(tasks), (os.cpu_count() or 4))
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
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
            hsi, labels = self._unpack_batch(batch_data)
            
            hsi = hsi.to(self.device)
            num_samples = min(4, hsi.shape[0])
            
            fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            
            col_titles = ['Input', 'Grad-CAM', 'Overlay']
            
            with torch.no_grad():
                outputs = self.model(hsi[:num_samples])
                preds = torch.argmax(outputs, dim=1)
            
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
    
    def _unpack_batch(self, batch_data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unpack batch data from DataLoader.
        
        Returns:
            Tuple[tensor, tensor]: (``hsi``, ``labels``)
        """
        if isinstance(batch_data, (list, tuple)):
            if len(batch_data) == 3:
                hsi, _, labels = batch_data
            elif len(batch_data) == 2:
                hsi, labels = batch_data
            else:
                raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
        else:
            raise TypeError(f"batch must be list or tuple, got: {type(batch_data)}")
        
        return hsi, labels
    
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
        
        model_path = os.path.join(self.output, 'models', f'{self.model_name}_best.pth')
        # Extract state_dict and remove 'module.' prefix if using DataParallel
        state_dict = self.best_model_state['model_state']
        if isinstance(self.model, nn.DataParallel):
            # Remove 'module.' prefix added by DataParallel
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        torch.save(state_dict, model_path)
        tprint(f"  Model saved to: {model_path}")
    
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
            # For segmentation output [B, K, H, W], sum spatial dims to get scalar
            if output.dim() == 4:
                loss = output[0, class_idx].sum()
            else:
                loss = output[0, class_idx]
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
