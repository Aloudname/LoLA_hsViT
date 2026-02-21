"""
trainer.py
Hyperspectral Image Classification Trainer
Optimization includes gradient accumulation, mixed precision training, learning rate warmup, and early stopping.
CPU/GPU supported with flexible memory configuration.

start train with:
    trainer.train()
"""

import os
import time
import warnings
import traceback
from typing import Tuple, Callable, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Module
from tqdm import tqdm
from munch import Munch

from pipeline.dataset import AbstractHSDataset

warnings.filterwarnings("ignore")


class hsTrainer:
    """
    Trainer for hyperspectral dataset.
    """
    
    def __init__(
        self,
        config: Munch = None,
        dataLoader: AbstractHSDataset = None,
        epochs: int = 20,
        model: Callable[..., Module] = None,
        model_name: str = "model",
        debug_mode: bool = False,
        num_gpus: int = 1
    ):
        
        self.config = config
        self.dataLoader = dataLoader
        self.epochs = epochs
        self.model = model
        self.model_name = model_name
        self.debug_mode = debug_mode
        self.num_gpus = num_gpus
        self.output = self.config.path.output + f'/{self.model_name}'
        
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(os.path.join(self.output, 'CAM'), exist_ok=True)
        os.makedirs(os.path.join(self.output, 'models'), exist_ok=True)
        
        # training workflow
        self._setup_device()
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

        print(f"Trainer Initialized successfully with:")
        print(f"  model: {self.model_name}")
        print(f"  epoch: {self.epochs}")
        print(f"  device: {self.device}")
        print(f"  num_gpus: {self.num_gpus}")
        print(f"  output_dir: {self.output}")
    
    def _setup_device(self) -> None:
        """setup device and validate multi-GPU configuration"""
        device_type = self.config.get('device_type', 'cuda')
        try:
            self.device = torch.device(device_type)
            if device_type == 'cuda':
                # Get available GPU count
                self.available_gpus = torch.cuda.device_count()
                
                # Validate num_gpus
                if self.num_gpus > self.available_gpus:
                    raise RuntimeError(
                        f"Requested {self.num_gpus} GPUs but only {self.available_gpus} available. "
                        f"Please set --parallel to a value <= {self.available_gpus}"
                    )
                
                if self.num_gpus < 1:
                    raise RuntimeError(f"num_gpus must be >= 1, got {self.num_gpus}")
                
                # Store GPU IDs to use
                self.gpu_ids = list(range(self.num_gpus))
                
                print(f"  Available GPUs: {self.available_gpus}")
                print(f"  Using {self.num_gpus} GPU(s): {self.gpu_ids}")
                for gpu_id in self.gpu_ids:
                    print(f"    GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)} - "
                          f"{torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3):.2f} GB")
                print(f"  CUDA version: {torch.version.cuda}")
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Initialization on CPU due to {e}")
            self.device = torch.device('cpu')
            self.num_gpus = 0
            self.gpu_ids = []
            self.available_gpus = 0
        
        # limit CPU threads to prevent oversubscription.
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'
    
    def _load_data(self) -> None:
        """load & create data loaders with optimized performance settings"""
        print("Loading data and creating data loaders with:")
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
                prefetch_factor = 2
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
        print("Creating model with:")
        try:
            self.model = self.model()
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
            print("Setting up DataParallel for multi-GPU training:")
            try:
                # Use DataParallel with explicit device IDs
                self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids, output_device=self.gpu_ids[0])
                print(f"  Model wrapped with DataParallel using {self.num_gpus} GPU(s): {self.gpu_ids}")
            except Exception as e:
                print(f"Warning: Failed to setup DataParallel: {e}")
                print(f"  Continuing with single GPU training")
    
    def _setup_training(self) -> None:
        """initialize training components including optimizer, scheduler, loss function, and mixed precision settings"""
        print("Initializing training components with:")
        
        # Optimizer
        lr = self.config.common.lr
        weight_decay = self.config.common.weight_decay
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        print(f"  optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
        
        # lr scheduler (Cosine Annealing with Warm Restarts)
        scheduler_config = self.config.common.scheduler
        T_0 = scheduler_config.T_0 if hasattr(scheduler_config, 'T_0') else 10
        T_mult = scheduler_config.T_mult if hasattr(scheduler_config, 'T_mult') else 2
        eta_min = scheduler_config.eta_min if hasattr(scheduler_config, 'eta_min') else 1e-6
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
        print(f"  lr scheduler: CosineAnnealingWarmRestarts (T_0={T_0}, T_mult={T_mult}, eta_min={eta_min})")
        
        # loss function
        label_smoothing = self.config.common.label_smoothing if hasattr(self.config.common, 'label_smoothing') else 0.1
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"  loss function: CrossEntropyLoss (label_smoothing={label_smoothing})")
        
        # mixed precision training on GPU
        use_amp = self.config.common.use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler() if use_amp else None
        self.use_amp = use_amp
        if use_amp:
            print(f"  mixed precision training enabled with GradScaler")
        
        # other hyperparameters
        self.grad_clip = self.config.common.grad_clip if hasattr(self.config.common, 'grad_clip') else 1.0
        self.warmup_epochs = self.config.common.warmup_epochs if hasattr(self.config.common, 'warmup_epochs') else 5
        self.patience = self.config.common.patience if hasattr(self.config.common, 'patience') else 20
        self.eval_interval = self.config.common.eval_interval if hasattr(self.config.common, 'eval_interval') else 1

    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for a single epoch.
        
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
        
        for batch_idx, batch_data in enumerate(pbar):
            hsi, labels = self._unpack_batch(batch_data)
            # non_blocking=True to transfer data parallel with last GPU caculation.
            hsi = hsi.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # check up data format
            if hsi.dim() == 4 and hsi.shape[-1] <= 16:  # [B, H, W, C]
                hsi = hsi.permute(0, 3, 1, 2)  # [B, C, H, W]
            
            # norm & forward pass
            hsi = self._normalize(hsi)
            
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
            
            # update lr after warmup
            if epoch >= self.warmup_epochs:
                self.scheduler.step(epoch + batch_idx / len(self.train_loader))
            
            total_loss += loss.item()
            with torch.no_grad():
                _, predicted = torch.max(outputs.detach(), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # tqdm
            acc = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # cache sweep
            if batch_idx % 5 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        evaluate on the test set.
        
        Return:
            Tuple (``loss``, ``accuracy``, ``kappa``, ``predictions``, ``labels``)
        """
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        eval_batch_size = self.config.common.eval_batch_size if hasattr(self.config.common, 'eval_batch_size') else 64
        if self.device.type == 'cpu':
            eval_batch_size = min(eval_batch_size, 16)
        
        eval_loader = torch.utils.data.DataLoader(
            self.test_loader.dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=self.config.memory.num_workers,
            pin_memory=False
        )
        
        pbar = tqdm(eval_loader, desc='Evaluating', leave=False)
        
        for batch_data in pbar:
            hsi, labels = self._unpack_batch(batch_data)
            hsi = hsi.to(self.device)
            labels = labels.to(self.device)
            
            # format check up
            if hsi.dim() == 4 and hsi.shape[-1] <= 16:
                hsi = hsi.permute(0, 3, 1, 2)
            
            hsi = self._normalize(hsi)
            
            outputs = self.model(hsi)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        acc = accuracy_score(targets, predictions) * 100
        kappa = cohen_kappa_score(targets, predictions) * 100
        loss = total_loss / len(eval_loader)
        
        return loss, acc, kappa, predictions, targets
    
    def train(self) -> Dict[str, float]:
        """
        training for epochs.
        
        Args:
            debug_mode: return CAM for every epoch if ``True``.
        
        Returns:
            Dict[``str``, ``float``] of final results.
        """
        print(f"\n{'='*20}")
        if self.debug_mode:
            print(f"Debug mode enabled: CAM and Eval enabled every epoch.")
        print(f"Training ({self.model_name})")
        print(f"{'='*20}\n")
        
        tic = time.perf_counter()
        
        for epoch in range(self.epochs):
            # training
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # validating
            should_eval = ((epoch + 1) % self.eval_interval == 0) or (epoch + 1 == self.epochs)
            
            if should_eval or self.debug_mode:
                eval_loss, eval_acc, kappa, pred, target = self.evaluate()
                self.eval_losses.append(eval_loss)
                self.eval_accs.append(eval_acc)
                
                print(f"\n[Epoch {epoch+1:3d}] "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:6.2f}% | "
                      f"Test Loss: {eval_loss:.4f} Acc: {eval_acc:6.2f}% Kappa: {kappa:6.2f}%")
                
                # save the best model
                if eval_acc > self.best_acc:
                    self.best_acc = eval_acc
                    self.best_epoch = epoch
                    self.best_model_state = {
                        'epoch': epoch,
                        'model_state': self.model.state_dict(),
                        'acc': eval_acc,
                        'kappa': kappa
                    }
                    self._save_model()
                    print(f"  Best model saved (Acc: {eval_acc:.2f}%) at {os.path.join(self.output, 'models', f'{self.model_name}_best.pth')}")
                else:
                    # early stopping check
                    if epoch - self.best_epoch > self.patience:
                        print(f"\n  Early stopping: {epoch - self.best_epoch} epochs without improvement")
                        break
            else:
                print(f"[Epoch {epoch+1:3d}] Train Loss: {train_loss:.4f} Acc: {train_acc:6.2f}%", end='')
            
            # generate CAM for debug mode or every 2 epochs
            if self.debug_mode or (should_eval and (epoch + 1) % 2 == 0):
                try:
                    self._generate_cam(epoch)
                except Exception as e:
                    print(f"  Error during CAM generating: {e}")
        
        toc = time.perf_counter()
        training_time = toc - tic
        
        print(f"\n{'='*20}")
        print(f"Training completed with:")
        print(f"  Best epoch: {self.best_epoch + 1} (Acc: {self.best_acc:.2f}%)")
        print(f"  Total time: {training_time:.2f}s")
        print(f"{'='*20}\n")
        
        # load best model for final evaluation
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state['model_state'])
        
        # final evaluation
        final_loss, final_acc, final_kappa, final_pred, final_target = self.evaluate()
        
        # visualization
        self._plot_training_curves()
        self._plot_confusion_matrix(final_target, final_pred)
        
        results = {
            'best_epoch': self.best_epoch + 1,
            'best_accuracy': self.best_acc,
            'final_accuracy': final_acc,
            'final_kappa': final_kappa,
            'training_time': training_time,
            'model_path': os.path.join(self.output, 'models', f'{self.model_name}_best.pth')
        }
        
        return results
    
    
    def _generate_cam(self, epoch: int) -> None:
        """
        Generate Class Activation Map (CAM),
        built-in method.
        """
        self.model.eval()
        
        try:
            # get a batch of test data
            test_iter = iter(self.test_loader)
            batch_data = next(test_iter)
            hsi, labels = self._unpack_batch(batch_data)
            
            hsi = hsi.to(self.device)
            num_samples = min(4, hsi.shape[0])
            
            fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            
            with torch.no_grad():
                outputs = self.model(hsi[:num_samples])
                preds = torch.argmax(outputs, dim=1)
            
            for i in range(num_samples):
                # RGB simulation
                sample = hsi[i].cpu().numpy()
                if sample.ndim == 4:
                    sample = sample[0]
                
                sample = (sample - sample.min()) / (sample.max() - sample.min() + 1e-8)
                
                if sample.shape[0] >= 3:
                    rgb = sample[:3].transpose(1, 2, 0)
                else:
                    rgb = np.repeat(sample[0:1].transpose(1, 2, 0), 3, axis=2)
                
                axes[i, 0].imshow(rgb)
                axes[i, 0].set_title(f'Sample {i+1}\nLabel: {labels[i].item()}')
                axes[i, 0].axis('off')
                
                # grayscale intensity
                gray = sample.mean(axis=0)
                axes[i, 1].imshow(gray, cmap='gray')
                axes[i, 1].set_title(f'Intensity\nPred: {preds[i].item()}')
                axes[i, 1].axis('off')
                
                # grayscale spectrum (center pixel)
                center_spec = sample[:, sample.shape[1]//2, sample.shape[2]//2]
                axes[i, 2].plot(center_spec, linewidth=2)
                axes[i, 2].set_title(f'Spectrum (center)')
                axes[i, 2].set_xlabel('Band')
                axes[i, 2].grid(True, alpha=0.3)
            
            plt.suptitle(f'Epoch {epoch+1} - Sample Visualization', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            cam_path = os.path.join(self.output, 'CAM', f'epoch_{epoch+1:03d}.png')
            plt.savefig(cam_path, dpi=100, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error during generating CAM: {e}")
            traceback.print_exc()
    
    def _plot_training_curves(self) -> None:
        """Plot training loss and accuracy curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        # loss curve
        axes[0].plot(self.train_losses, label='Train Loss', marker='o', markersize=4)
        if self.eval_losses:
            axes[0].plot(range(0, len(self.train_losses), self.eval_interval), 
                        self.eval_losses, label='Test Loss', marker='s', markersize=4)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # acc curve
        axes[1].plot(self.train_accs, label='Train Acc', marker='o', markersize=4)
        if self.eval_accs:
            axes[1].plot(range(0, len(self.train_accs), self.eval_interval), 
                        self.eval_accs, label='Test Acc', marker='s', markersize=4)
        axes[1].axhline(y=self.best_acc, color='r', linestyle='--', label=f'Best: {self.best_acc:.2f}%')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training Accuracy Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.output, 'training_curve.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Curve saved to {path}")
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """plotting confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = ax.text(j, i, cm[i, j],
                             ha="center", va="center",
                             color="white" if cm[i, j] > cm.max() / 2 else "black",
                             fontsize=8)
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        path = os.path.join(self.output, 'confusion_matrix.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Confusion matrix saved to {path}")
    
    
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
    
    def _normalize(self, hsi: torch.Tensor) -> torch.Tensor:
        """Normalize with shape:
            [B, C, H, W] -> [B, C, H, W]
        """
        mean = hsi.mean(dim=(2, 3), keepdim=True)
        std = hsi.std(dim=(2, 3), keepdim=True)
        return (hsi - mean) / (std + 1e-8)
    
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
        print(f"  Model saved to: {model_path}")
    
    def load_best_model(self) -> Module:
        model_path = os.path.join(self.output, 'models', f'{self.model_name}_best.pth')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Handle DataParallel model loading
            if isinstance(self.model, nn.DataParallel):
                # Add 'module.' prefix if loading into DataParallel model
                if not any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            else:
                # Remove 'module.' prefix if loading into non-DataParallel model
                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            self.model.load_state_dict(state_dict)
            print(f"  Loaded best model from: {model_path}")
        else:
            print(f"  Warning: Model file not found: {model_path}")
        
        return self.model
    
    def predict(self, hsi: torch.Tensor) -> torch.Tensor:
        """
        predict single sample/batch    
        hsi: shape of [B, C, H, W] or [B, H, W, C]

        Returns:
            tensor: predicted labels [B]
        """
        self.model.eval()
        
        with torch.no_grad():
            if hsi.dim() == 4 and hsi.shape[-1] <= 16:
                hsi = hsi.permute(0, 3, 1, 2)
            
            hsi = hsi.to(self.device)
            hsi = self._normalize(hsi)
            
            outputs = self.model(hsi)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions
