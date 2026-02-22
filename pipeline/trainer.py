"""
trainer.py
Hyperspectral Image Classification Trainer
Optimization includes gradient accumulation, mixed precision training, learning rate warmup, and early stopping.
CPU/GPU supported with flexible memory configuration.

start train with:
    trainer.train()
"""



import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from munch import Munch
from torch.nn import Module
from typing import Tuple, Callable, Dict
from pipeline.dataset import AbstractHSDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support, roc_curve, auc,
                             average_precision_score, precision_recall_curve)
import cv2, os, time, torch, traceback, warnings, json

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
        
        # enhanced tracking for richer visualizations
        self.lr_history = []          # lr per epoch
        self.grad_norms = []          # gradient L2 norm per epoch
        self.epoch_times = []         # wall-clock time per epoch

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
            prefetch_factor = 0
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
        """Initialize training components: optimizer, scheduler, loss, precision"""
        print("Initializing training components with:")
        
        # Optimizer - AdamW for better generalization
        lr = self.config.common.lr
        weight_decay = self.config.common.weight_decay
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
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
        
        # loss function — always use ignore_index=255 for background/padding
        label_smoothing = self.config.common.label_smoothing if hasattr(self.config.common, 'label_smoothing') else 0.1
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, ignore_index=255)
        print(f"  loss function: CrossEntropyLoss (label_smoothing={label_smoothing}, ignore_index=255)")
        
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
            
            # cache sweep
            if batch_idx % 5 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
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
    def evaluate(self, collect_extra: bool = False
                 ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Evaluate on the test set with optimized performance.
        
        Args:
            collect_extra: if True, also collect softmax probabilities and
                           penultimate-layer features (for ROC / t-SNE).
        
        Return:
            Tuple (``loss``, ``accuracy``, ``kappa``, ``predictions``, ``labels``)
            When collect_extra is True, also stores self._last_probas and
            self._last_features for consumption by the comparator.
        """
        self.model.eval()
        total_loss = 0.0
        predictions_list = []
        targets_list = []
        probas_list = [] if collect_extra else None
        features_list = [] if collect_extra else None
        
        # hook for feature extraction (penultimate layer)
        _feat_hook = None
        _feat_buffer = []
        if collect_extra:
            # Find the layer just before the classification head
            target_layer = None
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d)):
                    target_layer = module
            if target_layer is not None:
                def _hook(mod, inp, out):
                    _feat_buffer.append(out.detach().cpu())
                _feat_hook = target_layer.register_forward_hook(_hook)
        
        eval_loader = self.test_loader
        
        pbar = tqdm(eval_loader, desc='Evaluating', leave=False)
        
        with torch.no_grad():
            for batch_data in pbar:
                hsi, labels = self._unpack_batch(batch_data)
                hsi = hsi.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # format check up
                if hsi.dim() == 4 and hsi.shape[-1] <= 16:
                    hsi = hsi.permute(0, 3, 1, 2)
                
                hsi = self._normalize(hsi)
                
                outputs = self.model(hsi)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                
                # Flatten and filter out ignore pixels for metrics
                pred_np = predicted.cpu().numpy()
                label_np = labels.cpu().numpy()
                valid_mask = label_np != 255
                predictions_list.append(pred_np[valid_mask])
                targets_list.append(label_np[valid_mask])
                
                # Skip collecting probabilities to save memory (dense output)
                if collect_extra:
                    pass  # probabilities not collected for dense predictions
        
        if _feat_hook is not None:
            _feat_hook.remove()
        
        predictions = np.concatenate(predictions_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        
        acc = accuracy_score(targets, predictions) * 100
        kappa = cohen_kappa_score(targets, predictions) * 100
        loss = total_loss / len(eval_loader)
        
        # Compute mIoU
        model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        miou = self._compute_miou(targets, predictions, model_ref.num_classes)
        self._last_miou = miou
        
        if collect_extra:
            if probas_list:
                self._last_probas = np.concatenate(probas_list, axis=0)
            else:
                self._last_probas = None
            if _feat_buffer:
                raw = torch.cat(_feat_buffer, dim=0)
                self._last_features = raw.view(raw.size(0), -1).numpy()
                self._last_feature_labels = targets
            else:
                self._last_features = None
                self._last_feature_labels = None
        
        return loss, acc, kappa, predictions, targets
    
    def train(self) -> Dict[str, float]:
        """
        Training loop with optional data validation.
        
        Returns:
            Dict[``str``, ``float``] of final results.
        """
        # Validate data quality on first epoch
        if not hasattr(self, '_data_validated'):
            self._validate_data_quality()
            self._data_validated = True
        
        print(f"\n{'='*20}")
        if self.debug_mode:
            print(f"Debug mode enabled: CAM and visualizations enabled every epoch.")
        print(f"Training ({self.model_name})")
        print(f"{'='*20}\n")
        
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
                eval_loss, eval_acc, kappa, pred, target = self.evaluate()
                self.eval_losses.append(eval_loss)
                self.eval_accs.append(eval_acc)
                
                miou_str = ''
                if hasattr(self, '_last_miou'):
                    miou_str = f' mIoU: {self._last_miou:6.2f}%'
                
                print(f"\n[Epoch {epoch+1:3d}] "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:6.2f}% | "
                      f"Test Loss: {eval_loss:.4f} Acc: {eval_acc:6.2f}% Kappa: {kappa:6.2f}%{miou_str}")
                
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
            
            # Generate visualizations
            if self.debug_mode or (should_eval and (epoch + 1) % 2 == 0):
                try:
                    self._generate_cam(epoch)
                except Exception as e:
                    print(f"  CAM error: {e}")
        
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
        
        # final evaluation with extra data collection (probabilities, features)
        final_loss, final_acc, final_kappa, final_pred, final_target = self.evaluate(
            collect_extra=True)
        
        # measure inference time (average over a few batches)
        self._inference_time = self._measure_inference_time()
        
        # parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._param_counts = {'total': total_params, 'trainable': trainable_params}
        
        # single-model visualizations 
        self._plot_training_curves()
        self._plot_confusion_matrix(final_target, final_pred)
        self._plot_per_class_metrics(final_target, final_pred)
        self._plot_roc_curves(final_target)
        self._plot_grad_norm_curve()
        self._plot_lr_curve()
        self._plot_epoch_time_curve()
        
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
                # Dense labels: show center pixel label
                center_label = labels[i][labels[i].shape[0]//2, labels[i].shape[1]//2].item()
                axes[i, 0].set_title(f'Sample {i+1}\nCenter Label: {center_label}')
                axes[i, 0].axis('off')
                
                # grayscale intensity
                gray = sample.mean(axis=0)
                axes[i, 1].imshow(gray, cmap='gray')
                # Show center pixel prediction
                center_pred = preds[i][preds[i].shape[0]//2, preds[i].shape[1]//2].item()
                axes[i, 1].set_title(f'Intensity\nCenter Pred: {center_pred}')
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
        """Confusion matrix per class"""
        if not hasattr(self.config.clsf, 'targets'):
            return
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=range(self.config.clsf.num), zero_division=0
        )
        
        print("\n[Classification Metrics]")
        for i in range(self.config.clsf.num):
            name = self.config.clsf.targets[i] if i < len(self.config.clsf.targets) else f"Class_{i}"
            print(f"  {i:1d}. {name:10s} | P:{precision[i]:.4f} | R:{recall[i]:.4f} | F1:{f1[i]:.4f}")
        
        # Plot enhanced confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Count
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix (Count)')
        axes[0].set_ylabel('True')
        axes[0].set_xlabel('Predicted')
        
        # Normalized
        sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Greens', ax=axes[1])
        axes[1].set_title('Confusion Matrix (Normalized)')
        axes[1].set_ylabel('True')
        axes[1].set_xlabel('Predicted')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _visualize_layer_activations(self) -> None:
        """Visualize layer activation statistics for debugging"""
        self.model.eval()
        try:
            sample, _ = self.test_loader.dataset[0]
            sample = sample.unsqueeze(0).to(self.device)
            
            activation_stats = {}
            hooks = []
            
            def hook_fn(name):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        activation_stats[name] = {
                            'mean': output.mean().item(),
                            'std': output.std().item(),
                        }
                return hook
            
            # Register hooks for Conv and Linear layers
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    hooks.append(module.register_forward_hook(hook_fn(name)))
            
            with torch.no_grad():
                _ = self.model(sample)
            
            # Clean up hooks
            for h in hooks:
                h.remove()
            
            # Visualize if we have data
            if len(activation_stats) > 0:
                layers = list(activation_stats.keys())[:15]
                means = [activation_stats[l]['mean'] for l in layers]
                stds = [activation_stats[l]['std'] for l in layers]
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].bar(range(len(means)), means, alpha=0.7, color='steelblue')
                axes[0].set_title('Mean Activation by Layer')
                axes[0].set_xlabel('Layer Index')
                axes[1].bar(range(len(stds)), stds, alpha=0.7, color='coral')
                axes[1].set_title('Std Activation by Layer')
                axes[1].set_xlabel('Layer Index')
                plt.tight_layout()
                save_path = os.path.join(self.output, 'layer_activations.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"  Layer visualization skipped: {e}")
    
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
        Predict single sample/batch.
        
        Args:
            hsi: Tensor [B, C, H, W] or [B, H, W, C]
            
        Returns:
            Predicted dense labels [B, H, W]
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
    
    def _validate_data_quality(self) -> None:
        """Validate dataset quality: distribution, ranges, duplicates"""
        print("\n[Data Quality Check]")
        labels = self.dataLoader.patch_labels[1:]
        
        # Class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"  Label distribution:")
        for lbl, cnt in zip(unique_labels, counts):
            pct = cnt / len(labels) * 100
            print(f"    Class {lbl}: {cnt:6d} ({pct:5.1f}%)")
        
        # Imbalance ratio
        imbalance = counts.max() / counts.min()
        if imbalance > 10:
            print(f"   Class imbalance ratio: {imbalance:.1f}x")
        
        # Data range check
        sample_patch = self.dataLoader._get_patch(0)
        data_min, data_max = sample_patch.min(), sample_patch.max()
        print(f"  Data range: [{data_min:.4f}, {data_max:.4f}]")
        if data_max > 1.5 or data_min < -0.5:
            print(f"   Unexpected data range - check normalization")
    
    def _compute_gradcam(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """Compute Grad-CAM heatmap for model interpretability"""
        features = None
        gradients = None
        
        def forward_hook(module, input, output):
            nonlocal features
            features = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0].detach()
        
        # Find last Conv2d layer
        target_layer = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        
        if target_layer is None:
            return np.ones((x.shape[-2], x.shape[-1])) * 0.5
        
        # Register hooks
        h_f = target_layer.register_forward_hook(forward_hook)
        h_b = target_layer.register_backward_hook(backward_hook)
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(x)
        # For segmentation output [B, K, H, W], sum spatial dims to get scalar
        if output.dim() == 4:
            loss = output[0, class_idx].sum()
        else:
            loss = output[0, class_idx]
        loss.backward(retain_graph=True)
        
        # Compute CAM
        if features is not None and gradients is not None:
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (features * weights).sum(dim=1).squeeze(0)
            cam = torch.relu(cam).cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = cv2.resize(cam, (x.shape[-1], x.shape[-2])) if cam.size > 0 else np.ones((x.shape[-2], x.shape[-1])) * 0.5
        else:
            cam = np.ones((x.shape[-2], x.shape[-1])) * 0.5
        
        h_f.remove()
        h_b.remove()
        return cam

    def _plot_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Bar chart of per-class Precision / Recall / F1."""
        if not hasattr(self.config.clsf, 'targets'):
            return
        # remove BG.
        num_classes = self.config.clsf.num - 1
        names = self.config.clsf.targets[1:num_classes]

        p, r, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=range(num_classes), zero_division=0)

        x = np.arange(num_classes)
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - width, p, width, label='Precision', color='#2196F3', alpha=0.85)
        ax.bar(x, r, width, label='Recall', color='#FF5722', alpha=0.85)
        ax.bar(x + width, f1, width, label='F1-Score', color='#4CAF50', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.set_title(f'{self.model_name} — Per-Class Metrics', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output, 'per_class_metrics.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Per-class metrics saved to {path}")

    def _plot_roc_curves(self, y_true: np.ndarray) -> None:
        """Per-class ROC curves with AUC for the current model."""
        if not hasattr(self, '_last_probas') or self._last_probas is None:
            return
        # remove BG.
        num_classes = self.config.clsf.num - 1
        names = self.config.clsf.targets[1:num_classes]
        proba = self._last_probas

        y_onehot = np.zeros((len(y_true), num_classes))
        for i, t in enumerate(y_true):
            if 0 <= t < num_classes:
                y_onehot[i, t] = 1

        fig, ax = plt.subplots(figsize=(8, 7))
        for cls_i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_onehot[:, cls_i], proba[:, cls_i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{names[cls_i]} (AUC={roc_auc:.3f})', linewidth=1.3)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{self.model_name} — Per-Class ROC Curves', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output, 'roc_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ROC curves saved to {path}")

    def _plot_grad_norm_curve(self) -> None:
        """Gradient L2 norm across training epochs."""
        if not self.grad_norms:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.grad_norms, marker='o', markersize=3, linewidth=1.5, color='#FF5722')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient L2 Norm')
        ax.set_title(f'{self.model_name} — Gradient Norm', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output, 'gradient_norm.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Gradient norm curve saved to {path}")

    def _plot_lr_curve(self) -> None:
        """Learning rate schedule over training epochs."""
        if not self.lr_history:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.lr_history, linewidth=1.5, color='#2196F3')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.set_title(f'{self.model_name} — Learning Rate Schedule', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output, 'lr_schedule.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  LR schedule saved to {path}")

    def _plot_epoch_time_curve(self) -> None:
        """Wall-clock time per epoch."""
        if not self.epoch_times:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(self.epoch_times)), self.epoch_times,
               color='#4CAF50', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (s)')
        ax.set_title(f'{self.model_name} — Per-Epoch Wall-Clock Time', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output, 'epoch_times.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Epoch time chart saved to {path}")

    def _measure_inference_time(self, num_batches: int = 10) -> float:
        """
        Measure average inference latency in ms per batch.
        
        Returns:
            Average inference time in milliseconds.
        """
        self.model.eval()
        times = []
        try:
            test_iter = iter(self.test_loader)
            for _ in range(min(num_batches, len(self.test_loader))):
                batch_data = next(test_iter)
                hsi, _ = self._unpack_batch(batch_data)
                hsi = hsi.to(self.device, non_blocking=True)
                if hsi.dim() == 4 and hsi.shape[-1] <= 16:
                    hsi = hsi.permute(0, 3, 1, 2)
                hsi = self._normalize(hsi)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = self.model(hsi)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)  # ms
        except Exception:
            pass
        return float(np.mean(times)) if times else 0.0
