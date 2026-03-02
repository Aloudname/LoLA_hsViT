"""
visualize.py

Public API:
    - Visualizer: A wrapper class for all visualization functions.
    - plot_training_curves
    - plot_confusion_matrix
    - plot_roc_curves
    - plot_class_metrics
    - plot_cam_overlay

Usage:
    >>> from pipeline.visualize import Visualizer
    >>> viz = Visualizer(output_dir="./output")
    >>> viz.plot_training_curves(train_losses, eval_losses, model_name="exp1")
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_curve, auc)

__all__ = [
    'Visualizer',
    'plot_training_curves',
    'plot_confusion_matrix', 
    'plot_roc_curves',
    'plot_class_metrics',
    'plot_cam_overlay',
]


class Visualizer:
    """
    Wrapper class for visualization.
    
    Usage:
        - plot_training_curves()
        - plot_confusion_matrix()
        - plot_roc_curves()
        - plot_class_metrics()
        - plot_cam()
        - generate_all()
    """
    
    def __init__(self, 
                 output_dir: str,
                 model_name: str = "model",
                 class_names: Optional[List[str]] = None,
                 dpi: int = 150):
        """
        Args:
            output_dir: output_dir
            model_name: for plotting titles and filenames
            class_names: class names for confusion matrix and metrics
            dpi: figure DPI for saving
        """
        self._output_dir = output_dir
        self._model_name = model_name
        self._class_names = class_names or []
        self._dpi = dpi
        
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_training_curves(self,
                             train_losses: List[float],
                             eval_losses: List[float],
                             train_accs: Optional[List[float]] = None,
                             eval_accs: Optional[List[float]] = None,
                             eval_interval: int = 1,
                             best_acc: Optional[float] = None,
                             save: bool = True) -> plt.Figure:
        """  
        Args:
            train_losses: loss List of epochs
            eval_losses: validation loss List of epochs
            train_accs: acc list of epochs
            eval_accs: eval acc list of epochs
            eval_interval: from config, to align with x bar
            best_acc: best eval acc for reference line
            save: if save file
        
        Returns:
            plt.Figure object
        """
        ncols = 2 if train_accs else 1
        fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 5))
        if ncols == 1:
            axes = [axes]
        
        epochs = range(len(train_losses))
        eval_epochs = [e for e in epochs 
                      if (e + 1) % eval_interval == 0][:len(eval_losses)]
        
        # loss
        axes[0].plot(train_losses, 'b-o', markersize=3, label='Train')
        if eval_losses:
            axes[0].plot(eval_epochs, eval_losses, 'r-s', markersize=3, label='Test')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{self._model_name} — Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # acc
        if train_accs:
            axes[1].plot(train_accs, 'b-o', markersize=3, label='Train')
            if eval_accs:
                axes[1].plot(eval_epochs, eval_accs, 'r-s', markersize=3, label='Test')
            if best_acc:
                axes[1].axhline(y=best_acc, color='g', linestyle='--',
                               label=f'Best: {best_acc:.2f}%')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].set_title(f'{self._model_name} — Training Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            path = os.path.join(self._output_dir, 'training_curves.png')
            fig.savefig(path, dpi=self._dpi, bbox_inches='tight')
            plt.close(fig)
        
        return fig
    
    def plot_confusion_matrix(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              normalize: bool = True,
                              save: bool = True) -> plt.Figure:
        """
        Args:
            y_true: ndarray of shape (N,)
            y_pred: ndarray of shape (N,)
            normalize: if True, also plot normalized confusion matrix
            save: if save file
        
        Returns:
            plt.Figure object
        """
        num_classes = len(self._class_names) or max(y_true.max(), y_pred.max()) + 1
        labels = range(num_classes)
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        if normalize:
            cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # cm
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                       xticklabels=self._class_names,
                       yticklabels=self._class_names)
            axes[0].set_title('Confusion Matrix (Counts)')
            axes[0].set_ylabel('True')
            axes[0].set_xlabel('Predicted')
            
            # normalized cm
            sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Greens', ax=axes[1],
                       xticklabels=self._class_names,
                       yticklabels=self._class_names)
            axes[1].set_title('Confusion Matrix (Normalized)')
            axes[1].set_ylabel('True')
            axes[1].set_xlabel('Predicted')
        else:
            fig, ax = plt.subplots(figsize=(8, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=self._class_names,
                       yticklabels=self._class_names)
            ax.set_title(f'{self._model_name} — Confusion Matrix')
            ax.set_ylabel('True')
            ax.set_xlabel('Predicted')
        
        plt.tight_layout()
        
        if save:
            path = os.path.join(self._output_dir, 'confusion_matrix.png')
            fig.savefig(path, dpi=self._dpi, bbox_inches='tight')
            plt.close(fig)
        
        return fig
    
    def plot_roc_curves(self,
                        y_true: np.ndarray,
                        y_proba: np.ndarray,
                        save: bool = True) -> plt.Figure:
        """
        Args:
            y_true: ndarray of shape (N,)
            y_proba: (N, num_classes)
            save: if save file
        
        Returns:
            plt.Figure object
        """
        num_classes = y_proba.shape[1]
        class_names = self._class_names or [f'Class {i}' for i in range(num_classes)]
        
        fig, ax = plt.subplots(figsize=(8, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
        
        for i in range(num_classes):
            y_bin = (y_true == i).astype(int)
            if y_bin.sum() == 0:
                continue
            
            fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=1.5,
                   label=f'{class_names[i]} (AUC={roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{self._model_name} — ROC Curves')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            path = os.path.join(self._output_dir, 'roc_curves.png')
            fig.savefig(path, dpi=self._dpi, bbox_inches='tight')
            plt.close(fig)
        
        return fig
    
    def plot_class_metrics(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           save: bool = True) -> plt.Figure:
        """    
        Args:
            y_true: ndarray of shape (N,)
            y_pred: ndarray of shape (N,)
            save: if save file
        
        Returns:
            plt.Figure object
        """
        num_classes = len(self._class_names) or max(y_true.max(), y_pred.max()) + 1
        labels = range(num_classes)
        class_names = self._class_names or [f'Class {i}' for i in labels]
        
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0)
        
        x = np.arange(num_classes)
        w = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - w, p, w, label='Precision', color='#2196F3', alpha=0.85)
        ax.bar(x, r, w, label='Recall', color='#FF5722', alpha=0.85)
        ax.bar(x + w, f1, w, label='F1-Score', color='#4CAF50', alpha=0.85)
        
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.set_title(f'{self._model_name} — Per-Class Metrics')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save:
            path = os.path.join(self._output_dir, 'class_metrics.png')
            fig.savefig(path, dpi=self._dpi, bbox_inches='tight')
            plt.close(fig)
        
        return fig
    
    def plot_cam(self,
                 image: np.ndarray,
                 cam: np.ndarray,
                 alpha: float = 0.5,
                 save_name: Optional[str] = None) -> plt.Figure:
        """
        Args:
            image: img shape (H, W, 3) or (H, W)
            cam: CAM shape of (H, W), value range [0,1]
            alpha: overlay alpha (transparency)
            save_name: if None, do not save.
        
        Returns:
            plt.Figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # composed RGB
        if image.ndim == 2:
            rgb = np.stack([image] * 3, axis=-1)
        else:
            rgb = image
        
        # normalize RGB
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        
        # resize CAM to match image size
        if cam.shape[:2] != rgb.shape[:2]:
            cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
        
        # apply colormap to CAM
        heatmap = plt.cm.jet(cam)[..., :3]
        overlay = alpha * heatmap + (1 - alpha) * rgb
        overlay = np.clip(overlay, 0, 1)
        
        axes[0].imshow(rgb)
        axes[0].set_title('Input')
        axes[0].axis('off')
        
        axes[1].imshow(cam, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title('CAM')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_name:
            path = os.path.join(self._output_dir, 'CAM', save_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path, dpi=self._dpi, bbox_inches='tight')
            plt.close(fig)
        
        return fig
    
    def generate_all(self,
                     y_true: np.ndarray,
                     y_pred: np.ndarray,
                     y_proba: Optional[np.ndarray] = None,
                     train_losses: Optional[List[float]] = None,
                     eval_losses: Optional[List[float]] = None,
                     train_accs: Optional[List[float]] = None,
                     eval_accs: Optional[List[float]] = None,
                     parallel: bool = True) -> None:
        """
        Get all visualizations at once.
        
        Args:
            y_true: ndarray of shape (N,)
            y_pred: ndarray of shape (N,)
            y_proba: (N, num_classes) for ROC
            train_losses: shape (num_epochs,)
            eval_losses: shape (num_eval_points,)
            train_accs: shape (num_epochs,)
            eval_accs: shape (num_eval_points,)
            parallel: shape (num_epochs,)
        """
        tasks = []
        
        if train_losses:
            tasks.append(('training_curves', 
                         lambda: self.plot_training_curves(
                             train_losses, eval_losses or [],
                             train_accs, eval_accs)))
        
        tasks.append(('confusion_matrix',
                     lambda: self.plot_confusion_matrix(y_true, y_pred)))
        tasks.append(('class_metrics',
                     lambda: self.plot_class_metrics(y_true, y_pred)))
        
        if y_proba is not None:
            tasks.append(('roc_curves',
                         lambda: self.plot_roc_curves(y_true, y_proba)))
        
        if parallel and len(tasks) > 1:
            # NOTE: matplotlib is not fully thread-safe,
            # use process-based parallel safety.
            for name, fn in tasks:
                try:
                    fn()
                except Exception as e:
                    print(f"Warning: {name} failed: {e}")
        else:
            for name, fn in tasks:
                try:
                    fn()
                except Exception as e:
                    print(f"Warning: {name} failed: {e}")


# convenience functions to plot without init Visualizer instance.
def plot_training_curves(train_losses: List[float],
                         eval_losses: List[float],
                         output_path: str,
                         **kwargs) -> None:

    dir_path = os.path.dirname(output_path)
    viz = Visualizer(dir_path)
    viz.plot_training_curves(train_losses, eval_losses, **kwargs)


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          output_path: str,
                          class_names: Optional[List[str]] = None,
                          **kwargs) -> None:
    
    dir_path = os.path.dirname(output_path)
    viz = Visualizer(dir_path, class_names=class_names)
    viz.plot_confusion_matrix(y_true, y_pred, **kwargs)


def plot_roc_curves(y_true: np.ndarray,
                    y_proba: np.ndarray,
                    output_path: str,
                    class_names: Optional[List[str]] = None,
                    **kwargs) -> None:
    
    dir_path = os.path.dirname(output_path)
    viz = Visualizer(dir_path, class_names=class_names)
    viz.plot_roc_curves(y_true, y_proba, **kwargs)


def plot_class_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       output_path: str,
                       class_names: Optional[List[str]] = None,
                       **kwargs) -> None:

    dir_path = os.path.dirname(output_path)
    viz = Visualizer(dir_path, class_names=class_names)
    viz.plot_class_metrics(y_true, y_pred, **kwargs)


def plot_cam_overlay(image: np.ndarray,
                     cam: np.ndarray,
                     output_path: str,
                     **kwargs) -> None:

    dir_path = os.path.dirname(output_path)
    save_name = os.path.basename(output_path)
    viz = Visualizer(dir_path)
    viz.plot_cam(image, cam, save_name=save_name, **kwargs)
