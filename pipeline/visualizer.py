"""
visualizer.py
Cross-model comparison and enhanced visualization utilities.

Provides:
    - ModelComparator: Aggregates results from multiple models and generates
      comparative visualizations including overlay curves, radar charts,
      ROC-AUC comparison, parameter/speed bar charts, and summary tables.

Usage:
    from pipeline.visualizer import ModelComparator
    comparator = ModelComparator(output_dir, class_names)
    comparator.add_model(name, results_dict)
    comparator.plot_all()
"""

import os
import json
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    precision_recall_fscore_support, accuracy_score, cohen_kappa_score
)


class ModelComparator:
    """
    Aggregates training histories and evaluation results from multiple
    models and produces side-by-side comparative visualizations.
    """

    # Default aesthetics
    PALETTE = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
    MARKERS = ['o', 's', '^', 'D', 'v']

    def __init__(self, output_dir: str, class_names: List[str],
                 eval_interval: int = 1):
        self.output_dir = os.path.join(output_dir, 'comparison')
        os.makedirs(self.output_dir, exist_ok=True)
        self.class_names = class_names
        self.eval_interval = eval_interval

        # per-model storage
        self.model_names: List[str] = []
        self.train_losses: Dict[str, list] = {}
        self.train_accs: Dict[str, list] = {}
        self.eval_losses: Dict[str, list] = {}
        self.eval_accs: Dict[str, list] = {}
        self.results: Dict[str, dict] = {}         # final scalar results
        self.predictions: Dict[str, np.ndarray] = {}
        self.targets: Dict[str, np.ndarray] = {}
        self.probabilities: Dict[str, np.ndarray] = {}  # soft-max proba
        self.features: Dict[str, np.ndarray] = {}       # latent features
        self.feature_labels: Dict[str, np.ndarray] = {}
        self.param_counts: Dict[str, dict] = {}
        self.inference_times: Dict[str, float] = {}
        self.lr_histories: Dict[str, list] = {}

    def add_model(
        self,
        name: str,
        results: dict,
        train_losses: list,
        train_accs: list,
        eval_losses: list,
        eval_accs: list,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        feature_labels: Optional[np.ndarray] = None,
        param_counts: Optional[dict] = None,
        inference_time: Optional[float] = None,
        lr_history: Optional[list] = None,
    ):
        """Register a trained model's data for comparison."""
        self.model_names.append(name)
        self.results[name] = results
        self.train_losses[name] = train_losses
        self.train_accs[name] = train_accs
        self.eval_losses[name] = eval_losses
        self.eval_accs[name] = eval_accs
        self.predictions[name] = predictions
        self.targets[name] = targets
        if probabilities is not None:
            self.probabilities[name] = probabilities
        if features is not None:
            self.features[name] = features
        if feature_labels is not None:
            self.feature_labels[name] = feature_labels
        if param_counts is not None:
            self.param_counts[name] = param_counts
        if inference_time is not None:
            self.inference_times[name] = inference_time
        if lr_history is not None:
            self.lr_histories[name] = lr_history

    def plot_all(self):
        """Generate all comparative visualizations."""
        print(f"\n{'='*50}")
        print("Generating cross-model comparison visualizations")
        print(f"{'='*50}")

        self._plot_training_curves_overlay()
        self._plot_per_class_metrics_comparison()
        self._plot_radar_chart()
        self._plot_confusion_matrices_side_by_side()
        self._plot_model_complexity()

        if self.probabilities:
            self._plot_roc_comparison()
            self._plot_pr_curves_comparison()

        if self.features:
            self._plot_tsne_comparison()

        if self.lr_histories:
            self._plot_lr_comparison()

        self._generate_summary_table()
        self._save_results_json()

        print(f"\nAll comparison plots saved to: {self.output_dir}")

    def _plot_training_curves_overlay(self):
        """Overlay training loss & accuracy curves for all models."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        for idx, name in enumerate(self.model_names):
            c = self.PALETTE[idx % len(self.PALETTE)]
            m = self.MARKERS[idx % len(self.MARKERS)]

            # train loss
            axes[0, 0].plot(self.train_losses[name], label=name,
                            color=c, marker=m, markersize=3, linewidth=1.5)
            # eval loss
            if self.eval_losses[name]:
                x_eval = list(range(0, len(self.train_losses[name]),
                                    self.eval_interval))[:len(self.eval_losses[name])]
                axes[0, 1].plot(x_eval, self.eval_losses[name], label=name,
                                color=c, marker=m, markersize=3, linewidth=1.5)
            # train acc
            axes[1, 0].plot(self.train_accs[name], label=name,
                            color=c, marker=m, markersize=3, linewidth=1.5)
            # eval acc
            if self.eval_accs[name]:
                x_eval = list(range(0, len(self.train_accs[name]),
                                    self.eval_interval))[:len(self.eval_accs[name])]
                axes[1, 1].plot(x_eval, self.eval_accs[name], label=name,
                                color=c, marker=m, markersize=3, linewidth=1.5)

        titles = ['Train Loss', 'Eval Loss', 'Train Accuracy (%)', 'Eval Accuracy (%)']
        ylabels = ['Loss', 'Loss', 'Accuracy (%)', 'Accuracy (%)']
        for i, ax in enumerate(axes.flat):
            ax.set_title(titles[i], fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabels[i])
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Training Curves — All Models', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._save('training_curves_overlay.png')

    def _plot_per_class_metrics_comparison(self):
        """Grouped bar chart: Precision / Recall / F1 per class, per model."""
        num_classes = len(self.class_names)
        metrics_per_model = {}

        for name in self.model_names:
            p, r, f1, _ = precision_recall_fscore_support(
                self.targets[name], self.predictions[name],
                labels=range(num_classes), zero_division=0
            )
            metrics_per_model[name] = {'precision': p, 'recall': r, 'f1': f1}

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        metric_labels = ['Precision', 'Recall', 'F1-Score']
        metric_keys = ['precision', 'recall', 'f1']

        x = np.arange(num_classes)
        n = len(self.model_names)
        width = 0.8 / max(n, 1)

        for ax_idx, (ax, m_label, m_key) in enumerate(
                zip(axes, metric_labels, metric_keys)):
            for i, name in enumerate(self.model_names):
                vals = metrics_per_model[name][m_key]
                c = self.PALETTE[i % len(self.PALETTE)]
                ax.bar(x + i * width - (n - 1) * width / 2, vals,
                       width, label=name, color=c, alpha=0.85, edgecolor='white')
            ax.set_xticks(x)
            ax.set_xticklabels(self.class_names, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel(m_label)
            ax.set_title(f'Per-Class {m_label}', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle('Per-Class Metrics Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self._save('per_class_metrics.png')

    def _plot_radar_chart(self):
        """Radar chart comparing per-class accuracy for each model."""
        num_classes = len(self.class_names)
        angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False).tolist()
        angles += angles[:1]  # close polygon

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for idx, name in enumerate(self.model_names):
            p, r, f1, _ = precision_recall_fscore_support(
                self.targets[name], self.predictions[name],
                labels=range(num_classes), zero_division=0
            )
            values = f1.tolist() + [f1[0]]
            c = self.PALETTE[idx % len(self.PALETTE)]
            ax.plot(angles, values, linewidth=2, label=name, color=c)
            ax.fill(angles, values, alpha=0.1, color=c)

        ax.set_thetagrids(np.degrees(angles[:-1]), self.class_names, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title('Per-Class F1 Radar Chart', fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        plt.tight_layout()
        self._save('radar_chart.png')

    def _plot_confusion_matrices_side_by_side(self):
        """Normalized confusion matrices side by side."""
        from sklearn.metrics import confusion_matrix as cm_func

        n = len(self.model_names)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        for idx, name in enumerate(self.model_names):
            cm = cm_func(self.targets[name], self.predictions[name])
            cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
            sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
                        xticklabels=self.class_names,
                        yticklabels=self.class_names,
                        ax=axes[idx], cbar=False)
            acc = accuracy_score(self.targets[name], self.predictions[name]) * 100
            kappa = cohen_kappa_score(self.targets[name], self.predictions[name]) * 100
            axes[idx].set_title(f'{name}\nAcc={acc:.1f}% κ={kappa:.1f}%',
                                fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('True' if idx == 0 else '')
            axes[idx].set_xlabel('Predicted')
            axes[idx].tick_params(axis='both', labelsize=7)

        plt.suptitle('Confusion Matrices (Normalized)', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        self._save('confusion_matrices_comparison.png')

    def _plot_model_complexity(self):
        """Bar chart: total params, trainable params, inference time."""
        if not self.param_counts:
            return
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        names = list(self.param_counts.keys())
        total = [self.param_counts[n].get('total', 0) / 1e6 for n in names]
        trainable = [self.param_counts[n].get('trainable', 0) / 1e6 for n in names]
        colors = [self.PALETTE[i % len(self.PALETTE)] for i in range(len(names))]

        # total params
        bars0 = axes[0].bar(names, total, color=colors, alpha=0.85, edgecolor='white')
        axes[0].set_ylabel('Parameters (M)')
        axes[0].set_title('Total Parameters', fontsize=12, fontweight='bold')
        for b, v in zip(bars0, total):
            axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                         f'{v:.2f}M', ha='center', fontsize=9)
        axes[0].grid(axis='y', alpha=0.3)

        # trainable params
        bars1 = axes[1].bar(names, trainable, color=colors, alpha=0.85, edgecolor='white')
        axes[1].set_ylabel('Parameters (M)')
        axes[1].set_title('Trainable Parameters', fontsize=12, fontweight='bold')
        for b, v in zip(bars1, trainable):
            axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                         f'{v:.2f}M', ha='center', fontsize=9)
        axes[1].grid(axis='y', alpha=0.3)

        # inference time (if available)
        if self.inference_times:
            times = [self.inference_times.get(n, 0) for n in names]
            bars2 = axes[2].bar(names, times, color=colors, alpha=0.85, edgecolor='white')
            axes[2].set_ylabel('Time (ms/batch)')
            axes[2].set_title('Inference Latency', fontsize=12, fontweight='bold')
            for b, v in zip(bars2, times):
                axes[2].text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                             f'{v:.1f}ms', ha='center', fontsize=9)
            axes[2].grid(axis='y', alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'N/A', transform=axes[2].transAxes,
                         ha='center', fontsize=14)
            axes[2].set_title('Inference Latency', fontsize=12, fontweight='bold')

        plt.suptitle('Model Complexity Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        self._save('model_complexity.png')

    def _plot_roc_comparison(self):
        """Per-class & macro ROC curves for all models."""
        num_classes = len(self.class_names)
        n_models = len(self.model_names)

        # Macro ROC per model
        fig_macro, ax_macro = plt.subplots(figsize=(8, 7))

        # Per-class ROC: one subplot row per model
        fig_pc, axes_pc = plt.subplots(n_models, 1, figsize=(10, 5 * n_models))
        if n_models == 1:
            axes_pc = [axes_pc]

        for idx, name in enumerate(self.model_names):
            proba = self.probabilities[name]   # [N, K]
            tgt = self.targets[name]
            c = self.PALETTE[idx % len(self.PALETTE)]

            # One-hot encode targets
            y_onehot = np.zeros((len(tgt), num_classes))
            for i, t in enumerate(tgt):
                if 0 <= t < num_classes:
                    y_onehot[i, t] = 1

            fpr_all, tpr_all, roc_auc_all = {}, {}, {}
            for cls_i in range(num_classes):
                fpr_all[cls_i], tpr_all[cls_i], _ = roc_curve(
                    y_onehot[:, cls_i], proba[:, cls_i])
                roc_auc_all[cls_i] = auc(fpr_all[cls_i], tpr_all[cls_i])

            # Macro avg
            all_fpr = np.unique(np.concatenate(
                [fpr_all[i] for i in range(num_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for cls_i in range(num_classes):
                mean_tpr += np.interp(all_fpr, fpr_all[cls_i], tpr_all[cls_i])
            mean_tpr /= num_classes
            macro_auc = auc(all_fpr, mean_tpr)

            ax_macro.plot(all_fpr, mean_tpr, label=f'{name} (AUC={macro_auc:.3f})',
                          color=c, linewidth=2)

            # Per-class plot for this model
            for cls_i in range(num_classes):
                axes_pc[idx].plot(
                    fpr_all[cls_i], tpr_all[cls_i],
                    label=f'{self.class_names[cls_i]} ({roc_auc_all[cls_i]:.2f})',
                    linewidth=1.2)
            axes_pc[idx].plot([0, 1], [0, 1], 'k--', alpha=0.4)
            axes_pc[idx].set_title(f'{name} — Per-Class ROC', fontsize=11, fontweight='bold')
            axes_pc[idx].set_xlabel('FPR')
            axes_pc[idx].set_ylabel('TPR')
            axes_pc[idx].legend(fontsize=7, loc='lower right')
            axes_pc[idx].grid(True, alpha=0.3)

        ax_macro.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        ax_macro.set_title('Macro-Average ROC Comparison', fontsize=13, fontweight='bold')
        ax_macro.set_xlabel('False Positive Rate')
        ax_macro.set_ylabel('True Positive Rate')
        ax_macro.legend(fontsize=10)
        ax_macro.grid(True, alpha=0.3)
        fig_macro.tight_layout()
        fig_macro.savefig(os.path.join(self.output_dir, 'roc_macro_comparison.png'),
                          dpi=150, bbox_inches='tight')
        plt.close(fig_macro)

        fig_pc.suptitle('Per-Class ROC Curves', fontsize=14, fontweight='bold')
        fig_pc.tight_layout(rect=[0, 0, 1, 0.97])
        fig_pc.savefig(os.path.join(self.output_dir, 'roc_per_class.png'),
                       dpi=150, bbox_inches='tight')
        plt.close(fig_pc)
        print("  ROC curves saved.")

    def _plot_pr_curves_comparison(self):
        """Precision-Recall curves with Average Precision (AP) per model."""
        num_classes = len(self.class_names)

        fig, ax = plt.subplots(figsize=(8, 7))

        for idx, name in enumerate(self.model_names):
            proba = self.probabilities[name]
            tgt = self.targets[name]
            c = self.PALETTE[idx % len(self.PALETTE)]

            y_onehot = np.zeros((len(tgt), num_classes))
            for i, t in enumerate(tgt):
                if 0 <= t < num_classes:
                    y_onehot[i, t] = 1

            # Macro AP
            ap_per_class = []
            for cls_i in range(num_classes):
                ap_per_class.append(
                    average_precision_score(y_onehot[:, cls_i], proba[:, cls_i]))
            macro_ap = np.mean(ap_per_class)

            # Micro PR curve
            precision_vals, recall_vals, _ = precision_recall_curve(
                y_onehot.ravel(), proba.ravel())
            ax.plot(recall_vals, precision_vals,
                    label=f'{name} (mAP={macro_ap:.3f})',
                    color=c, linewidth=2)

        ax.set_title('Precision-Recall Curves (Micro)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save('precision_recall_comparison.png')
    
    def _plot_tsne_comparison(self):
        """t-SNE visualization of learned feature embeddings."""
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print("  sklearn TSNE not available, skipping.")
            return

        n = len(self.features)
        if n == 0:
            return

        fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
        if n == 1:
            axes = [axes]

        for idx, name in enumerate(self.model_names):
            if name not in self.features:
                continue
            feats = self.features[name]
            labels = self.feature_labels.get(name, self.targets.get(name))
            if labels is None:
                continue

            # Subsample for speed
            max_pts = 3000
            if len(feats) > max_pts:
                rng = np.random.RandomState(42)
                sel = rng.choice(len(feats), max_pts, replace=False)
                feats = feats[sel]
                labels = labels[sel]

            tsne = TSNE(n_components=2, perplexity=min(30, len(feats) - 1),
                        random_state=42, n_iter=1000)
            emb = tsne.fit_transform(feats)

            num_classes = len(self.class_names)
            cmap = plt.cm.get_cmap('tab10', num_classes)
            for cls_i in range(num_classes):
                mask = labels == cls_i
                if mask.sum() == 0:
                    continue
                axes[idx].scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.6,
                                  color=cmap(cls_i), label=self.class_names[cls_i])

            axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
            axes[idx].legend(fontsize=6, markerscale=2, loc='best')
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])

        plt.suptitle('t-SNE of Learned Feature Embeddings', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self._save('tsne_comparison.png')

    def _plot_lr_comparison(self):
        """Overlay learning rate schedules across models."""
        fig, ax = plt.subplots(figsize=(10, 4))
        for idx, name in enumerate(self.model_names):
            if name not in self.lr_histories:
                continue
            c = self.PALETTE[idx % len(self.PALETTE)]
            ax.plot(self.lr_histories[name], label=name, color=c, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()
        self._save('lr_schedule_comparison.png')


    def _generate_summary_table(self):
        """Generate a summary image table of key metrics for all models."""
        num_classes = len(self.class_names)
        rows = []
        headers = ['Model', 'Accuracy (%)', 'Kappa (%)', 'Macro-F1',
                    'Params (M)', 'Train Time (s)', 'Best Epoch']

        for name in self.model_names:
            acc = accuracy_score(self.targets[name], self.predictions[name]) * 100
            kappa = cohen_kappa_score(self.targets[name], self.predictions[name]) * 100
            _, _, f1, _ = precision_recall_fscore_support(
                self.targets[name], self.predictions[name],
                labels=range(num_classes), zero_division=0, average='macro')

            params_m = (self.param_counts.get(name, {}).get('total', 0) / 1e6
                        if name in self.param_counts else '-')
            train_time = self.results[name].get('training_time', '-')
            best_ep = self.results[name].get('best_epoch', '-')

            rows.append([
                name,
                f'{acc:.2f}',
                f'{kappa:.2f}',
                f'{f1:.4f}',
                f'{params_m:.2f}' if isinstance(params_m, float) else params_m,
                f'{train_time:.1f}' if isinstance(train_time, float) else train_time,
                str(best_ep)
            ])

        # matplotlib table
        fig, ax = plt.subplots(figsize=(14, 1.5 + 0.5 * len(rows)))
        ax.axis('off')
        table = ax.table(cellText=rows, colLabels=headers, loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.6)

        # header colour
        for j in range(len(headers)):
            table[0, j].set_facecolor('#2196F3')
            table[0, j].set_text_props(color='white', fontweight='bold')

        # highlight best accuracy row
        accs = [accuracy_score(self.targets[n], self.predictions[n])
                for n in self.model_names]
        best_idx = int(np.argmax(accs))
        for j in range(len(headers)):
            table[best_idx + 1, j].set_facecolor('#E3F2FD')

        ax.set_title('Model Comparison Summary', fontsize=14,
                      fontweight='bold', pad=15)
        plt.tight_layout()
        self._save('summary_table.png')

        # Print table to console
        print(f"\n{'='*80}")
        print("Model Comparison Summary")
        print(f"{'='*80}")
        header_str = ' | '.join(f'{h:>14s}' for h in headers)
        print(header_str)
        print('-' * len(header_str))
        for row in rows:
            print(' | '.join(f'{v:>14s}' for v in row))
        print(f"{'='*80}\n")

    def _save_results_json(self):
        """Persist all scalar results as JSON for programmatic consumption."""
        summary = {}
        num_classes = len(self.class_names)
        for name in self.model_names:
            acc = accuracy_score(self.targets[name], self.predictions[name]) * 100
            kappa = cohen_kappa_score(self.targets[name], self.predictions[name]) * 100
            p, r, f1, _ = precision_recall_fscore_support(
                self.targets[name], self.predictions[name],
                labels=range(num_classes), zero_division=0)
            summary[name] = {
                'accuracy': round(acc, 4),
                'kappa': round(kappa, 4),
                'macro_f1': round(float(f1.mean()), 4),
                'per_class_f1': {self.class_names[i]: round(float(f1[i]), 4)
                                 for i in range(num_classes)},
                'per_class_precision': {self.class_names[i]: round(float(p[i]), 4)
                                        for i in range(num_classes)},
                'per_class_recall': {self.class_names[i]: round(float(r[i]), 4)
                                     for i in range(num_classes)},
                'total_params': self.param_counts.get(name, {}).get('total', None),
                'trainable_params': self.param_counts.get(name, {}).get('trainable', None),
                'training_time': self.results[name].get('training_time', None),
                'best_epoch': self.results[name].get('best_epoch', None),
            }
        path = os.path.join(self.output_dir, 'comparison_results.json')
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Results JSON saved to {path}")

    def _save(self, filename: str, dpi: int = 150):
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
