# visualization methods for experiment analysis and paper figures.
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from __future__ import annotations
from pipeline.analyzer import MetricsBundle
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


class Visualizer:
    """figure writer for metrics, features, and segmentation outputs."""

    def __init__(self, output_dir: str, class_names: Sequence[str]) -> None:
        self.output_dir = Path(output_dir)
        self.class_names = list(class_names)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_curves(self, history: Mapping[str, Sequence[float]]) -> None:
        """plot train/eval loss and metric curves."""
        epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

        axes[0].plot(epochs, history.get("train_loss", []), label="train_loss", color="#1f77b4")
        axes[0].plot(epochs, history.get("eval_loss", []), label="eval_loss", color="#d62728")
        axes[0].set_title("loss curve")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].grid(alpha=0.3)
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

        axes[1].plot(epochs, history.get("train_dice", []), label="train_dice", color="#2ca02c")
        axes[1].plot(epochs, history.get("eval_dice", []), label="eval_dice", color="#9467bd")
        axes[1].set_title("dice curve")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("dice")
        axes[1].grid(alpha=0.3)
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

        fig.subplots_adjust(bottom=0.28, wspace=0.28)
        self._save_fig(fig, "curves/loss_dice_curves.png")

    def plot_prf(self, bundle: MetricsBundle) -> None:
        """plot precision/recall/f1 grouped bars by class."""
        p = [bundle.per_class[n]["precision"] for n in self.class_names]
        r = [bundle.per_class[n]["recall"] for n in self.class_names]
        f1 = [bundle.per_class[n]["f1"] for n in self.class_names]

        x = np.arange(len(self.class_names))
        width = 0.26

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - width, p, width, label="precision", color="#4e79a7")
        ax.bar(x, r, width, label="recall", color="#e15759")
        ax.bar(x + width, f1, width, label="f1", color="#59a14f")
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("score")
        ax.set_title("precision + recall + f1")
        ax.grid(axis="y", alpha=0.3)
        has_legend = self._apply_external_legend(fig, ax, anchor=(1.02, 0.5))
        if has_legend:
            fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
        else:
            fig.tight_layout()
        self._save_fig(fig, "metrics/prf_triplet.png")

    def plot_confusion_matrix(self, bundle: MetricsBundle) -> None:
        """plot confusion matrix heatmap."""
        cm = bundle.confusion_matrix.astype(np.float64)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axes[0].imshow(cm, cmap="Blues")
        im1 = axes[1].imshow(cm_norm, cmap="Greens", vmin=0.0, vmax=1.0)
        axes[0].set_title("confusion matrix (count)")
        axes[1].set_title("confusion matrix (normalized)")

        for ax, mat, fmt in [(axes[0], cm, "{:.0f}"), (axes[1], cm_norm, "{:.2f}")]:
            ax.set_xticks(np.arange(len(self.class_names)))
            ax.set_yticks(np.arange(len(self.class_names)))
            ax.set_xticklabels(self.class_names, rotation=45, ha="right")
            ax.set_yticklabels(self.class_names)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=8)

                fig.colorbar(im0, ax=axes[0], shrink=0.8)
                fig.colorbar(im1, ax=axes[1], shrink=0.8)
                fig.tight_layout()
        self._save_fig(fig, "metrics/confusion_matrix.png")

    def plot_roc(self, bundle: MetricsBundle) -> None:
        """plot per-foreground-class ROC curves with AUC in legend."""
        if not bundle.roc_curves:
            return

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        fg_names = self.class_names[1:] if len(self.class_names) > 1 else self.class_names
        cmap = plt.get_cmap("tab10")

        plotted = False
        for idx, name in enumerate(fg_names):
            if name not in bundle.roc_curves:
                continue
            curve = bundle.roc_curves[name]
            fpr = np.asarray(curve.get("fpr", np.empty((0,), dtype=np.float32)))
            tpr = np.asarray(curve.get("tpr", np.empty((0,), dtype=np.float32)))
            if fpr.size == 0 or tpr.size == 0:
                continue

            auc_v = float(bundle.roc_auc.get(name, float("nan")))
            auc_txt = f"{auc_v:.3f}" if np.isfinite(auc_v) else "nan"
            ax.plot(
                fpr,
                tpr,
                linewidth=2.0,
                color=cmap(idx % 10),
                label=f"{name} (AUC={auc_txt})",
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            return

        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#666666", linewidth=1.2, label="random")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title("foreground ROC curves")
        ax.grid(alpha=0.3)

        has_legend = self._apply_external_legend(fig, ax, anchor=(1.02, 0.5), fontsize=9)
        if has_legend:
            fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
        else:
            fig.tight_layout()
        self._save_fig(fig, "metrics/roc_auc.png")

    def plot_distribution(self, stats: Mapping[str, Any]) -> None:
        """plot split distribution before/after patch filtering and tracking."""
        class_names = list(stats.get("class_names", self.class_names))
        split_stats = stats.get("split", {})

        for split_name in ["train", "eval", "test"]:
            if split_name not in split_stats:
                continue
            cur = split_stats[split_name]
            raw = np.asarray(cur.get("raw_pixel_hist", []), dtype=np.float64)
            kept = np.asarray(cur.get("kept_patch_pixel_hist", []), dtype=np.float64)
            pre = np.asarray(cur.get("tracked_pre_hist", []), dtype=np.float64)
            post = np.asarray(cur.get("tracked_post_hist", []), dtype=np.float64)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            self._plot_hist_compare(axes[0], class_names, raw, kept, title=f"{split_name}: raw vs kept")
            self._plot_hist_compare(axes[1], class_names, pre, post, title=f"{split_name}: tracked pre vs post")
            fig.subplots_adjust(bottom=0.25, wspace=0.3)
            self._save_fig(fig, f"data/{split_name}_distribution_compare.png")

    def plot_spectral(self, spectra_by_class: Mapping[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """plot class spectral curves (mean +- std)."""
        if not spectra_by_class:
            return

        fig, ax = plt.subplots(figsize=(10, 4))
        for class_name, (mean_vec, std_vec) in spectra_by_class.items():
            x = np.arange(mean_vec.shape[0])
            ax.plot(x, mean_vec, label=class_name)
            ax.fill_between(x, mean_vec - std_vec, mean_vec + std_vec, alpha=0.15)

        ax.set_title("spectral curves by class")
        ax.set_xlabel("band index")
        ax.set_ylabel("normalized intensity")
        ax.grid(alpha=0.3)
        has_legend = self._apply_external_legend(fig, ax, anchor=(1.02, 0.5), ncol=1, fontsize=9)
        if has_legend:
            fig.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
        else:
            fig.tight_layout()
        self._save_fig(fig, "data/spectral_curves.png")

    def plot_tsne(self, features: np.ndarray, labels: np.ndarray, title: str = "feature tsne") -> None:
        """plot tsne for feature-level separability."""
        if features.size == 0:
            return
        if features.shape[0] < 10:
            return

        perplexity = min(30, max(5, features.shape[0] // 10))
        tsne = TSNE(n_components=2, random_state=3407, perplexity=perplexity, init="pca")
        z = tsne.fit_transform(features)

        fig, ax = plt.subplots(figsize=(7, 6))
        for cls_idx, class_name in enumerate(self.class_names):
            idx = labels == cls_idx
            if not np.any(idx):
                continue
            ax.scatter(z[idx, 0], z[idx, 1], s=8, alpha=0.7, label=class_name)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        has_legend = self._apply_external_legend(fig, ax, anchor=(1.02, 0.5), fontsize=8)
        if has_legend:
            fig.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
        else:
            fig.tight_layout()
        self._save_fig(fig, "features/tsne.png")

    def show_segmentation(
        self,
        images: Sequence[np.ndarray],
        preds: Sequence[np.ndarray],
        gts: Sequence[np.ndarray],
        max_items: int = 3,
        prefix: str = "test",
    ) -> None:
        """plot prediction/ground-truth side-by-side samples."""
        n = min(max_items, len(images), len(preds), len(gts))
        if n <= 0:
            return

        for i in range(n):
            img = np.asarray(images[i])
            pred = np.asarray(preds[i])
            gt = np.asarray(gts[i])

            if img.ndim == 3:
                if img.shape[0] >= 3:
                    disp = np.moveaxis(img[:3], 0, -1)
                else:
                    disp = np.repeat(np.moveaxis(img[:1], 0, -1), 3, axis=-1)
            else:
                disp = np.stack([img, img, img], axis=-1)

            disp = disp.astype(np.float32)
            disp = (disp - disp.min()) / max(1e-6, float(disp.max() - disp.min()))

            fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
            axes[0].imshow(disp)
            axes[0].set_title("image")
            axes[1].imshow(pred, vmin=0, vmax=len(self.class_names) - 1, cmap="tab10")
            axes[1].set_title("prediction")
            axes[2].imshow(gt, vmin=0, vmax=len(self.class_names) - 1, cmap="tab10")
            axes[2].set_title("ground truth")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            self._save_fig(fig, f"results/{prefix}_sample_{i+1}.png")

    def plot_attention_map(self, attention: Optional[np.ndarray]) -> None:
        """plot averaged cross-attention map."""
        if attention is None:
            return

        attn = np.asarray(attention)
        if attn.ndim != 4:
            return

        # expected shape: (b, heads, q, k).
        attn_mean = attn.mean(axis=(0, 1))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(attn_mean, cmap="magma", aspect="auto")
        ax.set_title("cross-attention map")
        ax.set_xlabel("spectral tokens")
        ax.set_ylabel("spatial tokens")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        self._save_fig(fig, "features/attention_map.png")

    def _plot_hist_compare(self, ax, class_names: Sequence[str], a: np.ndarray, b: np.ndarray, title: str) -> None:
        """helper for split histogram comparisons."""
        if a.size == 0 or b.size == 0:
            return
        x = np.arange(len(class_names))
        width = 0.35
        a_pct = a / np.maximum(a.sum(), 1.0)
        b_pct = b / np.maximum(b.sum(), 1.0)

        ax.bar(x - width / 2, a_pct, width, label="before")
        ax.bar(x + width / 2, b_pct, width, label="after")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=30, ha="right")
        ax.set_ylim(0.0, min(1.0, max(a_pct.max(), b_pct.max()) * 1.2 + 1e-4))
        ax.set_ylabel("ratio")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

    def _apply_external_legend(
        self,
        fig: plt.Figure,
        ax,
        anchor: Tuple[float, float] = (1.02, 0.5),
        ncol: int = 1,
        fontsize: int = 9,
    ) -> bool:
        """place legend outside the axis to avoid covering plotted data."""
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return False

        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=anchor,
            frameon=False,
            ncol=ncol,
            fontsize=fontsize,
            borderaxespad=0.0,
        )
        return True

    def _save_fig(self, fig: plt.Figure, relative_path: str) -> None:
        """save figure to output subtree and close handle."""
        out_path = self.output_dir / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
