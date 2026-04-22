from __future__ import annotations

import itertools
# visualization methods for experiment analysis and paper figures.
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.manifold import TSNE

from pipeline.analyzer import MetricsBundle


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
        if has_legend := self._apply_external_legend(fig, ax, anchor=(1.02, 0.5)):
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
            for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
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

        # Keep ROC legend inside the plot at lower-right as requested.
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        fig.tight_layout()
        self._save_fig(fig, "metrics/roc_auc.png")

    def plot_distribution(self, stats: Mapping[str, Any]) -> None:
        """plot class distribution summary as grouped bars for train/eval/test."""
        class_names = list(stats.get("class_names", self.class_names))
        split_stats = stats.get("split", {})
        splits = [name for name in ["train", "eval", "test"] if name in split_stats]
        if not splits:
            return

        split_ratio_rows: List[np.ndarray] = []
        for split_name in splits:
            cur = split_stats[split_name]
            # Prefer kept-patch distribution for final split composition;
            # fallback to raw histogram if kept stats are absent.
            hist = np.asarray(cur.get("kept_patch_pixel_hist", cur.get("raw_pixel_hist", [])), dtype=np.float64)
            if hist.size == 0:
                hist = np.zeros(len(class_names), dtype=np.float64)
            if hist.size < len(class_names):
                hist = np.pad(hist, (0, len(class_names) - hist.size), mode="constant")
            elif hist.size > len(class_names):
                hist = hist[: len(class_names)]

            split_ratio_rows.append(hist / np.maximum(hist.sum(), 1.0))

        ratio_table = np.stack(split_ratio_rows, axis=0)
        x = np.arange(len(class_names), dtype=np.float64)
        width = 0.78 / max(1, len(splits))

        fig, ax = plt.subplots(figsize=(11, 4.8))
        color_map = {
            "train": "#4e79a7",
            "eval": "#f28e2b",
            "test": "#59a14f",
        }

        for idx, split_name in enumerate(splits):
            offset = (idx - (len(splits) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                ratio_table[idx],
                width=width,
                label=split_name,
                color=color_map.get(split_name, "#999999"),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=0)
        ax.set_ylabel("ratio")
        ax.set_xlabel("class")
        ax.set_ylim(0.0, min(1.0, max(0.05, float(ratio_table.max()) * 1.15)))
        ax.set_title("class distribution across train/eval/test")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        self._save_fig(fig, "data/all_splits_distribution_compare.png")

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
        if has_legend := self._apply_external_legend(
            fig, ax, anchor=(1.02, 0.5), ncol=1, fontsize=9
        ):
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
        if has_legend := self._apply_external_legend(
            fig, ax, anchor=(1.02, 0.5), fontsize=8
        ):
            fig.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
        else:
            fig.tight_layout()
        self._save_fig(fig, f"features/{title.lower().replace(' ', '_')}.png")

    def plot_pca_lda_comparison(
        self,
        pca_features: np.ndarray,
        lda_features: np.ndarray,
        labels: np.ndarray,
        explained_variance_ratio: Optional[np.ndarray] = None,
        pca_dim: Optional[int] = None,
        lda_dim: Optional[int] = None,
        title: str = "PCA-LDA feature comparison",
    ) -> None:
        """plot paper-style comparison between PCA and PCA-LDA projections."""
        if pca_features.size == 0 or lda_features.size == 0 or labels.size == 0:
            return

        n = min(pca_features.shape[0], lda_features.shape[0], labels.shape[0])
        if n < 10:
            return

        pca_features = np.asarray(pca_features[:n], dtype=np.float32)
        lda_features = np.asarray(lda_features[:n], dtype=np.float32)
        labels = np.asarray(labels[:n], dtype=np.int64)

        def _to_2d(points: np.ndarray) -> np.ndarray:
            if points.ndim != 2 or points.shape[0] == 0:
                return np.empty((0, 2), dtype=np.float32)
            if points.shape[1] >= 2:
                return points[:, :2].astype(np.float32)
            pad = np.zeros((points.shape[0], 2 - points.shape[1]), dtype=np.float32)
            return np.concatenate([points.astype(np.float32), pad], axis=1)

        def _centroids(points: np.ndarray) -> Dict[int, np.ndarray]:
            out: Dict[int, np.ndarray] = {}
            for cls_idx in range(len(self.class_names)):
                idx = labels == cls_idx
                if np.any(idx):
                    out[cls_idx] = points[idx].mean(axis=0)
            return out

        def _distance_matrix(centroids: Dict[int, np.ndarray]) -> np.ndarray:
            mat = np.full((len(self.class_names), len(self.class_names)), np.nan, dtype=np.float32)
            for i in range(len(self.class_names)):
                if i not in centroids:
                    continue
                for j in range(len(self.class_names)):
                    if j not in centroids:
                        continue
                    mat[i, j] = float(np.linalg.norm(centroids[i] - centroids[j]))
            return mat

        pca_xy = _to_2d(pca_features)
        lda_xy = _to_2d(lda_features)
        pca_centroids = _centroids(pca_xy)
        lda_centroids = _centroids(lda_xy)
        pca_dist = _distance_matrix(pca_centroids)
        lda_dist = _distance_matrix(lda_centroids)

        if explained_variance_ratio is not None:
            explained_variance_ratio = np.asarray(explained_variance_ratio, dtype=np.float64)
            explained_variance_ratio = explained_variance_ratio[np.isfinite(explained_variance_ratio)]
            explained_variance_ratio = explained_variance_ratio[explained_variance_ratio > 0]

        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(self.class_names))]
        legend_handles = [
            Patch(facecolor=colors[i], edgecolor="none", label=f"{i}: {name}")
            for i, name in enumerate(self.class_names)
        ]

        fig, axes = plt.subplots(2, 2, figsize=(16.2, 9.2), gridspec_kw={"height_ratios": [1.0, 1.0], "width_ratios": [1.0, 1.0]})
        flat_axes = axes.flatten()
        fig.patch.set_facecolor("white")
        fig.suptitle(title, fontsize=14, fontweight="semibold", y=1.02)

        ax = flat_axes[0]
        ax.set_facecolor("#fbfbfd")
        if explained_variance_ratio is not None and explained_variance_ratio.size > 0:
            cum = np.cumsum(explained_variance_ratio)
            xs = np.arange(1, cum.shape[0] + 1)
            ax.plot(xs, cum, color="#4e79a7", linewidth=2.2, marker="o", markersize=3.5)
            ax.fill_between(xs, cum, 0.0, color="#4e79a7", alpha=0.12)
            ax.axhline(0.95, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.85)
            if pca_dim is not None and pca_dim > 0:
                mark_x = min(int(pca_dim), xs[-1])
                mark_y = cum[mark_x - 1]
                ax.axvline(mark_x, color="#2ca02c", linestyle=":", linewidth=1.4, alpha=0.9)
                ax.scatter([mark_x], [mark_y], s=42, color="#2ca02c", zorder=3)
                ax.text(
                    0.04,
                    0.08,
                    f"PCA dim = {mark_x}\nexplained = {mark_y:.3f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    va="bottom",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd", alpha=0.95),
                )
            ax.set_xlim(1, xs[-1])
            ax.set_ylim(0.0, 1.02)
        else:
            ax.text(0.5, 0.5, "PCA variance data unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("PCA variance retention")
        ax.set_xlabel("principal component")
        ax.set_ylabel("cumulative variance")
        ax.grid(alpha=0.25)
        ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="semibold")

        def _scatter_panel(ax, points: np.ndarray, centroids: Dict[int, np.ndarray], panel_label: str, panel_title: str, dim_label: str) -> None:
            ax.set_facecolor("#fbfbfd")
            for cls_idx, class_name in enumerate(self.class_names):
                idx = labels == cls_idx
                if not np.any(idx):
                    continue
                ax.scatter(
                    points[idx, 0],
                    points[idx, 1],
                    s=11,
                    alpha=0.72,
                    color=colors[cls_idx],
                    edgecolors="white",
                    linewidths=0.25,
                )
                if cls_idx in centroids:
                    cx, cy = centroids[cls_idx]
                    ax.scatter(
                        [cx],
                        [cy],
                        s=120,
                        marker="X",
                        color=colors[cls_idx],
                        edgecolors="black",
                        linewidths=0.7,
                        zorder=4,
                    )
                    ax.annotate(
                        class_name,
                        (cx, cy),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                        fontweight="semibold",
                        color="#1f1f1f",
                    )

            ax.set_title(panel_title)
            ax.set_xlabel(f"{dim_label}-1")
            ax.set_ylabel(f"{dim_label}-2")
            ax.grid(alpha=0.25)
            ax.set_aspect("equal", adjustable="datalim")
            ax.text(0.02, 0.97, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="semibold")

        _scatter_panel(flat_axes[1], pca_xy, pca_centroids, "(b)", "PCA projection", "PC")
        _scatter_panel(flat_axes[2], lda_xy, lda_centroids, "(c)", "PCA + LDA projection", "LD")

        heat_ax = flat_axes[3]
        heat_ax.axis("off")
        heat_gs = heat_ax.get_subplotspec().subgridspec(1, 2, wspace=0.18)
        hm_axes = [fig.add_subplot(heat_gs[0, i]) for i in range(2)]

        hm_titles = ["PCA centroid distance", "PCA + LDA centroid distance"]
        hm_mats = [pca_dist, lda_dist]
        hm_texts = ["(d1)", "(d2)"]
        for idx, (hax, mat, htitle, htxt) in enumerate(zip(hm_axes, hm_mats, hm_titles, hm_texts)):
            hax.set_facecolor("#fbfbfd")
            valid = mat[np.isfinite(mat)]
            if valid.size == 0:
                hax.text(0.5, 0.5, "no centroid data", ha="center", va="center", transform=hax.transAxes)
                hax.axis("off")
                continue

            vmax = float(np.nanmax(valid))
            im = hax.imshow(mat, cmap="magma", vmin=0.0, vmax=max(vmax, 1e-6))
            hax.set_title(htitle)
            hax.set_xticks(np.arange(len(self.class_names)))
            hax.set_yticks(np.arange(len(self.class_names)))
            hax.set_xticklabels(self.class_names, rotation=45, ha="right")
            hax.set_yticklabels(self.class_names)
            hax.tick_params(axis="both", labelsize=8)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    value = mat[i, j]
                    if np.isfinite(value):
                        color = "white" if value > 0.55 * vmax else "black"
                        hax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color=color)
            hax.text(0.02, 0.97, htxt, transform=hax.transAxes, ha="left", va="top", fontsize=11, fontweight="semibold")
            cbar = fig.colorbar(im, ax=hax, shrink=0.84, pad=0.02)
            cbar.ax.tick_params(labelsize=7)

        heat_ax.text(0.5, 1.02, "class-centroid pairwise distances", ha="center", va="bottom", transform=heat_ax.transAxes, fontsize=10)

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(4, len(legend_handles)),
            frameon=False,
            fontsize=8.5,
            bbox_to_anchor=(0.5, 0.01),
            title="classes",
            title_fontsize=8.5,
        )
        fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
        self._save_fig(fig, f"features/{title.lower().replace(' ', '_')}.png")

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

        cmap = plt.get_cmap("tab10", len(self.class_names))
        legend_handles = [
            Patch(facecolor=cmap(cls_idx), edgecolor="none", label=f"{cls_idx}: {name}")
            for cls_idx, name in enumerate(self.class_names)
        ]

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
            axes[1].imshow(pred, vmin=0, vmax=len(self.class_names) - 1, cmap=cmap)
            axes[1].set_title("prediction")
            axes[2].imshow(gt, vmin=0, vmax=len(self.class_names) - 1, cmap=cmap)
            axes[2].set_title("ground truth")
            for ax in axes:
                ax.axis("off")

            fig.legend(
                handles=legend_handles,
                loc="lower center",
                ncol=min(4, len(legend_handles)),
                frameon=False,
                fontsize=8,
                title="labels",
                title_fontsize=8,
            )
            fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
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
        ax.legend(loc="upper right", frameon=False, fontsize=8)

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
