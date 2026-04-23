from __future__ import annotations

import itertools
# visualization methods for experiment analysis and paper figures.
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Patch
from sklearn.decomposition import PCA
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
        val_loss = history.get("eval_loss", history.get("val_loss", []))
        axes[0].plot(epochs, val_loss, label="eval_loss", color="#d62728")
        stage1_total = [comp.get("stage1_total", np.nan) for comp in history.get("train_loss_components", [])]
        if len(stage1_total) == len(epochs):
            axes[0].plot(epochs, stage1_total, label="stage1_total", color="#17becf", alpha=0.8)
        axes[0].set_title("loss curve")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].grid(alpha=0.3)
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

        axes[1].plot(epochs, history.get("train_dice", []), label="train_dice", color="#2ca02c")
        val_dice = history.get("eval_dice", history.get("val_dice", []))
        axes[1].plot(epochs, val_dice, label="eval_dice", color="#9467bd")
        axes[1].set_title("dice curve")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("dice")
        axes[1].grid(alpha=0.3)
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
        
        stage_spans = history.get("stage_spans", [])
        stage_colors = ["#d9edf7", "#dff0d8", "#fcf8e3", "#f5e6ff"]
        for idx, span in enumerate(stage_spans):
            start = float(span.get("start", 1))
            end = float(span.get("end", start))
            color = stage_colors[idx % len(stage_colors)]
            for ax in axes:
                ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.25, linewidth=0)
        fig.subplots_adjust(bottom=0.28, wspace=0.28)
        self._save_fig(fig, "curves/loss_dice_curves.png")

    def plot_sampling_stats(self, stats: Mapping[str, Any]) -> None:
        """plot sampling distribution comparison"""
        split = stats.get("split", {}).get("train", {})
        class_names = list(stats.get("class_names", self.class_names))
        raw_hist = np.asarray(split.get("raw_pixel_hist", []), dtype=np.float64)
        kept_hist = np.asarray(split.get("kept_patch_pixel_hist", []), dtype=np.float64)
        sampled_hist = np.asarray(split.get("sampled_patch_pixel_hist", []), dtype=np.float64)
        if raw_hist.size == 0 or kept_hist.size == 0:
            return

        n_cls = len(class_names)
        raw_hist = np.pad(raw_hist[:n_cls], (0, max(0, n_cls - raw_hist[:n_cls].size)))
        kept_hist = np.pad(kept_hist[:n_cls], (0, max(0, n_cls - kept_hist[:n_cls].size)))
        sampled_hist = np.pad(sampled_hist[:n_cls], (0, max(0, n_cls - sampled_hist[:n_cls].size)))

        x = np.arange(n_cls)
        w = 0.25
        fig, ax = plt.subplots(figsize=(11.2, 4.5))
        ax.bar(x - w, raw_hist / np.maximum(raw_hist.sum(), 1.0), width=w, label="raw", color="#4e79a7")
        ax.bar(x, kept_hist / np.maximum(kept_hist.sum(), 1.0), width=w, label="kept", color="#f28e2b")
        ax.bar(x + w, sampled_hist / np.maximum(sampled_hist.sum(), 1.0), width=w, label="sampled", color="#59a14f")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("ratio")
        ax.set_title("train sampling distribution")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        self._save_fig(fig, "data/sampling_distribution.png")

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
        title: str = "spectral reducer feature comparison",
        pca_label: str = "PCA",
        lda_label: str = "Reducer",
    ) -> None:
        """plot paper-style comparison between reference projection and reducer projection."""
        if pca_features.size == 0 or lda_features.size == 0 or labels.size == 0:
            return

        n = min(pca_features.shape[0], lda_features.shape[0], labels.shape[0])
        if n < 10:
            return

        pca_features = np.asarray(pca_features[:n], dtype=np.float32)
        lda_features = np.asarray(lda_features[:n], dtype=np.float32)
        labels = np.asarray(labels[:n], dtype=np.int64)
        # Exclude background label from spectral projection visualization.
        non_bg = labels > 0
        if np.count_nonzero(non_bg) < 10:
            return
        pca_features = pca_features[non_bg]
        lda_features = lda_features[non_bg]
        labels = labels[non_bg]
        active_classes = sorted(np.unique(labels).tolist())
        class_names = [self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}" for idx in active_classes]
        cls_to_local = {cls_idx: i for i, cls_idx in enumerate(active_classes)}

        def _select_best_2d(points: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
            if points.ndim != 2 or points.shape[0] == 0:
                return np.empty((0, 2), dtype=np.float32), (0, 1)
            dim = points.shape[1]
            if dim == 1:
                pad = np.zeros((points.shape[0], 1), dtype=np.float32)
                return np.concatenate([points.astype(np.float32), pad], axis=1), (0, 0)
            if dim == 2:
                return points.astype(np.float32), (0, 1)

            # Fisher score per dimension: maximize between-class / within-class.
            global_mean = points.mean(axis=0)
            scores = np.zeros(dim, dtype=np.float64)
            present = 0
            for cls_idx in range(len(self.class_names)):
                idx = labels == cls_idx
                if not np.any(idx):
                    continue
                present += 1
                cls_points = points[idx]
                cls_mean = cls_points.mean(axis=0)
                cls_var = cls_points.var(axis=0)
                n = float(cls_points.shape[0])
                scores += n * (cls_mean - global_mean) ** 2 / np.maximum(cls_var, 1e-8)
            if present == 0:
                # fallback to first two dims if labels are empty/invalid
                return points[:, :2].astype(np.float32), (0, 1)
            best = np.argsort(-scores)[:2]
            best = np.sort(best)
            return points[:, best].astype(np.float32), (int(best[0]), int(best[1]))

        def _centroids(points: np.ndarray) -> Dict[int, np.ndarray]:
            out: Dict[int, np.ndarray] = {}
            for cls_idx in active_classes:
                idx = labels == cls_idx
                if np.any(idx):
                    out[cls_idx] = points[idx].mean(axis=0)
            return out

        def _distance_matrix(centroids: Dict[int, np.ndarray]) -> np.ndarray:
            n_cls = len(active_classes)
            mat = np.full((n_cls, n_cls), np.nan, dtype=np.float32)
            for i in active_classes:
                if i not in centroids:
                    continue
                for j in active_classes:
                    if j not in centroids:
                        continue
                    mat[cls_to_local[i], cls_to_local[j]] = float(np.linalg.norm(centroids[i] - centroids[j]))
            return mat

        pca_xy, pca_dims = _select_best_2d(pca_features)
        lda_xy, lda_dims = _select_best_2d(lda_features)
        pca_centroids = _centroids(pca_xy)
        lda_centroids = _centroids(lda_xy)
        # Distances are computed in full feature space to avoid hiding separability
        # that may lie outside 2D display coordinates.
        pca_dist = _distance_matrix(_centroids(pca_features))
        lda_dist = _distance_matrix(_centroids(lda_features))

        if explained_variance_ratio is not None:
            explained_variance_ratio = np.asarray(explained_variance_ratio, dtype=np.float64)
            explained_variance_ratio = explained_variance_ratio[np.isfinite(explained_variance_ratio)]
            explained_variance_ratio = explained_variance_ratio[explained_variance_ratio > 0]

        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(active_classes))]
        legend_handles = [
            Patch(facecolor=colors[i], edgecolor="none", label=f"{active_classes[i]}: {name}")
            for i, name in enumerate(class_names)
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
                    f"{pca_label} dim = {mark_x}\nexplained = {mark_y:.3f}",
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
        ax.set_title(f"{pca_label} variance retention")
        ax.set_xlabel("principal component")
        ax.set_ylabel("cumulative variance")
        ax.grid(alpha=0.25)
        ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="semibold")

        def _scatter_panel(
            ax,
            points: np.ndarray,
            centroids: Dict[int, np.ndarray],
            panel_label: str,
            panel_title: str,
            dim_label: str,
            dims_used: Tuple[int, int],
        ) -> None:
            ax.set_facecolor("#fbfbfd")
            for local_idx, cls_idx in enumerate(active_classes):
                class_name = class_names[local_idx]
                idx = labels == cls_idx
                if not np.any(idx):
                    continue
                ax.scatter(
                    points[idx, 0],
                    points[idx, 1],
                    s=11,
                    alpha=0.72,
                    color=colors[local_idx],
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
                        color=colors[local_idx],
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
            ax.set_xlabel(f"{dim_label}[{dims_used[0]}]")
            ax.set_ylabel(f"{dim_label}[{dims_used[1]}]")
            ax.grid(alpha=0.25)
            ax.set_aspect("equal", adjustable="datalim")
            ax.text(0.02, 0.97, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="semibold")

        _scatter_panel(flat_axes[1], pca_xy, pca_centroids, "(b)", f"{pca_label} projection (best 2D)", pca_label[:2].upper(), pca_dims)
        _scatter_panel(flat_axes[2], lda_xy, lda_centroids, "(c)", f"{lda_label} projection (best 2D)", lda_label[:2].upper(), lda_dims)

        heat_ax = flat_axes[3]
        heat_ax.axis("off")
        heat_gs = heat_ax.get_subplotspec().subgridspec(1, 2, wspace=0.18)
        hm_axes = [fig.add_subplot(heat_gs[0, i]) for i in range(2)]

        hm_titles = [f"{pca_label} centroid distance", f"{lda_label} centroid distance"]
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
            hax.set_xticks(np.arange(len(class_names)))
            hax.set_yticks(np.arange(len(class_names)))
            hax.set_xticklabels(class_names, rotation=45, ha="right")
            hax.set_yticklabels(class_names)
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

    def _select_best_2d_by_fisher(self, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        if points.ndim != 2 or points.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float32)
        if points.shape[1] == 1:
            return np.concatenate([points, np.zeros((points.shape[0], 1), dtype=np.float32)], axis=1)
        if points.shape[1] == 2:
            return points

        global_mean = points.mean(axis=0)
        scores = np.zeros(points.shape[1], dtype=np.float64)
        for cls_idx in np.unique(labels):
            if cls_idx <= 0:
                continue
            m = labels == cls_idx
            if not np.any(m):
                continue
            cls = points[m]
            cls_mean = cls.mean(axis=0)
            cls_var = cls.var(axis=0)
            scores += cls.shape[0] * (cls_mean - global_mean) ** 2 / np.maximum(cls_var, 1e-8)
        best = np.argsort(-scores)[:2]
        best = np.sort(best)
        return points[:, best].astype(np.float32)

    def plot_patient_shift(
        self,
        raw_points_2d: np.ndarray,
        norm_points_2d: np.ndarray,
        labels: np.ndarray,
        patient_ids: Sequence[str],
        selected_patients: Sequence[str],
        title: str = "patient spectral domain shift",
    ) -> None:
        """Plot per-class cross-patient shift with 90% clouds, raw vs norm (3x2)."""
        raw_pts = np.asarray(raw_points_2d, dtype=np.float32)
        norm_pts = np.asarray(norm_points_2d, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        pids = np.asarray([str(v) for v in patient_ids], dtype=object)
        if raw_pts.ndim != 2 or raw_pts.shape[1] != 2 or raw_pts.shape[0] < 20:
            return
        if norm_pts.ndim != 2 or norm_pts.shape[1] != 2 or norm_pts.shape[0] != raw_pts.shape[0]:
            return
        if y.shape[0] != raw_pts.shape[0] or pids.shape[0] != raw_pts.shape[0]:
            return

        fg_labels = [idx for idx in range(1, len(self.class_names))]
        if not fg_labels:
            return

        fig, axes = plt.subplots(len(fg_labels), 2, figsize=(12.5, 4.3 * len(fg_labels)))
        if len(fg_labels) == 1:
            axes = np.expand_dims(axes, axis=0)

        cmap = plt.get_cmap("tab10")
        color_map = {pid: cmap(i % 10) for i, pid in enumerate(selected_patients)}

        # chi-square quantile for 2 DoF at 90%
        chi2_q90 = 4.605170186

        for row, cls_idx in enumerate(fg_labels):
            cls_name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else f"class_{cls_idx}"
            for col, pts in enumerate([raw_pts, norm_pts]):
                ax = axes[row, col]
                col_name = "raw" if col == 0 else "norm"
                ax.set_title(f"{cls_name} ({col_name})")
                ax.set_xlabel("proj-dim1")
                ax.set_ylabel("proj-dim2")
                ax.grid(alpha=0.25)

                has_data = False
                for pid in selected_patients:
                    m = (y == cls_idx) & (pids == pid)
                    if np.count_nonzero(m) < 8:
                        continue
                    has_data = True
                    cur = pts[m]
                    color = color_map[pid]
                    ax.scatter(cur[:, 0], cur[:, 1], s=6, alpha=0.10, color=color, edgecolors="none")

                    center = cur.mean(axis=0)
                    cov = np.cov(cur.T)
                    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
                        continue
                    evals, evecs = np.linalg.eigh(cov)
                    evals = np.clip(evals, 1e-8, None)
                    order = np.argsort(evals)[::-1]
                    evals = evals[order]
                    evecs = evecs[:, order]
                    angle = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))
                    width = 2.0 * np.sqrt(chi2_q90 * evals[0])
                    height = 2.0 * np.sqrt(chi2_q90 * evals[1])
                    cloud = Ellipse(
                        xy=(float(center[0]), float(center[1])),
                        width=float(width),
                        height=float(height),
                        angle=angle,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.18,
                        linewidth=1.1,
                    )
                    ax.add_patch(cloud)
                if not has_data:
                    ax.text(0.5, 0.5, "insufficient points", transform=ax.transAxes, ha="center", va="center")

        legend_handles = [
            Patch(facecolor=color_map[pid], edgecolor="none", label=str(pid))
            for pid in selected_patients
            if pid in color_map
        ]
        if legend_handles:
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=min(5, len(legend_handles)),
                frameon=False,
                fontsize=8.5,
                title="patient",
                title_fontsize=9,
            )

        fig.suptitle(title, fontsize=12, y=0.995)
        fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.97))
        self._save_fig(fig, "data/patient_shift.png")

    def plot_patient_distance_distribution(
        self,
        features_hd: np.ndarray,
        labels: np.ndarray,
        patient_ids: Sequence[str],
        selected_patients: Sequence[str],
        title: str = " ", # leave none otherwise conflicts with legend
    ) -> None:
        x = np.asarray(features_hd, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        p = np.asarray([str(v) for v in patient_ids], dtype=object)
        if x.ndim != 2 or x.shape[0] < 20 or y.shape[0] != x.shape[0] or p.shape[0] != x.shape[0]:
            return

        fg_labels = [idx for idx in range(1, len(self.class_names))]
        fig, axes = plt.subplots(1, len(fg_labels), figsize=(5.4 * len(fg_labels), 4.2))
        if len(fg_labels) == 1:
            axes = [axes]
        rng = np.random.default_rng(3407)

        def _sample_pair_dist(a: np.ndarray, b: np.ndarray, n_pairs: int) -> np.ndarray:
            if a.shape[0] == 0 or b.shape[0] == 0:
                return np.empty((0,), dtype=np.float32)
            ia = rng.integers(0, a.shape[0], size=n_pairs)
            ib = rng.integers(0, b.shape[0], size=n_pairs)
            return np.linalg.norm(a[ia] - b[ib], axis=1).astype(np.float32)

        for ax, cls_idx in zip(axes, fg_labels):
            intra_vals: List[np.ndarray] = []
            inter_vals: List[np.ndarray] = []
            for pid in selected_patients:
                m = (y == cls_idx) & (p == pid)
                cur = x[m]
                if cur.shape[0] < 5:
                    continue
                intra_vals.append(_sample_pair_dist(cur, cur, n_pairs=800))
            for i in range(len(selected_patients)):
                for j in range(i + 1, len(selected_patients)):
                    mi = (y == cls_idx) & (p == selected_patients[i])
                    mj = (y == cls_idx) & (p == selected_patients[j])
                    ai = x[mi]
                    bj = x[mj]
                    if ai.shape[0] < 5 or bj.shape[0] < 5:
                        continue
                    inter_vals.append(_sample_pair_dist(ai, bj, n_pairs=800))

            intra = np.concatenate(intra_vals, axis=0) if intra_vals else np.empty((0,), dtype=np.float32)
            inter = np.concatenate(inter_vals, axis=0) if inter_vals else np.empty((0,), dtype=np.float32)
            cls_name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else str(cls_idx)
            if intra.size > 0:
                ax.hist(intra, bins=40, density=True, alpha=0.40, color="#4e79a7", label="intra-patient")
            if inter.size > 0:
                ax.hist(inter, bins=40, density=True, alpha=0.40, color="#e15759", label="inter-patient")
            ax.set_title(f"{cls_name}")
            ax.set_xlabel("L2 distance (high-dim)")
            ax.set_ylabel("density")
            ax.grid(alpha=0.25)
            if intra.size == 0 and inter.size == 0:
                ax.text(0.5, 0.5, "insufficient points", transform=ax.transAxes, ha="center", va="center")
        fig.legend(
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
            fontsize=9,
        )
        fig.suptitle(title, fontsize=12, y=0.995)
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.92))
        self._save_fig(fig, "data/patient_distance_distribution.png")

    def plot_patient_divergence_heatmaps(
        self,
        features_hd: np.ndarray,
        labels: np.ndarray,
        patient_ids: Sequence[str],
        selected_patients: Sequence[str],
        title: str = "cross-patient divergence heatmaps",
    ) -> None:
        x = np.asarray(features_hd, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        p = np.asarray([str(v) for v in patient_ids], dtype=object)
        if x.ndim != 2 or x.shape[0] < 20:
            return
        fg_labels = [idx for idx in range(1, len(self.class_names))]
        n_pat = len(selected_patients)
        if n_pat < 2:
            return
        fig, axes = plt.subplots(len(fg_labels), 2, figsize=(10.8, 3.7 * len(fg_labels)))
        if len(fg_labels) == 1:
            axes = np.expand_dims(axes, axis=0)
        rng = np.random.default_rng(3407)

        def _rbf_mmd2(a: np.ndarray, b: np.ndarray) -> float:
            na = min(256, a.shape[0]); nb = min(256, b.shape[0])
            a = a[rng.choice(a.shape[0], size=na, replace=False)]
            b = b[rng.choice(b.shape[0], size=nb, replace=False)]
            z = np.concatenate([a, b], axis=0)
            if z.shape[0] < 4:
                return float("nan")
            d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
            sigma2 = float(np.median(d2[np.triu_indices_from(d2, k=1)]))
            sigma2 = max(sigma2, 1e-6)
            kaa = np.exp(-np.sum((a[:, None, :] - a[None, :, :]) ** 2, axis=-1) / (2.0 * sigma2))
            kbb = np.exp(-np.sum((b[:, None, :] - b[None, :, :]) ** 2, axis=-1) / (2.0 * sigma2))
            kab = np.exp(-np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1) / (2.0 * sigma2))
            return float(kaa.mean() + kbb.mean() - 2.0 * kab.mean())

        def _bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
            mu1 = a.mean(axis=0); mu2 = b.mean(axis=0)
            c1 = np.cov(a, rowvar=False); c2 = np.cov(b, rowvar=False)
            dim = c1.shape[0]
            reg = 1e-4 * np.eye(dim)
            c1 = c1 + reg; c2 = c2 + reg
            c = 0.5 * (c1 + c2)
            invc = np.linalg.pinv(c)
            diff = (mu1 - mu2).reshape(-1, 1)
            t1 = 0.125 * float(diff.T @ invc @ diff)
            det_c = max(np.linalg.det(c), 1e-12)
            det_c1 = max(np.linalg.det(c1), 1e-12)
            det_c2 = max(np.linalg.det(c2), 1e-12)
            t2 = 0.5 * np.log(det_c / np.sqrt(det_c1 * det_c2))
            return float(t1 + t2)

        for row, cls_idx in enumerate(fg_labels):
            mmd_mat = np.full((n_pat, n_pat), np.nan, dtype=np.float64)
            bha_mat = np.full((n_pat, n_pat), np.nan, dtype=np.float64)
            for i, pi in enumerate(selected_patients):
                ai = x[(y == cls_idx) & (p == pi)]
                if ai.shape[0] < 8:
                    continue
                for j, pj in enumerate(selected_patients):
                    bj = x[(y == cls_idx) & (p == pj)]
                    if bj.shape[0] < 8:
                        continue
                    if i == j:
                        mmd_mat[i, j] = 0.0
                        bha_mat[i, j] = 0.0
                    else:
                        mmd_mat[i, j] = _rbf_mmd2(ai, bj)
                        bha_mat[i, j] = _bhattacharyya(ai, bj)

            cls_name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else str(cls_idx)
            for col, (mat, name, cmap) in enumerate([(mmd_mat, "MMD", "viridis"), (bha_mat, "Bhattacharyya", "magma")]):
                ax = axes[row, col]
                valid = mat[np.isfinite(mat)]
                if valid.size == 0:
                    ax.text(0.5, 0.5, "insufficient points", transform=ax.transAxes, ha="center", va="center")
                    ax.axis("off")
                    continue
                vmax = float(np.nanmax(valid))
                im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=max(vmax, 1e-8))
                ax.set_title(f"{cls_name} - {name}")
                ax.set_xticks(np.arange(n_pat)); ax.set_yticks(np.arange(n_pat))
                ax.set_xticklabels(selected_patients, rotation=45, ha="right", fontsize=7)
                ax.set_yticklabels(selected_patients, fontsize=7)
                fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        fig.suptitle(title, fontsize=12, y=0.995)
        fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.98))
        self._save_fig(fig, "data/patient_divergence_heatmaps.png")

    def plot_patient_feature_shift(
        self,
        feature_shift_by_class: Mapping[str, Sequence[Mapping[str, Any]]],
        title: str = "patient feature-wise shift (top dims)",
    ) -> None:
        """Visualize top shifted feature dimensions across patients for each class."""
        if not feature_shift_by_class:
            return
        class_keys = [name for name in self.class_names[1:] if name in feature_shift_by_class]
        if not class_keys:
            class_keys = list(feature_shift_by_class.keys())
        if not class_keys:
            return

        fig, axes = plt.subplots(1, len(class_keys), figsize=(5.2 * len(class_keys), 4.2))
        if len(class_keys) == 1:
            axes = [axes]
        cmap = plt.get_cmap("tab20")
        for ax, cls_name in zip(axes, class_keys):
            rows = list(feature_shift_by_class.get(cls_name, []))
            if not rows:
                ax.text(0.5, 0.5, "insufficient points", transform=ax.transAxes, ha="center", va="center")
                ax.axis("off")
                continue
            idxs = [int(r.get("feature_idx", 0)) for r in rows]
            vals = [float(r.get("shift_strength", 0.0)) for r in rows]
            x = np.arange(len(idxs))
            colors = [cmap(i % 20) for i in range(len(idxs))]
            ax.bar(x, vals, color=colors, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([str(i) for i in idxs], rotation=45, ha="right", fontsize=8)
            ax.set_title(cls_name)
            ax.set_xlabel("feature index (top shifted)")
            ax.set_ylabel("std of patient means")
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(title, fontsize=12, y=0.995)
        fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.97))
        self._save_fig(fig, "data/patient_feature_shift_topdims.png")

    def plot_patient_wise_band_shift(
        self,
        band_stats_by_class: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
        patient_order: Sequence[str],
        title: str = "patient-wise band shift",
    ) -> None:
        """
        For each foreground class, plot per-patient (band, mean) scatter+line with +/-2sigma range.
        """
        if not band_stats_by_class:
            return
        class_keys = [name for name in self.class_names[1:] if name in band_stats_by_class]
        if not class_keys:
            class_keys = list(band_stats_by_class.keys())
        if not class_keys:
            return

        fig, axes = plt.subplots(1, len(class_keys), figsize=(6.0 * len(class_keys), 4.7))
        if len(class_keys) == 1:
            axes = [axes]
        cmap = plt.get_cmap("tab10")
        colors = {pid: cmap(i % 10) for i, pid in enumerate(patient_order)}

        for ax, cls_name in zip(axes, class_keys):
            cls_stats = band_stats_by_class.get(cls_name, {})
            if not cls_stats:
                ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes, ha="center", va="center")
                ax.axis("off")
                continue
            for pid in patient_order:
                rows = cls_stats.get(pid, None)
                if rows is None:
                    continue
                mean = np.asarray(rows.get("mean", []), dtype=np.float32)
                std = np.asarray(rows.get("std", []), dtype=np.float32)
                if mean.size == 0 or std.size != mean.size:
                    continue
                x = np.arange(mean.size, dtype=np.int32)
                c = colors[pid]
                # scatter + dashed line for mean reflectance
                ax.scatter(x, mean, s=9, color=c, alpha=0.75)
                ax.plot(x, mean, linestyle="--", linewidth=1.1, color=c, alpha=0.9)
                # 2 sigma envelope
                low = mean - 2.0 * std
                high = mean + 2.0 * std
                ax.fill_between(x, low, high, color=c, alpha=0.10)
            ax.set_title(cls_name)
            ax.set_xlabel("band index")
            ax.set_ylabel("reflectance mean")
            ax.grid(alpha=0.25)

        legend_handles = [
            Patch(facecolor=colors[pid], edgecolor="none", label=str(pid))
            for pid in patient_order
            if pid in colors
        ]
        if legend_handles:
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=min(8, len(legend_handles)),
                frameon=False,
                fontsize=8.5,
                title="patient",
                title_fontsize=9,
            )
        fig.suptitle(title, fontsize=12, y=0.995)
        fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.97))
        self._save_fig(fig, "data/patient_wise_band_shift.png")

    def plot_patient_clustering(
        self,
        embedding_2d: np.ndarray,
        patient_ids: Sequence[str],
        cluster_ids: Sequence[int],
        title: str = "patient clustering",
    ) -> None:
        """Plot patient-level clustering scatter."""
        z = np.asarray(embedding_2d, dtype=np.float32)
        if z.ndim != 2 or z.shape[1] != 2 or z.shape[0] == 0:
            return
        pids = [str(v) for v in patient_ids]
        cids = [int(v) for v in cluster_ids]
        if len(pids) != z.shape[0] or len(cids) != z.shape[0]:
            return

        fig, ax = plt.subplots(figsize=(7.6, 5.8))
        cmap = plt.get_cmap("tab10")
        unique_clusters = sorted(set(cids))
        for cid in unique_clusters:
            idx = np.array([i for i, v in enumerate(cids) if v == cid], dtype=np.int64)
            if idx.size == 0:
                continue
            color = cmap(cid % 10)
            ax.scatter(z[idx, 0], z[idx, 1], s=64, color=color, alpha=0.85, label=f"cluster {cid}", edgecolors="black", linewidths=0.4)
            for i in idx.tolist():
                ax.annotate(pids[i], (z[i, 0], z[i, 1]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        self._save_fig(fig, "data/patient_clustering.png")

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
