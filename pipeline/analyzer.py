import os, numpy as np, warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List, Optional, Tuple

from pipeline.dataset import NpyHSDataset, tprint
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")


def compute_class_distribution(
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
) -> Dict:
    """
    Compute per-class sample counts and percentages.

    Args:
        labels: 1-D array of integer class labels (0-based).
        class_names: Optional human-readable names for each class.
        num_classes: Expected number of classes (auto-detected if None).

    Returns:
        Dict with keys: counts, percentages, class_names, num_classes,
        imbalance_ratio, total_samples.
    """
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    percentages = counts / total * 100 if total > 0 else np.zeros_like(counts, dtype=float)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    nonzero = counts[counts > 0]
    imbalance = float(nonzero.max() / nonzero.min()) if len(nonzero) > 1 else 1.0

    return {
        "counts": counts,
        "percentages": percentages,
        "class_names": class_names,
        "num_classes": num_classes,
        "imbalance_ratio": imbalance,
        "total_samples": int(total),
    }


def compute_patient_class_matrix(
    labels: np.ndarray,
    patient_groups: np.ndarray,
    patient_names: Optional[Dict[int, str]] = None,
    num_classes: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a (num_patients * num_classes) count matrix.

    Args:
        labels: 1-D array of integer class labels (0-based).
        patient_groups: 1-D array mapping each sample to a patient index.
        patient_names: Optional {group_idx: name} mapping.
        num_classes: Expected number of classes.

    Returns:
        matrix: shape (num_patients, num_classes), dtype int.
        patient_labels: list of patient display names (sorted by total samples desc).
    """
    if num_classes is None:
        num_classes = int(labels.max()) + 1

    unique_pids = np.unique(patient_groups)
    n_patients = len(unique_pids)
    matrix = np.zeros((n_patients, num_classes), dtype=np.int64)

    pid_to_row = {pid: i for i, pid in enumerate(unique_pids)}
    for pid in unique_pids:
        mask = patient_groups == pid
        cls_counts = np.bincount(labels[mask], minlength=num_classes)
        matrix[pid_to_row[pid]] = cls_counts

    # Build display names
    if patient_names:
        display = [patient_names.get(pid, f"P{pid}") for pid in unique_pids]
    else:
        display = [f"Patient {pid}" for pid in unique_pids]

    # Sort by total samples (descending) for better readability
    totals = matrix.sum(axis=1)
    order = np.argsort(-totals)
    matrix = matrix[order]
    display = [display[i] for i in order]

    return matrix, display


def plot_class_bar(dist: Dict, ax: Optional[plt.Axes] = None,
                   title: str = "Class Distribution") -> plt.Axes:
    """Horizontal bar chart of per-class sample counts with percentage labels."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.6 * dist["num_classes"] + 1.5))

    names = dist["class_names"]
    counts = dist["counts"]
    pcts = dist["percentages"]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, counts, color=colors, edgecolor="grey", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Sample Count")
    ax.set_title(title, fontweight="bold")

    # Labels: count + percentage
    max_count = counts.max()
    for bar, cnt, pct in zip(bars, counts, pcts):
        label = f" {cnt:,}  ({pct:.1f}%)"
        x_pos = bar.get_width()
        ax.text(x_pos + max_count * 0.01, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9)

    ax.set_xlim(0, max_count * 1.25)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    return ax

def plot_patient_heatmap(
    matrix: np.ndarray,
    patient_labels: List[str],
    class_names: List[str],
    ax: Optional[plt.Axes] = None,
    title: str = "Per-Patient Class Distribution",
    normalize_rows: bool = False,
) -> plt.Axes:
    """
    Heatmap showing sample counts (or proportions) per patient × class.

    Args:
        normalize_rows: If True, show row-wise percentages instead of counts.
    """
    show = matrix.copy().astype(float)
    fmt = ".0f"
    if normalize_rows:
        row_sums = show.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        show = show / row_sums * 100
        fmt = ".1f"

    n_patients, n_classes = show.shape
    if ax is None:
        fig_h = max(4, 0.4 * n_patients + 2)
        fig_w = max(6, 0.8 * n_classes + 3)
        _, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(show, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n_patients))
    ax.set_yticklabels(patient_labels, fontsize=8)
    ax.set_title(title, fontweight="bold", pad=12)

    # Annotate cells — skip if matrix too large
    if n_patients <= 60 and n_classes <= 15:
        for i in range(n_patients):
            for j in range(n_classes):
                val = show[i, j]
                color = "white" if val > show.max() * 0.6 else "black"
                if normalize_rows:
                    text = f"{val:.1f}" if val > 0 else ""
                else:
                    text = f"{int(val):,}" if val > 0 else ""
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=6 if n_patients > 30 else 7, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8,
                 label="%" if normalize_rows else "Count")
    return ax


def plot_patient_stacked_bar(
    matrix: np.ndarray,
    patient_labels: List[str],
    class_names: List[str],
    ax: Optional[plt.Axes] = None,
    title: str = "Per-Patient Sample Composition",
) -> plt.Axes:
    """Stacked horizontal bar: each patient's class composition."""
    n_patients, n_classes = matrix.shape
    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4, 0.35 * n_patients + 1.5)))

    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))
    y_pos = np.arange(n_patients)
    left = np.zeros(n_patients)

    for cls_idx in range(n_classes):
        vals = matrix[:, cls_idx].astype(float)
        ax.barh(y_pos, vals, left=left, color=colors[cls_idx],
                edgecolor="grey", linewidth=0.3, label=class_names[cls_idx])
        left += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(patient_labels, fontsize=7 if n_patients > 30 else 8)
    ax.invert_yaxis()
    ax.set_xlabel("Sample Count")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=min(4, n_classes))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    return ax


def plot_split_comparison(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    class_names: List[str],
    num_classes: int,
    ax: Optional[plt.Axes] = None,
    title: str = "Train / Test Class Distribution",
) -> plt.Axes:
    """Side-by-side bar chart comparing train vs test class proportions."""
    train_counts = np.bincount(train_labels, minlength=num_classes)
    test_counts = np.bincount(test_labels, minlength=num_classes)
    train_pct = train_counts / train_counts.sum() * 100
    test_pct = test_counts / test_counts.sum() * 100

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 0.6 * num_classes + 1.5))

    y = np.arange(num_classes)
    bar_h = 0.35
    ax.barh(y - bar_h / 2, train_pct, bar_h, label="Train", color="#4e79a7", edgecolor="grey", linewidth=0.5)
    ax.barh(y + bar_h / 2, test_pct, bar_h, label="Test", color="#e15759", edgecolor="grey", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Proportion (%)")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=10)

    # Annotate proportions
    for i in range(num_classes):
        ax.text(train_pct[i] + 0.3, i - bar_h / 2,
                f"{train_pct[i]:.1f}%", va="center", fontsize=8, color="#4e79a7")
        ax.text(test_pct[i] + 0.3, i + bar_h / 2,
                f"{test_pct[i]:.1f}%", va="center", fontsize=8, color="#e15759")

    return ax


def plot_multi_split_comparison(
    split_labels: List[Tuple[str, np.ndarray]],
    class_names: List[str],
    num_classes: int,
    ax: Optional[plt.Axes] = None,
    title: str = "Per-Class Proportions by Split",
) -> plt.Axes:
    """Compare proportions for multiple splits (train/val/test)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 0.7 * num_classes + 1.5))

    y = np.arange(num_classes)
    bar_h = 0.15
    offsets = np.linspace(-bar_h, bar_h, len(split_labels)) if len(split_labels) > 1 else [0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(split_labels)))

    for (name, labels), offset, color in zip(split_labels, offsets, colors):
        counts = np.bincount(labels, minlength=num_classes)
        pct = counts / counts.sum() * 100
        ax.barh(y + offset, pct, bar_h, label=name, color=color, edgecolor="grey", linewidth=0.5)
        for i, val in enumerate(pct):
            ax.text(val + 0.4, y[i] + offset, f"{val:.1f}%", va="center", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Proportion (%)")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    return ax

class DatasetAnalyzer:
    """Comprehensive dataset diagnostics and visualization for NpyHSDataset.

    Covers:
      1) overall class distribution
      2) per-patient label distribution
      3) raw-band class separability (Fisher score)
      4) PCA-band separability + original-band contribution to PCs
      5) split-wise class distribution (train/val/test)

    Results are saved to ``output/analysis`` by default.
    """

    def __init__(self, dataset: NpyHSDataset, output_dir: str = "output/analysis"):
        self.dataset = dataset
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _foreground_meta(self) -> Tuple[List[str], int]:
        """Return foreground-only class names and class count (exclude label 0/BG)."""
        class_names = list(self.dataset.targets)
        if class_names and str(class_names[0]).strip().upper() == "BG":
            class_names = class_names[1:]
        num_classes = len(class_names)
        return class_names, num_classes

    @staticmethod
    def _to_foreground_labels(labels: np.ndarray) -> np.ndarray:
        """Filter out background label 0 and remap foreground labels to 0-based."""
        labels = np.asarray(labels)
        fg = labels[labels > 0]
        return (fg - 1).astype(np.int64)

    @staticmethod
    def _to_foreground_labels_with_groups(
        labels: np.ndarray,
        groups: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter BG and keep aligned patient groups for foreground-only analysis."""
        labels = np.asarray(labels)
        groups = np.asarray(groups)
        mask = labels > 0
        return (labels[mask] - 1).astype(np.int64), groups[mask]

    def _split_indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        total_indices = np.arange(len(self.dataset.patch_indices))
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=350234)
        trainval_idx, test_idx = next(
            sgkf.split(total_indices, self.dataset.patch_labels, self.dataset.patch_patient_groups)
        )
        val_rate = self.dataset.config.split.val_rate
        gss = GroupShuffleSplit(n_splits=1, test_size=val_rate, random_state=350234)
        train_rel, val_rel = next(
            gss.split(
                np.arange(len(trainval_idx)),
                self.dataset.patch_labels[trainval_idx],
                groups=self.dataset.patch_patient_groups[trainval_idx],
            )
        )
        train_idx = trainval_idx[train_rel]
        val_idx = trainval_idx[val_rel]
        return train_idx, val_idx, test_idx

    @staticmethod
    def _fisher_score(class_sum: np.ndarray, class_sumsq: np.ndarray, class_count: np.ndarray) -> np.ndarray:
        """Compute Fisher score per feature.

        Args:
            class_sum: (K, D)
            class_sumsq: (K, D)
            class_count: (K,)
        """
        K, D = class_sum.shape
        total_count = class_count.sum() + 1e-8
        total_mean = class_sum.sum(axis=0) / total_count
        # within-class variance
        mean = class_sum / np.maximum(class_count[:, None], 1e-8)
        var = class_sumsq / np.maximum(class_count[:, None], 1e-8) - mean ** 2
        within = (var * class_count[:, None]).sum(axis=0) + 1e-8
        between = ((mean - total_mean) ** 2 * class_count[:, None]).sum(axis=0)
        return between / within

    def analyze_labels(self) -> Dict:
        labels, groups = self._to_foreground_labels_with_groups(
            self.dataset.patch_labels,
            self.dataset.patch_patient_groups,
        )
        class_names, num_classes = self._foreground_meta()

        dist = compute_class_distribution(labels, class_names, num_classes)

        matrix, patient_labels = compute_patient_class_matrix(
            labels, groups, getattr(self.dataset, "_patient_names", None), num_classes
        )

        # plots
        fig1, ax1 = plt.subplots(figsize=(9, 0.7 * num_classes + 1.5))
        plot_class_bar(dist, ax=ax1, title="Class Distribution (All)")
        fig1.tight_layout()
        fig1.savefig(os.path.join(self.output_dir, "class_distribution.png"), dpi=200)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(
            figsize=(max(7, num_classes + 2), max(5, 0.4 * len(patient_labels) + 2))
        )
        plot_patient_heatmap(matrix, patient_labels, class_names, ax=ax2, title="Per-Patient Counts")
        fig2.tight_layout()
        fig2.savefig(os.path.join(self.output_dir, "patient_heatmap.png"), dpi=200)
        plt.close(fig2)

        fig3, ax3 = plt.subplots(
            figsize=(max(7, num_classes + 2), max(5, 0.4 * len(patient_labels) + 2))
        )
        plot_patient_heatmap(
            matrix,
            patient_labels,
            class_names,
            ax=ax3,
            title="Per-Patient Class Proportions (%)",
            normalize_rows=True,
        )
        fig3.tight_layout()
        fig3.savefig(os.path.join(self.output_dir, "patient_heatmap_pct.png"), dpi=200)
        plt.close(fig3)

        return {"dist": dist, "patient_matrix": matrix, "patient_labels": patient_labels}

    def analyze_splits(self) -> Dict:
        train_idx, val_idx, test_idx = self._split_indices()
        class_names, num_classes = self._foreground_meta()

        rng = np.random.RandomState(350234)

        # raw labels per split
        train_raw = self._to_foreground_labels(self.dataset.patch_labels[train_idx])
        val_raw = self._to_foreground_labels(self.dataset.patch_labels[val_idx])
        test_raw = self._to_foreground_labels(self.dataset.patch_labels[test_idx])

        # simulate balanced re-sampling used by WeightedRandomSampler (weights=1/class_counts)
        raw_counts = np.bincount(train_raw, minlength=num_classes)
        if (raw_counts == 0).any():
            # fallback: keep raw if any class missing
            resampled_train = train_raw
            resampled_counts = raw_counts
        else:
            prob = np.ones(num_classes, dtype=np.float64) / num_classes
            resampled_counts = rng.multinomial(len(train_raw), prob)
            resampled_train = np.repeat(np.arange(num_classes), resampled_counts)

        split_labels = [
            ("Train", resampled_train),
            ("Val", val_raw),
            ("Test", test_raw),
        ]

        fig, ax = plt.subplots(figsize=(10, 0.7 * num_classes + 1.5))
        plot_multi_split_comparison(split_labels, class_names, num_classes, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "split_comparison.png"), dpi=200)
        plt.close(fig)

        stats = {}
        # include both raw and resampled stats for transparency
        stats["Train_raw"] = {
            "counts": raw_counts,
            "pct": raw_counts / raw_counts.sum() * 100,
        }
        stats["Train_resampled"] = {
            "counts": resampled_counts,
            "pct": resampled_counts / resampled_counts.sum() * 100,
        }
        for name, lbl in [("Val", val_raw), ("Test", test_raw)]:
            counts = np.bincount(lbl, minlength=num_classes)
            stats[name] = {
                "counts": counts,
                "pct": counts / counts.sum() * 100,
            }
        return stats

    def analyze_raw_bands(self, sample_cap: int = 3_000_000) -> Dict:
        """Compute per-band Fisher score on raw spectra via streaming.

        sample_cap limits total pixels aggregated to reduce memory/time.
        """
        pairs = self.dataset._pair_data_and_labels_()
        if not pairs:
            return {}

        remap = getattr(self.dataset, "_label_remap", None)
        if remap is None:
            raise ValueError("Label remap not found on dataset")

        first_data = np.load(pairs[0][0])
        n_bands = first_data.shape[2]
        num_classes = self.dataset.num

        class_sum = np.zeros((num_classes, n_bands), dtype=np.float64)
        class_sumsq = np.zeros_like(class_sum)
        class_count = np.zeros(num_classes, dtype=np.float64)

        rng = np.random.RandomState(123)
        total_used = 0
        for data_file, label_file in pairs:
            data = np.load(data_file).astype(np.float32)
            labels_raw = np.load(label_file).astype(np.int32)
            labels = remap[np.clip(labels_raw, 0, len(remap) - 1)]
            mask = labels > 0
            if not mask.any():
                continue
            lbl_flat = labels[mask] - 1  # 0-based classes
            specs = data.reshape(-1, n_bands)[mask.reshape(-1)]

            # subsample to respect sample_cap
            remain = max(sample_cap - total_used, 0)
            if remain <= 0:
                break
            if specs.shape[0] > remain:
                sel = rng.choice(specs.shape[0], remain, replace=False)
                specs = specs[sel]
                lbl_flat = lbl_flat[sel]
            total_used += specs.shape[0]

            for cls in np.unique(lbl_flat):
                cls_mask = lbl_flat == cls
                if not cls_mask.any():
                    continue
                vals = specs[cls_mask]
                class_sum[cls] += vals.sum(axis=0)
                class_sumsq[cls] += (vals ** 2).sum(axis=0)
                class_count[cls] += vals.shape[0]

        fisher = self._fisher_score(class_sum, class_sumsq, class_count)
        order = np.argsort(-fisher)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(np.arange(n_bands), fisher, color="#4e79a7", edgecolor="grey", linewidth=0.5)
        ax.set_title("Raw Band Fisher Score (higher = more discriminative)", fontweight="bold")
        ax.set_xlabel("Band Index")
        ax.set_ylabel("Fisher Score")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "raw_band_fisher.png"), dpi=200)
        plt.close(fig)

        topk = min(20, n_bands)
        top_idx = order[:topk]
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(np.arange(topk), fisher[top_idx], color="#59a14f", edgecolor="grey", linewidth=0.5)
        ax2.set_xticks(np.arange(topk))
        ax2.set_xticklabels(top_idx)
        ax2.set_title(f"Top-{topk} Raw Bands by Fisher Score", fontweight="bold")
        ax2.set_ylabel("Fisher Score")
        fig2.tight_layout()
        fig2.savefig(os.path.join(self.output_dir, "raw_band_fisher_top.png"), dpi=200)
        plt.close(fig2)

        return {
            "fisher": fisher,
            "class_sum": class_sum,
            "class_sumsq": class_sumsq,
            "class_count": class_count,
        }

    def analyze_pca(self, sample_patches: int = 50000) -> Dict:
        if self.dataset.pca_components is None or self.dataset.pca_explained_variance is None:
            return {}

        _, num_classes = self._foreground_meta()
        n_components, n_bands = self.dataset.pca_components.shape
        class_sum = np.zeros((num_classes, n_components), dtype=np.float64)
        class_sumsq = np.zeros_like(class_sum)
        class_count = np.zeros(num_classes, dtype=np.float64)

        margin = (self.dataset.patch_size - 1) // 2
        rng = np.random.RandomState(321)
        total_idx = len(self.dataset.patch_indices)
        sample_n = min(sample_patches, total_idx)
        sample_indices = rng.choice(total_idx, sample_n, replace=False)

        for idx in sample_indices:
            patch = self.dataset._get_patch_(idx)
            label = int(self.dataset.patch_labels[idx])
            if label <= 0:
                continue
            center = patch[margin, margin, :]
            cls = label - 1
            class_sum[cls] += center
            class_sumsq[cls] += center ** 2
            class_count[cls] += 1

        fisher_pc = self._fisher_score(class_sum, class_sumsq, class_count)

        # Explained variance plot
        fig, ax = plt.subplots(figsize=(8, 3.5))
        evr = self.dataset.pca_explained_variance
        ax.bar(np.arange(len(evr)), evr, color="#4e79a7", edgecolor="grey", linewidth=0.4)
        ax.set_title("PCA Explained Variance Ratio", fontweight="bold")
        ax.set_xlabel("Component")
        ax.set_ylabel("Variance Ratio")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "pca_explained_variance.png"), dpi=200)
        plt.close(fig)

        # PCA discriminative power
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(np.arange(n_components), fisher_pc, color="#f28e2c", edgecolor="grey", linewidth=0.5)
        ax2.set_title("PCA Band Fisher Score", fontweight="bold")
        ax2.set_xlabel("PCA Component")
        ax2.set_ylabel("Fisher Score")
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        fig2.tight_layout()
        fig2.savefig(os.path.join(self.output_dir, "pca_fisher.png"), dpi=200)
        plt.close(fig2)

        # Original-band contribution to PCs (absolute loadings heatmap for top PCs)
        top_pc = min(10, n_components)
        loadings = np.abs(self.dataset.pca_components[:top_pc])
        fig3, ax3 = plt.subplots(figsize=(10, 0.6 * top_pc + 2))
        im = ax3.imshow(loadings, aspect="auto", cmap="YlGnBu")
        ax3.set_yticks(np.arange(top_pc))
        ax3.set_yticklabels([f"PC{i}" for i in range(top_pc)])
        ax3.set_xticks(np.arange(n_bands))
        ax3.set_xticklabels(np.arange(n_bands))
        ax3.set_xlabel("Raw Band")
        ax3.set_title("Original Band Contribution to Top PCs (|loading|)", fontweight="bold")
        plt.colorbar(im, ax=ax3, shrink=0.8, label="|Loading|")
        fig3.tight_layout()
        fig3.savefig(os.path.join(self.output_dir, "pca_band_contribution.png"), dpi=200)
        plt.close(fig3)

        return {"fisher_pc": fisher_pc}

    def run_all(self) -> Dict:
        summary = {}
        summary["labels"] = self.analyze_labels()
        summary["splits"] = self.analyze_splits()
        summary["raw_bands"] = self.analyze_raw_bands()
        summary["pca"] = self.analyze_pca()
        return summary


def analyze_dataset(dataset: NpyHSDataset, show_split: bool = False,
                    loader_kwargs: Optional[Dict] = None) -> None:
    """
    Run full distribution analysis on a loaded NpyHSDataset.

    Generates:
        1. class_distribution.png    — overall class bar chart
        2. patient_heatmap.png       — patient * class count heatmap
        3. patient_heatmap_pct.png   — patient * class percentage heatmap
        4. patient_stacked.png       — stacked bar per patient
        5. (optional) split_comparison.png — train vs test class proportions
    """
    
    from config import load_config
    config = load_config()
    output_dir = config.path.output + "/analysis"
    
    tprint("Starting dataset analysis, output path:\n", output_dir)
    analyzer = DatasetAnalyzer(dataset, output_dir)
    summary = analyzer.run_all()
    # Optionally print summary stats to console
    dist = summary.get("labels", {}).get("dist", {})
    if dist:
        print(f"Class Distribution:")
        for name, cnt, pct in zip(dist["class_names"], dist["counts"], dist["percentages"]):
            print(f"  {name:>8s}: {cnt:>10,}  ({pct:5.1f}%)")
    if show_split:
        split_stats = summary.get("splits", {})
        if split_stats:
            print(f"\nSplit Distribution:")
            for split, stats in split_stats.items():
                print(f"  {split}:")
                for name, cnt, pct in zip(dist["class_names"], stats["counts"], stats["pct"]):
                    print(f"    {name:>8s}: {cnt:>10,}  ({pct:5.1f}%)")
    
    tprint("Analysis complete. Plots saved to:", output_dir)

__all__ = ["DatasetAnalyzer"]
