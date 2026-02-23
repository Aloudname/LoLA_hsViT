"""
analyze_distribution.py - Dataset class & patient distribution analysis.

Reusable diagnostic tool for understanding class imbalance and
per-patient sample bias. Generates publication-ready visualizations.

Usage:
    python analyze_distribution.py                  # full analysis
    python analyze_distribution.py --split          # also show train/test split
    python analyze_distribution.py -o analysis/     # custom output directory
Or in ablation.py:
    python ablation.py -a
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os, numpy as np, argparse, warnings

from datetime import datetime
from typing import Dict, List, Optional, Tuple

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

def analyze_dataset(dataset, output_dir: str = "output/analysis",
                    show_split: bool = False) -> None:
    """
    Run full distribution analysis on a loaded NpyHSDataset.

    Generates:
        1. class_distribution.png    — overall class bar chart
        2. patient_heatmap.png       — patient * class count heatmap
        3. patient_heatmap_pct.png   — patient * class percentage heatmap
        4. patient_stacked.png       — stacked bar per patient
        5. (optional) split_comparison.png — train vs test class proportions

    Also prints a text summary to console.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("[%H:%M:%S]")

    labels = dataset.patch_labels
    groups = dataset.patch_patient_groups
    class_names = list(dataset.targets)
    num_classes = dataset.num
    patient_names = getattr(dataset, "_patient_names", None)

    print(f"{ts} Dataset Distribution Analysis")
    print(f"  Total samples: {len(labels):,}")
    print(f"  Classes: {num_classes} ({', '.join(class_names)})")
    print(f"  Patients: {len(np.unique(groups))}")

    # overall class distribution
    dist = compute_class_distribution(labels, class_names, num_classes)
    print(f"\n  Imbalance ratio (max/min): {dist['imbalance_ratio']:.1f}x")
    for name, cnt, pct in zip(class_names, dist["counts"], dist["percentages"]):
        print(f"    {name:>8s}: {cnt:>10,}  ({pct:5.1f}%)")

    fig1, ax1 = plt.subplots(figsize=(9, 0.7 * num_classes + 1.5))
    plot_class_bar(dist, ax=ax1)
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=200)
    plt.close(fig1)

    # patient * class heatmaps (counts + percentages)
    matrix, p_labels = compute_patient_class_matrix(
        labels, groups, patient_names, num_classes)

    fig2, ax2 = plt.subplots(figsize=(max(7, num_classes + 2),
                                       max(5, 0.4 * len(p_labels) + 2)))
    plot_patient_heatmap(matrix, p_labels, class_names, ax=ax2,
                         title="Per-Patient Sample Counts")
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "patient_heatmap.png"), dpi=200)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(max(7, num_classes + 2),
                                       max(5, 0.4 * len(p_labels) + 2)))
    plot_patient_heatmap(matrix, p_labels, class_names, ax=ax3,
                         title="Per-Patient Class Proportions (%)",
                         normalize_rows=True)
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "patient_heatmap_pct.png"), dpi=200)
    plt.close(fig3)

    # stacked bar
    fig4, ax4 = plt.subplots(figsize=(11, max(5, 0.35 * len(p_labels) + 2)))
    plot_patient_stacked_bar(matrix, p_labels, class_names, ax=ax4)
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir, "patient_stacked.png"), dpi=200)
    plt.close(fig4)

    # train-test split comparison
    if show_split:
        print(f"\n  Performing train/test split for comparison...")
        train_loader, test_loader = dataset.create_data_loader(
            num_workers=0, batch_size=32)
        train_idx = train_loader.dataset.indices
        test_idx = test_loader.dataset.indices

        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        fig5, ax5 = plt.subplots(figsize=(9, 0.7 * num_classes + 1.5))
        plot_split_comparison(train_labels, test_labels, class_names, num_classes, ax=ax5)
        fig5.tight_layout()
        fig5.savefig(os.path.join(output_dir, "split_comparison.png"), dpi=200)
        plt.close(fig5)

        # per-patient: which patients in train vs test?
        train_patients = set(groups[train_idx])
        test_patients = set(groups[test_idx])
        print(f"  Train patients ({len(train_patients)}): "
              + ", ".join(patient_names.get(p, f"P{p}") for p in sorted(train_patients)))
        print(f"  Test patients  ({len(test_patients)}): "
              + ", ".join(patient_names.get(p, f"P{p}") for p in sorted(test_patients)))

        # Flag classes with extreme train/test skew
        tr_pct = np.bincount(train_labels, minlength=num_classes) / len(train_labels) * 100
        te_pct = np.bincount(test_labels, minlength=num_classes) / len(test_labels) * 100
        print(f"\n  Class proportion skew (train% - test%):")
        for i, name in enumerate(class_names):
            diff = tr_pct[i] - te_pct[i]
            flag = "  [Warning]" if abs(diff) > 5 else ""
            print(f"    {name:>8s}: train {tr_pct[i]:5.1f}%  test {te_pct[i]:5.1f}%  "
                  f"Δ={diff:+.1f}%{flag}")

    # summary patients that are missing certain classes
    print(f"\n  Patients missing classes (potential split risk):")
    for row_idx, p_name in enumerate(p_labels):
        missing = [class_names[c] for c in range(num_classes) if matrix[row_idx, c] == 0]
        if missing:
            print(f"    {p_name}: missing {', '.join(missing)}")

    print(f"\n{ts} Analysis saved to {os.path.abspath(output_dir)}/")


def main():
    parser = argparse.ArgumentParser(description="Dataset distribution analysis")
    parser.add_argument("--split", action="store_true",
                        help="Also analyze train/test split")
    parser.add_argument("-o", "--output", default="analysis",
                        help="Output directory for plots (default: analysis/)")
    args = parser.parse_args()

    from config import load_config
    from pipeline import NpyHSDataset

    config = load_config()
    print(f"Loading dataset from: {config.path.data}")
    dataset = NpyHSDataset(config=config)

    analyze_dataset(dataset, output_dir=args.output, show_split=args.split)

if __name__ == "__main__":
    main()
