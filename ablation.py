"""
ablation.py - Automated ablation experiment for model structure reduction.
Usage:
    python ablation.py -e <epoch> -m <model> -t <tag> (-s / -a)
e.g.:
    python ablation.py -e 4 -m common -t mini

Progressively reduces model structure (dim, depths, mlp_ratio, LoRA rank)
and re-trains to find the parameter-performance sweet spot where models
no longer overfit.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")

import numpy as np
import matplotlib.pyplot as plt
import os, gc, json, time, torch, argparse
from datetime import datetime

from config import load_config
from model import LoLA_hsViT, CommonViT
from pipeline import hsTrainer, NpyHSDataset, tprint
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

os.setpgrp() # prevent Ctrl+C from killing the whole terminal session when running long ablation schedules

@dataclass
class AblationConfig:
    """Single ablation experiment configuration."""
    tag: str                          # Human-readable tag (e.g. "compact_r8")
    model_type: str                   # "LoLA_hsViT" or "CommonViT"
    dim: int = 96                     # Base feature dimension
    depths: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_heads: List[int] = field(default_factory=lambda: [4, 8, 16])
    window_size: List[int] = field(default_factory=lambda: [7, 7, 7])
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.2
    r: int = 16                       # LoRA rank (LoLA_hsViT only)
    lora_alpha: int = 32              # LoRA alpha (LoLA_hsViT only)

    @property
    def run_name(self) -> str:
        """Unique name for this ablation run."""
        return f"{self.model_type}_{self.tag}"

    @property
    def short_label(self) -> str:
        """Short label for plots."""
        d = "x".join(map(str, self.depths))
        label = f"d{self.dim}_D{d}_m{self.mlp_ratio:.0f}"
        if self.model_type == "LoLA_hsViT":
            label += f"_r{self.r}"
        return label

    def build_model_fn(self):
        """Return a callable that constructs the model."""
        kwargs = dict(
            in_channels=15,   # from config.yaml: pca_components
            num_classes=8,     # from config.yaml: clsf.num (8 tissue classes, BG excluded)
            dim=self.dim,
            depths=self.depths,
            num_heads=self.num_heads,
            window_size=self.window_size[:len(self.depths)],
            mlp_ratio=self.mlp_ratio,
            drop_path_rate=self.drop_path_rate,
            spatial_size=15,   # from config.yaml: patch_size
            r=self.r,
            lora_alpha=self.lora_alpha,
        )
        if self.model_type == "LoLA_hsViT":
            return lambda: LoLA_hsViT(**kwargs)
        else:
            return lambda: CommonViT(**kwargs)

    def count_params(self) -> Tuple[int, int]:
        """Dry-run: create model, count params, delete."""
        model = self.build_model_fn()()
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        del model
        gc.collect()
        return total, trainable


@dataclass
class AblationResult:
    """Stores results for one ablation run."""
    config_tag: str
    model_type: str
    short_label: str
    total_params: int = 0
    trainable_params: int = 0
    best_epoch: int = 0
    best_train_acc: float = 0.0
    best_eval_acc: float = 0.0
    final_eval_acc: float = 0.0
    final_kappa: float = 0.0
    overfit_gap: float = 0.0          # train_acc - eval_acc at best epoch
    training_time: float = 0.0
    balance_score: float = 0.0        # composite metric
    output_dir: str = None


def build_common_vit_configs() -> List[AblationConfig]:
    """Define ablation configs for CommonViT."""
    return [
        AblationConfig(tag="full_stack",  model_type="CommonViT",
                       dim=96, depths=[3,4,5], num_heads=[4,8,16], mlp_ratio=4.0),
        AblationConfig(tag="shallow_dim",   model_type="CommonViT",
                       dim=96, depths=[2,3,3], num_heads=[4,8,16], mlp_ratio=4.0),
        AblationConfig(tag="reduced",   model_type="CommonViT",
                       dim=64, depths=[2,3,3], num_heads=[4,8,16], mlp_ratio=4.0),
        AblationConfig(tag="tiny",      model_type="CommonViT",
                       dim=32, depths=[2,2,2], num_heads=[4,8,16], mlp_ratio=2.0),
        AblationConfig(tag="mini",      model_type="CommonViT",
                       dim=32, depths=[1,1,2], num_heads=[2,4,8],  mlp_ratio=2.0),
        AblationConfig(tag="two_level", model_type="CommonViT",
                       dim=48, depths=[2,3], num_heads=[4,8], window_size=[7,7],
                       mlp_ratio=3.0),
    ]


def build_lola_vit_configs() -> List[AblationConfig]:
    """Define ablation configs for LoLA_hsViT."""
    return [
        AblationConfig(tag="full_stack",  model_type="LoLA_hsViT",
                       dim=96, depths=[3,4,5], num_heads=[4,8,16], mlp_ratio=4.0,
                       r=16, lora_alpha=32),
        AblationConfig(tag="shallow_dim",   model_type="LoLA_hsViT",
                       dim=96, depths=[2,3,3], num_heads=[4,8,16], mlp_ratio=4.0,
                       r=16, lora_alpha=32),
        AblationConfig(tag="reduced",   model_type="LoLA_hsViT",
                       dim=64, depths=[2,3,3], num_heads=[4,8,16], mlp_ratio=4.0,
                       r=16, lora_alpha=32),
        AblationConfig(tag="tiny",     model_type="LoLA_hsViT",
                       dim=32, depths=[2,2,2], num_heads=[4,8,16], mlp_ratio=2.0,
                       r=8, lora_alpha=16),
        AblationConfig(tag="mini",     model_type="LoLA_hsViT",
                       dim=32, depths=[1,1,2], num_heads=[2,4,8], window_size=[7,7,7],
                       mlp_ratio=2.0, r=4, lora_alpha=8),
        AblationConfig(tag="two_level", model_type="LoLA_hsViT",
                            dim=48, depths=[2,3], num_heads=[4,8], window_size=[7,7],
                            mlp_ratio=3.0, r=4, lora_alpha=8),
    ]

def compute_balance_score(eval_acc: float, overfit_gap: float,
                          total_params: int,
                          gap_threshold: float = 8.0,
                          param_penalty_weight: float = 2.0) -> float:
    """
    Composite score balancing accuracy, overfitting, and model size.

    balance = eval_acc
              - penalty_for_overfitting
              - penalty_for_large_size

    Higher is better.
    """
    # Penalize overfitting: 0 if gap <= threshold, else linearly increasing
    overfit_penalty = max(0.0, overfit_gap - gap_threshold) * 1.5

    # Penalize large model: use log10(params) scaled
    # log10(1M) ≈ 6, log10(80M) ≈ 7.9, log10(1K) ≈ 3
    size_penalty = param_penalty_weight * np.log10(max(total_params, 1))

    score = eval_acc - overfit_penalty - size_penalty
    return score

class AblationRunner:
    """Orchestrates the full ablation experiment."""

    def __init__(self, base_config, dataLoader, epochs: int, num_gpus: int,
                 gap_threshold: float = 8.0, resume_idx: int = 0,
                 n_folds: int = 0):
        self.base_config = base_config
        self.dataLoader = dataLoader
        self.epochs = epochs
        self.num_gpus = num_gpus
        self.gap_threshold = gap_threshold
        self.resume_idx = resume_idx
        self.n_folds = n_folds

        self.summary_dir = self.base_config.path.output
        os.makedirs(self.summary_dir, exist_ok=True)

        self.results: List[AblationResult] = []
        self.results_file = os.path.join(self.summary_dir, "ablation_results.json")

        # Load previous results if resuming
        if resume_idx > 0 and os.path.exists(self.results_file):
            self._load_results()

    def dry_run(self, configs: List[AblationConfig]) -> None:
        """Print parameter counts for all configs without training."""
        tprint(f"  DRY RUN: Parameter Count Preview")
        tprint(f"{'#':>3} {'Model':<12} {'Tag':<16} {'dim':>4} {'depths':<12} "
              f"{'mlp':>4} {'r':>3} {'Total Params':>14} {'Trainable':>14}")

        for i, cfg in enumerate(configs):
            total, trainable = cfg.count_params()
            depths_str = str(cfg.depths)
            r_str = str(cfg.r) if cfg.model_type == "LoLA_hsViT" else "-"
            tprint(f"{i:3d} {cfg.model_type:<12} {cfg.tag:<16} {cfg.dim:4d} "
                  f"{depths_str:<12} {cfg.mlp_ratio:4.1f} {r_str:>3} "
                  f"{total:>14,} {trainable:>14,}")

        tprint(f"Total configs: {len(configs)}")
        tprint(f"Estimated training time: ~{len(configs) * self.epochs * 2:.0f} min "
              f"(rough, depends on data & hardware)\n")

    def run_single(self, idx: int, cfg: AblationConfig) -> Optional[AblationResult]:
        """Train one configuration and return the result.

        When ``self.n_folds > 1``, this uses K-Fold cross-validation
        via ``hsTrainer.cross_validate()`` instead of a single hold-out
        train/test split.
        """
        tprint(f"  ABLATION [{idx+1}] {cfg.model_type} / {cfg.tag}")
        tprint(f"  dim={cfg.dim}, depths={cfg.depths}, mlp_ratio={cfg.mlp_ratio}", end="")
        if cfg.model_type == "LoLA_hsViT":
            tprint(f", r={cfg.r}, alpha={cfg.lora_alpha}", end="")
        if self.n_folds > 1:
            tprint(f" [CV={self.n_folds}]", end="")
        tprint()  # finish the line

        try:
            # Param counts (lightweight dry-run, no GPU needed)
            total_params, trainable_params = cfg.count_params()

            if self.n_folds > 1:
                # ── Cross-Validation mode ──
                results = hsTrainer.cross_validate(
                    config=self.base_config,
                    dataLoader=self.dataLoader,
                    n_folds=self.n_folds,
                    epochs=self.epochs,
                    model=cfg.build_model_fn(),
                    model_name=cfg.run_name,
                    num_gpus=self.num_gpus,
                    debug_mode=False,
                )

                best_eval_acc = results['cv_accuracy_mean']
                best_train_acc = results.get('cv_train_acc_mean', best_eval_acc)
                overfit_gap = results.get('cv_overfit_gap_mean', 0.0)
                best_epoch = results.get('best_epoch', 0)
                output_dir = results.get('output_dir', '')
            else:
                # ── Single hold-out mode (original behaviour) ──
                trainer = hsTrainer(
                    config=self.base_config,
                    dataLoader=self.dataLoader,
                    epochs=self.epochs,
                    model=cfg.build_model_fn(),
                    model_name=cfg.run_name,
                    debug_mode=False,
                    num_gpus=self.num_gpus,
                )

                results = trainer.train()

                # Overwrite param counts from actual model (includes DP wrapper)
                total_params = sum(p.numel() for p in trainer.model.parameters())
                trainable_params = sum(
                    p.numel() for p in trainer.model.parameters()
                    if p.requires_grad)

                best_epoch_idx = trainer.best_epoch
                if best_epoch_idx < len(trainer.train_accs):
                    best_train_acc = trainer.train_accs[best_epoch_idx]
                else:
                    best_train_acc = (
                        trainer.train_accs[-1] if trainer.train_accs else 0.0)
                best_eval_acc = trainer.best_acc
                overfit_gap = max(0.0, best_train_acc - best_eval_acc)
                best_epoch = trainer.best_epoch + 1
                output_dir = trainer.output

            balance = compute_balance_score(
                best_eval_acc, overfit_gap, total_params, self.gap_threshold)

            result = AblationResult(
                config_tag=cfg.tag,
                model_type=cfg.model_type,
                short_label=cfg.short_label,
                total_params=total_params,
                trainable_params=trainable_params,
                best_epoch=best_epoch,
                best_train_acc=best_train_acc,
                best_eval_acc=best_eval_acc,
                final_eval_acc=results.get('final_accuracy', 0.0),
                final_kappa=results.get('final_kappa', 0.0),
                overfit_gap=overfit_gap,
                training_time=results.get('training_time', 0.0),
                balance_score=balance,
                output_dir=output_dir,
            )

            self.results.append(result)
            self._save_results()

            # Print summary
            cv_tag = f" (CV {self.n_folds}-fold mean)" if self.n_folds > 1 else ""
            tprint(f"\n Result{cv_tag}: Acc={best_eval_acc:.2f}%, "
                  f"Gap={overfit_gap:.1f}%, Params={total_params:,}, "
                  f"Balance={balance:.2f}")

        except Exception as e:
            tprint(f"\n FAILED: {e}")
            import traceback
            traceback.print_exc()
            result = None

        finally:
            # Aggressive memory cleanup
            if 'trainer' in locals():
                del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result

    def run_all(self, configs: List[AblationConfig]) -> None:
        """Execute the full ablation schedule."""
        total = len(configs)
        tprint("\n")
        tprint(f"  Starting Ablation: {total} experiments, {self.epochs} epochs each")
        tprint(f"  Gap threshold: {self.gap_threshold}%")
        tprint("\n")

        tic = time.perf_counter()

        # run in sequence.
        for i, cfg in enumerate(configs):
            if i < self.resume_idx:
                tprint(f"  [Skipping {i+1}/{total}] {cfg.run_name} (resume)")
                continue
            self.run_single(i, cfg)

        elapsed = time.perf_counter() - tic
        tprint(f"\n")
        tprint(f"  Ablation Complete: {len(self.results)} runs in {elapsed:.1f}s")
        tprint(f"\n")

    def run_tag(self, tag: str, all_configs: List[AblationConfig]) -> None:
        """
        Train only the two configs (one per model_type) that match the given tag.

        This is useful for quickly re-running or comparing a single architecture
        variant across LoLA_hsViT and CommonViT without running the full ablation.

        Args:
            tag: The config tag to match (e.g. 'full_stack', 'tiny').
            all_configs: The complete list of ablation configs to search.
        """
        matched = [c for c in all_configs if c.tag == tag]
        if not matched:
            available_tags = sorted(set(c.tag for c in all_configs))
            tprint(f"  ERROR: No config found with tag '{tag}'.")
            tprint(f"  Available tags: {available_tags}")
            return

        model_types_found = [c.model_type for c in matched]
        tprint(f"\n")
        tprint(f"  Tag Run: '{tag}' — {len(matched)} config(s) matched")
        for c in matched:
            tprint(f"    {c.model_type}/{c.tag}  dim={c.dim} depths={c.depths}")
        tprint(f"\n")

        tic = time.perf_counter()
        for i, cfg in enumerate(matched):
            self.run_single(i, cfg)
        elapsed = time.perf_counter() - tic

        tprint(f"\n")
        tprint(f"  Tag Run Complete: {len(matched)} run(s) in {elapsed:.1f}s")
        tprint(f"\n")

    def analyze(self) -> Dict[str, Optional[AblationResult]]:
        """
        Analyze results:
          1. Select best balanced model per model_type
          2. Generate comparison visualizations
          3. Save summary
          4. Clean up non-optimal model checkpoints

        Returns dict of {model_type: best AblationResult}.
        """
        if not self.results:
            tprint("No results to analyze.")
            return {}

        best_models = {}

        for model_type in ["CommonViT", "LoLA_hsViT"]:
            type_results = [r for r in self.results if r.model_type == model_type]
            if not type_results:
                continue

            # Sort by balance score (higher is better)
            type_results.sort(key=lambda r: r.balance_score, reverse=True)
            best = type_results[0]
            best_models[model_type] = best

            tprint(f"  Best {model_type}: [{best.config_tag}]")
            tprint(f"    Acc: {best.best_eval_acc:.2f}%, "
                  f"Gap: {best.overfit_gap:.1f}%, "
                  f"Params: {best.total_params:,}")
            tprint(f"    Balance Score: {best.balance_score:.2f}")

        # Generate visualizations
        self._plot_param_vs_accuracy()
        self._plot_overfit_analysis()
        self._plot_ablation_table()
        self._plot_pareto_frontier()
        self._plot_balance_scores()

        # Save summary JSON
        self._save_summary(best_models)

        # Cleanup: remove non-optimal model checkpoint files
        self._cleanup_models(best_models)

        return best_models

    def _plot_param_vs_accuracy(self) -> None:
        """Scatter: parameter count vs eval accuracy, colored by overfit gap."""
        fig, ax = plt.subplots(figsize=(12, 7))

        for model_type, marker, color_base in [
            ("CommonViT", "o", "Blues"),
            ("LoLA_hsViT", "s", "Oranges"),
        ]:
            subset = [r for r in self.results if r.model_type == model_type]
            if not subset:
                continue

            params = [r.total_params / 1e6 for r in subset]
            accs = [r.best_eval_acc for r in subset]
            gaps = [r.overfit_gap for r in subset]

            sc = ax.scatter(params, accs, c=gaps, cmap=color_base,
                           marker=marker, s=120, edgecolors='k',
                           linewidths=0.8, label=model_type,
                           vmin=0, vmax=max(gaps) + 5)

            for r, x, y in zip(subset, params, accs):
                ax.annotate(r.config_tag, (x, y), fontsize=7,
                           ha='center', va='bottom', textcoords="offset points",
                           xytext=(0, 6))

        ax.set_xlabel("Total Parameters (M)", fontsize=12)
        ax.set_ylabel("Best Eval Accuracy (%)", fontsize=12)
        ax.set_title("Ablation: Parameters vs Accuracy", fontsize=14)
        ax.legend(fontsize=10)
        plt.colorbar(sc, ax=ax, label="Overfit Gap (%)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(self.summary_dir, "param_vs_accuracy.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        tprint(f"  Saved: {path}")

    def _plot_overfit_analysis(self) -> None:
        """Bar chart: overfit gap per config, with threshold line."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax, model_type in zip(axes, ["CommonViT", "LoLA_hsViT"]):
            subset = [r for r in self.results if r.model_type == model_type]
            if not subset:
                ax.set_visible(False)
                continue

            subset.sort(key=lambda r: r.total_params, reverse=True)
            labels = [r.config_tag for r in subset]
            gaps = [r.overfit_gap for r in subset]
            colors = ['#e74c3c' if g > self.gap_threshold else '#2ecc71' for g in gaps]

            bars = ax.barh(labels, gaps, color=colors, edgecolor='k', linewidth=0.5)
            ax.axvline(x=self.gap_threshold, color='k', linestyle='--',
                      linewidth=1.5, label=f'Threshold ({self.gap_threshold}%)')
            ax.set_xlabel("Overfit Gap (Train Acc - Eval Acc) %")
            ax.set_title(f"{model_type} — Overfit Gap")
            ax.legend()
            ax.invert_yaxis()

        fig.tight_layout()
        path = os.path.join(self.summary_dir, "overfit_analysis.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        tprint(f"  Saved: {path}")

    def _plot_pareto_frontier(self) -> None:
        """Pareto frontier: parameter count vs eval accuracy."""
        fig, ax = plt.subplots(figsize=(12, 7))

        for model_type, marker, color in [
            ("CommonViT", "o", "#2196F3"),
            ("LoLA_hsViT", "s", "#FF5722"),
        ]:
            subset = [r for r in self.results if r.model_type == model_type]
            if not subset:
                continue

            # Sort by params ascending
            subset.sort(key=lambda r: r.total_params)
            params = [r.total_params / 1e6 for r in subset]
            accs = [r.best_eval_acc for r in subset]

            ax.scatter(params, accs, marker=marker, s=100, color=color,
                      edgecolors='k', linewidths=0.6, label=model_type, zorder=3)

            # Compute Pareto frontier
            pareto_params, pareto_accs = [], []
            max_acc = -1
            for p, a in zip(params, accs):
                if a > max_acc:
                    pareto_params.append(p)
                    pareto_accs.append(a)
                    max_acc = a

            if pareto_params:
                ax.plot(pareto_params, pareto_accs, '--', color=color, alpha=0.7,
                       linewidth=1.5, label=f"{model_type} Pareto")

            for r, x, y in zip(subset, params, accs):
                ax.annotate(r.config_tag, (x, y), fontsize=7,
                           ha='center', va='bottom', textcoords="offset points",
                           xytext=(0, 6))

        ax.set_xlabel("Total Parameters (M)", fontsize=12)
        ax.set_ylabel("Best Eval Accuracy (%)", fontsize=12)
        ax.set_title("Pareto Frontier: Params vs Accuracy", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(self.summary_dir, "pareto_frontier.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        tprint(f"  Saved: {path}")

    def _plot_balance_scores(self) -> None:
        """Bar chart of balance scores per config."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax, model_type in zip(axes, ["CommonViT", "LoLA_hsViT"]):
            subset = [r for r in self.results if r.model_type == model_type]
            if not subset:
                ax.set_visible(False)
                continue

            subset.sort(key=lambda r: r.balance_score, reverse=True)
            labels = [r.config_tag for r in subset]
            scores = [r.balance_score for r in subset]
            colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(scores)))

            # Highlight best
            edge_colors = ['gold' if i == 0 else 'k' for i in range(len(scores))]
            edge_widths = [3 if i == 0 else 0.5 for i in range(len(scores))]

            for j, (label, score) in enumerate(zip(labels, scores)):
                ax.barh(label, score, color=colors[j],
                       edgecolor=edge_colors[j], linewidth=edge_widths[j])

            ax.set_xlabel("Balance Score (higher = better)")
            ax.set_title(f"{model_type} — Balance Scores")
            ax.invert_yaxis()

        fig.tight_layout()
        path = os.path.join(self.summary_dir, "balance_scores.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        tprint(f"  Saved: {path}")

    def _plot_ablation_table(self) -> None:
        """Generate a styled summary table as an image."""
        if not self.results:
            return

        fig, ax = plt.subplots(figsize=(18, max(4, len(self.results) * 0.55 + 1.5)))
        ax.axis('off')

        headers = ["#", "Model", "Tag", "dim", "depths", "mlp", "r",
                   "Params(M)", "Train%", "Eval%", "Gap%", "Kappa%",
                   "Time(s)", "Balance"]

        table_data = []
        for i, r in enumerate(sorted(self.results,
                                      key=lambda x: (x.model_type, -x.balance_score))):
            row = [
                str(i + 1),
                r.model_type,
                r.config_tag,
                str(r.short_label.split("_")[0].replace("d", "")),
                r.short_label.split("_")[1].replace("D", "") if "_D" in r.short_label else "-",
                r.short_label.split("_")[2].replace("m", "") if len(r.short_label.split("_")) > 2 else "-",
                r.short_label.split("_r")[-1] if "_r" in r.short_label else "-",
                f"{r.total_params / 1e6:.2f}",
                f"{r.best_train_acc:.1f}",
                f"{r.best_eval_acc:.1f}",
                f"{r.overfit_gap:.1f}",
                f"{r.final_kappa:.1f}",
                f"{r.training_time:.0f}",
                f"{r.balance_score:.1f}",
            ]
            table_data.append(row)

        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)

        # Style header
        for j, header in enumerate(headers):
            table[0, j].set_facecolor('#34495e')
            table[0, j].set_text_props(color='white', fontweight='bold')

        # Highlight rows with best balance per model type
        best_per_type = {}
        for i, r in enumerate(sorted(self.results,
                                      key=lambda x: (x.model_type, -x.balance_score))):
            if r.model_type not in best_per_type:
                best_per_type[r.model_type] = i
                for j in range(len(headers)):
                    table[i + 1, j].set_facecolor('#d5f5e3')

        # Color overfit gap cells
        for i, r in enumerate(sorted(self.results,
                                      key=lambda x: (x.model_type, -x.balance_score))):
            gap_col = headers.index("Gap%")
            if r.overfit_gap > self.gap_threshold:
                table[i + 1, gap_col].set_facecolor('#f5b7b1')
            else:
                table[i + 1, gap_col].set_facecolor('#abebc6')

        fig.tight_layout()
        path = os.path.join(self.summary_dir, "ablation_table.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        tprint(f"  Saved: {path}")

    def _cleanup_models(self, best_models: Dict[str, AblationResult]) -> None:
        """
        Remove model checkpoint files for non-optimal runs
        but preserve all visualization outputs (plots, CAMs, etc.).
        """
        best_dirs = set()
        for r in best_models.values():
            if r:
                best_dirs.add(r.output_dir)

        removed = 0
        for r in self.results:
            if r.output_dir in best_dirs:
                continue
            model_dir = os.path.join(r.output_dir, "models")
            if os.path.exists(model_dir):
                for f in os.listdir(model_dir):
                    fpath = os.path.join(model_dir, f)
                    if os.path.isfile(fpath) and f.endswith('.pth'):
                        os.remove(fpath)
                        removed += 1

        tprint(f"\n  Cleanup: removed {removed} non-optimal model checkpoint(s).")
        tprint(f"  All visualization outputs are preserved.")

    def _save_results(self) -> None:
        """Save current results to JSON."""
        data = [asdict(r) for r in self.results]
        with open(self.results_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_results(self) -> None:
        """Load previous results from JSON."""
        try:
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            self.results = [AblationResult(**d) for d in data]
            tprint(f"  Loaded {len(self.results)} previous results from {self.results_file}")
        except Exception as e:
            tprint(f"  Warning: Could not load previous results: {e}")

    def _save_summary(self, best_models: Dict[str, AblationResult]) -> None:
        """Save human-readable summary."""
        path = os.path.join(self.summary_dir, "best_models.txt")
        with open(path, 'w') as f:
            f.write("\n")
            f.write("  ABLATION EXPERIMENT SUMMARY\n")
            f.write(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Epochs per run: {self.epochs}\n")
            f.write(f"  Gap threshold: {self.gap_threshold}%\n")
            f.write(f"  Total runs: {len(self.results)}\n")
            f.write("\n")

            for model_type, best in best_models.items():
                f.write(f"Best {model_type}:\n")
                f.write(f"  Config tag:      {best.config_tag}\n")
                f.write(f"  Structure:       {best.short_label}\n")
                f.write(f"  Total params:    {best.total_params:,}\n")
                f.write(f"  Trainable params:{best.trainable_params:,}\n")
                f.write(f"  Best eval acc:   {best.best_eval_acc:.2f}%\n")
                f.write(f"  Overfit gap:     {best.overfit_gap:.1f}%\n")
                f.write(f"  Final kappa:     {best.final_kappa:.2f}%\n")
                f.write(f"  Balance score:   {best.balance_score:.2f}\n")
                f.write(f"  Training time:   {best.training_time:.1f}s\n")
                f.write(f"  Output dir:      {best.output_dir}\n")
                f.write("\n")

            f.write("\nAll results (sorted by balance score):\n\n")
            sorted_results = sorted(self.results,
                                    key=lambda r: r.balance_score, reverse=True)
            for i, r in enumerate(sorted_results):
                f.write(f"  {i+1:2d}. [{r.model_type}] {r.config_tag:16s} | "
                        f"Acc={r.best_eval_acc:5.1f}% | "
                        f"Gap={r.overfit_gap:4.1f}% | "
                        f"Params={r.total_params/1e6:6.2f}M | "
                        f"Balance={r.balance_score:6.1f}\n")

        tprint(f"  Saved: {path}")


def _run_smoke_test(args) -> bool:
    """
    light-weight smoke test.

    Returns:
        True if all phases pass, False otherwise.
    """
    import copy, tempfile, shutil
    from torch.utils.data import DataLoader, Dataset, Subset
    from sklearn.model_selection import GroupKFold

    tprint("  Running quick test")

    errors = []
    tmpdir = tempfile.mkdtemp(prefix="smoke_test_")
    tprint(f"  Temp dir: {tmpdir}")

    try:
        # Config
        tprint("\n Config loading...")
        config = load_config()
        config.path.output = tmpdir
        config.split.batch_size = 4
        config.split.eval_batch_size = 4
        config.memory.num_workers = 0
        config.memory.pin_memory = False
        config.common.gradient_accumulation_steps = 1
        config.common.eval_interval = 1
        config.common.patience = 999           # No early stopping
        config.common.use_amp = False
        config.memory.use_amp = False
        tprint("   Config loaded and overridden")

        # Synthetic Dataset
        tprint("\n Creating synthetic dataset...")
        n_samples = 80
        n_channels = config.preprocess.pca_components   # 15
        patch_size = config.split.patch_size             # 15
        num_classes = config.clsf.num                    # 8

        class SyntheticHSDataset(Dataset):
            """Minimal synthetic dataset mimicking NpyHSDataset interface."""
            def __init__(self):
                # Per-patch label (for CV split & class-weight computation)
                self.patch_labels = np.random.randint(0, num_classes, size=n_samples)
                for c in range(num_classes):
                    self.patch_labels[c] = c
                np.random.shuffle(self.patch_labels)
                self.patch_patient_groups = np.array([i % 10 for i in range(n_samples)])
                self._num_patients = len(np.unique(self.patch_patient_groups))
                self.data = torch.randn(n_samples, n_channels, patch_size, patch_size)
                # (B, H, W)
                self.labels = torch.tensor(self.patch_labels, dtype=torch.long) \
                    .unsqueeze(-1).unsqueeze(-1).expand(-1, patch_size, patch_size).clone()
                self.num = num_classes

            def __len__(self):
                return n_samples

            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]

            def _get_patch(self, idx):
                """Return raw patch data for given index."""
                return self.data[idx]

        dataset = SyntheticHSDataset()
        assert len(dataset) == n_samples
        assert len(np.unique(dataset.patch_labels)) == num_classes
        tprint(f"    Synthetic dataset: {n_samples} samples, {num_classes} classes, "
               f"{dataset._num_patients} patients")

        # DataLoader
        tprint("\n  DataLoader construction...")
        train_idx = np.arange(0, 60)
        test_idx = np.arange(60, 80)

        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=4,
            shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(
            Subset(dataset, test_idx), batch_size=4,
            shuffle=False, num_workers=0, drop_last=False)

        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape == (4, n_channels, patch_size, patch_size), \
            f"Unexpected batch shape: {batch_x.shape}"
        tprint(f"    Train loader: {len(train_loader)} batches, "
               f"batch shape: {tuple(batch_x.shape)}")
        tprint(f"    Test loader:  {len(test_loader)} batches")

        # Model Instantiation
        tprint("\n  Model instantiation & forward/backward...")
        tiny_kwargs = dict(
            in_channels=n_channels,
            num_classes=num_classes,
            dim=16,
            depths=[1, 1],
            num_heads=[2, 4],
            window_size=[7, 7],
            mlp_ratio=2.0,
            drop_path_rate=0.1,
            spatial_size=patch_size,
            r=4,
            lora_alpha=8,
        )

        for model_cls, name in [(CommonViT, "CommonViT"), (LoLA_hsViT, "LoLA_hsViT")]:
            model = model_cls(**tiny_kwargs)
            n_params = sum(p.numel() for p in model.parameters())
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Forward pass
            out = model(batch_x)
            loss = torch.nn.functional.cross_entropy(out, batch_y)

            # Backward pass
            loss.backward()

            tprint(f"    {name}: {n_params:,} params ({n_trainable:,} trainable), "
                   f"output={list(out.shape)}, loss={loss.item():.4f}")
            del model, out, loss
            gc.collect()

        # single training, run 1 epoch
        tprint("\nSingle training run (1 epoch)...")
        model_fn = lambda: LoLA_hsViT(**tiny_kwargs)

        trainer = hsTrainer(
            config=config,
            dataLoader=dataset,
            epochs=1,
            model=model_fn,
            model_name="smoke_single",
            debug_mode=False,
            num_gpus=min(config.memory.parallel_gpu, 1),
            train_loader=train_loader,
            test_loader=test_loader,
        )

        results = trainer.train()
        assert 'final_accuracy' in results, "Missing final_accuracy in results"
        assert 'final_kappa' in results, "Missing final_kappa in results"
        tprint(f"    Training done: acc={results['final_accuracy']:.2f}%, "
               f"kappa={results['final_kappa']:.2f}%")

        model_dir = os.path.join(trainer.output, 'smoke_test')
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')] \
            if os.path.isdir(model_dir) else []
        tprint(f"    Model saved: {len(model_files)} checkpoint(s)")

        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Cross-Validation for 2 folds
        tprint("\n Cross-validation for 2 folds, 1 epoch each")

        gkf = GroupKFold(n_splits=2)
        fold_loaders = []
        for tr_i, te_i in gkf.split(
                np.arange(n_samples), dataset.patch_labels,
                dataset.patch_patient_groups):
            tr_ld = DataLoader(Subset(dataset, tr_i), batch_size=4,
                               shuffle=True, num_workers=0, drop_last=True)
            te_ld = DataLoader(Subset(dataset, te_i), batch_size=4,
                               shuffle=False, num_workers=0, drop_last=False)
            fold_loaders.append((tr_ld, te_ld))

        cv_config = copy.deepcopy(config)
        cv_config.path.output = tmpdir

        cv_results = hsTrainer.cross_validate(
            config=cv_config,
            dataLoader=dataset,
            n_folds=2,
            epochs=1,
            model=model_fn,
            model_name="smoke_cv",
            num_gpus=min(config.memory.parallel_gpu, 1),
            debug_mode=False,
            _fold_loaders_override=fold_loaders,
        )

        assert 'cv_accuracy_mean' in cv_results, "Missing cv_accuracy_mean"
        assert 'cv_accuracy_std' in cv_results, "Missing cv_accuracy_std"
        tprint(f"    CV done: acc={cv_results['cv_accuracy_mean']:.2f}% "
               f"± {cv_results['cv_accuracy_std']:.2f}%")

        del cv_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Ablation Runner (dry-run) 
        tprint("\n  Ablation runner (dry-run)...")
        smoke_configs = [
            AblationConfig(tag="smoke", model_type="CommonViT",
                           dim=16, depths=[1, 1], num_heads=[2, 4],
                           window_size=[7, 7], mlp_ratio=2.0),
            AblationConfig(tag="smoke", model_type="LoLA_hsViT",
                           dim=16, depths=[1, 1], num_heads=[2, 4],
                           window_size=[7, 7], mlp_ratio=2.0, r=4, lora_alpha=8),
        ]

        runner = AblationRunner(
            base_config=config,
            dataLoader=None,
            epochs=1,
            num_gpus=1,
            n_folds=0,
        )
        runner.dry_run(smoke_configs)
        tprint("    Dry-run complete")

        # Check Outputs 
        tprint("\n Checking outputs...")
        all_files = []
        for root, dirs, files in os.walk(tmpdir):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), tmpdir)
                all_files.append(rel)

        png_files = [f for f in all_files if f.endswith('.png')]
        json_files = [f for f in all_files if f.endswith('.json')]
        pth_files = [f for f in all_files if f.endswith('.pth')]

        tprint(f"  Output files: {len(all_files)} total")
        tprint(f"    .png plots:  {len(png_files)}")
        tprint(f"    .json data:  {len(json_files)}")
        tprint(f"    .pth models: {len(pth_files)}")

        if png_files:
            for f in sorted(png_files)[:12]:
                tprint(f"      {f}")
            if len(png_files) > 12:
                tprint(f"      ... and {len(png_files) - 12} more")

    except Exception as e:
        import traceback as tb
        errors.append(f"{type(e).__name__}: {e}")
        tprint(f"\n    ERROR: {e}")
        tb.print_exc()

    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
            tprint(f"\n  Temp dir cleaned: {tmpdir}")
        except Exception:
            pass

    if errors:
        tprint(f"    SMOKE TEST FAILED — {len(errors)} error(s):")
        for e in errors:
            tprint(f"    {e}")
    else:
        tprint("    SMOKE TEST PASSED — All phases OK")

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(
        description='Auto-ablation experiment for LoLA_hsViT & CommonViT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Epochs per ablation run (default: 10)')
    parser.add_argument('--model', '-m', type=str, default='both',
                        choices=['lola', 'common', 'both'],
                        help='Which model(s) to ablate (default: both)')
    parser.add_argument('--tag', '-t', type=str, default=None,
                        help='Train only the configs with this tag from both model types '
                             '(e.g. --tag full_stack trains LoLA_hsViT/full_stack + CommonViT/full_stack)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only show parameter counts; do not train')
    parser.add_argument('--resume', type=int, default=0,
                        help='Resume from config index (0-based)')
    parser.add_argument('--gap-threshold', type=float, default=8.0,
                        help='Max acceptable overfit gap %% (default: 8)')
    parser.add_argument('--smoke-test', '-s', action='store_true',
                        help='Quick end-to-end validation with synthetic data. '
                             'Runs 1 epoch, 2-fold CV, tiny model. Finishes in <60s.')
    parser.add_argument('--analyze_dataset', '-a', action='store_true',
                        help='Analyze dataset distribution and generate plots.')


    args = parser.parse_args()

    if args.analyze_dataset:
        tprint("Analyzing mode enabled.")
        from config import load_config
        from pipeline import NpyHSDataset
        from pipeline import analyze_dataset
        
        config = load_config()
        print(f"Loading dataset from: {config.path.data}")
        dataset = NpyHSDataset(config=config)
        analyze_dataset(dataset, show_split=True)
        tprint("Dataset analysis complete.")
        raise SystemExit(0)

    if args.smoke_test:
        tprint("Smoke test mode enabled.")
        ok = _run_smoke_test(args)
        tprint("Smoke test complete.")
        raise SystemExit(0 if ok else 1)

    all_configs: List[AblationConfig] = []
    all_configs.extend(build_common_vit_configs())
    all_configs.extend(build_lola_vit_configs())

    configs: List[AblationConfig] = []
    if args.model in ('common', 'both'):
        configs.extend(build_common_vit_configs())
    if args.model in ('lola', 'both'):
        configs.extend(build_lola_vit_configs())

    if not configs:
        tprint("No configs to run.")
        return

    config = load_config()

    # Resolve n_folds: CLI --cv overrides config.yaml if non-zero
    n_folds = config.common.cv_folds
    parallel = config.memory.parallel_gpu

    # Dry-run: skip data loading, just show parameter counts
    if args.dry_run:
        show_configs = [c for c in all_configs if c.tag == args.tag] if args.tag else configs
        runner = AblationRunner(
            base_config=config,
            dataLoader=None,
            epochs=args.epoch,
            num_gpus=parallel,
            gap_threshold=args.gap_threshold,
            resume_idx=args.resume,
            n_folds=n_folds,
        )
        runner.dry_run(show_configs)
        return

    # Full run: load data
    dataLoader = NpyHSDataset(config=config, transform=None)

    runner = AblationRunner(
        base_config=config,
        dataLoader=dataLoader,
        epochs=args.epoch,
        num_gpus=config.memory.parallel_gpu,
        gap_threshold=args.gap_threshold,
        resume_idx=args.resume,
        n_folds=n_folds,
    )

    # Tag mode: train only the matching configs
    if args.tag:
        runner.run_tag(args.tag, all_configs)
        if len(runner.results) >= 2:
            runner.analyze()
        else:
            tprint(f"  Skipping full analysis (need >=2 results, got {len(runner.results)})")
            for r in runner.results:
                tprint(f"  {r.model_type}/{r.config_tag}: "
                      f"Acc={r.best_eval_acc:.2f}% Gap={r.overfit_gap:.1f}% "
                      f"Params={r.total_params:,}")
        return

    # Full ablation: run all experiments
    runner.run_all(configs)

    # Analyze, visualize, and select best models
    best = runner.analyze()

    # Final summary
    print("\n")
    tprint("  ABLATION COMPLETE")
    for model_type, result in best.items():
        if result:
            tprint(f"  {model_type}: [{result.config_tag}] "
                  f"Acc={result.best_eval_acc:.2f}% "
                  f"Params={result.total_params:,} "
                  f"Gap={result.overfit_gap:.1f}%")

if __name__ == "__main__":
    main()
