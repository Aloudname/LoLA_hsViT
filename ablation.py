"""
ablation.py - Automated ablation experiment for model structure reduction.

Progressively reduces model structure (dim, depths, mlp_ratio, LoRA rank)
and re-trains to find the parameter-performance sweet spot where models
no longer overfit.
"""

import numpy as np
import matplotlib.pyplot as plt
import os, gc, json, time, torch, argparse

from config import load_config
from model import LoLA_hsViT, CommonViT
from pipeline import hsTrainer, NpyHSDataset
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

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
            num_classes=9,     # from config.yaml: clsf.num
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
                 gap_threshold: float = 8.0, resume_idx: int = 0):
        self.base_config = base_config
        self.dataLoader = dataLoader
        self.epochs = epochs
        self.num_gpus = num_gpus
        self.gap_threshold = gap_threshold
        self.resume_idx = resume_idx

        self.summary_dir = self.base_config.path.output
        os.makedirs(self.summary_dir, exist_ok=True)

        self.results: List[AblationResult] = []
        self.results_file = os.path.join(self.summary_dir, "ablation_results.json")

        # Load previous results if resuming
        if resume_idx > 0 and os.path.exists(self.results_file):
            self._load_results()

    def dry_run(self, configs: List[AblationConfig]) -> None:
        """Print parameter counts for all configs without training."""
        print(f"  DRY RUN: Parameter Count Preview")
        print(f"{'#':>3} {'Model':<12} {'Tag':<16} {'dim':>4} {'depths':<12} "
              f"{'mlp':>4} {'r':>3} {'Total Params':>14} {'Trainable':>14}")

        for i, cfg in enumerate(configs):
            total, trainable = cfg.count_params()
            depths_str = str(cfg.depths)
            r_str = str(cfg.r) if cfg.model_type == "LoLA_hsViT" else "-"
            print(f"{i:3d} {cfg.model_type:<12} {cfg.tag:<16} {cfg.dim:4d} "
                  f"{depths_str:<12} {cfg.mlp_ratio:4.1f} {r_str:>3} "
                  f"{total:>14,} {trainable:>14,}")

        print(f"Total configs: {len(configs)}")
        print(f"Estimated training time: ~{len(configs) * self.epochs * 2:.0f} min "
              f"(rough, depends on data & hardware)\n")

    def run_single(self, idx: int, cfg: AblationConfig) -> Optional[AblationResult]:
        """Train one configuration and return the result."""
        print(f"  ABLATION [{idx+1}] {cfg.model_type} / {cfg.tag}")
        print(f"  dim={cfg.dim}, depths={cfg.depths}, mlp_ratio={cfg.mlp_ratio}", end="")
        if cfg.model_type == "LoLA_hsViT":
            print(f", r={cfg.r}, alpha={cfg.lora_alpha}", end="")

        try:
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

            # Collect metrics
            total_params = sum(p.numel() for p in trainer.model.parameters())
            trainable_params = sum(
                p.numel() for p in trainer.model.parameters() if p.requires_grad)

            # Compute overfit gap at best epoch
            best_epoch_idx = trainer.best_epoch
            if best_epoch_idx < len(trainer.train_accs):
                best_train_acc = trainer.train_accs[best_epoch_idx]
            else:
                best_train_acc = trainer.train_accs[-1] if trainer.train_accs else 0.0
            best_eval_acc = trainer.best_acc
            overfit_gap = max(0.0, best_train_acc - best_eval_acc)

            balance = compute_balance_score(
                best_eval_acc, overfit_gap, total_params, self.gap_threshold)

            result = AblationResult(
                config_tag=cfg.tag,
                model_type=cfg.model_type,
                short_label=cfg.short_label,
                total_params=total_params,
                trainable_params=trainable_params,
                best_epoch=trainer.best_epoch + 1,
                best_train_acc=best_train_acc,
                best_eval_acc=best_eval_acc,
                final_eval_acc=results.get('final_accuracy', 0.0),
                final_kappa=results.get('final_kappa', 0.0),
                overfit_gap=overfit_gap,
                training_time=results.get('training_time', 0.0),
                balance_score=balance,
                output_dir=trainer.output,
            )

            self.results.append(result)
            self._save_results()

            # Print summary
            print(f"\n Result: Acc={best_eval_acc:.2f}%, "
                  f"Gap={overfit_gap:.1f}%, Params={total_params:,}, "
                  f"Balance={balance:.2f}")

        except Exception as e:
            print(f"\n FAILED: {e}")
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
        print("\n")
        print(f"  Starting Ablation: {total} experiments, {self.epochs} epochs each")
        print(f"  Gap threshold: {self.gap_threshold}%")
        print("\n")

        tic = time.perf_counter()

        # run in sequence.
        for i, cfg in enumerate(configs):
            if i < self.resume_idx:
                print(f"  [Skipping {i+1}/{total}] {cfg.run_name} (resume)")
                continue
            self.run_single(i, cfg)

        elapsed = time.perf_counter() - tic
        print(f"\n")
        print(f"  Ablation Complete: {len(self.results)} runs in {elapsed:.1f}s")
        print(f"\n")

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
            print("No results to analyze.")
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

            print(f"  Best {model_type}: [{best.config_tag}]")
            print(f"    Acc: {best.best_eval_acc:.2f}%, "
                  f"Gap: {best.overfit_gap:.1f}%, "
                  f"Params: {best.total_params:,}")
            print(f"    Balance Score: {best.balance_score:.2f}")

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
        print(f"  Saved: {path}")

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
        print(f"  Saved: {path}")

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
        print(f"  Saved: {path}")

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
        print(f"  Saved: {path}")

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
        print(f"  Saved: {path}")

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

        print(f"\n  Cleanup: removed {removed} non-optimal model checkpoint(s).")
        print(f"  All visualization outputs are preserved.")

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
            print(f"  Loaded {len(self.results)} previous results from {self.results_file}")
        except Exception as e:
            print(f"  Warning: Could not load previous results: {e}")

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

        print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Auto-ablation experiment for LoLA_hsViT & CommonViT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Epochs per ablation run (default: 10)')
    parser.add_argument('--parallel', '-p', type=int, default=1,
                        help='Number of GPUs (default: 1)')
    parser.add_argument('--model', '-m', type=str, default='both',
                        choices=['lola', 'common', 'both'],
                        help='Which model(s) to ablate (default: both)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only show parameter counts; do not train')
    parser.add_argument('--resume', type=int, default=0,
                        help='Resume from config index (0-based)')
    parser.add_argument('--gap-threshold', type=float, default=8.0,
                        help='Max acceptable overfit gap %% (default: 8)')

    args = parser.parse_args()

    # Build config list
    configs: List[AblationConfig] = []
    if args.model in ('common', 'both'):
        configs.extend(build_common_vit_configs())
    if args.model in ('lola', 'both'):
        configs.extend(build_lola_vit_configs())

    if not configs:
        print("No configs to run.")
        return

    # Load base config
    config = load_config()

    # Dry-run: skip data loading, just show parameter counts
    if args.dry_run:
        runner = AblationRunner(
            base_config=config,
            dataLoader=None,
            epochs=args.epoch,
            num_gpus=args.parallel,
            gap_threshold=args.gap_threshold,
            resume_idx=args.resume,
        )
        runner.dry_run(configs)
        return

    # Full run: load data
    dataLoader = NpyHSDataset(config=config, transform=None)

    runner = AblationRunner(
        base_config=config,
        dataLoader=dataLoader,
        epochs=args.epoch,
        num_gpus=args.parallel,
        gap_threshold=args.gap_threshold,
        resume_idx=args.resume,
    )

    # Run all experiments
    runner.run_all(configs)

    # Analyze, visualize, and select best models
    best = runner.analyze()

    # Final summary
    print("\n")
    print("  ABLATION COMPLETE")
    for model_type, result in best.items():
        if result:
            print(f"  {model_type}: [{result.config_tag}] "
                  f"Acc={result.best_eval_acc:.2f}% "
                  f"Params={result.total_params:,} "
                  f"Gap={result.overfit_gap:.1f}%")

if __name__ == "__main__":
    main()
