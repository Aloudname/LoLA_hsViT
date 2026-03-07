#!/usr/bin/env python3
"""
run.py — unified entry point for training, evaluation, and dataset analysis.

Usage examples:
    # Train LoLA_hsViT_mini for 10 epochs
    python run.py -t mini -m lola -e 10

    # Train LoLA_hsViT_mini and CommonViT_mini sequentially (10 epochs each)
    python run.py -t mini -m lola common -e 10

    # Train all tags of CommonViT
    python run.py -t full reduced tiny mini 2layer -m common -e 10

    # Train multiple models x multiple tags (cartesian product)
    python run.py -t mini tiny -m lola common -e 10

    # 5-fold cross-validation
    python run.py -t mini -m lola -e 30 -cv 5

    # Multi-seed evaluation
    python run.py -t mini -m lola -e 10 --seeds 350234,350235,350236

    # Analyze dataset only
    python run.py -a
"""

import argparse
import copy
import gc
import json
import numpy as np
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from pipeline import (NpyHSDataset, TrainPipeline, ModelFactory,
                       TrainResult, CVResult, tprint, analyze_dataset)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified entry point for LoLA_hsViT / CommonViT training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument('--model', '-m', nargs='+', type=str, default=['lola'],
                        help='Model family: lola, common, unet. '
                             'Multiple values run sequentially.')
    parser.add_argument('--tag', '-t', nargs='+', type=str, default=['full'],
                        help='Architecture variant tag: full, reduced, tiny, mini, 2layer. '
                             'Multiple values expand as cartesian product with --model.')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Training epochs (default: 50)')
    parser.add_argument('--cv', '-cv', type=int, default=0,
                        help='Cross-validation folds (0 = single hold-out training)')
    parser.add_argument('--name', '-n', type=str, default=None,
                        help='Custom experiment name (default: auto-generated)')
    parser.add_argument('--debug', '-debug', action='store_true',
                        help='Debug mode: generate CAM per epoch')
    parser.add_argument('--gpus', type=int, default=None,
                        help='GPU count (default: auto-detect)')
    parser.add_argument('--analyze_dataset', '-a', action='store_true',
                        help='Analyze dataset distribution and exit')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated split seeds for multi-seed evaluation, '
                             'e.g. "350234,350235,350236"')

    return parser.parse_args()


def _parse_seed_list(seed_text: str):
    if seed_text is None:
        return []
    out = []
    for token in seed_text.split(','):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def _build_run_queue(models, tags):
    """Build a list of (registry_key, display_name) from model x tag cartesian product."""
    queue = []
    for model in models:
        for tag in tags:
            registry_key = ModelFactory.resolve_name(model, tag)
            display_name = registry_key
            queue.append((registry_key, display_name))
    return queue


def train_single_model(config, dataset, model_name: str,
                       epochs: int, cv_folds: int = 0,
                       exp_name: str = None, debug: bool = False,
                       num_gpus: int = None):
    """Train a single model and return the result.

    Args:
        config: Munch config.
        dataset: NpyHSDataset instance.
        model_name: registry key, e.g. 'lola_mini', 'common'.
        epochs: training epochs.
        cv_folds: CV folds (0 = hold-out).
        exp_name: experiment name.
        debug: debug mode flag.
        num_gpus: GPU count, None = auto.

    Returns:
        TrainResult or CVResult.
    """
    model_fn = ModelFactory.create(model_name, config)
    name = exp_name or f"{model_name}_exp"

    pipeline = TrainPipeline(
        config=config,
        dataset=dataset,
        model_fn=model_fn,
        name=name,
        num_gpus=num_gpus,
        debug_mode=debug
    )

    pipeline.summary()

    if cv_folds > 0:
        pipeline.fit_cv(n_folds=cv_folds, epochs=epochs)
    else:
        pipeline.fit(epochs=epochs)

    return pipeline.result_


def _release_resources():
    """Aggressive cleanup between sequential model runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    tprint("Loading configuration...")
    config = load_config()

    tprint("Loading dataset...")
    dataset = NpyHSDataset(config)
    tprint(f"Dataset loaded: {len(dataset)} patches")

    # Dataset analysis mode
    if args.analyze_dataset:
        tprint("Analyzing mode enabled.")
        analyze_dataset(dataset, show_split=True)
        tprint("Dataset analysis complete.")
        raise SystemExit(0)

    # Validate tags
    for tag in args.tag:
        if tag not in ModelFactory.VALID_TAGS:
            raise ValueError(f"Invalid tag '{tag}'. "
                             f"Valid tags: {ModelFactory.VALID_TAGS}")

    # Build run queue (cartesian product of models x tags)
    run_queue = _build_run_queue(args.model, args.tag)

    # Validate all model names before starting
    available = ModelFactory.available_models()
    for registry_key, _ in run_queue:
        if registry_key not in available:
            raise ValueError(f"Unknown model '{registry_key}'. "
                             f"Available: {', '.join(available)}")

    seed_list = _parse_seed_list(args.seeds)

    # -- Multi-seed evaluation --
    if seed_list:
        tprint(f"Multi-seed evaluation: {len(seed_list)} seeds x "
               f"{len(run_queue)} model(s)")

        for registry_key, display_name in run_queue:
            all_results = []
            tprint(f"\n{'='*60}")
            tprint(f"Model: {display_name}, seeds: {seed_list}")

            for seed in seed_list:
                run_cfg = copy.deepcopy(config)
                run_cfg.split.split_seed = int(seed)
                dataset.config = run_cfg

                seed_exp_name = f"{registry_key}_seed{seed}"
                tprint(f"\n[Seed {seed}] start")
                result = train_single_model(
                    config=run_cfg, dataset=dataset,
                    model_name=registry_key, epochs=args.epochs,
                    cv_folds=args.cv, exp_name=seed_exp_name,
                    debug=args.debug, num_gpus=args.gpus)

                if isinstance(result, CVResult):
                    all_results.append({
                        'seed': seed, 'mode': 'cv',
                        'accuracy_mean': float(result.accuracy_mean),
                        'accuracy_std': float(result.accuracy_std),
                        'kappa_mean': float(result.kappa_mean),
                        'kappa_std': float(result.kappa_std),
                        'miou_mean': float(result.miou_mean),
                        'output_dir': result.output_dir,
                    })
                else:
                    all_results.append({
                        'seed': seed, 'mode': 'holdout',
                        'best_accuracy': float(result.best_accuracy),
                        'final_accuracy': float(result.final_accuracy),
                        'final_kappa': float(result.final_kappa),
                        'final_miou': float(result.final_miou),
                        'output_dir': result.output_dir,
                    })
                _release_resources()

            # Print summary
            print(f"\nMulti-seed summary for {display_name}:")
            if all_results[0]['mode'] == 'cv':
                acc_means = np.array([r['accuracy_mean'] for r in all_results])
                print(f"  CV acc over seeds: {acc_means.mean():.2f}% "
                      f"+- {acc_means.std():.2f}%")
            else:
                final_accs = np.array([r['final_accuracy'] for r in all_results])
                print(f"  final_acc over seeds: {final_accs.mean():.2f}% "
                      f"+- {final_accs.std():.2f}%")

            summary_path = os.path.join(
                config.path.output, f"{registry_key}_multiseed_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({'seeds': seed_list, 'results': all_results}, f, indent=2)
            print(f"  saved: {summary_path}")
            _release_resources()
        return

    # -- Sequential model training --
    total = len(run_queue)
    all_results = {}

    tprint(f"\nRun queue: {total} model(s)")
    for i, (registry_key, display_name) in enumerate(run_queue):
        tprint(f"\n{'='*60}")
        tprint(f"[{i+1}/{total}] Training: {display_name}")
        tprint(f"{'='*60}")

        exp_name = args.name if (total == 1 and args.name) else f"{registry_key}_exp"

        try:
            result = train_single_model(
                config=config, dataset=dataset,
                model_name=registry_key, epochs=args.epochs,
                cv_folds=args.cv, exp_name=exp_name,
                debug=args.debug, num_gpus=args.gpus)

            all_results[registry_key] = result
            print(f"\n  {display_name} complete: {result}")

        except Exception as e:
            tprint(f"  ERROR training {display_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            _release_resources()

    # -- Summary --
    if total > 1:
        print(f"\n{'='*60}")
        print(f"  SUMMARY ({total} models)")
        print(f"{'='*60}")
        for name, r in all_results.items():
            if isinstance(r, CVResult):
                print(f"  {name:<20} Acc={r.accuracy_mean:.2f}% "
                      f"+- {r.accuracy_std:.2f}%  Kappa={r.kappa_mean:.2f}%")
            elif isinstance(r, TrainResult):
                print(f"  {name:<20} Acc={r.final_accuracy:.2f}%  "
                      f"Kappa={r.final_kappa:.2f}%  mIoU={r.final_miou:.2f}%")
        print()
    elif all_results:
        name, r = next(iter(all_results.items()))
        print(f"\nTraining complete!")
        print(r)
        out = r.output_dir if hasattr(r, 'output_dir') else 'N/A'
        print(f"Output: {out}")


if __name__ == '__main__':
    main()
