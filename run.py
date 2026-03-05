#!/usr/bin/env python3
"""
run.py
entrypoint for whole pipeline.

Usage:
    # single model training
    python run.py --model LoLA_hsViT --epochs 50
    
    # 5 fold CV
    python run.py --model LoLA_hsViT --cv 5 --epochs 30
    
    # model comparison (all models with 5 fold CV)
    python run.py --compare --epochs 10
"""

import argparse
import copy
import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from pipeline import (NpyHSDataset, TrainPipeline, ModelFactory,
                       TrainResult, CVResult, tprint, analyze_dataset)


def parse_args():
    parser = argparse.ArgumentParser(
        description='entry point in terminal',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--model', '-m', type=str, default='lola',
                        choices=['lola', 'common', 'unet'],
                        help='model name')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='train epochs')
    parser.add_argument('--cv', '-cv', type=int, default=0,
                        help='cv folds, 0 for single training')
    parser.add_argument('--name', '-n', type=str, default=None,
                        help='experiment name (default: modelname_exp)')
    parser.add_argument('--compare', '-cp', action='store_true',
                        help='compare all models with CV, overrides --model and --name')
    parser.add_argument('--debug', '-debug', action='store_true',
                        help='debug mode')
    parser.add_argument('--gpus', type=int, default=None,
                        help='GPU num, auto for default.')
    parser.add_argument('--analyze_dataset', '-a', action='store_true',
                        help='Analyze dataset distribution and generate plots.')
    parser.add_argument('--seeds', type=str, default=None,
                        help='comma-separated split seeds, e.g. "350234,350235,350236"')
    
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


def train_single_model(config, dataset, model_name: str, 
                       epochs: int, cv_folds: int = 0,
                       exp_name: str = None, debug: bool = False,
                       num_gpus: int = None):
    """
    Args:
        config: Munch instance.
        dataset: torch.utils.data.Dataset / AbstractHSDataset instance.
        model_name: model name in ModelFactory.available_models()
        epochs: training epochs
        cv_folds: folds in cv (0=single training)
        exp_name: experiment name, default to "{model_name}_exp"
        debug: debug mode
        num_gpus: default None for auto-detect, 0 for CPU.
    
    Returns:
        TrainResult obj /  CVResult obj
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


def compare_models(config, dataset, epochs: int, cv_folds: int = 0,
                   debug: bool = False, num_gpus: int = None):
    """
    get all models in ModelFactory, train and compare them with CV.
    
    Args:
        config: Munch instance
        dataset: torch.utils.data.Dataset / AbstractHSDataset instance
        epochs: training epochs
        cv_folds: folds in cv (0=single training)
        debug: debug mode
        num_gpus: default None for auto-detect, 0 for CPU.
    
    Returns:
        Dict[str, Union[TrainResult, CVResult]]
    """
    results = {}
    
    for model_name in ModelFactory.available_models():
        tprint(f"Training: {model_name}")
        
        try:
            result = train_single_model(
                config=config,
                dataset=dataset,
                model_name=model_name,
                epochs=epochs,
                cv_folds=cv_folds,
                exp_name=f"{model_name}_compare",
                debug=debug,
                num_gpus=num_gpus
            )
            results[model_name] = result
        except Exception as e:
            tprint(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("model comparison results:")
    
    if cv_folds > 0:
        print(f"{'Model':<15} {'Acc Mean':<12} {'Acc Std':<10} "
              f"{'Kappa':<10} {'Time':<10}")
        for name, r in results.items():
            if isinstance(r, CVResult):
                print(f"{name:<15} {r.accuracy_mean:>10.2f}% "
                      f"{r.accuracy_std:>8.2f}% "
                      f"{r.kappa_mean:>8.2f}% "
                      f"{r.total_time:>8.0f}s")
    else:
        print(f"{'Model':<15} {'Accuracy':<12} {'Kappa':<10} "
              f"{'mIoU':<10} {'Time':<10}")
        for name, r in results.items():
            if isinstance(r, TrainResult):
                print(f"{name:<15} {r.final_accuracy:>10.2f}% "
                      f"{r.final_kappa:>8.2f}% "
                      f"{r.final_miou:>8.2f}% "
                      f"{r.training_time:>8.0f}s")
    print("\n")
    return results


def main():
    args = parse_args()
    
    tprint("Loading configuration...")
    config = load_config()
    
    tprint("Loading dataset...")
    dataset = NpyHSDataset(config)
    tprint(f"Dataset loaded: {len(dataset)} patches")
    
    if args.analyze_dataset:
        tprint("Analyzing mode enabled.")
        analyze_dataset(dataset, show_split=True)
        tprint("Dataset analysis complete.")
        raise SystemExit(0)
    
    seed_list = _parse_seed_list(args.seeds)

    if seed_list:
        if args.compare:
            raise ValueError("--seeds currently supports single-model mode only (no --compare)")

        all_results = []
        tprint(f"Running multi-seed evaluation with {len(seed_list)} seeds: {seed_list}")

        for seed in seed_list:
            run_cfg = copy.deepcopy(config)
            run_cfg.split.split_seed = int(seed)
            dataset.config = run_cfg

            seed_exp_name = args.name or f"{args.model}_seed{seed}"
            tprint(f"\n[Seed {seed}] start")
            result = train_single_model(
                config=run_cfg,
                dataset=dataset,
                model_name=args.model,
                epochs=args.epochs,
                cv_folds=args.cv,
                exp_name=seed_exp_name,
                debug=args.debug,
                num_gpus=args.gpus
            )

            if isinstance(result, CVResult):
                all_results.append({
                    'seed': seed,
                    'mode': 'cv',
                    'accuracy_mean': float(result.accuracy_mean),
                    'accuracy_std': float(result.accuracy_std),
                    'kappa_mean': float(result.kappa_mean),
                    'kappa_std': float(result.kappa_std),
                    'miou_mean': float(result.miou_mean),
                    'output_dir': result.output_dir,
                })
            else:
                all_results.append({
                    'seed': seed,
                    'mode': 'holdout',
                    'best_accuracy': float(result.best_accuracy),
                    'final_accuracy': float(result.final_accuracy),
                    'final_kappa': float(result.final_kappa),
                    'final_miou': float(result.final_miou),
                    'output_dir': result.output_dir,
                })

        print("\n")
        print("Multi-seed summary:")
        if all_results[0]['mode'] == 'cv':
            acc_means = np.array([r['accuracy_mean'] for r in all_results], dtype=np.float64)
            print(f"  CV accuracy_mean over seeds: {acc_means.mean():.2f}% ± {acc_means.std():.2f}%")
            for r in all_results:
                print(f"  seed={r['seed']}: acc={r['accuracy_mean']:.2f}% ± {r['accuracy_std']:.2f}%")
        else:
            final_accs = np.array([r['final_accuracy'] for r in all_results], dtype=np.float64)
            best_accs = np.array([r['best_accuracy'] for r in all_results], dtype=np.float64)
            print(f"  final_accuracy over seeds: {final_accs.mean():.2f}% ± {final_accs.std():.2f}%")
            print(f"  best(val)_accuracy over seeds: {best_accs.mean():.2f}% ± {best_accs.std():.2f}%")
            for r in all_results:
                print(f"  seed={r['seed']}: final={r['final_accuracy']:.2f}%, best_val={r['best_accuracy']:.2f}%")

        summary_path = os.path.join(config.path.output, f"{args.model}_multiseed_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({'seeds': seed_list, 'results': all_results}, f, indent=2)
        print(f"  saved: {summary_path}")
        return

    if args.compare:
        results = compare_models(
            config=config,
            dataset=dataset,
            epochs=args.epochs,
            cv_folds=args.cv,
            debug=args.debug,
            num_gpus=args.gpus
        )
    else:
        result = train_single_model(
            config=config,
            dataset=dataset,
            model_name=args.model,
            epochs=args.epochs,
            cv_folds=args.cv,
            exp_name=args.name,
            debug=args.debug,
            num_gpus=args.gpus
        )
        
        print("\n")
        print("Training complete!")
        print(result)
        print(f"Output path: {result.output_dir if hasattr(result, 'output_dir') else 'N/A'}")

if __name__ == '__main__':
    main()
