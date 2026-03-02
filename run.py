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
    
    return parser.parse_args()


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
        result = pipeline.run_cv(n_folds=cv_folds, epochs=epochs)
    else:
        result = pipeline.run(epochs=epochs)
    
    return result


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
        config = load_config()
        print(f"Loading dataset from: {config.path.data}")
        dataset = NpyHSDataset(config=config)
        analyze_dataset(dataset, show_split=True)
        tprint("Dataset analysis complete.")
        raise SystemExit(0)
    
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
