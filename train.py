"""
train.py - entrance script.
Run training with:
    python train.py --m lola -e 50
    python train.py --model
"""

import argparse
from config import load_config
from pipeline import hsTrainer
from pipeline import NpyHSDataset
from model import Unet, CommonViT, LoLA_hsViT


def _run_trainer(trainer):
    """Run a single trainer and return (results, trainer) or None on failure."""
    try:
        results = trainer.train()
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:8.4f}")
            else:
                print(f"  {key:20s}: {value}")
        return results, trainer
    except Exception as e:
        print(f"Training during {trainer.model_name} failed: {e}")
        return None


def main() -> bool:
    parser = argparse.ArgumentParser(
        description='Train through here',
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--debug', '-d', type=bool, default=False,
                        help='Whether to run in debug mode (default: False). '
                        'Debug mode uses fewer epochs and smaller batch size for quick testing.')
    parser.add_argument('--parallel', '-p', type=int, default=1,
                       help='number of GPUs to use for parallel training (default: 1). '
                            'Each number uses that many GPUs. Raises error if more than available.')
    
    args = parser.parse_args()
    
    # Validate parallel argument
    if args.parallel < 1:
        print(f"Error: --parallel must be >= 1, got {args.parallel}")
        return False
    
    config = load_config()
    dataLoader = NpyHSDataset(config=config, transform=None)
    
    model_specs = [
        ('LoLA_hsViT', lambda: LoLA_hsViT()),
        ('CommonViT',  lambda: CommonViT()),
        ('Unet',       lambda: Unet()),
    ]
    
    for model_name, model_fn in model_specs:
        trainer = hsTrainer(
            config=config,
            dataLoader=dataLoader,
            epochs=args.epoch,
            model=model_fn,
            model_name=model_name,
            debug_mode=args.debug,
            num_gpus=args.parallel,
        )
        outcome = _run_trainer(trainer)
        if outcome is None:
            return False
    return True
            
if __name__ == "__main__":
    main()
