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


def main():

    parser = argparse.ArgumentParser(
        description='hsViT',
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    

    parser.add_argument('--model', '-m', type=str, default='lola',
                       choices=['unet', 'vit', 'lola', 'custom'],
                       help='model type from {unet, vit, lola, custom}')
    parser.add_argument('--epoch', '-e', type=int, default=50)
    parser.add_argument('--debug', '-d', type=bool, default=False)
    config = load_config()
    
    dataLoader = NpyHSDataset(config=config, transform=None)
    
    trainer = hsTrainer(
        config=config,
        dataLoader=dataLoader,
        epochs=parser.parse_args().epoch,
        model=lambda: LoLA_hsViT(),
        model_name='LoLA_hsViT',
        debug_mode=parser.parse_args().debug
    )
    
    results = trainer.train()
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:8.4f}")
        else:
            print(f"  {key:20s}: {value}")
    
    # trainer2
    trainer2 = hsTrainer(
        config=config,
        dataLoader=dataLoader,
        epochs=50,
        model=lambda: CommonViT(),
        model_name='CommonViT',
        debug_mode=parser.parse_args().debug
    )
    
    results2 = trainer2.train()
    for key, value in results2.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:8.4f}")
        else:
            print(f"  {key:20s}: {value}")

if __name__ == "__main__":
    main()
