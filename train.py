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


def main() -> bool:
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
    try:  
        results = trainer.train()
        for key, value in results.items():
                if isinstance(value, float):
                    print(f"  {key:20s}: {value:8.4f}")
                else:
                    print(f"  {key:20s}: {value}")
                    
        del trainer
    except Exception as e:
        print(f"Training during {trainer.model_name} failed: {e}")
        return False
    
    # trainer2
    trainer2 = hsTrainer(
            config=config,
            dataLoader=dataLoader,
            epochs=parser.parse_args().epoch,
            model=lambda: CommonViT(),
            model_name='CommonViT',
            debug_mode=parser.parse_args().debug
        )
    try:    
        results2 = trainer2.train()
        for key, value in results2.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:8.4f}")
            else:
                print(f"  {key:20s}: {value}")
                
        del trainer2
    except Exception as e:
        print(f"Training during {trainer2.model_name} failed: {e}")
        return False
    return True
            
if __name__ == "__main__":
    main()
