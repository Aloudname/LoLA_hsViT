"""
训练器使用示例

展示如何用 HyperspectralTrainer 训练任意模型
"""

import torch
import argparse
import torch.nn as nn
from config import load_config
from pipeline.trainer import HyperspectralTrainer
from pipeline.dataset import MatHyperspectralDataset


class SimpleHSICNN(nn.Module):
    """CNN"""
    
    def __init__(self, num_classes: int = 9, num_bands: int = 15):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_bands, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SimpleViT(nn.Module):
    """Vision Transformer"""
    
    def __init__(self, num_classes: int = 9, patch_size: int = 4, dim: int = 64):
        super().__init__()
        self.patch_embed = nn.Conv2d(15, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (31//patch_size)**2 + 1, dim))
        
        self.transformer = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=256, batch_first=True),
            nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=256, batch_first=True),
        )
        
        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, dim, h, w]
        x = x.flatten(2).transpose(1, 2)  # [B, n_patches, dim]
        
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        
        x = self.transformer(x)
        x = x[:, 0]  # cls token
        x = self.head(x)
        return x



def main():

    parser = argparse.ArgumentParser(
        description='hsViT',
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    

    parser.add_argument('--model', type=str, default='lola',
                       choices=['unet', 'vit', 'lola', 'custom'],
                       help='model type from {unet, vit, lola, custom}')
    parser.add_argument('--epoch', type=int, default=50)
    config = load_config()
    
    dataLoader = MatHyperspectralDataset(config=config, transform=None)
    
    trainer = HyperspectralTrainer(
        config=config,
        dataLoader=dataLoader,
        epochs=parser.parse_args().epoch,
        model_fn=lambda: SimpleHSICNN(num_classes=9, num_bands=15),
        model_name='SimpleHSICNN',
        output_dir='./output/hsi_cnn'
    )
    
    results = trainer.train(debug_mode=False)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:8.4f}")
        else:
            print(f"  {key:20s}: {value}")
    
    # trainer2
    # trainer2 = HyperspectralTrainer(
    #     config=config,
    #     dataLoader=dataLoader,
    #     epochs=50,
    #     model_fn=lambda: SimpleViT(num_classes=9),
    #     model_name='SimpleViT',
    #     output_dir='./output/hsi_vit'
    # )
    # results2 = trainer2.train()


if __name__ == "__main__":
    main()
