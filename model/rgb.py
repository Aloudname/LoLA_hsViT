from __future__ import annotations

# rgb vit baseline model for comparison experiments.
from munch import Munch
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.builder import build_backbone


class RGBViT(nn.Module):
    """rgb vit segmentation baseline.

    shape flow:
    x (b, 3, h, w)
      -> stem (b, d, h, w)
      -> patch embed (b, d, h/4, w/4)
      -> vit-style encoder tokens (b, h/4*w/4, d)
      -> decoder (b, num_classes, h, w)
    """

    def __init__(self, config: Munch) -> None:
        super().__init__()
        in_channels = 3
        num_classes = int(config.data.get("num_classes"))
        embed_dim = int(config.model.get("embed_dim", 128))
        depth = int(config.model.get("depth", 4))
        num_heads = int(config.model.get("num_heads", 4))
        mlp_ratio = float(config.model.get("mlp_ratio", 4.0))
        decoder_dim = int(config.model.get("decoder_dim", 96))
        dropout = float(config.model.get("dropout", 0.1))
        freeze_backbone = bool(config.model.get("freeze_backbone", False))

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        self.patch_embed = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        backbone_cfg = {
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "dropout": dropout,
            "freeze_backbone": freeze_backbone,
            "use_pretrained": bool(config.get("use_pretrained", config.get("pretrained_backbone", False))),
            "backbone_name": str(config.get("backbone_name", "vit_small_patch16_224")),
            "pretrained_weights": bool(config.get("pretrained_weights", config.get("backbone_pretrained", True))),
            "pretrained_cache_dir": str(config.get("pretrained_cache_dir", "")),
        }
        self.backbone = build_backbone(backbone_cfg)

        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.seg_head = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

        # backbone freezing/unfreezing is handled by the backbone builder and trainer.

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """extract low-resolution token feature map.

        input:
            x (b, 3, h, w)
        output:
            feature map (b, d, h/4, w/4)
        """
        x = self.stem(x)
        x = self.patch_embed(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.backbone(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """run rgb segmentation forward.

        input:
            x (b, 3, h, w)
        output:
            logits (b, num_classes, h, w)
        """
        if x.ndim != 4:
            raise ValueError(f"expected 4d input, got shape={tuple(x.shape)}")

        h, w = x.shape[2], x.shape[3]
        feat_map = self.forward_features(x)
        dec = self.decoder(feat_map)
        dec = F.interpolate(dec, size=(h, w), mode="bilinear", align_corners=False)
        return self.seg_head(dec)
