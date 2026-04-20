# rgb vit baseline model for comparison experiments.
from typing import Any, Mapping
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBViT(nn.Module):
    """rgb vit segmentation baseline.

    shape flow:
    x (b, 3, h, w)
      -> stem (b, d, h, w)
      -> patch embed (b, d, h/4, w/4)
      -> vit-style encoder tokens (b, h/4*w/4, d)
      -> decoder (b, num_classes, h, w)
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        in_channels = int(config.get("in_channels", 3))
        num_classes = int(config["num_classes"])
        embed_dim = int(config.get("embed_dim", 128))
        depth = int(config.get("depth", 4))
        num_heads = int(config.get("num_heads", 4))
        mlp_ratio = float(config.get("mlp_ratio", 4.0))
        decoder_dim = int(config.get("decoder_dim", 96))
        dropout = float(config.get("dropout", 0.1))
        freeze_backbone = bool(config.get("freeze_backbone", False))

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

        ff_dim = int(embed_dim * mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.backbone_norm = nn.LayerNorm(embed_dim)

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

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

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
        tokens = self.backbone_norm(self.backbone(tokens))
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
