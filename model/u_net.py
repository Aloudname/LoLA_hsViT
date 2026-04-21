from __future__ import annotations

# 2d unet baseline for segmentation comparison.
from munch import Munch
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """two-layer conv block.

    x (b, c_in, h, w) --(conv-bn-relu)x2--> y (b, c_out, h, w)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """compact 2d unet baseline.

    shape flow:
    x (b, c, h, w)
      -> encoder: [c->b, b->2b, 2b->4b]
      -> bottleneck: 8b
      -> decoder with skip concatenation
      -> logits (b, num_classes, h, w)
    """

    def __init__(self, config: Munch) -> None:
        super().__init__()
        in_channels = int(config.preprocess.get("output_dim"))
        num_classes = int(config.data.get("num_classes"))
        base_channels = int(config.model.get("base_channels", 32))

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """extract final decoder feature map before segmentation head."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return d1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """run unet segmentation forward.

        input:
            x (b, c, h, w)
        output:
            logits (b, num_classes, h, w)
        """
        if x.ndim != 4:
            raise ValueError(f"expected 4d input, got shape={tuple(x.shape)}")
        return self.head(self.forward_features(x))
