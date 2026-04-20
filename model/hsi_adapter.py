# hsi adapter model with spectral-spatial token fusion.
from __future__ import annotations
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralEncoder(nn.Module):
    """learnable spectral projection.

    x (b, c_in, h, w) --(1x1 conv + bn + gelu)--> y (b, c_out, h, w)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CrossAttentionFusion(nn.Module):
    """cross attention from spatial tokens to spectral tokens.

    q: spatial tokens (b, n_spatial, d)
    k,v: spectral tokens (b, n_spectral, d)
    output: fused tokens (b, n_spatial, d)
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, spatial_tokens: torch.Tensor, spectral_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(spatial_tokens)
        kv = self.norm_kv(spectral_tokens)
        fused, attn_weights = self.attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attention = attn_weights.detach()
        return spatial_tokens + self.proj(fused)


class HSIAdapter(nn.Module):
    """hsi adapter segmentation model.

    overall shape flow:
    x (b, c, h, w)
      -> spectral encoder (b, s, h, w)
      -> spatial patch embed (b, d, h/4, w/4)
      -> tokenization:
         spectral tokens (b, h*w, s) -> linear -> (b, h*w, d)
         spatial tokens  (b, h/4*w/4, d)
      -> cross attention fusion (b, h/4*w/4, d)
      -> vit-style backbone (b, h/4*w/4, d)
      -> decoder + skip (b, num_classes, h, w)
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        in_channels = int(config["in_channels"])
        num_classes = int(config["num_classes"])
        spectral_dim = int(config.get("spectral_dim", 32))
        embed_dim = int(config.get("embed_dim", 128))
        depth = int(config.get("depth", 4))
        num_heads = int(config.get("num_heads", 4))
        mlp_ratio = float(config.get("mlp_ratio", 4.0))
        decoder_dim = int(config.get("decoder_dim", 96))
        dropout = float(config.get("dropout", 0.1))
        freeze_backbone = bool(config.get("freeze_backbone", True))

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.spectral_dim = spectral_dim
        self.embed_dim = embed_dim

        self.spectral_encoder = SpectralEncoder(in_channels, spectral_dim)
        self.spectral_token_proj = nn.Linear(spectral_dim, embed_dim)

        self.patch_embed = nn.Sequential(
            nn.Conv2d(spectral_dim, embed_dim, kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        self.fusion = CrossAttentionFusion(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

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

        self.skip_proj = nn.Sequential(
            nn.Conv2d(spectral_dim, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.fuse_head = nn.Sequential(
            nn.Conv2d(decoder_dim * 2, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
        )
        self.seg_head = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """extract fused feature map.

        input:
            x (b, c, h, w)
        output:
            feature map (b, d, h/4, w/4)
        """
        spectral_map = self.spectral_encoder(x)
        return self._forward_features_from_spectral(spectral_map)

    def _forward_features_from_spectral(self, spectral_map: torch.Tensor) -> torch.Tensor:
        """encode fused feature map from spectral map."""
        b = spectral_map.shape[0]
        spatial_map = self.patch_embed(spectral_map)
        hs, ws = spatial_map.shape[2], spatial_map.shape[3]

        spatial_tokens = spatial_map.flatten(2).transpose(1, 2)

        spectral_tokens = spectral_map.flatten(2).transpose(1, 2)
        spectral_tokens = self.spectral_token_proj(spectral_tokens)

        # align token length by pooled sampling when needed.
        if spectral_tokens.shape[1] != spatial_tokens.shape[1]:
            pooled = F.adaptive_avg_pool1d(
                spectral_tokens.transpose(1, 2),
                output_size=spatial_tokens.shape[1],
            )
            spectral_tokens = pooled.transpose(1, 2)

        fused_tokens = self.fusion(spatial_tokens, spectral_tokens)
        fused_tokens = self.backbone_norm(self.backbone(fused_tokens))

        return fused_tokens.transpose(1, 2).reshape(b, self.embed_dim, hs, ws)

    def get_attention_map(self) -> Optional[torch.Tensor]:
        """return last cross-attention map with shape (b, heads, q, k)."""
        return self.fusion.last_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """run dense segmentation forward.

        input:
            x (b, c, h, w)
        output:
            logits (b, num_classes, h, w)
        """
        if x.ndim != 4:
            raise ValueError(f"expected 4d input, got shape={tuple(x.shape)}")

        h, w = x.shape[2], x.shape[3]
        spectral_map = self.spectral_encoder(x)
        skip = self.skip_proj(spectral_map)

        feat_map = self._forward_features_from_spectral(spectral_map)
        dec = self.decoder(feat_map)
        dec = F.interpolate(dec, size=(h, w), mode="bilinear", align_corners=False)

        fused = torch.cat([dec, skip], dim=1)
        logits = self.seg_head(self.fuse_head(fused))
        return logits
