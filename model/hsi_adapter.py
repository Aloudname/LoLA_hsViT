from __future__ import annotations

from munch import Munch
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.builder import build_backbone


class SpectralTransformerEncoder(nn.Module):
    """
    enhanced spectral encoder with local conv and light-weight Transformer:
    - local conv: depthwise conv with kernel (1,3) to enhance local spectral features
    - light-weight Transformer: 2 layers, small hidden dim, few heads
        no downsampling, out.shape = in.shape
    """
    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 4):
        super().__init__()
        # initial 1x1 conv to project to out_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        # position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, out_channels, 1, 1) * 0.02)
        # 1d conv for local spectral feature enhancement
        self.conv1d = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(1, 3), padding=(0, 1),
                      groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )
        # light-weight spectral Transformer
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=out_channels,
                nhead=num_heads,
                dim_feedforward=out_channels * 2,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=2,
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.stem(x)
        x = x + self.pos_embed
        # 1d conv enhancement
        x = self.conv1d(x) + x
        B, C, H, W = x.shape
        # (B*H*W, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C).unsqueeze(1)  # (B*H*W, 1, C)
        return x


class SimpleSpectralEncoder(nn.Module):
    """
    lightweight spectral encoder:
        (1x1 conv + BN + GELU)*3
    no downsampling, out.shape = in.shape
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SpectralSE(nn.Module):
    """Squeeze-and-Excitation for spectral-wise attention."""
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MultiScaleSpectralTokens(nn.Module):
    """
    multiscale spectral tokens generator: global token + region tokens
     - global token: 1 token from global average pooling
     - region tokens: num_regions^2 tokens from spatially pooled regions
     - final output: (B, 1+R*R, embed_dim)
    """
    def __init__(self, spectral_dim: int, embed_dim: int, num_regions: int = 4):
        super().__init__()
        self.num_regions = num_regions
        # global pool -> 1 token
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # region pool -> num_regions^2 tokens
        self.region_pool = nn.AdaptiveAvgPool2d((num_regions, num_regions))
        self.proj = nn.Linear(spectral_dim, embed_dim)

    def forward(self, spectral_map: torch.Tensor):
        B = spectral_map.shape[0]
        # global token
        global_token = self.global_pool(spectral_map).flatten(2).transpose(1, 2)  # (B, 1, spectral_dim)
        # region token
        region_tokens = self.region_pool(spectral_map)  # (B, spectral_dim, R, R)
        region_tokens = region_tokens.flatten(2).transpose(1, 2)  # (B, R*R, spectral_dim)
        # merge + project
        all_tokens = torch.cat([global_token, region_tokens], dim=1)  # (B, 1+R*R, spectral_dim)
        all_tokens = self.proj(all_tokens)  # (B, 1+R*R, embed_dim)
        return all_tokens


class CrossAttentionFusion(nn.Module):
    """spatial tokens (Q) inquire multi scale spectral tokens (K/V)"""
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

    def forward(self,
                spatial_tokens: torch.Tensor,
                spectral_tokens: torch.Tensor) -> torch.Tensor:
        
        q = self.norm_q(spatial_tokens)                     # (B, N_spatial, D)
        kv = self.norm_kv(spectral_tokens)                  # (B, N_spectral, D)
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
    """Spectral-Spatial fusion adapter."""
    def __init__(self, config: Munch) -> None:
        super().__init__()
        in_channels = int(config.data.preprocess.get("output_dim")) \
            if config.data.preprocess.get("mode") != "none" \
            else int(config.data.get("hsi_bands"))

        num_classes = int(config.data.get("num_classes"))
        spectral_dim = int(config.model.get("spectral_dim", 32))
        embed_dim = int(config.model.get("embed_dim", 128))
        num_heads = int(config.model.get("num_heads", 4))
        decoder_dim = int(config.model.get("decoder_dim", 96))
        dropout = float(config.model.get("dropout", 0.1))

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.spectral_dim = spectral_dim
        self.embed_dim = embed_dim

        # spectral encoder + SE
        self.spectral_encoder = SimpleSpectralEncoder(in_channels, spectral_dim)
        self.spectral_se = SpectralSE(spectral_dim)

        # spectral tokens
        self.spectral_tokens_gen = MultiScaleSpectralTokens(spectral_dim,
                                                            embed_dim,
                                                            num_regions=4)

        # spatial embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(spectral_dim, embed_dim, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # fusion
        self.fusion = CrossAttentionFusion(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           dropout=dropout)

        # ViT bb
        self.backbone = build_backbone(config)

        # decoder
        self.skip_proj = nn.Sequential(
            nn.Conv2d(spectral_dim, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_dim,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
            
            nn.Conv2d(decoder_dim, decoder_dim,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
            
            nn.Dropout2d(dropout),
        )
        
        self.fuse_head = nn.Sequential(
            nn.Conv2d(decoder_dim * 2, decoder_dim,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
        )
        self.seg_head = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected 4d input, got shape={tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            if x.shape[-1] == self.in_channels:
                x = x.permute(0, 3, 1, 2).contiguous()
            else:
                raise ValueError(f"channel mismatch: got {x.shape[1]}, expected {self.in_channels}")
        spectral_map = self.spectral_encoder(x)
        spectral_map = self.spectral_se(spectral_map)
        return self._forward_features_from_spectral(spectral_map)

    def _forward_features_from_spectral(self, spectral_map: torch.Tensor) -> torch.Tensor:
        b = spectral_map.shape[0]
        spatial_map = self.patch_embed(spectral_map)
        hs, ws = spatial_map.shape[2], spatial_map.shape[3]

        # spatial token: (B, D, hs, ws) -> (B, hs*ws, D)
        spatial_tokens = spatial_map.flatten(2).transpose(1, 2)

        # spectral token: (B, N_spec, D)
        spectral_tokens = self.spectral_tokens_gen(spectral_map)

        # fuse + bb
        fused_tokens = self.fusion(spatial_tokens, spectral_tokens)
        fused_tokens = self.backbone(fused_tokens)

        # (B, N, D) -> (B, D, hs, ws)
        return fused_tokens.transpose(1, 2).reshape(b, self.embed_dim, hs, ws)

    def get_attention_map(self) -> Optional[torch.Tensor]:
        return self.fusion.last_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected 4d input, got shape={tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            if x.shape[-1] == self.in_channels:
                x = x.permute(0, 3, 1, 2).contiguous()
            else:
                raise ValueError(f"channel mismatch: got {x.shape[1]}, expected {self.in_channels}")

        H_orig, W_orig = x.shape[2], x.shape[3]

        spectral_map = self.spectral_encoder(x)
        spectral_map = self.spectral_se(spectral_map)

        skip = self.skip_proj(spectral_map)

        feat_map = self._forward_features_from_spectral(spectral_map)
        dec = self.decoder(feat_map)
        dec = F.interpolate(dec, size=(H_orig, W_orig),
                            mode='bilinear', align_corners=False)

        fused = torch.cat([dec, skip], dim=1)
        logits = self.seg_head(self.fuse_head(fused))

        # ensure shape
        if logits.shape[2:] != (H_orig, W_orig):
            logits = F.interpolate(logits, size=(H_orig, W_orig),
                                    mode='bilinear', align_corners=False)
        return logits