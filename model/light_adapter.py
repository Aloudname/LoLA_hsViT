from __future__ import annotations

from munch import Munch
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.builder import build_backbone


class BandAttentionEncoder(nn.Module):
    """band-wise self-attention encoder."""
    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        # band positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, out_channels, 1, 1) * 0.02)
        # band-wise self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels * 2, out_channels),
        )
        self.norm2 = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.stem(x)
        x = x + self.pos_embed
        B, C, H, W = x.shape
        # reshape to (B*H*W, C) for attention
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C).unsqueeze(1)  # (N, 1, C)

        x_conv = x.reshape(B, C, H * W).transpose(1, 2)  # (B, N, C)
        x_conv = self.norm(x_conv)
        attn_out, _ = self.attn(x_conv, x_conv, x_conv)
        x_conv = x_conv + attn_out
        x_conv = self.norm2(x_conv)
        ffn_out = self.ffn(x_conv)
        x_conv = x_conv + ffn_out
        x_out = x_conv.transpose(1, 2).reshape(B, C, H, W)
        return x_out


class SpatialEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, H*W, D)
        x = self.norm(x)
        return x, (H, W)


class SimpleFusion(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, spatial_tokens: torch.Tensor, spectral_tokens: torch.Tensor) -> torch.Tensor:
        global_spectral = spectral_tokens.mean(dim=1, keepdim=True)      # (B, 1, D)
        gate = self.gate(global_spectral)
        return spatial_tokens * (1 - gate) + spectral_tokens * gate


class UNetDecoder(nn.Module):
    def __init__(self, embed_dim: int, skip_channels: int, decoder_dim: int, num_classes: int):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(embed_dim, decoder_dim, kernel_size=2, stride=2)
        self.skip_conv = nn.Conv2d(skip_channels, decoder_dim, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(decoder_dim * 2, decoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(decoder_dim, decoder_dim, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.seg_head = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, feat: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up1(feat)
        skip_scaled = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        skip_proj = self.skip_conv(skip_scaled)
        x = torch.cat([x, skip_proj], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = self.conv2(x)
        return self.seg_head(x)


class LightAdapter(nn.Module):
    """Light weighted adapter."""
    def __init__(self, config: Munch):
        super().__init__()
        in_channels = int(config.data.preprocess.get("output_dim")) \
            if config.data.preprocess.get("mode") != "none" \
            else int(config.data.get("hsi_bands"))

        num_classes = int(config.data.get("num_classes"))
        spectral_dim = int(config.model.get("spectral_dim", 32))
        embed_dim = int(config.model.get("embed_dim", 128))
        decoder_dim = int(config.model.get("decoder_dim", 96))

        self.in_channels = in_channels

        self.spectral_encoder = BandAttentionEncoder(in_channels, spectral_dim)
        self.spatial_embed = SpatialEmbed(spectral_dim, embed_dim, patch_size=4)
        self.spectral_token_proj = nn.Linear(spectral_dim, embed_dim)
        self.fusion = SimpleFusion(embed_dim)
        self.backbone = build_backbone(config)

        self.decoder = UNetDecoder(
            embed_dim=embed_dim,
            skip_channels=spectral_dim,
            decoder_dim=decoder_dim,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got {x.shape}")
        if x.shape[1] != self.in_channels:
            if x.shape[-1] == self.in_channels:
                x = x.permute(0, 3, 1, 2).contiguous()
            else:
                raise ValueError(f"Input channel mismatch: got {x.shape[1]}, expected {self.in_channels}")

        H_orig, W_orig = x.shape[2], x.shape[3]

        # spectral encode
        spec_feat = self.spectral_encoder(x)                    # (B, spectral_dim, H, W)

        # spatial embed
        spatial_tokens, (Hf, Wf) = self.spatial_embed(spec_feat)  # (B, N, embed_dim), (Hf, Wf)

        # spectral token
        spec_down = F.adaptive_avg_pool2d(spec_feat, (Hf, Wf))   # (B, spectral_dim, Hf, Wf)
        spec_tokens = spec_down.flatten(2).transpose(1, 2)       # (B, N, spectral_dim)
        spec_tokens = self.spectral_token_proj(spec_tokens)      # (B, N, embed_dim)

        # fuse + bb
        fused_tokens = self.fusion(spatial_tokens, spec_tokens)  # (B, N, embed_dim)
        backbone_out = self.backbone(fused_tokens)               # (B, N, embed_dim)

        # reshape backbone output to feature map
        feat_map = backbone_out.transpose(1, 2).reshape(-1, self.spatial_embed.proj.out_channels, Hf, Wf)

        # decoder
        logits = self.decoder(feat_map, spec_feat)               # (B, num_classes, H_dec, W_dec)

        # interpolate to original size if needed
        if logits.shape[2:] != (H_orig, W_orig):
            logits = F.interpolate(logits, size=(H_orig, W_orig),
                                   mode='bilinear', align_corners=False)

        return logits