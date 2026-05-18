from munch import Munch
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.builder import build_backbone


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    current = cfg
    for key in path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, None)
        else:
            current = getattr(current, key, None)
    return default if current is None else current


def _cfg_first(cfg: Any, paths: Sequence[str], default: Any = None) -> Any:
    for path in paths:
        value = _cfg_get(cfg, path, None)
        if value is not None:
            return value
    return default


def _to_int_tuple(value: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            size = int(value[0])
            return size, size
        if len(value) >= 2:
            return int(value[0]), int(value[1])
    size = int(value)
    return size, size


def _to_int_list(value: Any, default: Sequence[int]) -> Sequence[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return list(default)
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _make_gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


def _resolve_heads(embed_dim: int, num_heads: int) -> int:
    heads = min(max(1, num_heads), embed_dim)
    while heads > 1 and embed_dim % heads != 0:
        heads -= 1
    return max(1, heads)


class ConvNormAct1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, groups: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, groups=groups, bias=False),
            _make_gn(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvNormAct2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            _make_gn(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpatialRefineStage(nn.Module):
    def __init__(self, channels: int, depth: int = 1, residual: bool = True):
        super().__init__()
        self.residual = bool(residual)
        self.blocks = nn.ModuleList([ConvNormAct2d(channels, channels, kernel_size=3) for _ in range(max(1, int(depth)))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            updated = block(out)
            out = out + updated if self.residual else updated
        return out


class SimpleSpectralEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, hidden_dim: Optional[int] = None, depth: int = 2):
        super().__init__()
        hidden_dim = hidden_dim or max(embed_dim // 2, 32)

        layers = [ConvNormAct2d(in_channels, hidden_dim, kernel_size=1)]
        for _ in range(max(0, depth - 1)):
            layers.append(ConvNormAct2d(hidden_dim, hidden_dim, kernel_size=1))
        layers.append(nn.Conv2d(hidden_dim, embed_dim, kernel_size=1, bias=False))
        layers.append(_make_gn(embed_dim))
        layers.append(nn.GELU())
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SpectralResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, expansion: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden_channels = int(channels * expansion)

        self.norm1 = _make_gn(channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.norm2 = _make_gn(channels)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.depthwise(x)
        x = F.gelu(self.pointwise(x))
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


class LightSpectralTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        num_heads = _resolve_heads(embed_dim, num_heads)
        hidden_dim = int(embed_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def _safe_attention(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.attn(x, x, x, need_weights=False)[0]
        except RuntimeError as exc:
            msg = str(exc).lower()
            sdp_err = ("invalid configuration argument" in msg) or ("scaled_dot_product_attention" in msg)
            if x.is_cuda and sdp_err and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
                # Fallback to math SDP with fp32 to avoid CUDA kernel launch issues on some driver/runtime combos.
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    x32 = x.float()
                    out = self.attn(x32, x32, x32, need_weights=False)[0]
                return out.to(dtype=x.dtype)
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self._safe_attention(x)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        return x


class SpectralTransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        spectral_tokens: int = 12,
        conv_depth: int = 2,
        transformer_depth: int = 1,
        num_heads: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(embed_dim // 2, 32)
        self.spectral_tokens = max(4, int(spectral_tokens))

        self.stem = ConvNormAct1d(1, hidden_dim, kernel_size=kernel_size)
        self.conv_blocks = nn.Sequential(
            *[
                SpectralResidualBlock(hidden_dim, kernel_size=max(3, kernel_size - 2), dropout=dropout)
                for _ in range(max(1, conv_depth))
            ]
        )
        self.token_proj = nn.Conv1d(hidden_dim, embed_dim, kernel_size=1, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.spectral_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer_blocks = nn.ModuleList(
            [
                LightSpectralTransformerBlock(embed_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(max(1, transformer_depth))
            ]
        )
        self.token_norm = nn.LayerNorm(embed_dim)
        self.attn_pool = nn.Linear(embed_dim, 1)
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def _get_pos_embed(self, token_count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos_embed = self.pos_embed
        if pos_embed.shape[1] == token_count:
            return pos_embed.to(device=device, dtype=dtype)

        pos_embed = pos_embed.transpose(1, 2)
        pos_embed = F.interpolate(pos_embed, size=token_count, mode="linear", align_corners=False)
        pos_embed = pos_embed.transpose(1, 2)
        return pos_embed.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, bands, height, width = x.shape
        spectra = x.permute(0, 2, 3, 1).contiguous().view(batch_size * height * width, 1, bands)

        x_seq = self.stem(spectra)
        x_seq = self.conv_blocks(x_seq)

        token_count = min(self.spectral_tokens, bands)
        avg_tokens = F.adaptive_avg_pool1d(x_seq, token_count)
        max_tokens = F.adaptive_max_pool1d(x_seq, token_count)
        x_tokens = 0.5 * avg_tokens + 0.5 * max_tokens
        x_tokens = self.token_proj(x_tokens).transpose(1, 2)

        x_tokens = x_tokens + self._get_pos_embed(token_count, x_tokens.device, x_tokens.dtype)
        for block in self.transformer_blocks:
            x_tokens = block(x_tokens)

        x_tokens = self.token_norm(x_tokens)
        attn = torch.softmax(self.attn_pool(x_tokens).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (x_tokens * attn).sum(dim=1) + 0.5 * x_tokens.max(dim=1).values
        pooled = self.out_proj(pooled)

        pooled = pooled.view(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        return pooled


class MultiScaleSpectralEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        kernel_sizes: Sequence[int] = (3, 5, 9),
        dilations: Sequence[int] = (1, 2),
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(embed_dim // 2, 48)

        self.stem = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1, bias=False),
            _make_gn(hidden_dim),
            nn.GELU(),
        )

        branches = []
        for kernel_size in kernel_sizes:
            kernel_size = max(3, int(kernel_size))
            for dilation in dilations:
                dilation = max(1, int(dilation))
                padding = ((kernel_size - 1) // 2) * dilation
                branches.append(
                    nn.Sequential(
                        nn.Conv1d(
                            hidden_dim,
                            hidden_dim,
                            kernel_size=kernel_size,
                            padding=padding,
                            dilation=dilation,
                            groups=hidden_dim,
                            bias=False,
                        ),
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                        _make_gn(hidden_dim),
                        nn.GELU(),
                    )
                )
        self.branches = nn.ModuleList(branches)
        mix_channels = hidden_dim * (1 + len(self.branches))
        self.mix = nn.Sequential(
            nn.Conv1d(mix_channels, hidden_dim, kernel_size=1, bias=False),
            _make_gn(hidden_dim),
            nn.GELU(),
        )

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=1, bias=False),
            _make_gn(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, bands, height, width = x.shape
        spectra = x.permute(0, 2, 3, 1).contiguous().view(batch_size * height * width, 1, bands)

        base = self.stem(spectra)
        multi_scale_feats = [base]
        for branch in self.branches:
            multi_scale_feats.append(branch(base))
        fused = self.mix(torch.cat(multi_scale_feats, dim=1))

        gate = self.channel_gate(fused)
        fused = fused * gate + base
        fused = self.dropout(fused)

        pooled_avg = F.adaptive_avg_pool1d(fused, 1).squeeze(-1)
        pooled_max = F.adaptive_max_pool1d(fused, 1).squeeze(-1)
        pooled = 0.7 * pooled_avg + 0.3 * pooled_max

        encoded = self.out_proj(pooled.unsqueeze(-1)).squeeze(-1)
        encoded = encoded.view(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        return encoded


class DualScaleSpectralConvEncoder(nn.Module):
    """
    Lightweight spectral encoder for raw spectra.
    Uses short/long receptive-field 1D depthwise convolutions to capture local and global band differences.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        short_kernel: int = 3,
        long_kernel: int = 11,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(embed_dim // 2, 48)
        short_kernel = max(3, int(short_kernel))
        long_kernel = max(short_kernel + 2, int(long_kernel))
        if short_kernel % 2 == 0:
            short_kernel += 1
        if long_kernel % 2 == 0:
            long_kernel += 1

        self.stem = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=1, bias=False),
            _make_gn(hidden_dim),
            nn.GELU(),
        )

        self.short_branch = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=short_kernel,
                padding=short_kernel // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            _make_gn(hidden_dim),
            nn.GELU(),
        )
        self.long_branch = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=long_kernel,
                padding=long_kernel // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            _make_gn(hidden_dim),
            nn.GELU(),
        )

        self.scale_gate = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False),
            _make_gn(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 2, kernel_size=1),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=1, bias=False),
            _make_gn(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, bands, height, width = x.shape
        spectra = x.permute(0, 2, 3, 1).contiguous().view(batch_size * height * width, 1, bands)

        base = self.stem(spectra)
        short_feat = self.short_branch(base)
        long_feat = self.long_branch(base)

        mix_logits = self.scale_gate(torch.cat([short_feat, long_feat], dim=1))
        mix_weights = torch.softmax(mix_logits, dim=1)
        fused = mix_weights[:, 0:1, :] * short_feat + mix_weights[:, 1:2, :] * long_feat
        fused = fused + base
        fused = self.dropout(fused)

        pooled_avg = F.adaptive_avg_pool1d(fused, 1).squeeze(-1)
        pooled_max = F.adaptive_max_pool1d(fused, 1).squeeze(-1)
        pooled = 0.6 * pooled_avg + 0.4 * pooled_max

        encoded = self.out_proj(pooled.unsqueeze(-1)).squeeze(-1)
        encoded = encoded.view(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        return encoded


class LightSpectralTokens(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        target_grid: Tuple[int, int] = (8, 8),
        max_mix_ratio: float = 0.35,
    ):
        super().__init__()
        self.target_grid = target_grid
        self.max_mix_ratio = float(max_mix_ratio)

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            _make_gn(embed_dim),
            nn.GELU(),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            _make_gn(embed_dim),
            nn.GELU(),
        )
        self.token_norm = nn.LayerNorm(embed_dim)

    def _resolve_target_grid(self, height: int, width: int) -> Tuple[int, int]:
        target_h = max(1, min(self.target_grid[0], height))
        target_w = max(1, min(self.target_grid[1], width))
        return target_h, target_w

    def forward(self, x: torch.Tensor):
        projected = self.proj(x)
        _, _, height, width = projected.shape
        target_h, target_w = self._resolve_target_grid(height, width)

        avg_map = F.adaptive_avg_pool2d(projected, output_size=(target_h, target_w))
        max_map = F.adaptive_max_pool2d(projected, output_size=(target_h, target_w))
        pooled = (1.0 - self.max_mix_ratio) * avg_map + self.max_mix_ratio * max_map
        fused = self.mix(pooled) + avg_map

        tokens = fused.flatten(2).transpose(1, 2).contiguous()
        tokens = self.token_norm(tokens)

        metadata = {
            "grid_size": (target_h, target_w),
            "token_count": target_h * target_w,
        }
        return tokens, metadata, fused


class HSIAdapter(nn.Module):
    def __init__(self, config: Munch):
        super().__init__()
        self.config = config

        preprocess_mode = str(_cfg_first(config, ["data.preprocess.mode"], "none")).lower()
        self.preprocess_mode = preprocess_mode
        # Only raw-spectrum (mode=none) uses supervised spectral encoder branch.
        self.use_supervised_encoder = preprocess_mode == "none"
        configured_in_channels = _cfg_first(config, ["model.in_channels"], None)
        if configured_in_channels is not None:
            self.in_channels = int(configured_in_channels)
        else:
            reduced_channels = int(_cfg_first(config, ["data.preprocess.output_dim"], _cfg_first(config, ["data.hsi_bands"], 96)))
            raw_channels = int(_cfg_first(config, ["data.hsi_bands"], 96))
            self.in_channels = reduced_channels if preprocess_mode != "none" else raw_channels

        self.num_classes = int(_cfg_first(config, ["model.num_classes", "data.num_classes", "num_classes"], 2))
        self.embed_dim = int(_cfg_first(config, ["model.embed_dim", "model.hidden_dim"], 256))
        self.decoder_dim = int(_cfg_first(config, ["model.decoder_dim"], self.embed_dim))

        default_encoder_type = "spectral_dual_scale" if self.use_supervised_encoder else "simple"
        spectral_encoder_type = str(_cfg_first(config, ["model.spectral_encoder.type"], default_encoder_type)).lower()
        spectral_hidden_dim = int(_cfg_first(config, ["model.spectral_encoder.hidden_dim"], max(self.embed_dim // 2, 64)))
        spectral_tokens = int(_cfg_first(config, ["model.spectral_encoder.spectral_tokens"], 12))
        spectral_conv_depth = int(_cfg_first(config, ["model.spectral_encoder.conv_depth"], 2))
        spectral_transformer_depth = int(_cfg_first(config, ["model.spectral_encoder.transformer_depth"], 1))
        spectral_heads = int(_cfg_first(config, ["model.spectral_encoder.num_heads"], 4))
        spectral_kernel = int(_cfg_first(config, ["model.spectral_encoder.kernel_size"], 7))
        spectral_kernel_sizes = _to_int_list(_cfg_first(config, ["model.spectral_encoder.kernel_sizes"], [3, 5, 9]), [3, 5, 9])
        spectral_dilations = _to_int_list(_cfg_first(config, ["model.spectral_encoder.dilations"], [1, 2]), [1, 2])
        spectral_short_kernel = int(_cfg_first(config, ["model.spectral_encoder.short_kernel_size"], 3))
        spectral_long_kernel = int(_cfg_first(config, ["model.spectral_encoder.long_kernel_size"], 11))
        spectral_dropout = float(_cfg_first(config, ["model.spectral_encoder.dropout"], 0.0))

        if self.use_supervised_encoder:
            if spectral_encoder_type in {"simple"}:
                self.spectral_encoder = SimpleSpectralEncoder(
                    in_channels=self.in_channels,
                    embed_dim=self.embed_dim,
                    hidden_dim=spectral_hidden_dim,
                    depth=max(1, spectral_conv_depth),
                )
            elif spectral_encoder_type in {"spectral_transformer", "transformer"}:
                self.spectral_encoder = SpectralTransformerEncoder(
                    embed_dim=self.embed_dim,
                    hidden_dim=spectral_hidden_dim,
                    spectral_tokens=spectral_tokens,
                    conv_depth=spectral_conv_depth,
                    transformer_depth=spectral_transformer_depth,
                    num_heads=spectral_heads,
                    kernel_size=spectral_kernel,
                    dropout=spectral_dropout,
                )
            elif spectral_encoder_type in {"spectral_dual_scale", "dual_scale", "dual"}:
                self.spectral_encoder = DualScaleSpectralConvEncoder(
                    embed_dim=self.embed_dim,
                    hidden_dim=spectral_hidden_dim,
                    short_kernel=spectral_short_kernel,
                    long_kernel=spectral_long_kernel,
                    dropout=spectral_dropout,
                )
            else:
                self.spectral_encoder = MultiScaleSpectralEncoder(
                    embed_dim=self.embed_dim,
                    hidden_dim=spectral_hidden_dim,
                    kernel_sizes=spectral_kernel_sizes,
                    dilations=spectral_dilations,
                    dropout=spectral_dropout,
                )
        else:
            # Reduced-spectrum branch: no extra encoder, only light conv reshaping.
            self.spectral_encoder = nn.Sequential(
                ConvNormAct2d(self.in_channels, spectral_hidden_dim, kernel_size=1),
                ConvNormAct2d(spectral_hidden_dim, spectral_hidden_dim, kernel_size=3),
                nn.Conv2d(spectral_hidden_dim, self.embed_dim, kernel_size=1, bias=False),
                _make_gn(self.embed_dim),
                nn.GELU(),
            )

        target_grid = _to_int_tuple(
            _cfg_first(config, ["model.tokenizer.target_grid", "model.tokenizer.grid_size"], (8, 8)),
            (8, 8),
        )
        max_mix_ratio = float(_cfg_first(config, ["model.tokenizer.max_mix_ratio"], 0.35))
        spatial_enhance_cfg = _cfg_first(config, ["model.spatial_enhance"], {})
        if spatial_enhance_cfg is None:
            spatial_enhance_cfg = {}
        if not isinstance(spatial_enhance_cfg, dict):
            try:
                spatial_enhance_cfg = dict(spatial_enhance_cfg)
            except Exception:
                spatial_enhance_cfg = {}
        self.shape_assertions = bool(spatial_enhance_cfg.get("shape_assertions", True))
        self.enable_spatial_refine = bool(spatial_enhance_cfg.get("enable_refine", True))
        self.refine_depth = max(1, int(spatial_enhance_cfg.get("refine_depth", 2)))
        self.use_residual_refine = bool(spatial_enhance_cfg.get("use_residual_refine", True))
        self.use_shallow_skip = bool(spatial_enhance_cfg.get("use_shallow_skip", True))
        self.skip_fusion = str(spatial_enhance_cfg.get("skip_fusion", "add")).lower()
        if self.skip_fusion not in {"add", "concat"}:
            self.skip_fusion = "add"

        self.tokenizer = LightSpectralTokens(
            in_channels=self.embed_dim,
            embed_dim=self.embed_dim,
            target_grid=target_grid,
            max_mix_ratio=max_mix_ratio,
        )

        self.backbone = build_backbone(config, embed_dim=self.embed_dim)

        self.token_refine = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1, bias=False),
            _make_gn(self.embed_dim),
            nn.GELU(),
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(self.embed_dim * 2, self.embed_dim, kernel_size=1, bias=False),
            _make_gn(self.embed_dim),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.decoder_dim, kernel_size=1, bias=False),
            _make_gn(self.decoder_dim),
            nn.GELU(),
        )
        self.pre_decoder_refine = (
            SpatialRefineStage(self.embed_dim, depth=self.refine_depth, residual=self.use_residual_refine)
            if self.enable_spatial_refine
            else nn.Identity()
        )
        self.post_decoder_refine = (
            SpatialRefineStage(self.decoder_dim, depth=max(1, self.refine_depth - 1), residual=self.use_residual_refine)
            if self.enable_spatial_refine
            else nn.Identity()
        )
        if self.use_shallow_skip:
            self.shallow_skip_proj = nn.Sequential(
                nn.Conv2d(self.embed_dim, self.decoder_dim, kernel_size=1, bias=False),
                _make_gn(self.decoder_dim),
                nn.GELU(),
            )
            self.skip_fusion_proj = (
                nn.Sequential(
                    nn.Conv2d(self.decoder_dim * 2, self.decoder_dim, kernel_size=1, bias=False),
                    _make_gn(self.decoder_dim),
                    nn.GELU(),
                )
                if self.skip_fusion == "concat"
                else nn.Identity()
            )
        else:
            self.shallow_skip_proj = None
            self.skip_fusion_proj = nn.Identity()
        self.classifier = nn.Conv2d(self.decoder_dim, self.num_classes, kernel_size=1)

        self.spectral_aux_head = nn.Sequential(
            ConvNormAct2d(self.embed_dim, self.embed_dim, kernel_size=3),
            nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=1),
        ) if self.use_supervised_encoder else None
        self.token_aux_head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=1) if self.use_supervised_encoder else None

        self.feature_dim = self.decoder_dim
        self._last_aux_outputs: Dict[str, torch.Tensor] = {}

    def _reshape_tokens_to_map(self, tokens: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        batch_size, token_count, channels = tokens.shape
        grid_h, grid_w = grid_size
        if token_count != grid_h * grid_w:
            raise ValueError(f"Token count {token_count} does not match grid {grid_size}.")
        return tokens.transpose(1, 2).contiguous().view(batch_size, channels, grid_h, grid_w)

    def _assert_hw(self, name: str, tensor: torch.Tensor, expected_hw: Tuple[int, int]) -> None:
        if not self.shape_assertions:
            return
        current_hw = tuple(tensor.shape[-2:])
        if current_hw != tuple(expected_hw):
            raise ValueError(f"{name} spatial mismatch: got {current_hw}, expected {tuple(expected_hw)}")

    def _update_aux_outputs(
        self,
        spectral_features: torch.Tensor,
        token_features: torch.Tensor,
        fused_features: torch.Tensor,
    ) -> None:
        if not self.use_supervised_encoder or self.spectral_aux_head is None or self.token_aux_head is None:
            self._last_aux_outputs = {}
            return
        spatial_size = spectral_features.shape[-2:]
        token_logits = self.token_aux_head(token_features)
        token_logits = F.interpolate(token_logits, size=spatial_size, mode="bilinear", align_corners=False)

        self._last_aux_outputs = {
            "spectral_logits": self.spectral_aux_head(spectral_features),
            "token_logits": token_logits,
            "spectral_features": spectral_features,
            "token_features": token_features,
            "fused_features": fused_features,
        }

    def _forward_features_from_spectral(self, spectral_features: torch.Tensor) -> torch.Tensor:
        tokens, metadata, token_map = self.tokenizer(spectral_features)
        if self.shape_assertions:
            token_count = int(metadata.get("token_count", tokens.shape[1]))
            if token_count != int(tokens.shape[1]):
                raise ValueError(f"Tokenizer token_count mismatch: metadata={token_count}, actual={tokens.shape[1]}")
        tokens = self.backbone(tokens)
        if self.shape_assertions and tokens.ndim != 3:
            raise ValueError(f"Backbone output must be [B, N, D], got {tuple(tokens.shape)}")

        token_map = self._reshape_tokens_to_map(tokens, metadata["grid_size"])
        token_map = self.token_refine(token_map)
        token_map = F.interpolate(
            token_map,
            size=spectral_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        self._assert_hw("token_map_upsampled", token_map, tuple(spectral_features.shape[-2:]))

        fusion_context = torch.cat([spectral_features, token_map], dim=1)
        token_weight = self.fusion_gate(fusion_context)
        fused_features = spectral_features + token_weight * token_map
        fused_features = self.pre_decoder_refine(fused_features)
        fused_features = self.fusion_proj(fused_features)
        if self.use_shallow_skip and self.shallow_skip_proj is not None:
            shallow_skip = self.shallow_skip_proj(spectral_features)
            if shallow_skip.shape[-2:] != fused_features.shape[-2:]:
                shallow_skip = F.interpolate(shallow_skip, size=fused_features.shape[-2:], mode="bilinear", align_corners=False)
            if self.skip_fusion == "concat":
                fused_features = self.skip_fusion_proj(torch.cat([fused_features, shallow_skip], dim=1))
            else:
                fused_features = fused_features + shallow_skip
        fused_features = self.post_decoder_refine(fused_features)
        self._assert_hw("fused_features", fused_features, tuple(spectral_features.shape[-2:]))
        self._update_aux_outputs(spectral_features, token_map, fused_features)
        return fused_features

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        spectral_features = self.spectral_encoder(x)
        return self._forward_features_from_spectral(spectral_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        logits = self.classifier(features)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        self._assert_hw("logits", logits, tuple(x.shape[-2:]))
        return logits

    def get_aux_outputs(self) -> Dict[str, torch.Tensor]:
        return self._last_aux_outputs

    def get_spectral_supervision_tensors(self) -> Dict[str, torch.Tensor]:
        if not self.use_supervised_encoder:
            return {}
        return self.get_aux_outputs()

    def forward_with_aux(self, x: torch.Tensor):
        logits = self.forward(x)
        return logits, self.get_aux_outputs()