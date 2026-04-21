# import models from timm through huggingface.
from __future__ import annotations

from pathlib import Path
from typing import Optional

from model.backbones.base import BaseBackbone
import os, torch, torch.nn as nn, torch.nn.functional as F


# internal mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class TimmViTBackbone(BaseBackbone):
    """timm ViT block wrapper for token inputs [B, N, D].

    timm ViT blocks do not add position embeddings by themselves. Position
    embeddings are added manually before entering block stacks.
    """

    def __init__(
        self,
        input_dim: int,
        name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        freeze: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()

        try:
            import timm
        except Exception as exc:  # pragma: no cover - environment specific
            raise RuntimeError("timm is required for pretrained ViT backbone") from exc

        resolved_cache_dir: Optional[str] = None
        if cache_dir is not None and str(cache_dir).strip() != "":
            resolved_cache_dir = str(Path(cache_dir).expanduser())
            Path(resolved_cache_dir).mkdir(parents=True, exist_ok=True)

        vit = timm.create_model(name, pretrained=pretrained, cache_dir=resolved_cache_dir)
        if not hasattr(vit, "blocks") or not hasattr(vit, "norm"):
            raise ValueError(f"{name} is not a supported timm ViT-like model")

        vit_dim = int(getattr(vit, "embed_dim", input_dim))
        self.input_dim = int(input_dim)
        self.vit_dim = vit_dim
        self.hidden_dim = self.input_dim

        self.in_proj = nn.Identity() if self.input_dim == self.vit_dim else nn.Linear(self.input_dim, self.vit_dim)
        self.out_proj = nn.Identity() if self.vit_dim == self.input_dim else nn.Linear(self.vit_dim, self.input_dim)

        self.blocks = vit.blocks
        self.norm = vit.norm
        self.pos_drop = vit.pos_drop if hasattr(vit, "pos_drop") else nn.Identity()

        self.num_prefix_tokens = int(getattr(vit, "num_prefix_tokens", 1))

        self.use_pos_embed = bool(hasattr(vit, "pos_embed") and (vit.pos_embed is not None))
        if self.use_pos_embed:
            pos_embed = vit.pos_embed.detach().clone()
            if pos_embed.ndim != 3 or pos_embed.shape[-1] != self.vit_dim:
                self.use_pos_embed = False
            else:
                if self.num_prefix_tokens > 0 and pos_embed.shape[1] > self.num_prefix_tokens:
                    pos_embed = pos_embed[:, self.num_prefix_tokens :, :]
                elif self.num_prefix_tokens > 0 and pos_embed.shape[1] <= self.num_prefix_tokens:
                    self.use_pos_embed = False

                if self.use_pos_embed and pos_embed.shape[1] > 0:
                    self.register_buffer("base_pos_embed", pos_embed, persistent=False)
                else:
                    self.use_pos_embed = False

        if freeze:
            for p in self.blocks.parameters():
                p.requires_grad_(False)
            for p in self.norm.parameters():
                p.requires_grad_(False)

    def _get_pos_embed(self, token_len: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if not self.use_pos_embed:
            return None

        pos = self.base_pos_embed.to(device=device, dtype=dtype)
        if pos.shape[1] == token_len:
            return pos

        # 1D interpolation is robust for variable token lengths.
        pos_1d = pos.transpose(1, 2)
        pos_1d = F.interpolate(pos_1d, size=token_len, mode="linear", align_corners=False)
        return pos_1d.transpose(1, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"expected [B, N, D] tokens, got shape={tuple(tokens.shape)}")
        if tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected token dim D={self.input_dim}, got D={tokens.shape[-1]}"
            )

        x = self.in_proj(tokens)
        pos_embed = self._get_pos_embed(token_len=x.shape[1], device=x.device, dtype=x.dtype)
        if pos_embed is not None:
            x = x + pos_embed

        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.out_proj(x)
        return x
