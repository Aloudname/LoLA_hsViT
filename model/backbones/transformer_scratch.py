from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.base import BaseBackbone


class ScratchEncoderBlock(nn.Module):
    """Lightweight pre-norm transformer block with explicit ops."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        ff_dim = int(embed_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def _safe_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        try:
            return self.attn(q, k, v, need_weights=False)[0]
        except RuntimeError as exc:
            msg = str(exc).lower()
            sdp_err = ("invalid configuration argument" in msg) or ("scaled_dot_product_attention" in msg)
            if q.is_cuda and sdp_err and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    q32 = q.float()
                    k32 = k.float()
                    v32 = v.float()
                    out = self.attn(q32, k32, v32, need_weights=False)[0]
                return out.to(dtype=q.dtype)
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.norm1(x)
        # During ONNX export, avoid native fastpath by passing distinct tensors.
        if torch.onnx.is_in_onnx_export():
            k = q + 0.0
            v = q + 0.0
        else:
            k = q
            v = q
        attn_out = self._safe_attention(q, k, v)
        x = x + self.drop1(attn_out)

        y = self.norm2(x)
        y = self.fc1(y)
        y = F.gelu(y)
        y = self.drop2(y)
        y = self.fc2(y)
        y = self.drop2(y)
        return x + y


class TransformerBackbone(BaseBackbone):
    """Transformer token backbone trained from scratch."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        freeze: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(embed_dim)
        self.blocks = nn.ModuleList(
            [
                ScratchEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        if freeze:
            for p in self.blocks.parameters():
                p.requires_grad_(False)
            for p in self.norm.parameters():
                p.requires_grad_(False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"expected [B, N, D] tokens, got shape={tuple(tokens.shape)}")
        if (not torch.onnx.is_in_onnx_export()) and tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"expected token dim D={self.hidden_dim}, got D={tokens.shape[-1]}"
            )

        x = tokens
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
