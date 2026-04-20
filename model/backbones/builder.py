from __future__ import annotations

from typing import Any, Mapping

from model.backbones.base import BaseBackbone
from model.backbones.transformer_scratch import TransformerBackbone
from model.backbones.vit_timm import TimmViTBackbone


def build_backbone(config: Mapping[str, Any]) -> BaseBackbone:
    """Backbone factory.

    If use_pretrained=True, build a timm ViT wrapper.
    If use_pretrained=False, use scratch Transformer backbone.

    Pretrained loading failures are raised as RuntimeError to keep experiment
    behavior explicit and reproducible.
    """

    embed_dim = int(config.get("embed_dim", 128))
    depth = int(config.get("depth", 4))
    num_heads = int(config.get("num_heads", 4))
    mlp_ratio = float(config.get("mlp_ratio", 4.0))
    dropout = float(config.get("dropout", 0.1))
    freeze = bool(config.get("freeze_backbone", False))

    use_pretrained = bool(config.get("use_pretrained", config.get("pretrained_backbone", False)))
    backbone_name = str(config.get("backbone_name", "vit_small_patch16_224"))
    pretrained_weights = bool(config.get("pretrained_weights", config.get("backbone_pretrained", True)))
    pretrained_cache_dir = str(config.get("pretrained_cache_dir", "")).strip()

    if use_pretrained:
        try:
            return TimmViTBackbone(
                input_dim=embed_dim,
                name=backbone_name,
                pretrained=pretrained_weights,
                freeze=freeze,
                cache_dir=pretrained_cache_dir if pretrained_cache_dir else None,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load pretrained backbone '{backbone_name}': {exc}") from exc

    return TransformerBackbone(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        freeze=freeze,
    )
