from __future__ import annotations

from munch import Munch
from typing import Any, Mapping

from model.backbones.base import BaseBackbone
from model.backbones.transformer_scratch import TransformerBackbone
from model.backbones.vit_timm import TimmViTBackbone


def build_backbone(config: Munch) -> BaseBackbone:
    """Backbone factory.

    If use_pretrained=True, build a timm ViT wrapper.
    If use_pretrained=False, use scratch Transformer backbone.

    Pretrained loading failures are raised as RuntimeError to keep experiment
    behavior explicit and reproducible.
    """

    embed_dim = int(config.model.get("embed_dim", 128))
    depth = int(config.model.get("depth", 4))
    num_heads = int(config.model.get("num_heads", 4))
    mlp_ratio = float(config.model.get("mlp_ratio", 4.0))
    dropout = float(config.model.get("dropout", 0.1))
    freeze = bool(config.model.get("freeze_backbone", False))

    use_pretrained = bool(config.model.get("use_pretrained", False))
    backbone_name = str(config.model.get("backbone_name", "vit_small_patch16_224"))
    pretrained_weights = bool(config.model.get("pretrained_weights", True))
    pretrained_cache_dir = str(config.path.get("pretrained_cache_dir", "")).strip()

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
