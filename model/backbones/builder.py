import torch.nn as nn

from .vit_timm import TimmViTBackbone
from typing import Any, Iterable, Optional

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


def _cfg_first(cfg: Any, paths: Iterable[str], default: Any = None) -> Any:
    for path in paths:
        value = _cfg_get(cfg, path, None)
        if value is not None:
            return value
    return default


class IdentityTokenBackbone(nn.Module):
    def __init__(self, embed_dim: Optional[int] = None):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, tokens):
        return tokens


def build_backbone(config: Any, embed_dim: Optional[int] = None) -> nn.Module:
    backbone_type = str(
        _cfg_first(
            config,
            ["model.backbone.type", "model.backbone_type"],
            "timm_vit",
        )
    ).lower()

    if backbone_type in {"none", "identity"}:
        return IdentityTokenBackbone(embed_dim=embed_dim)

    if backbone_type not in {"timm_vit", "vit_timm", "timm", "vit"}:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")

    model_name = _cfg_first(
        config,
        ["model.backbone_name", "model.backbone.name", "model.vit_name"],
        "vit_base_patch16_224",
    )
    pretrained = bool(
        _cfg_first(
            config,
            ["model.pretrained_weights", "model.backbone.pretrained_weights", "model.pretrained_backbone"],
            True,
        )
    )
    freeze = bool(_cfg_first(config, ["model.freeze_backbone", "model.backbone.freeze"], False))
    unfreeze_last_n = int(_cfg_first(config, ["model.unfreeze_last_n", "model.backbone.unfreeze_last_n"], 0))
    pretrained_cache_dir = _cfg_first(config, ["path.pretrained_cache_dir"], None)
    pretrained_local_path = _cfg_first(config, ["model.pretrained_local_path", "model.backbone.pretrained_local_path"], None)
    grad_checkpointing = bool(_cfg_first(config, ["model.backbone.grad_checkpointing"], False))
    target_embed_dim = embed_dim or int(_cfg_first(config, ["model.embed_dim", "model.hidden_dim"], 256))
    drop_rate = float(_cfg_first(config, ["model.backbone.drop_rate"], 0.0))
    attn_drop_rate = float(_cfg_first(config, ["model.backbone.attn_drop_rate"], 0.0))
    drop_path_rate = float(_cfg_first(config, ["model.backbone.drop_path_rate"], 0.0))

    return TimmViTBackbone(
        model_name=model_name,
        pretrained=pretrained,
        embed_dim=target_embed_dim,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        freeze=freeze,
        unfreeze_last_n=unfreeze_last_n,
        trainable_patterns=None,
        unfreeze_patterns=None,
        train_pos_embed=False,
        pos_embed_interp="2d",
        strict_input_dim=False,
        pretrained_cache_dir=pretrained_cache_dir,
        pretrained_local_path=pretrained_local_path,
        grad_checkpointing=grad_checkpointing,
    )