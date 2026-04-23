import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import load_file as load_safetensors_file
except Exception:
    load_safetensors_file = None


def _to_pattern_list(patterns: Optional[Iterable[str]]) -> Sequence[str]:
    if patterns is None:
        return []
    if isinstance(patterns, str):
        text = patterns.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return [part.strip().strip("'\"") for part in text.split(",") if part.strip()]
    return [str(pattern).strip() for pattern in patterns if str(pattern).strip()]


class TimmViTBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        embed_dim: Optional[int] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        freeze: bool = False,
        unfreeze_last_n: int = 0,
        trainable_patterns: Optional[Iterable[str]] = None,
        unfreeze_patterns: Optional[Iterable[str]] = None,
        train_pos_embed: bool = False,
        pos_embed_interp: str = "2d",
        strict_input_dim: bool = False,
        pretrained_cache_dir: Optional[str] = None,
        pretrained_local_path: Optional[str] = None,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        use_pretrained = bool(pretrained and not pretrained_local_path)
        timm_kwargs = {
            "pretrained": use_pretrained,
            "num_classes": 0,
            "global_pool": "",
            "drop_rate": drop_rate,
            "attn_drop_rate": attn_drop_rate,
            "drop_path_rate": drop_path_rate,
        }
        if pretrained_cache_dir:
            timm_kwargs["cache_dir"] = str(pretrained_cache_dir)
        self.vit = timm.create_model(
            model_name,
            **timm_kwargs,
        )
        if pretrained_local_path:
            local_state_path = self._resolve_local_weights_path(pretrained_local_path)
            self._load_local_pretrained(local_state_path)

        self.model_name = model_name
        self.vit_embed_dim = int(getattr(self.vit, "embed_dim"))
        self.input_embed_dim = int(embed_dim or self.vit_embed_dim)
        self.output_embed_dim = self.input_embed_dim
        self.pos_embed_interp = str(pos_embed_interp).lower()
        self.strict_input_dim = bool(strict_input_dim)

        self.input_proj = (
            nn.Identity() if self.input_embed_dim == self.vit_embed_dim else nn.Linear(self.input_embed_dim, self.vit_embed_dim)
        )
        self.output_proj = (
            nn.Identity() if self.output_embed_dim == self.vit_embed_dim else nn.Linear(self.vit_embed_dim, self.output_embed_dim)
        )

        prefix_tokens = int(getattr(self.vit, "num_prefix_tokens", 0))
        if prefix_tokens == 0:
            prefix_tokens = 1 if getattr(self.vit, "cls_token", None) is not None else 0
            if getattr(self.vit, "dist_token", None) is not None:
                prefix_tokens += 1
        self.num_prefix_tokens = prefix_tokens
        if grad_checkpointing and hasattr(self.vit, "set_grad_checkpointing"):
            self.vit.set_grad_checkpointing(True)

        self._configure_trainability(
            freeze=freeze,
            unfreeze_last_n=unfreeze_last_n,
            trainable_patterns=_to_pattern_list(trainable_patterns),
            unfreeze_patterns=_to_pattern_list(unfreeze_patterns),
            train_pos_embed=train_pos_embed,
        )

    @staticmethod
    def _resolve_local_weights_path(pretrained_local_path: str) -> Path:
        path = Path(pretrained_local_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.is_dir():
            candidates = [
                path / "model.safetensors",
                path / "pytorch_model.bin",
                path / "pytorch_model.pt",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"no supported weights file found in directory: {path}")
        if not path.exists():
            fallback_hidden = (Path.cwd() / ".pretrained" / path.name).resolve()
            if fallback_hidden.exists():
                return fallback_hidden
            raise FileNotFoundError(f"local pretrained weights not found: {path}")
        return path

    def _load_local_pretrained(self, local_state_path: Path) -> None:
        suffix = local_state_path.suffix.lower()
        if suffix == ".safetensors":
            if load_safetensors_file is None:
                raise RuntimeError("safetensors is required to load .safetensors weights")
            state_dict = load_safetensors_file(str(local_state_path), device="cpu")
        else:
            payload = torch.load(str(local_state_path), map_location="cpu")
            if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
                state_dict = payload["state_dict"]
            elif isinstance(payload, dict):
                state_dict = payload
            else:
                raise RuntimeError(f"unsupported checkpoint payload type: {type(payload)}")
        self.vit.load_state_dict(state_dict, strict=False)

    def _configure_trainability(
        self,
        freeze: bool,
        unfreeze_last_n: int,
        trainable_patterns: Sequence[str],
        unfreeze_patterns: Sequence[str],
        train_pos_embed: bool,
    ) -> None:
        for parameter in self.vit.parameters():
            parameter.requires_grad = not freeze

        if freeze:
            if hasattr(self.vit, "blocks") and unfreeze_last_n > 0:
                for block in self.vit.blocks[-int(unfreeze_last_n):]:
                    for parameter in block.parameters():
                        parameter.requires_grad = True

                if hasattr(self.vit, "norm"):
                    for parameter in self.vit.norm.parameters():
                        parameter.requires_grad = True

            active_patterns = list(trainable_patterns) + list(unfreeze_patterns)
            if active_patterns:
                for name, parameter in self.vit.named_parameters():
                    if any(pattern in name for pattern in active_patterns):
                        parameter.requires_grad = True

            if train_pos_embed:
                for name in ("pos_embed", "cls_token", "dist_token"):
                    tensor = getattr(self.vit, name, None)
                    if isinstance(tensor, nn.Parameter):
                        tensor.requires_grad = True

        for parameter in self.input_proj.parameters():
            parameter.requires_grad = True
        for parameter in self.output_proj.parameters():
            parameter.requires_grad = True

    def _infer_source_grid(self, token_count: int) -> Optional[Tuple[int, int]]:
        patch_embed = getattr(self.vit, "patch_embed", None)
        grid_size = getattr(patch_embed, "grid_size", None)
        if isinstance(grid_size, tuple) and len(grid_size) == 2 and grid_size[0] * grid_size[1] == token_count:
            return int(grid_size[0]), int(grid_size[1])

        side = int(round(math.sqrt(token_count)))
        if side * side == token_count:
            return side, side

        for height in range(int(math.sqrt(token_count)), 0, -1):
            if token_count % height == 0:
                return height, token_count // height
        return None

    def _infer_target_grid(self, token_count: int) -> Optional[Tuple[int, int]]:
        side = int(round(math.sqrt(token_count)))
        if side * side == token_count:
            return side, side

        for height in range(int(math.sqrt(token_count)), 0, -1):
            if token_count % height == 0:
                return height, token_count // height
        return None

    def _resize_pos_embed(self, token_count: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        pos_embed = getattr(self.vit, "pos_embed", None)
        if pos_embed is None:
            return None

        prefix_tokens = self.num_prefix_tokens
        prefix = pos_embed[:, :prefix_tokens]
        spatial = pos_embed[:, prefix_tokens:]

        if spatial.shape[1] == token_count:
            full_pos = torch.cat([prefix, spatial], dim=1)
            return full_pos.to(device=device, dtype=dtype)

        source_grid = self._infer_source_grid(spatial.shape[1])
        target_grid = self._infer_target_grid(token_count)

        use_2d = (
            self.pos_embed_interp in {"2d", "bicubic", "bilinear"}
            and source_grid is not None
            and target_grid is not None
            and source_grid[0] * source_grid[1] == spatial.shape[1]
            and target_grid[0] * target_grid[1] == token_count
        )

        if use_2d:
            mode = "bicubic" if self.pos_embed_interp in {"2d", "bicubic"} else "bilinear"
            spatial = spatial.reshape(1, source_grid[0], source_grid[1], -1).permute(0, 3, 1, 2)
            spatial = F.interpolate(spatial, size=target_grid, mode=mode, align_corners=False)
            spatial = spatial.permute(0, 2, 3, 1).reshape(1, token_count, -1)
        else:
            spatial = spatial.transpose(1, 2)
            spatial = F.interpolate(spatial, size=token_count, mode="linear", align_corners=False)
            spatial = spatial.transpose(1, 2)

        full_pos = torch.cat([prefix, spatial], dim=1)
        return full_pos.to(device=device, dtype=dtype)

    def _prefix_tokens(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        prefixes = []
        cls_token = getattr(self.vit, "cls_token", None)
        if cls_token is not None:
            prefixes.append(cls_token.expand(batch_size, -1, -1))
        dist_token = getattr(self.vit, "dist_token", None)
        if dist_token is not None:
            prefixes.append(dist_token.expand(batch_size, -1, -1))
        if not prefixes:
            return torch.empty(batch_size, 0, self.vit_embed_dim, device=device, dtype=dtype)
        return torch.cat(prefixes, dim=1).to(device=device, dtype=dtype)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"TimmViTBackbone expects [B, N, D] tokens, got shape {tuple(tokens.shape)}")

        if self.strict_input_dim and tokens.shape[-1] != self.input_embed_dim:
            raise ValueError(
                f"Expected token dim {self.input_embed_dim}, received {tokens.shape[-1]} for backbone {self.model_name}."
            )

        x = self.input_proj(tokens)
        batch_size, token_count, _ = x.shape

        prefix = self._prefix_tokens(batch_size, x.device, x.dtype)
        if prefix.shape[1] > 0:
            x = torch.cat([prefix, x], dim=1)

        pos_embed = self._resize_pos_embed(token_count, x.device, x.dtype)
        if pos_embed is not None and pos_embed.shape[1] == x.shape[1]:
            x = x + pos_embed

        if hasattr(self.vit, "pos_drop"):
            x = self.vit.pos_drop(x)

        if hasattr(self.vit, "patch_drop"):
            x = self.vit.patch_drop(x)

        for block in self.vit.blocks:
            x = block(x)

        if hasattr(self.vit, "norm"):
            x = self.vit.norm(x)

        if prefix.shape[1] > 0:
            x = x[:, prefix.shape[1] :, :]

        x = self.output_proj(x)
        return x