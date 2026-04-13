"""RGBViT model for RGB image segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """DropPath layer compatible with timm.layers.DropPath API."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class StandardMultiHeadAttention(nn.Module):
    """Standard multi-head self-attention without windowing."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer block (LN + MHSA + MLP)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = StandardMultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RGBViT(nn.Module):
    """A conventional ViT-like segmentation backbone for RGB images."""

    def __init__(self, in_channels=3, num_classes=4, patch_size=31,
                 dim=96, depths=(3, 4, 5), num_heads=(4, 8, 16),
                 mlp_ratio=4.0, drop_path_rate=0.2,
                 hierarchical_head=True,
                 eval_fg_gate_threshold: float = -1.0,
                 window_size=None, r=None, lora_alpha=None):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.patch_size = int(patch_size)
        self.dim = int(dim)
        self.depths = list(depths)
        self.num_heads = list(num_heads)
        self.mode = 'segmentation'
        self.hierarchical_head = bool(hierarchical_head)
        self.eval_fg_gate_threshold = float(eval_fg_gate_threshold)

        if self.num_classes <= 1:
            raise ValueError("RGBViT requires num_classes > 1 for segmentation supervision")

        # Shallow RGB stem (pure 2D conv).
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )

        self.patch_embed = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )

        self.patch_resolution = self.patch_size // 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_resolution, self.patch_resolution, self.dim))
        self.pos_drop = nn.Dropout(p=0.1)

        self.dims = [self.dim * (2 ** i) for i in range(len(self.depths))]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]

        self.transformer_levels = nn.ModuleList()
        curr_dpr_idx = 0
        for level_idx in range(len(self.depths)):
            current_dim = self.dims[level_idx]
            current_depth = self.depths[level_idx]
            current_heads = self.num_heads[level_idx]

            blocks = nn.ModuleList()
            for _ in range(current_depth):
                block = TransformerBlock(
                    dim=current_dim,
                    num_heads=current_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=0.1,
                    attn_drop=0.1,
                    drop_path=dpr[curr_dpr_idx],
                    norm_layer=nn.LayerNorm,
                )
                curr_dpr_idx += 1
                blocks.append(block)

            self.transformer_levels.append(blocks)

            if level_idx < len(self.depths) - 1:
                self.register_module(
                    f'downsample_{level_idx}',
                    nn.Sequential(
                        nn.Conv2d(current_dim, self.dims[level_idx + 1], kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(self.dims[level_idx + 1]),
                        nn.ReLU(inplace=True),
                    )
                )

        # Keep these members for compatibility with existing trainer/CAM workflow.
        self.final_feature_map = None
        self.norm = nn.LayerNorm(self.dims[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.dims[-1], self.num_classes)

        self.seg_decoder = nn.Sequential(
            nn.Conv2d(self.dims[-1], 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.seg_head = nn.Conv2d(128, self.num_classes, kernel_size=1)

        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def freeze_all_but_lora(self):
        """Compatibility API: freeze backbone and keep heads trainable."""
        for name, param in self.named_parameters():
            if 'head' not in name and 'norm' not in name and 'seg_' not in name:
                param.requires_grad = False

    def generate_cam(self, class_idx=None):
        weights = self.head.weight
        feature_map = self.final_feature_map
        cam = torch.einsum('kc,bchw->bkhw', weights, feature_map)

        cam_min = cam.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        cam_max = cam.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        if class_idx is not None:
            cam = cam[:, class_idx:class_idx + 1]
        return cam

    def forward_features(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B, C, H, W], got {x.dim()}D")

        x = self.rgb_stem(x)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for level_idx, blocks in enumerate(self.transformer_levels):
            b, h, w, c = x.shape
            x_seq = x.view(b, h * w, c)
            for block in blocks:
                x_seq = block(x_seq)
            x = x_seq.view(b, h, w, c)

            if level_idx < len(self.transformer_levels) - 1:
                x_conv = x.permute(0, 3, 1, 2)
                downsample = getattr(self, f'downsample_{level_idx}')
                x_conv = downsample(x_conv)
                x = x_conv.permute(0, 2, 3, 1)

        b, h, w, c = x.shape
        x = x.view(b, h * w, c)
        x = self.norm(x)
        self.final_feature_map = x.view(b, h, w, c).permute(0, 3, 1, 2).detach()
        return x

    def forward(self, x, pretrained_input=None, return_cam=False):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B, C, H, W], got {x.dim()}D tensor")

        b, c, h, w = x.shape
        if c != self.in_channels:
            print(
                f"WARNING: input has {c} channels, model expects {self.in_channels}."
            )

        x = self.forward_features(x)
        h_feat, w_feat = self.final_feature_map.shape[2], self.final_feature_map.shape[3]
        x = x.permute(0, 2, 1).contiguous().view(b, -1, h_feat, w_feat)

        x = self.seg_decoder(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        output = self.seg_head(x)

        if return_cam:
            return output, self.generate_cam()
        return output


class RGBViT_reduced(RGBViT):
    _defaults = dict(dim=64, depths=[2, 3, 3], num_heads=[4, 8, 16], mlp_ratio=4.0, drop_path_rate=0.2)

    def __init__(self, **kwargs):
        merged = {**self._defaults, **kwargs}
        super().__init__(**merged)


class RGBViT_tiny(RGBViT):
    _defaults = dict(dim=64, depths=[2, 2, 2], num_heads=[4, 8, 16], mlp_ratio=4.0, drop_path_rate=0.2)

    def __init__(self, **kwargs):
        merged = {**self._defaults, **kwargs}
        super().__init__(**merged)


class RGBViT_mini(RGBViT):
    _defaults = dict(dim=48, depths=[1, 1, 2], num_heads=[2, 4, 8], mlp_ratio=4.0, drop_path_rate=0.2)

    def __init__(self, **kwargs):
        merged = {**self._defaults, **kwargs}
        super().__init__(**merged)


class RGBViT_2layer(RGBViT):
    _defaults = dict(dim=32, depths=[1, 1], num_heads=[2, 4], mlp_ratio=4.0, drop_path_rate=0.2)

    def __init__(self, **kwargs):
        merged = {**self._defaults, **kwargs}
        super().__init__(**merged)
