"""
Standard ViT.
"""

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


class AdaptiveSqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        self.adaptive_pool_2d = nn.AdaptiveAvgPool2d(1)
        self.adaptive_pool_3d = nn.AdaptiveAvgPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 5:  # 3D: (B, C, D, H, W)
            b, c, d, h, w = x.size()
            y = self.adaptive_pool_3d(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1, 1)
            return x * y.expand_as(x)
        elif x.dim() == 4:  # 2D: (B, C, H, W)
            b, c, h, w = x.size()
            y = self.adaptive_pool_2d(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)
        else:
            raise ValueError(f"Unsupported tensor shape: {x.dim()}")


class BandDropout(nn.Module):
    """Dropout for spectral bands C"""
    def __init__(self, drop_rate=0.1):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if not self.training or self.drop_rate == 0:
            return x
            
        if x.dim() == 5:  # (b, c, d, h, w)
            b, c, d, h, w = x.shape
            mask = torch.bernoulli(torch.ones(b, c, d, 1, 1, device=x.device) * (1 - self.drop_rate))
        elif x.dim() == 4:  # (b, c, h, w)
            b, c, h, w = x.shape
            mask = torch.bernoulli(torch.ones(b, c, 1, 1, device=x.device) * (1 - self.drop_rate))
        else:
            return x
        
        x = x * mask / (1 - self.drop_rate)
        return x


class StandardMultiHeadAttention(nn.Module):
    """
    Standard multi-head self-attention without windowing.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    """
    Standard Transformer
    
    Includes:
    - LayerNorm + multihead self-attention
    - LayerNorm + MLP
    - residual connection + DropPath
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., 
                norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = StandardMultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CommonViT(nn.Module):
    """
    Standard, common ViT for hyperspectral image pixel-level segmentation.
    Same API and params with LoLA are remaining.
    """
    
    def __init__(self, in_channels=None, num_classes=None, patch_size=None, dim=96, depths=[3, 4, 5],
                 num_heads=[4, 8, 16], mlp_ratio=4., drop_path_rate=0.2,
                 window_size=None, r=None, lora_alpha=None,
                 hierarchical_head=True,
                 eval_fg_gate_threshold: float = -1.0):
        """
        Standard, common ViT for hyperspectral image pixel-level segmentation.
        in_channel num_classes and patch_size are necessary.
        
        params:
            in_channels (int): 15 default.
            num_classes (int): 8 default.
            dim (int): dim of feature maps, 96 default.
            depths (list): num of transformer blocks, [3, 4, 5] default.
            num_heads (list): attention head for layers, [4, 8, 16] default.
            window_size (list): only for compatibility, ignored.
            mlp_ratio (float): num of MLP hidden layes, 4 default.
            drop_path_rate (float): drop path rate, 0.2 default.
            patch_size (int): input spatial size (pixel), 15 default.
            r (int): only for compatibility, ignored.
            lora_alpha (float): only for compatibility, ignored.
        """
        super().__init__()
        # Note: window_size, r, lora_alpha are accepted for API compatibility but ignored
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dim = dim
        self.depths = depths
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.mode = 'segmentation'  # pixel-level only
        self.hierarchical_head = True
        self.eval_fg_gate_threshold = float(eval_fg_gate_threshold)
        
        print(f"StandardHSITransformer initialized with {in_channels} input channels, "
              f"{num_classes} classes, depths {depths}, {patch_size} patch_size.")
        
        # preprocessing layers.
        self.spectral_conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True)
        )
        
        self.band_dropout = BandDropout(drop_rate=0.1)
        self.spectral_attention = AdaptiveSqueezeExcitation(dim)
        
        
        # patch & pos embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        self.patch_resolution = patch_size // 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_resolution,
                                                   self.patch_resolution, dim))
        self.pos_drop = nn.Dropout(p=0.1)
        
        
        # transformer main blocks
        self.dims = [dim * (2 ** i) for i in range(len(depths))]
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.transformer_levels = nn.ModuleList()
        curr_dpr_idx = 0
        
        for level_idx in range(len(depths)):
            current_dim = self.dims[level_idx]
            current_depth = depths[level_idx]
            current_num_heads = num_heads[level_idx]
            
            blocks = nn.ModuleList()
            for block_idx in range(current_depth):
                drop_path_prob = dpr[curr_dpr_idx]
                curr_dpr_idx += 1
                
                block = TransformerBlock(
                    dim=current_dim,
                    num_heads=current_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=0.1,
                    attn_drop=0.1,
                    drop_path=drop_path_prob,
                    norm_layer=nn.LayerNorm
                )
                blocks.append(block)
            
            self.transformer_levels.append(blocks)
            
            # down-sampling except for the last level
            if level_idx < len(depths) - 1:
                # down-sampling by 2x2 conv
                self.register_module(
                    f'downsample_{level_idx}',
                    nn.Sequential(
                        nn.Conv2d(current_dim, self.dims[level_idx + 1], 
                                kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(self.dims[level_idx + 1]),
                        nn.ReLU(inplace=True)
                    )
                )
        
        # Kept for CAM generation compatibility
        self.final_feature_map = None  # for CAM
        self.norm = nn.LayerNorm(self.dims[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.dims[-1], num_classes)
        
        # Segmentation decoder: upsample from final feature map to input resolution
        self.seg_decoder = nn.Sequential(
            nn.Conv2d(self.dims[-1], 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # 2-stage segmentation heads:
        # 1) fg_bg_head predicts BG/FG logits (2 channels)
        # 2) fg_class_head predicts foreground subclass logits (num_classes - 1)
        # Fused output remains [B, num_classes, H, W].
        if self.num_classes <= 1:
            raise ValueError("CommonViT requires num_classes > 1 for two-stage segmentation supervision")
        
        # 2-stage fg/bg -> cls heads.
        self.fg_bg_head = nn.Conv2d(128, 2, kernel_size=1)
        self.fg_class_head = nn.Conv2d(128, num_classes - 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(128, 1, kernel_size=1)
        self.aux_fg_bg_head = nn.Conv2d(self.dims[-1], 2, kernel_size=1)
        self.aux_fg_class_head = nn.Conv2d(self.dims[-1], num_classes - 1, kernel_size=1)
        self.seg_head = None
        
        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        
        print(f"StandardHSITransformer initialized successfully.")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def freeze_all_but_lora(self):
        """
        Freeze all parameters except the heads, norm layers, and seg decoder/head.
        """
        for name, param in self.named_parameters():
            if 'head' not in name and 'norm' not in name and 'seg_' not in name:
                param.requires_grad = False

    def generate_cam(self, class_idx=None):
        """
        Return:
            ``cam`` of [B, num_classes, H, W] or [B, 1, H, W] if class_idx
        """
        # weights of classification heads: [num_classes, C]
        weights = self.head.weight  
        
        # shape of last feat. map: [B, C, H, W]
        feature_map = self.final_feature_map
        B, C_feat, H, W = feature_map.shape
        
        # CAM: weights @ feature_map -> [B, num_classes, H, W]
        cam = torch.einsum('kc, bchw -> bkhw', weights, feature_map)
        
        # norm to [0, 1]
        cam_min = cam.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        cam_max = cam.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        if class_idx is not None:
            cam = cam[:, class_idx:class_idx + 1]
        
        return cam

    def forward_features(self, x):
        """        
        input shape: [B, C, H, W]
        """
        if x.dim() == 4:  # [B, C, H, W]
            B, C, H, W = x.shape
            x = x.unsqueeze(1)  # [B, 1, C, H, W]
        
        x = self.spectral_conv(x)  # [B, dim, D, H, W]
        x = self.band_dropout(x)
        x = self.spectral_attention(x)
        
        # ave pooling to dim.C -> [B, dim, H, W]
        x = x.mean(dim=2)
        
        x = self.patch_embed(x)  # [B, dim, H/2, W/2]
        x = x.permute(0, 2, 3, 1)  # [B, H/2, W/2, dim]
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for level_idx, blocks in enumerate(self.transformer_levels):
            B, H, W, C = x.shape
            
            # reshape to [B, N, C]
            x_seq = x.view(B, H * W, C)
            
            # block stacking
            for block in blocks:
                x_seq = block(x_seq)
            
            # reshape to [B, H, W, C]
            x = x_seq.view(B, H, W, C)
            
            # downsampling except for the last level
            if level_idx < len(self.transformer_levels) - 1:
                # to [B, C, H, W] for downsampling
                x_conv = x.permute(0, 3, 1, 2)
                downsample = getattr(self, f'downsample_{level_idx}')
                x_conv = downsample(x_conv)  # [B, 2*C, H/2, W/2]
                
                # to [B, H', W', 2*C]
                x = x_conv.permute(0, 2, 3, 1)
        
        # final norm
        B, H, W, C = x.shape
        x = x.view(B, H * W, C)
        x = self.norm(x)

        self.final_feature_map = x.view(B, H, W, C).permute(0, 3, 1, 2).detach()
        
        return x

    def forward(self, x, pretrained_input=None, return_cam=False):
        """
        Pixel-level segmentation forward pass.
        
        Args:
            x: input [B, C, H, W]
            pretrained_input: Optional pretrained input (unused, kept for API compat).
            return_cam: if True, also return CAM as second element.
        
        Return:
            [B, num_classes, H, W] dense per-pixel logits.
            If return_cam: (logits, cam)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B, C, H, W], got {x.dim()}D tensor")
        
        B, C, H, W = x.shape
        
        if C != self.in_channels:
            print(f"\nWARNING: Input has {C} channels, but model expects {self.in_channels}.\n"
                  "This usually caused by incorrect data shape, which requires [B, C, H, W]. \n")
        
        x = self.forward_features(x)  # [B, N, C_final]
        
        # Reshape to spatial for gradient flow: [B, C_final, H_feat, W_feat]
        H_feat, W_feat = self.final_feature_map.shape[2], self.final_feature_map.shape[3]
        # .contiguous() required to avoid CUDA misaligned address under AMP + DataParallel
        x = x.permute(0, 2, 1).contiguous().view(B, -1, H_feat, W_feat)
        aux_fg_bg_logits = self.aux_fg_bg_head(x)
        aux_fg_class_logits = self.aux_fg_class_head(x)

        x = self.seg_decoder(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        fg_bg_logits = self.fg_bg_head(x)          # [B, 2, H, W], [bg, fg]
        fg_class_logits = self.fg_class_head(x)    # [B, C-1, H, W], foreground subclasses
        boundary_logits = self.boundary_head(x)    # [B, 1, H, W]

        # Conditional fusion: P(class)=P(FG)*P(class|FG), P(BG)=1-P(FG).
        p_bgfg = F.softmax(fg_bg_logits, dim=1)
        p_bg = p_bgfg[:, 0:1, :, :]
        p_fg = p_bgfg[:, 1:2, :, :]
        p_fg_cond = F.softmax(fg_class_logits, dim=1)
        if (not self.training) and (self.eval_fg_gate_threshold >= 0.0):
            fg_mask = (p_fg >= self.eval_fg_gate_threshold).float()
            p_fg = p_fg * fg_mask
        p_fg_cls = p_fg_cond * p_fg
        probs = torch.cat([p_bg, p_fg_cls], dim=1)
        output = torch.log(probs + 1e-8)
        staged_output = {
            'fused_logits': output,
            'fg_bg_logits': fg_bg_logits,
            'fg_class_logits': fg_class_logits,
            'boundary_logits': boundary_logits,
            'aux_fg_bg_logits': aux_fg_bg_logits,
            'aux_fg_class_logits': aux_fg_class_logits,
        }
        
        if return_cam:
            cam = self.generate_cam()
            return staged_output, cam
        
        return staged_output


class CommonViT_reduced(CommonViT):
    """Reduced CommonViT: dim=64, depths=[2,3,3]."""
    _defaults = dict(dim=64, depths=[2, 3, 3], num_heads=[4, 8, 16], mlp_ratio=4., drop_path_rate=0.2)

    def __init__(self, **kwargs):
        merged = {**self._defaults, **kwargs}
        super().__init__(**merged)


class CommonViT_tiny(CommonViT):
    """Tiny CommonViT: dim=64, depths=[2,2,2]."""
    _defaults = dict(dim=64, depths=[2, 2, 2], num_heads=[4, 8, 16], mlp_ratio=4., drop_path_rate=0.2)

    def __init__(self, **kwargs):
        merged = {**self._defaults, **kwargs}
        super().__init__(**merged)


class CommonViT_mini(CommonViT):
    """Mini CommonViT: dim=48, depths=[1,1,2]."""
    _defaults = dict(dim=48, depths=[1, 1, 2], num_heads=[2, 4, 8], mlp_ratio=4., drop_path_rate=0.2)

    def __init__(self, **kwargs):
        merged = {**self._defaults, **kwargs}
        super().__init__(**merged)


class CommonViT_2layer(CommonViT):
    """2-layer CommonViT: dim=32, depths=[1,1]."""
    _defaults = dict(dim=32, depths=[1, 1], num_heads=[2, 4], mlp_ratio=4., drop_path_rate=0.2)

    def __init__(self, **kwargs):
        merged = {**self._defaults, **kwargs}
        super().__init__(**merged)

if __name__ == "__main__":
    model = CommonViT(
        in_channels=15, num_classes=8, dim=96, depths=[3, 4, 5],
        num_heads=[4, 8, 16], window_size=[7, 7, 7], mlp_ratio=4.,
        drop_path_rate=0.2, patch_size=15, r=16, lora_alpha=32
    )
    
    dummy_input = torch.randn(2, 15, 15, 15)  # [B, C, H, W]
    output = model(dummy_input)
    print(f"  Output shape: {output.shape}")  # Expected: [2, 8, 15, 15]
    
    output_with_cam = model(dummy_input, return_cam=True)
    if isinstance(output_with_cam, tuple):
        output, cam = output_with_cam
        print(f"  CAM shape: {cam.shape}")
    
    model.freeze_all_but_lora()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
