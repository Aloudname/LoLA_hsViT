"""
Standard ViT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath


class Swish(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta
    
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class SwiGLU(nn.Module):
    """SwiGLU for MLP"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.W = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        gate = F.silu(self.w1(x))
        return self.W(self.drop(gate * self.w2(x)))

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
            # SwiGLU(mlp_hidden_dim, mlp_hidden_dim, mlp_hidden_dim),
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
    Standard, common ViT for hyperspectral image classification.
    Same API and params with LoLA are remaining.
    """
    
    def __init__(self, in_channels=15, num_classes=9, dim=96, depths=[3, 4, 5],
                 num_heads=[4, 8, 16], window_size=[7, 7, 7], mlp_ratio=4.,
                 drop_path_rate=0.2, spatial_size=15, r=16, lora_alpha=32):
        """
        Standard, common ViT for hyperspectral image classification.
        
        params:
            in_channels (int): 15 default.
            num_classes (int): 9 default.
            dim (int): dim of feature maps, 96 default.
            depths (list): num of transformer blocks, [3, 4, 5] default.
            num_heads (list): attention head for layers, [4, 8, 16] default.
            window_size (list): only for compatibility.
            mlp_ratio (float): num of MLP hidden layes, 4 default.
            drop_path_rate (float): drop path rate, 0.2 default.
            spatial_size (int): input spatial size (pixel), 15 default.
            r (int): only for compatibility.
            lora_alpha (float): only for compatibility.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dim = dim
        self.depths = depths
        self.num_heads = num_heads
        self.spatial_size = spatial_size
        
        print(f"StandardHSITransformer initialized with {in_channels} input channels, "
              f"{num_classes} classes, depths {depths}")
        
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
        
        self.patch_resolution = spatial_size // 2
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
        
        
        # clsf heads      
        self.final_feature_map = None  # for CAM
        self.norm = nn.LayerNorm(self.dims[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.dims[-1], num_classes)
        
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
        False name. This method actually freezes
        all parameters except the cls head and norm layers.
        """
        for name, param in self.named_parameters():
            if 'head' not in name and 'norm' not in name:
                param.requires_grad = False

    def merge_all_lora_into_linear(self):
        """
        Null compatible with LoLA interface.
        """
        pass

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
        
        current_spatial_size = self.patch_resolution
        
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
                current_spatial_size = current_spatial_size // 2
        
        # final norm
        B, H, W, C = x.shape
        x = x.view(B, H * W, C)
        x = self.norm(x)

        self.final_feature_map = x.view(B, H, W, C).permute(0, 3, 1, 2).detach()
        
        return x

    def forward(self, x, pretrained_input=None, return_cam=False):
        """
        Args:
            x: input [B, C, H, W]
            pretrained_input: Optional pretrained input.
            return_cam: if True, return CAM
        
        Return:
            output: classification [B, num_classes]
            or
            (output, cam) if return_cam.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B, C, H, W], got {x.dim()}D tensor")
        
        B, C, H, W = x.shape
        
        if C != self.in_channels:
            print(f"WARNING: Input has {C} channels, but model expects {self.in_channels}")
        
        x = self.forward_features(x)
        
        # global ave pooling
        x = x.permute(0, 2, 1)  # [B, C, N]
        x = self.avgpool(x)     # [B, C, 1]
        x = x.flatten(1)        # [B, C]
        
        # classification head
        output = self.head(x)
        
        if return_cam:
            cam = self.generate_cam()
            return output, cam
        
        return output


if __name__ == "__main__":
    model = CommonViT(
        in_channels=15, 
        num_classes=9, 
        dim=96, 
        depths=[3, 4, 5],
        num_heads=[4, 8, 16], 
        window_size=[7, 7, 7],
        mlp_ratio=4.,
        drop_path_rate=0.2, 
        spatial_size=15, 
        r=16, 
        lora_alpha=32
    )
    
    dummy_input = torch.randn(2, 15, 15, 15)  # [B, C, H, W]
    output = model(dummy_input)
    print(f" Output shape: {output.shape}")  # Expected: [2, 9]
    
    output_with_cam = model(dummy_input, return_cam=True)
    if isinstance(output_with_cam, tuple):
        output, cam = output_with_cam
        print(f"  CAM shape: {cam.shape}")  # Expected: [2, 9, H', W']
    
    model.freeze_all_but_lora()
    print(f"  Model parameters frozen (except head and norm)")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
