# LoLA_hsViT.py defines model + its components.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath


class Swish(nn.Module):
    """
    Leveled up to ``SwiGLU`` in next class.
    """
    def __init__(self, beta = 1):
        super().__init__()
        self.beta = beta
    
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class SwiGLU(nn.Module):
    r"""
    The activation function mainly utilized in the LoRA blocks of the model.

    ``SwiGLU(x) = Swish(W1*x) @ σ(W2*x)``, Hadamard product.

    Figure of SwiGLU seen by running as ``__main__``.
    """
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

# LoRA Linear Layer with gating and residual mechanisms
class LoRALinear(nn.Module):
    r"""
    Block conducts low rank adaptation(LoRA) for param efficient fine-tuning (PEFT).
    Can be used to replace standard linear layers as enhanced ones in models.

    This block contains:
        - a linear layer frozen during training.
        - a LoRA component of paired down and up projections, core for PEFT.

    Optional is a gating residual connection mechanism for enhanced expressiveness.
    """
    def __init__(self, in_features, out_features,
                r=16, lora_alpha=32, lora_dropout=0.1,
                bias=True, enable_gate_residual=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.enable_gate_residual = enable_gate_residual
        
        # Main linear layer (frozen during training)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.requires_grad_(False)
        
        # down and up projections with dim of down(in * r), up(r * out).
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        
        # Enhanced LoRA components
        if self.enable_gate_residual:
            self.lora_gate = nn.Linear(in_features, out_features, bias=False)
            self.lora_residual = nn.Linear(in_features, out_features, bias=False)
        
        # Scaling factor
        self.scaling = lora_alpha / r
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        if self.enable_gate_residual:
            nn.init.kaiming_uniform_(self.lora_gate.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_residual.weight, a=math.sqrt(5))

    def forward(self, x):
        # part1: Main linear transformation.
        main_output = self.linear(x)
        
        # part2: LoRA component.
        lora_output = self.lora_up(self.lora_down(x)) * self.scaling
        
        if self.enable_gate_residual:
            gate = torch.sigmoid(self.lora_gate(x))
            residual = self.lora_residual(x)

            # if gate_res: Y = (W*x + b) + σ(G*x) * scale*(B@A) + R*x
            output = main_output + gate * lora_output + residual
        else:
            # if not gate_res: Y = (W*x + b) + scale*(B@A), mergeable.
            output = main_output + lora_output
        
        return output

    def can_merge_exactly(self) -> bool:
        return self.linear is not None and not getattr(self, 'enable_gate_residual', True)

    def merge_into_linear_(self):
        r"""
        In LoRA, update weight matrix using:
            ``W_eff = W + ΔW = W + scale * (B @ A)``
        
        Therefore, merge to fold LoRA weights into base weights.
        Exact merge only if gate disabled.
        Residual (if present) is always foldable.
        """
        with torch.no_grad():
            # Merge residual if available
            if hasattr(self, 'lora_residual') and self.enable_gate_residual:
                self.linear.weight.add_(self.lora_residual.weight)
            # Merge LoRA only if mergeable
            if self.can_merge_exactly():
                # W_eff = W + scale * (B @ A)
                update = torch.matmul(self.lora_up.weight, self.lora_down.weight) * self.scaling
                self.linear.weight.add_(update)
                # After merge, disable LoRA contribution by zeroing
                self.lora_down.weight.zero_()
                self.lora_up.weight.zero_()

# Adaptive Squeeze-and-Excitation handles both 2D and 3D tensors
class AdaptiveSqueezeExcitation(nn.Module):
    r"""
    Squeeze-and-Excitation sub-block used in down-sampling blocks.

    SE is a typical channel attention mechanism to adjust channel-wise weights with:
       - Squeeze layer: ``[B, C, H, W (,D)]`` -> ``[B, C, 1, 1 (,1)]`` by pooling.
       - Excitation layer: ``[B, C, 1, 1 (,1)]`` -> ``[B, C, 1, 1 (,1)]`` by FC layers.
       - Activation: Sigmoid ``[B, C, 1, 1 (,1)]`` to get adjusted channel weights.

    By squeezing spatial dimensions, SE captures global context per channel.
    By multiplying the original input with the weights, SE enhances important channels.
    Adaptive Squeeze-and-Excitation that works with both 2D and 3D tensors.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Squeeze: Adaptive pooling layers
        self.adaptive_pool_2d = nn.AdaptiveAvgPool2d(1)
        self.adaptive_pool_3d = nn.AdaptiveAvgPool3d(1)
        
        # Excitation: FC layers
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Determine input dimensions and apply appropriate pooling
        if x.dim() == 5:  # 3D input: (batch, channel, depth, height, width)
            b, c, d, h, w = x.size()
            y = self.adaptive_pool_3d(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1, 1)
            return x * y.expand_as(x)
        elif x.dim() == 4:  # 2D input: (batch, channel, height, width)
            b, c, h, w = x.size()
            y = self.adaptive_pool_2d(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)
        else:
            raise ValueError(f"Unsupported tensor shape: {x.dim()}. Expected [B, C, H, W] or [B, C, D, H, W].")

# (Pretrained model and cross-attention modules removed for submission)

# FIXED: Enhanced Band Dropout
class BandDropout(nn.Module):
    """
    Dropout layer that randomly drops entire spectral bands (channels) during training.
    """
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

# Local(Window) Attention with LoRA-replace and relative position bias.
class PEFTWindowAttention(nn.Module):
    r"""
        Describes a window-based multi-head self attention (W-MSA) module.
        
        This local-attention module features:
            - LoRA replaced normal linear layers.
            - Relative position bias in attention score calculation.
        
        Hence, final attention is described as: 
            ``attn = Softmax( (Q@K'/sqrt(d) + B) ) @ V``

        Structure of the module:
            input ``x`` --(LoRA QKV projection)--> ``Q``, ``K``, ``V``
            ``Q @ K'/sqrt(d)`` --(Bias + mask + Softmax)--> ``attn``
            ``attn @ V`` --(LoRA projection + drop)--> output ``x``

    """
    def __init__(self, dim, window_size, num_heads,
                qkv_bias=True, qk_scale=None, attn_drop=0.,
                proj_drop=0., r=16, lora_alpha=32):
        """
        key learnable vars:
           - ``qkv``: [B\_, N, 3, num_heads, C // num_heads]
           - ``q``, ``k``, ``v``: [B\_, num_heads, N, C // num_heads], where C // num_heads = head_dim
           - ``relative_position_bias_table``: [ (2W-1)^2, num_heads], indexed to get bias B during forwarding.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Linear layer for QKV generation replaced by LoRA.
        self.qkv = LoRALinear(dim, dim * 3, r=r, lora_alpha=lora_alpha, bias=qkv_bias)
        
        # Linear layer for output generation replaced by LoRA.
        self.proj = LoRALinear(dim, dim, r=r, lora_alpha=lora_alpha)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position bias table of [(2W-1)^2, num_heads].
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window.
        coords_h = coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        r"""
        params:
        - ``x``: input features with shape of ``[B\_, N, C]``
           - ``B\_``: B\_ = B * num_Windows.
           - ``N``: tokens in a window (W^2).
           - ``C``: number of channels.
        - ``mask``: (0/-inf) mask with shape of ``(num_windows, N, N)`` or ``None``.
        """

        # get q, k, v by linear projection (LoRA replaced linear).
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # get Q@K'/sqrt(d).
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # get (Q@K'/sqrt(d) + B) as final attention score.
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Enhanced PEFT LoLA_hsViT Block.
class PEFTLoLA_hsViTBlock(nn.Module):
    r"""
        Core block of the model with residual-like structure.
        
        This block contains:
              - LayerNorm layers for normalization.
              - MLP with LoRA as stick-and-stuff layer.
              - residual-like connections and DropPath for regularization.
              - Local window attention with LoRA and relative position bias.
        Takes input ``x`` of [B, H, W, C], outputs same shape.

        Structure of the module:
            input ``x`` --(LayerNorm + win_attn)--> ``x_win_attn``
            ``x_win_attn`` --( +x )--> ``x_buffer``
            ``x_buffer`` --(LayerNorm + MLP)--> ``x_mlp``
            output ``x + x_mlp``
    """
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., 
                 act_layer=Swish, norm_layer=nn.LayerNorm, r=16, lora_alpha=32):  # Fixed LoRA rank to 16
        """
        Args:
            ``dim``: Number of input channels.
            ``num_heads``: Number of attention heads.
            ``window_size``: Size of the local attention window.
            ``mlp_ratio``: Ratio for MLP hidden dimension.
            ``qkv_bias``: If True, add bias to QKV projections.
            ``qk_scale``: Optional scaling factor for QK.
            ``drop``: Dropout rate after MLP and attention.
            ``attn_drop``: Dropout rate within attention.
            ``drop_path``: Stochastic depth rate.
            ``act_layer``: Activation function for MLP.
            ``norm_layer``: Normalization layer.
        """
        
        super().__init__()
        self.window_size = window_size
        self.dim = dim
        
        # All normalization layers.
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # Local attention with LoRA and relative position bias.
        self.attn = PEFTWindowAttention(
            dim, num_heads=num_heads, window_size=window_size,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
            r=r, lora_alpha=lora_alpha
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        # MLP with LoRA as stick-and-stuff layer.
        self.mlp = nn.Sequential(
            LoRALinear(dim, mlp_hidden_dim, r=r, lora_alpha=lora_alpha),
            SwiGLU(mlp_hidden_dim, mlp_hidden_dim, mlp_hidden_dim),
            nn.Dropout(drop),
            LoRALinear(mlp_hidden_dim, dim, r=r, lora_alpha=lora_alpha),
            nn.Dropout(drop)
        )

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x
        
        # Apply layer norm
        x = x.view(B, H * W, C)
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Window partition implemented in class EnhancedReduceSize.
        x_windows, (H_padded, W_padded) = window_partition(x, self.window_size)
        
        # Window attention
        attn_windows = self.attn(x_windows)
        
        # Reverse window partition
        x = window_reverse(attn_windows, self.window_size, H, W)
        
        # FFN with residual-like connection
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x.view(B, H * W, C))).view(B, H, W, C))
        
        return x

# ReduceSize module with SE used in down-sampling block.
class EnhancedReduceSize(nn.Module):
    r"""
        Down-sampling module that reduces spatial dimensions by half.

        This block contains:
            - Swish activation for non-linearity.
            - Pointwise Conv2d for channel adjustment.
            - Depthwise Conv2d for spatial feature extraction.
            - Adaptive Squeeze-and-Excitation for channel attention.

        In this block, parameter of the model is reduced and limited due to:
            - Conv2d to reduce the size before the next LoLA_hsViT Block.
            - Squeeze-Excitation mechanism to screen important channel.

        Structure of the module:
            feature extract: input ``x`` --(depthwise conv + Swish + SE + pointwise conv)--> ``x_conv``
            down-sampling: ``x_conv`` --(BatchNorm + reduction)--> output ``x_reduced``
        Down-sampling outcome:
            ``x`` [B, C, H, W] -> ``x_reduced`` [B, C or 2C, H/2, W/2]
    """

    def __init__(self, dim, keep_dim=False):
        r"""
        Args:
            ``dim``: dim of input var.
            ``keep_dim``: boolean to keep input dim from down-sampling.
        """
        super().__init__()
        self.keep_dim = keep_dim
        dim_out = dim if keep_dim else dim * 2
        
        # Only for feature extraction.
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            Swish(),  # Swish activate for Conv2d.
            AdaptiveSqueezeExcitation(dim),  # Using SE for channel-wise attention.
            nn.Conv2d(dim, dim_out, 1, 1, 0, bias=False),
        )

        # Dim reduction.
        self.norm = nn.BatchNorm2d(dim_out)
        self.reduction = nn.Conv2d(dim_out, dim_out, 3, 2, 1, bias=False) if not keep_dim else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x

# Window partition and reverse functions.
def window_partition(x, window_size):
    r"""
    Core function of local attention.
    ``H,W`` may not be integer factors of ``window_size``,
    hence partition without padding may encounter overlapping windows.
    Partition into non-overlapping windows with padding if needed.
    Args:
        ``x``: [B, H, W, C]
        ``window_size``: window size
    Returns:
        ``windows``: [B*num_windows, window_size*window_size, C]
        ``(Hp, Wp)``: padded height and width
    """
    B, H, W, C = x.shape
    
    # Calculate padding. pad = 0 if H, W is integer factor of window_size.
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    
    # Updated dimensions after padding.
    Hp, Wp = H + pad_h, W + pad_w
    
    # Ensure dimensions are valid.
    if Hp < window_size or Wp < window_size:
        pad_h = max(0, window_size - Hp)
        pad_w = max(0, window_size - Wp)
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        # padded windows parameter.
        Hp, Wp = H + pad_h, W + pad_w
    
    # Reshape to windows.
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size * window_size, C)
    
    return windows, (Hp, Wp)

def window_reverse(windows, window_size, H, W):
    r"""
    Inverse function of window_partition.
    Remove padding to reverse window partition.
    Args:
        ``windows``: [B*num_windows, window_size*window_size, C]
        ``window_size``: Window size
        ``H,W``: Original height and width
    Returns:
        ``x``: [B, H, W, C]
    """
    # Calculate number of windows in each dimension
    num_h = (H + window_size - 1) // window_size
    num_w = (W + window_size - 1) // window_size
    
    # Calculate batch size
    B = windows.shape[0] // (num_h * num_w)
    C = windows.shape[2]
    
    # Reshape back to window grid
    x = windows.view(B, num_h, num_w, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    
    # Reshape to original size
    x = x.view(B, num_h * window_size, num_w * window_size, C)
    
    # Remove padding if necessary
    if x.shape[1] > H or x.shape[2] > W:
        x = x[:, :H, :W, :]
    
    return x

# Final implementation of LoLA_hsViT.
class LoLA_hsViT(nn.Module):
    r"""
    LoLA_hsViT model for hyperspectral image classification with:
        - SE for channel attention.
        - Band Dropout for robustness.
        - LoRA in attention and MLP layers.
        - Local Attention for more efficient.
        - Note: Without cross-attention.

    See parameters initialization in ``__init__`` method.

    Structure of the module:
        (1) Feature extract of ``[B, C, H, W] -> [B, dim, H, W]``:
                input ``x`` --( Conv3d + BN3d + Swish ) *3 --> ``X'``
                ``X'`` --( BandDropout + SE + Avepool(C) ) --> ``X_1``

        (2) Patch & positional embedding of ``[B, dim, H, W] -> [B, HW/4, dim]``:
                ``X_1`` --( Conv2d + BN2d + Swish )--> ``patch_embedded_X``
                ``patch_embedded_X`` + ( ``zeros[1, H, W, C]`` )--> ``pos_embedded_X``
                ``pos_embedded_X`` --( Dropout )--> ``X_2``

        (3) Main LAViT Blocks of ``[B, HW/4, dim] -> [B, HW/64, 4dim]``:
                ``X_2`` --( LAViT *depth[0] + down-sampling )--> ``X' ``
                ``X' `` --( LAViT *depth[1] + down-sampling )--> ``X''``
                ``X''`` --( LAViT *depth[2] )--> ``X_3``

        (4) Classification Head of ``[B, HW/64, 4dim] -> [B, K]``:
                ``X_3`` --(Flatten + LN + AvePool + LoRA)--> output ``Y``
    """
    def __init__(self, in_channels=15, num_classes=9, dim=96, depths=[3, 4, 5],
                 num_heads=[4, 8, 16], window_size=[7, 7, 7], mlp_ratio=4.,
                 drop_path_rate=0.2, spatial_size=15, r=16, lora_alpha=32):
        """
        Args:
            ``in_channels``: Number of input channels (spectral bands).
            ``num_classes``: Number of output classes for classification.
            ``dim``: Base dimension for model.
            ``depths``: List of depths for each LoLA_hsViT block.
            ``num_heads``: List of number of attention heads for each LoLA_hsViT block.
            ``window_size``: List of window sizes for each LoLA_hsViT block.
            ``mlp_ratio``: Ratio for MLP hidden dimension.
            ``drop_path_rate``: Stochastic depth rate.
            ``spatial_size``: Spatial size.
            ``r``: LoRA rank.
            ``lora_alpha``: LoRA scaling factor.
        """
        
        super().__init__()
        self.in_channels = in_channels
        print(f"Model initialized with {in_channels} input channels and {depths} depths.")
        
        # Block1: Spectral processing.
        self.spectral_conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(7,3,3), padding=(3,1,1)),
            nn.BatchNorm3d(32),
            Swish(),
            nn.Conv3d(32, 64, kernel_size=(5,3,3), padding=(2,1,1)),
            nn.BatchNorm3d(64),
            Swish(),
            nn.Conv3d(64, dim, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(dim),
            Swish()
        )
        
        self.band_dropout = BandDropout(drop_rate=0.1)
        self.spectral_attention = AdaptiveSqueezeExcitation(dim)
        
        # Store LoRA layers for CLR updates
        self.lora_layers = []
        

        # Block2: Patch embedding.
        self.patch_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            Swish()
        )
        
        # Position embedding
        self.patch_resolution = spatial_size // 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_resolution, 
                                                 self.patch_resolution, dim))
        self.pos_drop = nn.Dropout(p=0.1)
        
        # Track dimensions through network
        self.dims = [dim]
        for i in range(1, len(depths)):
            self.dims.append(self.dims[-1] * 2)
        

        # Main part: LAViT backbone with PEFT and self-attention.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        curr_idx = 0
        
        for i in range(len(depths)):
            # Create downsample layer if not last level
            downsample = None if i == len(depths) - 1 else EnhancedReduceSize(self.dims[i])
            
            # Create blocks for current LEVEL.
            level = nn.ModuleList()
            for j in range(depths[i]):
                block = PEFTLoLA_hsViTBlock(
                    dim=self.dims[i],
                    num_heads=num_heads[i],
                    window_size=window_size[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=dpr[curr_idx + j],
                    norm_layer=nn.LayerNorm,
                    r=r,
                    lora_alpha=lora_alpha
                )
                level.append(block)
            curr_idx += depths[i]
            
            # Add downsampling if needed
            if downsample is not None:
                level.append(downsample)
            
            self.levels.append(level)
        
        # layers in final block.
        # register a buffer of final feature map for CAM.
        self.final_feature_map = None
        self.norm = nn.LayerNorm(self.dims[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = LoRALinear(
            self.dims[-1], num_classes, r=r, 
            lora_alpha=lora_alpha, enable_gate_residual=False)
        self.lora_layers.append(self.head)
        
        
        # Initialize weights
        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        
        # Collect all LoRA layers for CLR updates
        self._collect_lora_layers()
        
        print(f"LoLA_hsViT initialized.")

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, LoRALinear)):
            if isinstance(m, LoRALinear):
                trunc_normal_(m.linear.weight, std=.02)
                if m.linear.bias is not None:
                    nn.init.constant_(m.linear.bias, 0)
            else:
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _collect_lora_layers(self):
        """Collect all LoRA layers for CLR updates"""
        self.lora_layers = []
        for module in self.modules():
            if isinstance(module, LoRALinear):
                self.lora_layers.append(module)

    def freeze_all_but_lora(self):
        # First, freeze everything
        for p in self.parameters():
            p.requires_grad_(False)

        # Then, enable only LoRA adapter parameters (keep base linear frozen)
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.linear.requires_grad_(False)
                for p in module.lora_down.parameters():
                    p.requires_grad_(True)
                for p in module.lora_up.parameters():
                    p.requires_grad_(True)
                if hasattr(module, 'lora_gate'):
                    for p in module.lora_gate.parameters():
                        p.requires_grad_(True)
                if hasattr(module, 'lora_residual'):
                    for p in module.lora_residual.parameters():
                        p.requires_grad_(True)

    def merge_all_lora_into_linear(self):
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.merge_into_linear_()
                
    def update_lora_scale(self, factor):
        """Update scaling factor for all LoRA layers based on CLR cycle"""
        for layer in self.lora_layers:
            if hasattr(layer, 'set_cycle_factor'):
                layer.set_cycle_factor(factor)

    def generate_cam(self, class_idx=None):
        """
        Generate Class Activation Map (CAM) for the last forward pass.
        
        Args:
            class_idx: Index of target class (``None`` for all classes).
        
        Returns:
            cam: Tensor of shape [B, num_classes, H, W] (or [B, 1, H, W] if ``class_idx`` is specified)
        """
        # Get weights from classification head
        weights = self.head.linear.weight  # [num_classes, C]
        
        # Feature map shape: [B, C, H, W]
        feature_map = self.final_feature_map
        B, C_feat, H, W = feature_map.shape    # C_feat = feature channels(384).
        C_weights, _ =weights.shape  # C_weights = num_classes(9).
        
        # Compute CAM: weights @ feature_map -> [B, num_classes, H, W]
        cam = torch.einsum('kc, bchw -> bkhw', weights, feature_map)  # Matrix multiplication over channels
        
        # Normalize CAM to [0, 1] per class
        cam_min = cam.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        cam_max = cam.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        cam = (cam - cam_min) / (cam - cam_min + 1e-8)
        
        # Return specific class if requested.
        if class_idx is not None:
            cam = cam[:, class_idx:class_idx+1]  # [B, 1, H, W]
        return cam

    def forward_features(self, x):
        """
        Forward pass with proper tensor handling.
        Expected input: [B, C, H, W] where C is the number of channels.
        """
        # FIXED: Handle input tensor properly for hyperspectral data
        if x.dim() == 4:  # [B, C, H, W]
            B, C, H, W = x.shape
            # Reshape to add spectral dimension: [B, 1, C, H, W] for 3D conv
            x = x.unsqueeze(1)  # [B, 1, C, H, W]
            
        # Enhanced spectral processing with 3D convolution
        x = self.spectral_conv(x)  # [B, dim, D, H, W]
        
        x = self.band_dropout(x)
        x = self.spectral_attention(x)  # Now uses adaptive attention
        
        # Average over spectral dimension -> [B, dim, H, W]
        x = x.mean(dim=2)
        

        # Patch embedding
        x = self.patch_embed(x)  # [B, dim, H/2, W/2]
        x = x.permute(0, 2, 3, 1)  # [B, H/2, W/2, dim]
        
        # Add position embedding and dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Process through LoLA_hsViT backbone.
        for i, level in enumerate(self.levels):
            for j, block in enumerate(level):
                if isinstance(block, EnhancedReduceSize):
                    x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
                    x = block(x)
                    x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
                else:
                    x = block(x)
        
        # Final norm
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = self.norm(x)

        # Reshape back to spatial dims as final feature map.
        self.final_feature_map = x.reshape(B, H, W, C).permute(0, 3, 1, 2).detach()  # [B, C, H, W]
        return x

    def forward(self, x, pretrained_input=None, return_cam=False):
        """
        FIXED: Forward pass with proper input validation for hyperspectral data
        """
        # Validate input shape
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B, C, H, W], got {x.dim()}D tensor with shape {x.shape}")
        
        B, C, H, W = x.shape
        
        # Check if input channels match expected
        if C != self.in_channels:
            print(f"WARNING: Input has {C} channels, but model expects {self.in_channels}")
            print("This might cause issues if the number of channels is significantly different.")
        
        # Original LoLA_hsViT forward pass
        x = self.forward_features(x)
        
        # Global pooling
        x = x.permute(0, 2, 1)  # [B, C, N]
        x = self.avgpool(x)  # [B, C, 1]
        x = x.flatten(1)  # [B, C]
        
        # Original classification path (pretrained and cross-attention removed)
        output = self.head(x)

        if return_cam:
            cam = self.generate_cam()
            return output, cam
        return output

if __name__ == "__main__":
    # Example usage and testing of the model and scheduler.
    model = LoLA_hsViT(in_channels=15, num_classes=9, dim=96, depths=[3, 4, 5],
                       num_heads=[4, 8, 16], window_size=[7, 7, 7], mlp_ratio=4.,
                       drop_path_rate=0.2, spatial_size=15, r=16, lora_alpha=32)
    
    # Create dummy input for testing
    dummy_input = torch.randn(2, 15, 15, 15)  # [B, C, H, W]
    
    # Test forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Expected: [B, num_classes]
    
    # Test CAM generation
    output_with_cam = model(dummy_input, return_cam=True)
    if isinstance(output_with_cam, tuple):
        output, cam = output_with_cam
        print(f"CAM shape: {cam.shape}")  # Expected: [B, num_classes, H', W']