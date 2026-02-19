# common_vit.py defines model + its components.
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

# Final implementation of CommonViT.
class CommonViT(nn.Module):
    r"""
    CommonViT model for hyperspectral image classification.
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
        self.register_buffer('final_feature_map', torch.zeros(1))
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