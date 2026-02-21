"""
U-net.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        self.adaptive_pool_2d = nn.AdaptiveAvgPool2d(1)
        self.adaptive_pool_3d = nn.AdaptiveAvgPool3d(1)
        
        self.fc = nn.Sequential(
            # Squeeze
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            
            # Excitation
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
    

class Unet(nn.Module):
    """
    Hyperspectral imaging U-net.
    Same API and params with LoLA are remaining.
    """
    
    def __init__(self, in_channels=15, num_classes=9, dim=96, 
                 spatial_size=15):
        """
        Hyperspectral U-net for classification.
        
        params:
            in_channels (int): 15 default.
            num_classes (int): 9 default.
            dim (int): dim of feature maps, 96 default.
            spatial_size (int): input spatial size (pixel), 15 default.
            r (int): only for compatibility.
            lora_alpha (float): only for compatibility.
            depths (list): only for compatibility.
            num_heads (list): only for compatibility.
            window_size (list): only for compatibility.
            mlp_ratio (float): only for compatibility.
            drop_path_rate (float): only for compatibility.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dim = dim
        self.spatial_size = spatial_size
        
        print(f"U-net initialized with {in_channels} input channels, "
              f"{num_classes} classes, dim {dim}, spatial_size {spatial_size}")
        
        # preprocessing layers.
        self.encode_conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.SEBlock1 = SEBlock(64)
        self.max_pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.encode_conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.SEBlock2 = SEBlock(128)
        self.max_pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.encode_conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        self.SEBlock3 = SEBlock(256)
        self.max_pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.encode_conv4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU()
        )
        self.SEBlock4 = SEBlock(512)
        self.max_pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.plain_conv = nn.Sequential(
            nn.Conv3d(512, 1024, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(1024),
            nn.ReLU(),
            nn.Conv3d(1024, 1024, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(1024),
            nn.ReLU()
        )
        
        self.decode_conv4 = nn.Sequential(
            nn.Conv3d(1024, 512, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU()
        )
        
        self.decode_conv3 = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        
        self.decode_conv2 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        
        self.decode_conv1 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        
        # classification head
        self.final_feature_map = None  # for CAM
        self.norm = nn.LayerNorm(64)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(64, num_classes)
        
        self.apply(self._init_weights)
        
        print(f"U-net initialized successfully.")

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
        Freeze all parameters except the classification head and norm layers.
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
            x = x.unsqueeze(2)  # [B, C, 1, H, W]
        
        B, C, D, H, W = x.shape
        x = self.encode_conv1(x)
        x = self.SEBlock1(x)
        x = self.max_pool1(x)
        
        x = self.encode_conv2(x)
        x = self.SEBlock2(x)
        x = self.max_pool2(x)
        
        x = self.encode_conv3(x)
        x = self.SEBlock3(x)
        x = self.max_pool3(x)
        
        x = self.encode_conv4(x)
        x = self.SEBlock4(x)
        x = self.max_pool4(x)
        
        x = self.plain_conv(x)
        
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.decode_conv4(x)
        
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.decode_conv3(x)
        
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.decode_conv2(x)
        
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.decode_conv1(x)
        
        # ave pooling -> [B, 64, 1, 1, 1]
        x_pooled = F.adaptive_avg_pool3d(x, (1, 1, 1))
        # for CAM: [B, 64, H, W]
        self.final_feature_map = x.reshape(B, H, W, C).permute(0, 3, 1, 2).detach()
        
        return x_pooled

    def forward(self, x, return_cam=True):
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
        
        # flatten
        x = x.view(B, -1)
        
        # classification head
        output = self.head(x)
        
        if return_cam:
            cam = self.generate_cam()
            return output, cam
        
        return output


if __name__ == "__main__":
    model = Unet(
        in_channels=15, 
        num_classes=9, 
        spatial_size=15
    )
    
    dummy_input = torch.randn(2, 15, 15, 15)  # [B, C, H, W]
    output = model(dummy_input)
    print(f"  Output shape: {output.shape}")  # Expected: [2, 9]
    
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