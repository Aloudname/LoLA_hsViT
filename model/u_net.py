"""
U-net.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_



class DoubleConv(nn.Module):
    """(Conv3D -> IN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1,bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# down-sampling
class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), ceil_mode=True),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)

# up-sampling
class Up(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear = True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor = 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size = 2, stride = 2)

        self.conv = DoubleConv(in_channels, out_channels)
        self.downc = nn.Conv3d(in_channels, out_channels, kernel_size = 3, stride=1, padding=1, bias=True)
        self.downr = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x1 = self.downr(self.downc(x1))

        x = torch.cat([x2, x1], dim = 1)
        return self.conv(x)

# Output layer
class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)
    

class Unet(nn.Module):
    """
    Hyperspectral imaging U-net for pixel-level segmentation.
    Same API and params with LoLA are remaining.
    """
    
    def __init__(self, in_channels=15, num_classes=9, dim=64):
        """
        U-net for hyperspectral image pixel-level segmentation.
        
        params:
            in_channels (int): 15 default.
            num_classes (int): 9 default.
            dim (int): dim of feature maps, 64 default.
        """
        
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dim = dim
        self.mode = 'segmentation'  # pixel-level only
        
        print(f"U-net initialized with {in_channels} input channels, "
              f"{num_classes} classes")
        
        
        self.in_conv = DoubleConv(in_channels, dim)  # [B, dim, D, H, W]
        
        self.encoder_conv1 = Down(dim, dim * 2)  # [B, 2*dim, D/2, H/2, W/2]
        self.encoder_conv2 = Down(dim * 2, dim * 4)  # [B, 4*dim, D/4, H/4, W/4]
        self.encoder_conv3 = Down(dim * 4, dim * 8)  # [B, 8*dim, D/8, H/8, W/8]
        self.encoder_conv4 = Down(dim * 8, dim * 16)  # [B, 16*dim, D/16, H/16, W/16]
        
        self.decoder_conv4 = Up(dim * 16, dim * 8)  # [B, 8*dim, D/8, H/8, W/8]
        self.decoder_conv3 = Up(dim * 8, dim * 4)  # [B, 4*dim, D/4, H/4, W/4]
        self.decoder_conv2 = Up(dim * 4, dim * 2)  # [B, 2*dim, D/2, H/2, W/2]
        self.decoder_conv1 = Up(dim * 2, dim)  # [B, dim, D, H, W]
        self.out_conv = Out(dim, num_classes)  # [B, num_classes, D, H, W]
        
        # Kept for CAM generation compatibility
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(dim, num_classes)
        
        # Segmentation head: 1x1 conv for per-pixel prediction
        self.seg_head = nn.Conv2d(dim, num_classes, kernel_size=1)
        
        # useful components
        self.final_feature_map = None  # for CAM
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
        Null compatible with LoLA interface.
        """
        pass

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
            x = x.unsqueeze(2)  # [B, C, 1, H, W] - add depth dimension for Conv3d
            
        x1 = self.in_conv(x)  # [B, dim, D, H, W]
        x2 = self.encoder_conv1(x1)  # [B, 2*dim, D/2, H/2, W/2]
        x3 = self.encoder_conv2(x2)  # [B, 4*dim, D/4, H/4, W/4]
        x4 = self.encoder_conv3(x3)  # [B, 8*dim, D/8, H/8, W/8]
        x5 = self.encoder_conv4(x4)  # [B, 16*dim, D/16, H/16, W/16]
        
        mask = self.decoder_conv4(x5, x4)  # [B, 8*dim, D/8, H/8, W/8]
        mask = self.decoder_conv3(mask, x3)  # [B, 4*dim, D/4, H/4, W/4]
        mask = self.decoder_conv2(mask, x2)  # [B, 2*dim, D/2, H/2, W/2]
        x = self.decoder_conv1(mask, x1)  # [B, dim, D, H, W]
        x = x.mean(dim=2)  # global spectral pooling -> [B, dim(new C), H, W]
        
        # Store feature map for CAM generation
        self.final_feature_map = x.detach()  # [B, C, H, W]
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
            print(f"WARNING: Input has {C} channels, but model expects {self.in_channels}")
        
        x = self.forward_features(x)  # [B, dim, H, W]
        output = self.seg_head(x)     # [B, num_classes, H, W]
        
        if return_cam:
            cam = self.generate_cam()
            return output, cam
        
        return output


if __name__ == "__main__":
    model = Unet(in_channels=15, num_classes=9, dim=64)
    
    dummy_input = torch.randn(4, 15, 15, 15)  # [B, C, H, W]
    output = model(dummy_input)
    print(f"  Output shape: {output.shape}")  # Expected: [4, 9, 15, 15]
    
    output_with_cam = model(dummy_input, return_cam=True)
    if isinstance(output_with_cam, tuple):
        output, cam = output_with_cam
        print(f"  CAM shape: {cam.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")