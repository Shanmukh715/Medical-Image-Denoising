import torch
import torch.nn as nn
import timm

class SwinPrior(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # Use 'tiny' for CPU (faster than 'small')
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=False, 
            in_chans=in_channels, 
            features_only=True
        )
        
        # Decoder with UPSAMPLING to fix the size mismatch
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False), # <--- FIX
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        features = self.swin(x)
        last_map = features[-1].permute(0, 3, 1, 2)
        out = self.decoder(last_map)
        return x + out