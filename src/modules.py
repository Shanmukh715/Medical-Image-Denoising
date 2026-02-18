import torch
import torch.nn as nn
import timm

class SwinPrior(nn.Module):
    """
    The 'Heavyweight' ML component (Swin Transformer).
    Acts as the learned Proximal Operator (z-update) in ADMM.
    
    CONFIGURATION: 'swin_small' (Heavy) for RTX 4060.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        # We use a heavier backbone (Swin Small) for better feature extraction
        self.swin = timm.create_model(
            'swin_small_patch4_window7_224',
            pretrained=True,
            in_chans=in_channels, 
            features_only=True
        )
        
        # FIX: The Swin Small output is 7x7 (downsampled by 32).
        # We must UPSAMPLE it back to 224x224.
        self.decoder = nn.Sequential(
            # 1. Reduce channels
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # 2. UPSAMPLE (The missing link!) - Restores size from 7x7 to 224x224
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False),
            
            # 3. Refine features
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Swin Transformer acts as a global context extractor
        # x shape: [Batch, 1, 224, 224]
        features = self.swin(x)
        
        # Take the last feature map. Swin outputs [Batch, H, W, C] -> Permute to [Batch, C, H, W]
        # Shape here is [Batch, 768, 7, 7]
        last_map = features[-1].permute(0, 3, 1, 2)
        
        # Project back to image domain (Upsample happens here)
        # Shape becomes [Batch, 1, 224, 224]
        out = self.decoder(last_map)
        
        # Residual connection
        return x + out