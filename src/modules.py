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
            'swin_small_patch4_window7_224',  # Heavier than 'tiny'
            pretrained=True,                  # Start with ImageNet knowledge
            in_chans=in_channels, 
            features_only=True
        )
        
        # The Swin Small output at the last stage has 768 channels.
        # We project this deep semantic feature map back to the image space.
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x shape: [Batch, 1, 224, 224]
        # Swin Transformer acts as a global context extractor
        features = self.swin(x)
        
        # Take the last feature map. Swin outputs [Batch, H, W, C], so permute to [Batch, C, H, W]
        last_map = features[-1].permute(0, 3, 1, 2)
        
        # Project back to image domain
        out = self.decoder(last_map)
        
        # Residual connection: The network learns the *correction* to add/subtract
        return x + out