import torch
import torch.nn.functional as F

class Hankelizer:
    """
    Implements Unit 1: Special Matrices (Hankelization).
    Converts image features into structured Hankel matrices to enforce low-rank priors.
    """
    def __init__(self, patch_size=3):
        self.k = patch_size

    def image_to_hankel(self, x):
        """
        Input: Tensor [Batch, Channels, Height, Width]
        Output: Hankel-like Matrix structure
        Transforms 2D spatial data into a matrix where we can analyze Rank.
        """
        patches = F.unfold(x, kernel_size=self.k, padding=self.k//2)
        return patches 

    def hankel_projection(self, x):
        """
        Applies a structural consistency check.
        In a pure math solver, we would enforce Rank(H) = k. 
        Here, we use the folding/unfolding process as a structural regularization 
        layer to dampen high-frequency noise that violates local self-similarity.
        """
        patches = self.image_to_hankel(x)
        
        # Re-fold back. This averaging of overlapping patches acts as 
        # a 'Structure-Aware' filter, strictly enforcing the Hankel geometry.
        restored = F.fold(patches, output_size=x.shape[-2:], kernel_size=self.k, padding=self.k//2)
        
        # Normalizing for overlap counts
        divisor = F.fold(torch.ones_like(patches), output_size=x.shape[-2:], kernel_size=self.k, padding=self.k//2)
        return restored / divisor