import torch
import torch.nn as nn
from utils_math import Hankelizer
from modules import SwinPrior

class ADMM_Stage(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoiser = SwinPrior(in_channels=1)
        self.hankel = Hankelizer()
        
        # Learnable Parameters for the Optimization Algorithm (Unit 2)
        # rho: Penalty parameter (controls how strict the constraints are)
        # eta: Step size for the Gradient Descent (x-update)
        self.rho = nn.Parameter(torch.tensor(0.1)) 
        self.eta = nn.Parameter(torch.tensor(0.5)) 

    def forward(self, x, z, u, y_noisy):
        """
        One iteration of ADMM unrolled into a Deep Layer.
        """
        # --- Step 1: x-update (Reconstruction via Gradient Descent) ---
        # Solving: min ||x - y||^2 + rho ||x - z + u||^2
        # Gradient: (x - y) + rho * (x - z + u)
        gradient = (x - y_noisy) + self.rho * (x - z + u)
        x_new = x - self.eta * gradient

        # --- Step 2: z-update (Denoising via Swin + Hankel) ---
        v = x_new + u
        
        # A. Apply Linear Algebra Constraint (Unit 1)
        v_structured = self.hankel.hankel_projection(v)
        
        # B. Apply Deep Learning Prior (Swin Small)
        z_new = self.denoiser(v_structured)

        # --- Step 3: u-update (Multiplier Update) ---
        u_new = u + (x_new - z_new)

        return x_new, z_new, u_new

class DeepUnrolledADMM(nn.Module):
    def __init__(self, stages=8):  # 8 Stages = Deep Math Unrolling
        super().__init__()
        self.stages = nn.ModuleList([ADMM_Stage() for _ in range(stages)])

    def forward(self, y_noisy):
        # Initialize ADMM variables
        x = y_noisy.clone()
        z = y_noisy.clone()
        u = torch.zeros_like(y_noisy)

        # Run the Unrolled Optimization Loop
        for stage in self.stages:
            x, z, u = stage(x, z, u, y_noisy)
        
        return x # The Mathematically Optimized Image