import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from model_admm import DeepUnrolledADMM

# --- CONFIGURATION (CPU MODE) ---
BATCH_SIZE = 4        # Reduced batch size for CPU stability
LEARNING_RATE = 1e-4
EPOCHS = 50           # You can reduce this to 5 or 10 for a quick demo
DEVICE = torch.device("cpu") # <--- FORCED CPU

# Universal path to find your dataset automatically
DATASET_PATH = os.path.join(os.getcwd(), "brain_tumor_dataset", "yes")

# --- HELPER: CUSTOM DATASET ---
class MRIDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        
        # Load as Grayscale
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        
        # Handle cases where image might not load
        if image is None:
            return torch.zeros(1, 224, 224), torch.zeros(1, 224, 224)
            
        image = cv2.resize(image, (224, 224)) 
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        clean_tensor = torch.from_numpy(image).unsqueeze(0) # Shape: [1, 224, 224]
        
        # Add Synthetic Gaussian Noise
        noise = torch.randn_like(clean_tensor) * 0.1
        noisy_tensor = clean_tensor + noise
        
        return noisy_tensor, clean_tensor

# --- MAIN TRAINING LOOP ---
def train():
    print(f"--- 🐢 Starting Training on {DEVICE} (Reliability Mode) ---")
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: Dataset path not found: {DATASET_PATH}")
        print("Please ensure 'brain_tumor_dataset/yes' exists in your project folder.")
        return

    dataset = MRIDataset(DATASET_PATH)
    
    # CRITICAL FIX: num_workers=0 prevents Windows freezing
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"✅ Loaded {len(dataset)} images. Starting loop...")

    # Initialize Model (6 stages is enough for a CPU demo)
    model = DeepUnrolledADMM(stages=6).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for i, (noisy, clean) in enumerate(dataloader):
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward Pass
            restored = model(noisy)
            
            # Loss Calculation
            loss = criterion(restored, clean)
            
            # Backprop
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print status every 10 batches so you know it's working
            if i % 10 == 0:
                print(f"   Batch {i}/{len(dataloader)}...", end='\r')
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")
        
        # Save checkpoints every 5 epochs
        if (epoch + 1) % 5 == 0:
            if not os.path.exists("checkpoints"): os.makedirs("checkpoints")
            torch.save(model.state_dict(), f"checkpoints/admm_cpu_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()