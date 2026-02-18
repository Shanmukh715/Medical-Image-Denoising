import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from model_admm import DeepUnrolledADMM

# --- CONFIGURATION FOR HIGH-SPEC MACHINE ---
BATCH_SIZE = 8       # Optimized for RTX 4060 8GB VRAM
LEARNING_RATE = 1e-4
EPOCHS = 100         # Deep training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# !!! UPDATE THIS PATH TO YOUR ACTUAL FOLDER !!!
# Automatically finds the 'brain_tumor_dataset/yes' folder in your project
DATASET_PATH = os.path.join(os.getcwd(), "brain_tumor_dataset", "yes")
class MRIDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        # Load Grayscale
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224)) 
        
        # Normalize [0,1]
        image = image.astype(np.float32) / 255.0
        clean_tensor = torch.from_numpy(image).unsqueeze(0) 
        
        # Add Heavy Gaussian Noise for Robust Training
        noise = torch.randn_like(clean_tensor) * 0.15 
        noisy_tensor = clean_tensor + noise
        
        return noisy_tensor, clean_tensor

def train():
    print(f"--- 🚀 Starting Training on {DEVICE} ---")
    print(f"--- Configuration: 8 Stages | Swin Small | Batch {BATCH_SIZE} ---")
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: Dataset path not found: {DATASET_PATH}")
        print("Please open src/train.py and update DATASET_PATH")
        return

    dataset = MRIDataset(DATASET_PATH)
    # Set num_workers to 0 to fix the freezing issue on Windows
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = DeepUnrolledADMM(stages=8).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for i, (noisy, clean) in enumerate(dataloader):
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            
            optimizer.zero_grad()
            restored = model(noisy)
            loss = criterion(restored, clean)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            if not os.path.exists("checkpoints"): os.makedirs("checkpoints")
            path = f"checkpoints/admm_swin_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), path)
            print(f"💾 Checkpoint saved: {path}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True # Boosts RTX 4060 performance
    train()