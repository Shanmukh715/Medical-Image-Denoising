import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from model_admm import DeepUnrolledADMM

# --- CONFIGURATION ---
# Use the same path as your train.py
DATASET_PATH = os.path.join(os.getcwd(), "brain_tumor_dataset", "yes")
CHECKPOINT_PATH = "checkpoints/admm_cpu_epoch_50.pth" # <--- Check your 'checkpoints' folder for the latest file
DEVICE = torch.device("cpu") # Keep CPU for testing to avoid memory issues

def load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return img

def evaluate():
    print(f"--- 📊 Starting Evaluation ---")
    
    # 1. Load the Model
    model = DeepUnrolledADMM(stages=6).to(DEVICE)
    
    # Load Weights (Handle errors if file name is different)
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"✅ Loaded weights from {CHECKPOINT_PATH}")
    except FileNotFoundError:
        print(f"❌ Checkpoint not found at {CHECKPOINT_PATH}. Please check the filename in 'checkpoints/' folder.")
        return

    model.eval()

    # 2. Get Test Images (Take first 3 images)
    image_files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.jpg')][:3]
    
    total_psnr = 0
    total_ssim = 0

    print("\n--- Generating Visual Results ---")

    for i, img_file in enumerate(image_files):
        # Prepare Data
        clean_np = load_image(os.path.join(DATASET_PATH, img_file))
        clean_tensor = torch.from_numpy(clean_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Add Noise (Same as training)
        torch.manual_seed(42) # Fixed seed for fair comparison
        noise = torch.randn_like(clean_tensor) * 0.1
        noisy_tensor = clean_tensor + noise
        
        # Run Model
        with torch.no_grad():
            denoised_tensor = model(noisy_tensor)
        
        # Convert back to Numpy for Metrics
        noisy_np = noisy_tensor.squeeze().cpu().numpy()
        denoised_np = denoised_tensor.squeeze().cpu().numpy()
        
        # Clip values to valid range [0, 1]
        denoised_np = np.clip(denoised_np, 0, 1)
        noisy_np = np.clip(noisy_np, 0, 1)

        # Calculate Metrics
        current_psnr = psnr(clean_np, denoised_np, data_range=1.0)
        current_ssim = ssim(clean_np, denoised_np, data_range=1.0)
        
        total_psnr += current_psnr
        total_ssim += current_ssim
        
        print(f"Image {i+1}: PSNR = {current_psnr:.2f} dB | SSIM = {current_ssim:.4f}")

        # --- PLOT & SAVE RESULTS ---
        plt.figure(figsize=(15, 5))
        
        # 1. Noisy Input
        plt.subplot(1, 3, 1)
        plt.imshow(noisy_np, cmap='gray')
        plt.title(f"Noisy Input\n(Simulated Noise)", fontsize=10)
        plt.axis('off')
        
        # 2. Our Model Output
        plt.subplot(1, 3, 2)
        plt.imshow(denoised_np, cmap='gray')
        plt.title(f"Deep Unrolled ADMM\nPSNR: {current_psnr:.2f} dB", fontsize=10, fontweight='bold')
        plt.axis('off')
        
        # 3. Ground Truth
        plt.subplot(1, 3, 3)
        plt.imshow(clean_np, cmap='gray')
        plt.title(f"Original / Ground Truth", fontsize=10)
        plt.axis('off')
        
        plt.tight_layout()
        save_path = f"result_image_{i+1}.png"
        plt.savefig(save_path)
        print(f"   Saved comparison to {save_path}")
        plt.close()

    # Final Average
    print(f"\n--- 🏆 Final Report Stats ---")
    print(f"Average PSNR: {total_psnr/3:.2f} dB")
    print(f"Average SSIM: {total_ssim/3:.4f}")
    print("Use these numbers in your 'Results' section.")

if __name__ == "__main__":
    evaluate()