import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ==============================
# PATH TO YOUR IMAGE FOLDER
# ==============================
IMAGE_FOLDER = r"C:\Users\shanm\OneDrive\Desktop\coding\Project\SEM 3 AND 4\SEM 4\Maths\brain_tumor_dataset\yes"

# ==============================
# IMAGE PROCESSING FUNCTIONS
# ==============================

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    return img

def add_gaussian_noise(image, mean=0, var=0.01):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = image + gauss * 255
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, amount=0.05):
    noisy = image.copy()
    num_salt = int(amount * image.size * 0.5)
    num_pepper = int(amount * image.size * 0.5)

    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy[coords[0], coords[1]] = 255

    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy

def mean_filter(img):
    return cv2.blur(img, (5,5))

def median_filter(img):
    return cv2.medianBlur(img, 5)

def bilateral_filter(img):
    return cv2.bilateralFilter(img, 9, 75, 75)

def wavelet_denoising(img):
    coeffs = pywt.wavedec2(img, 'db1', level=2)

    cA, (cH, cV, cD), (cH2, cV2, cD2) = coeffs

    threshold = np.std(cD) * 0.5

    cH = pywt.threshold(cH, threshold, mode='soft')
    cV = pywt.threshold(cV, threshold, mode='soft')
    cD = pywt.threshold(cD, threshold, mode='soft')

    denoised = pywt.waverec2([cA, (cH, cV, cD), (cH2, cV2, cD2)], 'db1')

    # 🔥 FIX: Resize to original image size
    denoised = cv2.resize(denoised, (img.shape[1], img.shape[0]))

    return np.clip(denoised, 0, 255).astype(np.uint8)

# ==============================
# MAIN EXECUTION
# ==============================

images = os.listdir(IMAGE_FOLDER)

for img_name in images[:5]:   # process first 5 images
    path = os.path.join(IMAGE_FOLDER, img_name)

    original = load_image(path)
    noisy = add_gaussian_noise(original)

    mean_img = mean_filter(noisy)
    median_img = median_filter(noisy)
    bilateral_img = bilateral_filter(noisy)
    wavelet_img = wavelet_denoising(noisy)

    print(f"\nResults for {img_name}")
    print("Mean PSNR:", psnr(original, mean_img))
    print("Median PSNR:", psnr(original, median_img))
    print("Bilateral PSNR:", psnr(original, bilateral_img))
    print("Wavelet PSNR:", psnr(original, wavelet_img))

    # Display results
    titles = ["Original", "Noisy", "Mean", "Median", "Bilateral", "Wavelet"]
    imgs = [original, noisy, mean_img, median_img, bilateral_img, wavelet_img]

    plt.figure(figsize=(12, 6))
    for i in range(len(imgs)):
        plt.subplot(2, 3, i+1)
        plt.imshow(imgs[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.suptitle(f"Results for {img_name}")
    plt.tight_layout()
    plt.show()
