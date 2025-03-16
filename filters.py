# filters.py

import numpy as np
import scipy.ndimage as ndimage
import cv2

def apply_gaussian_filter(image, sigma=1.0):
    """Applies a Gaussian filter to smooth the image."""
    return ndimage.gaussian_filter(image, sigma=sigma)

def apply_median_filter(image, size=3):
    """Applies a median filter to reduce noise."""
    return ndimage.median_filter(image, size=size)

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Applies a bilateral filter for edge-preserving smoothing."""
    return cv2.bilateralFilter(image.astype(np.float32), d, sigma_color, sigma_space)

def apply_sharpening_filter(image, alpha=1.5, beta=-0.5):
    """Applies a sharpening filter using an unsharp masking technique."""
    blurred = apply_gaussian_filter(image, sigma=1)
    sharpened = alpha * image + beta * blurred
    return np.clip(sharpened, 0, 1)

if __name__ == "__main__":
    # Example usage
    sample_image = np.random.rand(256, 256)
    gaussian_smoothed = apply_gaussian_filter(sample_image, sigma=2.0)
    median_filtered = apply_median_filter(sample_image, size=5)
    bilateral_filtered = apply_bilateral_filter(sample_image)
    sharpened_image = apply_sharpening_filter(sample_image)
    
    print("Filtering applied successfully.")

