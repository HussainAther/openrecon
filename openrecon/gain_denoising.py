# openrecon/gan_denoising.py
"""
AI-based denoising module using pretrained GAN or U-Net models.
"""
import torch
import numpy as np
from openrecon.ai_models import load_pytorch_model, run_pytorch_model

class GANDenoiser:
    def __init__(self, model_path, use_cuda=False):
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model = load_pytorch_model(model_path).to(self.device)

    def denoise(self, input_image):
        """
        Apply the AI model to denoise a single 2D image.

        Args:
            input_image (np.ndarray): Noisy input image (2D float32)

        Returns:
            np.ndarray: Denoised image (2D)
        """
        input_tensor = torch.tensor(input_image, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output.squeeze().cpu().numpy()

if __name__ == "__main__":
    print("GANDenoiser module ready.")
