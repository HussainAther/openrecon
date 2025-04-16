# openrecon/inpainting.py
"""
AI-powered inpainting module for limited-angle CT reconstruction.
"""
import torch
import numpy as np
from openrecon.ai_models import load_pytorch_model

class InpaintingModel:
    def __init__(self, model_path, use_cuda=False):
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model = load_pytorch_model(model_path).to(self.device)

    def inpaint(self, sinogram):
        """
        Inpaint a 2D sinogram with missing angles or partial views.

        Args:
            sinogram (np.ndarray): 2D array with missing-angle sinogram [angles x detectors]

        Returns:
            np.ndarray: Inpainted (filled-in) sinogram
        """
        sino_tensor = torch.tensor(sinogram, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = self.model(sino_tensor)
        return output.squeeze().cpu().numpy()

if __name__ == "__main__":
    print("InpaintingModel module ready.")

