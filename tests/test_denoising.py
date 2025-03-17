# test_denoising.py

import torch
import pytest
from openrecon.gan_denoising import DenoisingModel

def test_denoising_model():
    """Test the AI-based denoising model on synthetic noisy data."""
    model = DenoisingModel()
    model.load_pretrained("models/denoising_model.pth")  # Ensure this file exists
    model.eval()
    
    # Generate synthetic noisy image
    noisy_image = torch.randn(1, 1, 64, 64)  # Simulated noisy CT scan
    with torch.no_grad():
        denoised_image = model(noisy_image)
    
    # Check that the output shape matches the input shape
    assert denoised_image.shape == noisy_image.shape, "Denoised image shape mismatch"
    
    # Ensure pixel values remain within expected range (assuming normalized [0,1])
    assert torch.all(denoised_image >= 0) and torch.all(denoised_image <= 1), "Denoised image has invalid pixel values"

if __name__ == "__main__":
    pytest.main(["-v", "test_denoising.py"])

