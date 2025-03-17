# test_inpainting.py

import torch
import pytest
from openrecon.inpainting import InpaintingModel

def test_inpainting_model():
    """Test the AI-based inpainting model on synthetic masked data."""
    model = InpaintingModel()
    model.load_pretrained("models/inpainting_model.pth")  # Ensure this file exists
    model.eval()
    
    # Generate synthetic masked image
    image = torch.randn(1, 1, 64, 64)  # Simulated CT scan
    mask = (torch.rand(1, 1, 64, 64) > 0.5).float()  # Random binary mask (some missing data)
    
    with torch.no_grad():
        inpainted_image = model(image, mask)
    
    # Check that the output shape matches the input shape
    assert inpainted_image.shape == image.shape, "Inpainted image shape mismatch"
    
    # Ensure pixel values remain within expected range (assuming normalized [0,1])
    assert torch.all(inpainted_image >= 0) and torch.all(inpainted_image <= 1), "Inpainted image has invalid pixel values"

if __name__ == "__main__":
    pytest.main(["-v", "test_inpainting.py"])

