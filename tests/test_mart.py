# test_mart.py

import numpy as np
import pytest
from openrecon.mart_reconstruction import MARTReconstruction

def test_mart_reconstruction():
    """Test MART reconstruction algorithm with synthetic data."""
    projections = np.random.rand(64, 64)  # Simulated projection data
    mart = MARTReconstruction(num_iterations=5)
    reconstructed_image = mart.reconstruct(projections)
    
    # Check that the output shape matches the input shape
    assert reconstructed_image.shape == projections.shape, "Reconstructed image shape mismatch"
    
    # Check that values are non-negative
    assert np.all(reconstructed_image >= 0), "Reconstructed image has negative values"

if __name__ == "__main__":
    pytest.main(["-v", "test_mart.py"])

