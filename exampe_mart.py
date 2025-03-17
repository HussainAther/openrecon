# example_mart.py

import numpy as np
import matplotlib.pyplot as plt
from openrecon.mart_reconstruction import MARTReconstruction

def main():
    """Run a basic example of MART-based CT reconstruction."""
    
    # Generate synthetic projection data (64x64 matrix of random values)
    projections = np.random.rand(64, 64)
    
    # Initialize MART reconstruction model with 10 iterations
    reconstructor = MARTReconstruction(num_iterations=10)
    
    # Perform the reconstruction
    reconstructed_image = reconstructor.reconstruct(projections)
    
    # Display the results
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title("MART Reconstructed Image")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()

