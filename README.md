# OpenRecon: AI-Powered CT Reconstruction Toolkit 🚀

## 📌 Overview
**OpenRecon** is an open-source AI-driven Computed Tomography (CT) reconstruction toolkit designed for **low-dose, high-precision imaging**. It includes:
✅ **MART-based iterative reconstruction**  
✅ **AI-powered denoising (GANs, Autoencoders)**  
✅ **Inpainting models for missing-angle CT**  
✅ **Advanced filtering (Gaussian, Bilateral, Median)**  

OpenRecon is part of the **OpenRBYR** ecosystem, focusing on high-fidelity image reconstruction techniques.

---

## 🚀 Features
- **MART Reconstruction** – Iterative Multiplicative Algebraic Reconstruction Technique
- **GAN-Based Denoising** – AI-based noise removal for low-dose CT scans
- **Inpainting Algorithms** – Restores missing projection data
- **Custom Filters** – Gaussian, Median, and Bilateral filtering
- **Pre-trained AI Models** – Load and fine-tune models easily
- **Extensible API** – Can be integrated into cloud & real-time imaging pipelines

---

## 🔧 Installation
To install OpenRecon, simply run:
```bash
pip install openrecon
```
Or, clone the repository and install dependencies:
```bash
git clone https://github.com/YourGitHub/OpenRecon.git
cd OpenRecon
pip install -r requirements.txt
```

---

## 📂 Project Structure
```
OpenRecon/
│── openrecon/              # Main package source code
│   │── mart_reconstruction.py  # Iterative CT reconstruction
│   │── gan_denoising.py        # AI-based denoising model
│   │── inpainting.py           # AI-powered inpainting for missing angles
│   │── filters.py              # Filtering techniques
│   │── utils.py                # Helper functions
│
│── models/                  # Pre-trained AI models
│   │── denoising_model.pth
│   │── inpainting_model.pth
│
│── examples/                # Sample scripts
│   │── example_mart.py
│   │── example_gan_denoising.py
│   │── example_inpainting.py
│   │── example_filters.py
│
│── tests/                   # Unit tests
│   │── test_mart.py
│   │── test_denoising.py
│   │── test_inpainting.py
│
│── docs/                    # Documentation
│
│── LICENSE                  # Open-source license (MIT)
│── README.md                # Project documentation
│── requirements.txt         # Dependencies
│── setup.py                 # Package setup
```

---

## 🛠 Usage
### **1️⃣ Run a Basic MART Reconstruction**
```python
from openrecon.mart_reconstruction import MARTReconstruction
import numpy as np

class MARTReconstruction:
    def __init__(self, num_iterations=10):
        self.num_iterations = num_iterations

    def reconstruct(self, projections):
        reconstructed_image = np.ones_like(projections)
        for _ in range(self.num_iterations):
            reconstructed_image *= projections / (np.sum(reconstructed_image, axis=0) + 1e-8)
        return reconstructed_image

    def visualize_reconstruction(self, image):
        import matplotlib.pyplot as plt
        plt.imshow(image, cmap='gray')
        plt.title("Reconstructed Image")
        plt.colorbar()
        plt.show()
```

### **2️⃣ Apply AI-Based Denoising**
```python
from openrecon.gan_denoising import DenoisingModel
import torch

class DenoisingModel:
    def __init__(self):
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.model(x)

    def load_pretrained(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
```

### **3️⃣ Inpainting Missing Data**
```python
from openrecon.inpainting import InpaintingModel
import torch

class InpaintingModel:
    def __init__(self):
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, mask):
        x = x * mask  # Apply mask for inpainting simulation
        return self.model(x)
```

### **4️⃣ Apply Image Filters**
```python
from openrecon.filters import apply_gaussian_filter, apply_median_filter
import numpy as np

image = np.random.rand(256, 256)
gaussian_filtered = apply_gaussian_filter(image, sigma=1.5)
median_filtered = apply_median_filter(image, size=3)
```

---

## 🔬 Research & Development
This project is actively developed as part of the **OpenRBYR** ecosystem, focusing on AI-enhanced medical imaging.

Contributions, feedback, and collaborations are welcome! 🚀

---

## 📜 License
This project is licensed under the **MIT License** – free to use and modify.

---

## 🤝 Contributing
1. Fork the repository
2. Clone your fork
3. Create a new branch (`git checkout -b new-feature`)
4. Commit your changes (`git commit -m 'Add new feature'`)
5. Push the branch (`git push origin new-feature`)
6. Submit a Pull Request 🎉

---

## 🌎 Stay Connected
📢 Follow the project on [GitHub](https://github.com/YourGitHub/OpenRecon)  
💬 Join the discussion on **Discord / Reddit / Substack** *(Coming Soon!)*

🔥 **Let’s build the future of AI-powered medical imaging together!**

