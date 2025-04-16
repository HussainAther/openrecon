# openrecon/ai_models.py
"""
Manage loading of PyTorch and ONNX models for reconstruction and postprocessing.
"""
import os
import torch
import onnxruntime as ort


def load_pytorch_model(path):
    """Load a PyTorch model from .pth file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = torch.load(path, map_location=torch.device('cpu'))
    model.eval()
    return model


def run_pytorch_model(model, input_tensor):
    """Run inference using a loaded PyTorch model."""
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))
        return output.squeeze(0)


def load_onnx_model(path):
    """Load an ONNX model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return ort.InferenceSession(path)


def run_onnx_model(session, input_array):
    """Run ONNX model inference."""
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_array.astype('float32')})
    return output[0]

