"""
optimizations.py
Utility functions to apply different optimization techniques on DenseNet.
"""

import torch
from torchvision import models


def apply_amp(model: torch.nn.Module):
    """
    Mixed Precision (AMP) optimization.
    Note: AMP is not a model transformation, it's used during inference/training.
    So we just return the model here, and handle autocast in benchmark.py.
    """
    return model


def apply_jit(model: torch.nn.Module, sample_input: torch.Tensor):
    """
    TorchScript (JIT) optimization using tracing.
    Args:
        model: The PyTorch model.
        sample_input: A representative input tensor.
    Returns:
        TorchScript traced model.
    """
    model.eval()
    scripted_model = torch.jit.trace(model, sample_input)
    return scripted_model


def apply_quantization(model: torch.nn.Module):
    """
    Dynamic Quantization.
    Best for CPU inference (weight-only quantization).
    Use on CPU.
    """
    model_cpu = model.to("cpu")
    model_cpu.eval()
    try:
        # quantize only Linear layers
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model_cpu,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        return quantized_model
    except Exception as e:
        print(f"[WARN] quantization failed: {e}")
        # fallback to original model (on CPU)
        return model_cpu


def get_model(optimization: str, device: str, sample_input: torch.Tensor):
    """
    Load DenseNet-121 with pretrained weights and apply optimization.
    Args:
        optimization: one of ['baseline', 'amp', 'jit', 'quantization']
        device: 'cpu' or 'cuda'
        sample_input: A representative input tensor (for JIT)
    """
    weights = models.DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights).to(device)

    if optimization == "baseline":
        return model
    elif optimization == "amp":
        return apply_amp(model)
    elif optimization == "jit":
        return apply_jit(model, sample_input.to(device))
    elif optimization == "quantization":
        # quantization only works on CPU
        return apply_quantization(model.to("cpu"))
    else:
        raise ValueError(f"Unknown optimization: {optimization}")
