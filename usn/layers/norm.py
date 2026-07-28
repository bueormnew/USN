"""Normalization layers for USN architecture.

Implements RMSNorm (default) and LayerNorm with a factory function
for configurable selection. RMSNorm is preferred for its computational
efficiency and similar performance to LayerNorm.
"""

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Computes y = x / RMS(x) × γ where RMS(x) = √(mean(x²) + ε).
    More efficient than LayerNorm as it skips mean subtraction.

    Args:
        d_model: Feature dimension to normalize over.
        eps: Small constant for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize input tensor.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Normalized tensor of same shape.
        """
        # Compute in float32 for stability, cast back
        input_dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return (self.weight * x).to(input_dtype)


class USNLayerNorm(nn.Module):
    """Standard Layer Normalization.

    Computes y = (x - mean(x)) / √(var(x) + ε) × γ + β.

    Args:
        d_model: Feature dimension to normalize over.
        eps: Small constant for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize input tensor.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Normalized tensor of same shape.
        """
        input_dtype = x.dtype
        x = x.float()
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return (self.weight * x + self.bias).to(input_dtype)


def create_norm(norm_type: str, d_model: int, eps: float = 1e-6) -> nn.Module:
    """Factory function for normalization layers.

    Args:
        norm_type: Type of normalization ("rmsnorm" or "layernorm").
        d_model: Feature dimension.
        eps: Epsilon for numerical stability.

    Returns:
        Instantiated normalization module.

    Raises:
        ValueError: If norm_type is not recognized.
    """
    if norm_type == "rmsnorm":
        return RMSNorm(d_model, eps)
    elif norm_type == "layernorm":
        return USNLayerNorm(d_model, eps)
    else:
        raise ValueError(f"Unknown norm_type '{norm_type}'. Must be 'rmsnorm' or 'layernorm'.")
