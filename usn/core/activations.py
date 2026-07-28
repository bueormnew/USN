"""Activation function registry for USN.

Provides a factory function and registry pattern for activation functions
used in the Channel Mixing MLP. Supports GELU, SiLU/Swish, and ReLU
using PyTorch built-in implementations for maximum hardware efficiency.

Custom activations can be registered via register_activation().
"""

from collections.abc import Callable

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Global registry mapping activation names to factory functions
_ACTIVATION_REGISTRY: dict[str, Callable[[], nn.Module]] = {}


class _GELUActivation(nn.Module):
    """GELU activation using PyTorch built-in."""

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x)


class _SiLUActivation(nn.Module):
    """SiLU (Swish) activation using PyTorch built-in."""

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x)


class _ReLUActivation(nn.Module):
    """ReLU activation using PyTorch built-in."""

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)


def _register_defaults() -> None:
    """Register the default activation functions."""
    _ACTIVATION_REGISTRY["gelu"] = _GELUActivation
    _ACTIVATION_REGISTRY["silu"] = _SiLUActivation
    _ACTIVATION_REGISTRY["swish"] = _SiLUActivation  # Alias
    _ACTIVATION_REGISTRY["relu"] = _ReLUActivation


# Register defaults on module load
_register_defaults()


def get_activation(name: str) -> nn.Module:
    """Get an activation function module by name.

    Args:
        name: Name of the activation function. Supported: "gelu", "silu", "swish", "relu".

    Returns:
        An nn.Module implementing the activation function.

    Raises:
        ValueError: If the activation name is not registered.
    """
    name_lower = name.lower()
    if name_lower not in _ACTIVATION_REGISTRY:
        available = ", ".join(sorted(_ACTIVATION_REGISTRY.keys()))
        raise ValueError(
            f"Unknown activation '{name}'. Available: {available}. "
            f"Register custom activations with register_activation()."
        )
    return _ACTIVATION_REGISTRY[name_lower]()


def register_activation(name: str, factory: Callable[[], nn.Module]) -> None:
    """Register a custom activation function.

    Args:
        name: Name to register the activation under.
        factory: Callable that returns an nn.Module implementing the activation.

    Example:
        >>> class MyActivation(nn.Module):
        ...     def forward(self, x):
        ...         return x * torch.sigmoid(x * 1.7)
        >>> register_activation("my_act", MyActivation)
        >>> act = get_activation("my_act")
    """
    _ACTIVATION_REGISTRY[name.lower()] = factory


def list_activations() -> list[str]:
    """List all registered activation function names."""
    return sorted(_ACTIVATION_REGISTRY.keys())
