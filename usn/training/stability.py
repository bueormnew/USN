"""Training stability utilities for NaN detection, loss spikes, and state monitoring.

These are lightweight utility functions that the trainer can optionally call
to detect and handle common training instabilities. They do not modify
training state themselves — they only report issues so the caller can decide
how to respond (skip batch, reduce LR, clip state, etc.).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def has_nan(tensor: Tensor) -> bool:
    """Check if a tensor contains any NaN values.

    Args:
        tensor: Any PyTorch tensor to inspect.

    Returns:
        True if the tensor contains at least one NaN, False otherwise.
    """
    return bool(torch.isnan(tensor).any().item())


def has_inf(tensor: Tensor) -> bool:
    """Check if a tensor contains any infinite values.

    Args:
        tensor: Any PyTorch tensor to inspect.

    Returns:
        True if the tensor contains at least one ±inf, False otherwise.
    """
    return bool(torch.isinf(tensor).any().item())


def detect_loss_spike(
    current_loss: float,
    loss_history: list[float],
    threshold: float = 5.0,
    window: int = 100,
) -> bool:
    """Detect whether the current loss represents a spike relative to recent history.

    A spike is detected when the current loss exceeds the running average of
    the most recent `window` entries by more than `threshold` times.

    Args:
        current_loss: The loss value for the current step.
        loss_history: List of past loss values (most recent at end).
        threshold: Multiplier above the running average to flag as spike.
        window: Number of recent entries to average over.

    Returns:
        True if the current loss is a spike, False otherwise.
        Returns False if there is insufficient history (fewer than 1 entry).
    """
    if not loss_history:
        return False

    recent = loss_history[-window:]
    avg = sum(recent) / len(recent)

    if avg <= 0 or math.isnan(avg) or math.isinf(avg):
        return False

    return current_loss > threshold * avg


def check_state_magnitude(
    state: Tensor,
    max_norm: float = 1000.0,
) -> tuple[bool, float]:
    """Monitor the magnitude of a state tensor and flag if it exceeds a threshold.

    Args:
        state: A state tensor (semantic vector or relational matrix).
        max_norm: Maximum allowed L2 norm before flagging.

    Returns:
        Tuple of (is_exceeded, actual_norm) where is_exceeded is True
        if the norm exceeds max_norm.
    """
    norm = float(torch.norm(state.float()).item())
    return norm > max_norm, norm


def clip_state_norm(
    state: Tensor,
    max_norm: float = 1000.0,
) -> Tensor:
    """Clip a state tensor's norm to a maximum value in-place style.

    If the state norm exceeds max_norm, the tensor is scaled down to
    have exactly max_norm as its L2 norm. Otherwise returned unchanged.

    Args:
        state: A state tensor to potentially clip.
        max_norm: Maximum allowed L2 norm.

    Returns:
        The (possibly clipped) state tensor. Does not modify in-place;
        returns a new tensor if clipping is applied.
    """
    norm = torch.norm(state.float())
    if norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        return state * scale.to(state.dtype)
    return state
