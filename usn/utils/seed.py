"""Reproducibility utilities for setting random seeds across all libraries.

Ensures deterministic behavior when needed for debugging and reproducibility.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for Python, NumPy, PyTorch, and CUDA.

    Ensures reproducible results across runs by seeding all relevant
    random number generators.

    Args:
        seed: Integer seed value. Must be non-negative.
        deterministic: If True, also sets PyTorch to use deterministic
            algorithms (may reduce performance). Sets
            ``torch.backends.cudnn.deterministic = True`` and
            ``torch.backends.cudnn.benchmark = False``.

    Example:
        >>> from usn.utils import set_seed
        >>> set_seed(42)
        >>> # All subsequent random operations are reproducible
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    # Python stdlib random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic mode
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # PyTorch 1.8+ deterministic algorithms flag
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
