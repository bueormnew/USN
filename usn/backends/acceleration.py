"""Acceleration level management for the USN library.

Implements a 4-level acceleration hierarchy with automatic detection and
graceful fallback. The levels are:

1. TRITON  — Custom Triton fused kernels (fastest, requires triton + CUDA)
2. COMPILE — torch.compile with inductor backend (requires PyTorch 2.0+ CUDA)
3. AUTOGRAD — Custom autograd functions (requires CUDA)
4. EAGER   — Standard PyTorch eager mode (always available, baseline)

The system auto-detects the best level at import time and NEVER fails due to
unavailable acceleration — it always falls back to EAGER as the last resort.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import IntEnum

import torch

logger = logging.getLogger(__name__)


class AccelerationLevel(IntEnum):
    """4-level acceleration hierarchy for USN kernels.

    Lower numeric value = faster / more specialized.
    Higher numeric value = more compatible / always available.
    """

    TRITON = 1
    """Custom Triton fused kernels — maximum performance."""

    COMPILE = 2
    """torch.compile with inductor — good performance, no Triton needed."""

    AUTOGRAD = 3
    """Custom autograd functions — CUDA required, no JIT compilation."""

    EAGER = 4
    """Standard PyTorch eager mode — always available, baseline performance."""


def _detect_best_level() -> AccelerationLevel:
    """Detect the best available acceleration level.

    Decision tree:
    1. Is Triton installed AND CUDA available? → TRITON
    2. Is torch.compile available (PyTorch ≥ 2.0) AND CUDA available? → COMPILE
    3. Is CUDA available? → AUTOGRAD
    4. Otherwise → EAGER

    This function NEVER raises — it always returns a valid level.
    """
    # Check Level 1: Triton
    if torch.cuda.is_available():
        try:
            import triton  # noqa: F401
            import triton.language  # noqa: F401

            return AccelerationLevel.TRITON
        except (ImportError, RuntimeError, OSError):
            pass

    # Check Level 2: torch.compile with inductor
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        try:

            @torch.compile(backend="inductor")
            def _test_fn(x: torch.Tensor) -> torch.Tensor:
                return x + 1

            _test_fn(torch.zeros(1, device="cuda"))
            return AccelerationLevel.COMPILE
        except Exception:
            pass

    # Check Level 3: Custom autograd (CUDA available but no compile)
    if torch.cuda.is_available():
        return AccelerationLevel.AUTOGRAD

    # Level 4: Eager (CPU or unsupported hardware)
    return AccelerationLevel.EAGER


class AccelerationManager:
    """Manages the 4-level acceleration hierarchy with graceful fallback.

    This is a class-level (singleton-style) manager. All methods are classmethods
    that operate on shared state. The acceleration level is auto-detected at
    import time but can be overridden manually.

    Usage:
        # Auto-detected at import
        level = AccelerationManager.get_level()

        # Manual override
        AccelerationManager.set_level(AccelerationLevel.EAGER)

        # Get kernel for current level
        kernel = AccelerationManager.get_kernel("projections")
    """

    _current_level: AccelerationLevel = AccelerationLevel.EAGER
    _detected_level: AccelerationLevel = AccelerationLevel.EAGER
    _initialized: bool = False
    _kernel_registry: dict[str, dict[AccelerationLevel, Callable]] = {}

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Lazily initialize by detecting the best level."""
        if not cls._initialized:
            cls._detected_level = _detect_best_level()
            cls._current_level = cls._detected_level
            cls._initialized = True
            logger.info(
                "USN acceleration level auto-detected: %s",
                cls._current_level.name,
            )

    @classmethod
    def detect_best_level(cls) -> AccelerationLevel:
        """Detect and return the best available acceleration level.

        This re-runs detection (useful if hardware state has changed).

        Returns:
            The best AccelerationLevel available on this system.
        """
        cls._detected_level = _detect_best_level()
        cls._current_level = cls._detected_level
        cls._initialized = True
        return cls._detected_level

    @classmethod
    def set_level(cls, level: int | AccelerationLevel) -> None:
        """Manually set the acceleration level.

        Use this to override auto-detection, e.g., for debugging or
        benchmarking different backends.

        Args:
            level: The desired acceleration level (int 1-4 or AccelerationLevel).

        Raises:
            ValueError: If level is not a valid AccelerationLevel value.
        """
        cls._ensure_initialized()
        try:
            cls._current_level = AccelerationLevel(level)
        except ValueError:
            valid = [f"{l.name}={l.value}" for l in AccelerationLevel]
            raise ValueError(
                f"Invalid acceleration level: {level!r}. Valid levels: {', '.join(valid)}"
            ) from None
        logger.info("USN acceleration level set to: %s", cls._current_level.name)

    @classmethod
    def get_level(cls) -> AccelerationLevel:
        """Get the current acceleration level.

        Returns:
            The currently active AccelerationLevel.
        """
        cls._ensure_initialized()
        return cls._current_level

    @classmethod
    def get_kernel(cls, kernel_name: str) -> Callable:
        """Get the appropriate kernel implementation for the current level.

        Looks up the kernel in the registry for the current acceleration level.
        If the kernel is not available at the current level, falls back to the
        next available level (higher number = more compatible).

        Args:
            kernel_name: Name of the kernel to retrieve (e.g., "projections",
                "temporal_gate", "state_core", "channel_mlp").

        Returns:
            Callable implementing the requested kernel at the best available level.

        Raises:
            KeyError: If kernel_name is not registered at any level.
        """
        cls._ensure_initialized()

        if kernel_name not in cls._kernel_registry:
            raise KeyError(
                f"Unknown kernel '{kernel_name}'. "
                f"Registered kernels: {list(cls._kernel_registry.keys())}"
            )

        level_map = cls._kernel_registry[kernel_name]

        # Try current level, then fall back to higher (more compatible) levels
        for level in sorted(AccelerationLevel, key=lambda l: l.value):
            if level.value >= cls._current_level.value and level in level_map:
                return level_map[level]

        # Should not reach here if EAGER is always registered
        raise KeyError(
            f"No implementation found for kernel '{kernel_name}' "
            f"at level {cls._current_level.name} or below."
        )

    @classmethod
    def register_kernel(
        cls,
        kernel_name: str,
        level: AccelerationLevel,
        implementation: Callable,
    ) -> None:
        """Register a kernel implementation at a specific acceleration level.

        This is used by the backend modules (triton_kernels, fallbacks) to
        populate the registry.

        Args:
            kernel_name: Name of the kernel (e.g., "projections").
            level: The acceleration level this implementation targets.
            implementation: The callable implementing the kernel.
        """
        if kernel_name not in cls._kernel_registry:
            cls._kernel_registry[kernel_name] = {}
        cls._kernel_registry[kernel_name][level] = implementation

    @classmethod
    def reset(cls) -> None:
        """Reset manager to uninitialized state. Useful for testing."""
        cls._current_level = AccelerationLevel.EAGER
        cls._detected_level = AccelerationLevel.EAGER
        cls._initialized = False
        cls._kernel_registry = {}
