"""Timing and memory tracking context managers for profiling.

Provides lightweight utilities for measuring execution time and
GPU/CPU memory consumption of code blocks.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class TimerResult:
    """Result from a timer context manager."""

    name: str
    elapsed_seconds: float = 0.0

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_seconds * 1000.0

    def __repr__(self) -> str:
        return f"TimerResult(name='{self.name}', elapsed={self.elapsed_ms:.2f}ms)"


@dataclass
class MemoryResult:
    """Result from a memory_tracker context manager."""

    peak_allocated_bytes: int = 0
    allocated_delta_bytes: int = 0
    peak_reserved_bytes: int = 0
    device: str = "cpu"

    @property
    def peak_allocated_mb(self) -> float:
        """Peak allocated memory in megabytes."""
        return self.peak_allocated_bytes / (1024 * 1024)

    @property
    def allocated_delta_mb(self) -> float:
        """Change in allocated memory in megabytes."""
        return self.allocated_delta_bytes / (1024 * 1024)

    @property
    def peak_reserved_mb(self) -> float:
        """Peak reserved memory in megabytes."""
        return self.peak_reserved_bytes / (1024 * 1024)

    def __repr__(self) -> str:
        return (
            f"MemoryResult(device='{self.device}', "
            f"peak_allocated={self.peak_allocated_mb:.2f}MB, "
            f"delta={self.allocated_delta_mb:.2f}MB)"
        )


@contextmanager
def timer(name: str = "block", sync_cuda: bool = True) -> Generator[TimerResult, None, None]:
    """Context manager for measuring wall-clock execution time.

    If CUDA is available and ``sync_cuda`` is True, synchronizes CUDA
    before measuring to ensure accurate GPU timing.

    Args:
        name: Descriptive name for the timed block.
        sync_cuda: Whether to call ``torch.cuda.synchronize()`` before
            start and end measurements.

    Yields:
        A TimerResult object that is populated with elapsed time on exit.

    Example:
        >>> from usn.utils.timing import timer
        >>> with timer("forward_pass") as t:
        ...     output = model(input_ids)
        >>> print(f"{t.name}: {t.elapsed_ms:.1f}ms")
    """
    result = TimerResult(name=name)

    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    try:
        yield result
    finally:
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        result.elapsed_seconds = time.perf_counter() - start


@contextmanager
def memory_tracker(device: str | torch.device = "cuda") -> Generator[MemoryResult, None, None]:
    """Context manager for tracking GPU memory usage of a code block.

    Resets peak memory statistics before the block and records peak
    and delta memory consumption on exit.

    Args:
        device: The device to track memory for. Only CUDA devices
            are supported for detailed tracking. If CUDA is unavailable,
            yields a zeroed result.

    Yields:
        A MemoryResult populated with memory statistics on exit.

    Example:
        >>> from usn.utils.timing import memory_tracker
        >>> with memory_tracker() as mem:
        ...     tensor = torch.randn(1024, 1024, device="cuda")
        >>> print(f"Peak: {mem.peak_allocated_mb:.1f}MB")
    """
    device_str = str(device)
    result = MemoryResult(device=device_str)

    if not torch.cuda.is_available() or "cuda" not in device_str:
        yield result
        return

    # Ensure any pending ops are done
    torch.cuda.synchronize()

    # Record starting memory
    start_allocated = torch.cuda.memory_allocated(device)

    # Reset peak stats to get accurate peak for this block
    torch.cuda.reset_peak_memory_stats(device)

    try:
        yield result
    finally:
        torch.cuda.synchronize()
        end_allocated = torch.cuda.memory_allocated(device)
        result.peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
        result.allocated_delta_bytes = end_allocated - start_allocated
        result.peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
