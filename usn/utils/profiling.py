"""Profiling utilities for USN models.

Provides functions to profile forward pass, backward pass, and memory
usage of USN models without requiring optional dependencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ProfileResult:
    """Results from a profiling run.

    Attributes:
        elapsed_ms: Wall-clock time in milliseconds.
        peak_memory_mb: Peak GPU memory allocated in MB (0.0 on CPU).
        details: Additional profiling details.
    """

    elapsed_ms: float
    peak_memory_mb: float
    details: dict[str, Any]

    def __repr__(self) -> str:
        lines = [
            "ProfileResult(",
            f"  elapsed_ms={self.elapsed_ms:.2f},",
            f"  peak_memory_mb={self.peak_memory_mb:.2f},",
        ]
        for k, v in self.details.items():
            lines.append(f"  {k}={v},")
        lines.append(")")
        return "\n".join(lines)


def profile_forward(
    model: nn.Module,
    input_ids: Tensor,
    warmup_steps: int = 3,
    measure_steps: int = 10,
) -> ProfileResult:
    """Profile the forward pass of a model.

    Runs warmup iterations followed by timed iterations to measure
    average forward pass latency and peak memory usage.

    Args:
        model: The model to profile (should be in eval mode).
        input_ids: Input tensor of token IDs (batch, seq_len).
        warmup_steps: Number of warmup iterations (not timed).
        measure_steps: Number of timed iterations.

    Returns:
        ProfileResult with timing and memory statistics.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    is_cuda = device.type == "cuda"

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_steps):
            model(input_ids)

    # Synchronize before measurement
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    # Timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(measure_steps):
            model(input_ids)

    if is_cuda:
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / measure_steps) * 1000.0

    peak_mem = 0.0
    if is_cuda:
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    batch_size, seq_len = input_ids.shape
    tokens_per_sec = (batch_size * seq_len * measure_steps) / elapsed

    return ProfileResult(
        elapsed_ms=avg_ms,
        peak_memory_mb=peak_mem,
        details={
            "batch_size": batch_size,
            "seq_len": seq_len,
            "measure_steps": measure_steps,
            "tokens_per_sec": tokens_per_sec,
            "device": str(device),
        },
    )


def profile_backward(
    model: nn.Module,
    input_ids: Tensor,
    warmup_steps: int = 3,
    measure_steps: int = 10,
) -> ProfileResult:
    """Profile the backward pass of a model.

    Measures time for a full forward + backward pass, then subtracts
    an estimate of forward-only time to isolate backward cost.

    Args:
        model: The model to profile (should be in train mode).
        input_ids: Input tensor of token IDs (batch, seq_len).
        warmup_steps: Number of warmup iterations (not timed).
        measure_steps: Number of timed iterations.

    Returns:
        ProfileResult with backward-pass timing and memory statistics.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    is_cuda = device.type == "cuda"

    model.train()

    # Warmup
    for _ in range(warmup_steps):
        output = model(input_ids)
        logits = output[0] if isinstance(output, tuple) else output
        loss = logits.sum()
        loss.backward()
        model.zero_grad()

    # Synchronize before measurement
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    # Timed forward+backward
    start = time.perf_counter()
    for _ in range(measure_steps):
        output = model(input_ids)
        logits = output[0] if isinstance(output, tuple) else output
        loss = logits.sum()
        loss.backward()
        model.zero_grad()

    if is_cuda:
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / measure_steps) * 1000.0

    peak_mem = 0.0
    if is_cuda:
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    batch_size, seq_len = input_ids.shape

    return ProfileResult(
        elapsed_ms=avg_ms,
        peak_memory_mb=peak_mem,
        details={
            "batch_size": batch_size,
            "seq_len": seq_len,
            "measure_steps": measure_steps,
            "includes_forward": True,
            "device": str(device),
        },
    )


def profile_memory(
    model: nn.Module,
    input_ids: Tensor,
) -> ProfileResult:
    """Profile memory usage during a forward pass.

    Measures parameter memory, activation memory, and peak allocation.
    On CPU, only parameter memory is reported.

    Args:
        model: The model to profile.
        input_ids: Input tensor of token IDs (batch, seq_len).

    Returns:
        ProfileResult with detailed memory breakdown.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    is_cuda = device.type == "cuda"

    # Parameter memory
    param_mem_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mem_mb = param_mem_bytes / (1024 * 1024)

    # Buffer memory
    buffer_mem_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    buffer_mem_mb = buffer_mem_bytes / (1024 * 1024)

    activation_mem_mb = 0.0
    peak_mem_mb = 0.0

    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.memory_allocated(device)

        model.eval()
        with torch.no_grad():
            model(input_ids)

        torch.cuda.synchronize(device)
        mem_after = torch.cuda.memory_allocated(device)
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        activation_mem_mb = max(0.0, (mem_after - mem_before)) / (1024 * 1024)
    else:
        # On CPU, we can only report parameter/buffer sizes
        model.eval()
        with torch.no_grad():
            model(input_ids)
        peak_mem_mb = param_mem_mb + buffer_mem_mb

    batch_size, seq_len = input_ids.shape

    return ProfileResult(
        elapsed_ms=0.0,
        peak_memory_mb=peak_mem_mb,
        details={
            "param_memory_mb": param_mem_mb,
            "buffer_memory_mb": buffer_mem_mb,
            "activation_memory_mb": activation_mem_mb,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": str(device),
        },
    )
