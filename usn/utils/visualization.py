"""Visualization utilities for USN models.

Provides simple print-based visualization of states and gate activations
without requiring matplotlib or other optional dependencies.
"""

from __future__ import annotations

import sys
from typing import IO, TextIO

import torch
from torch import Tensor

from usn.core.types import UnifiedState


def visualize_state(
    state: UnifiedState,
    layer_idx: int = 0,
    max_dims: int = 16,
    output: TextIO | None = None,
) -> str:
    """Visualize the semantic and relational state components.

    Prints a text-based summary with statistics and ASCII bar charts
    for the semantic state vector and relational state matrix.

    Args:
        state: UnifiedState containing semantic and relational tensors.
        layer_idx: Layer index for labeling (used in output header).
        max_dims: Maximum number of dimensions to display individually.
        output: Optional text stream to write to. If None, prints to stdout.

    Returns:
        The formatted visualization string.
    """
    out = output or sys.stdout
    lines: list[str] = []

    semantic = state.semantic.detach().float()
    relational = state.relational.detach().float()

    # Handle batched state — show first sample
    if semantic.dim() == 2:
        semantic = semantic[0]
    if relational.dim() == 3:
        relational = relational[0]

    lines.append(f"╔══ Layer {layer_idx} State ══╗")
    lines.append("")

    # Semantic state summary
    lines.append(f"  Semantic State (dim={semantic.shape[0]}):")
    lines.append(
        f"    mean={semantic.mean().item():.6f}  "
        f"std={semantic.std().item():.6f}  "
        f"min={semantic.min().item():.6f}  "
        f"max={semantic.max().item():.6f}"
    )
    lines.append(
        f"    norm={semantic.norm().item():.6f}  "
        f"nonzero={semantic.nonzero().shape[0]}/{semantic.shape[0]}"
    )

    # ASCII bar chart for semantic state (first max_dims elements)
    display_dims = min(max_dims, semantic.shape[0])
    if semantic.shape[0] > 0:
        abs_max = semantic.abs().max().item()
        scale = abs_max if abs_max > 0 else 1.0
        lines.append(f"    Values (first {display_dims}):")
        for i in range(display_dims):
            val = semantic[i].item()
            bar_len = int(abs(val) / scale * 20)
            if val >= 0:
                bar = " " * 20 + "│" + "█" * bar_len
            else:
                bar = " " * (20 - bar_len) + "█" * bar_len + "│"
            lines.append(f"    [{i:3d}] {bar} {val:+.4f}")

    lines.append("")

    # Relational state summary
    k = relational.shape[0]
    lines.append(f"  Relational State ({k}×{k}):")
    lines.append(
        f"    mean={relational.mean().item():.6f}  "
        f"std={relational.std().item():.6f}  "
        f"min={relational.min().item():.6f}  "
        f"max={relational.max().item():.6f}"
    )
    fro_norm = relational.norm().item()
    lines.append(
        f"    frobenius_norm={fro_norm:.6f}  "
        f"spectral_norm={torch.linalg.norm(relational, ord=2).item():.6f}"
    )

    # Simple matrix heatmap (text-based)
    display_k = min(k, 8)
    if k > 0:
        lines.append(f"    Matrix (top-left {display_k}×{display_k}):")
        abs_max_r = relational[:display_k, :display_k].abs().max().item()
        scale_r = abs_max_r if abs_max_r > 0 else 1.0
        # Use characters for intensity
        chars = " ░▒▓█"
        header = "      " + "".join(f"{j:^3d}" for j in range(display_k))
        lines.append(header)
        for i in range(display_k):
            row_chars = []
            for j in range(display_k):
                val = relational[i, j].item()
                intensity = min(int(abs(val) / scale_r * 4), 4)
                c = chars[intensity]
                if val < 0:
                    row_chars.append(f"-{c} ")
                else:
                    row_chars.append(f" {c} ")
            lines.append(f"    {i:2d}| {''.join(row_chars)}")

    lines.append("")
    lines.append("╚" + "═" * 24 + "╝")

    result = "\n".join(lines)
    out.write(result + "\n")
    return result


def visualize_gates(
    activations: dict[str, Tensor],
    max_timesteps: int = 20,
    max_dims: int = 8,
    output: TextIO | None = None,
) -> str:
    """Visualize gate activations from a USN block.

    Displays statistics and per-timestep values for gate tensors
    such as alpha (temporal), lambda (semantic decay), rho (relational
    decay), g (write gate), and c (confidence).

    Args:
        activations: Dictionary mapping gate names to tensors.
            Expected keys include any subset of: "alpha", "lambda",
            "rho", "g", "c". Tensors should have shape
            (batch, seq, dim) or (batch, seq, 1).
        max_timesteps: Maximum number of timesteps to display.
        max_dims: Maximum number of dimensions to show per gate.
        output: Optional text stream. Defaults to stdout.

    Returns:
        The formatted visualization string.
    """
    out = output or sys.stdout
    lines: list[str] = []

    lines.append("╔══ Gate Activations ══╗")
    lines.append("")

    for name, tensor in activations.items():
        tensor = tensor.detach().float()

        # Take first batch sample
        if tensor.dim() == 3:
            tensor = tensor[0]  # (seq, dim)
        elif tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)  # (seq, 1)

        seq_len, dim = tensor.shape
        display_steps = min(max_timesteps, seq_len)
        display_dims = min(max_dims, dim)

        lines.append(f"  Gate: {name} (seq={seq_len}, dim={dim})")
        lines.append(
            f"    mean={tensor.mean().item():.4f}  "
            f"std={tensor.std().item():.4f}  "
            f"min={tensor.min().item():.4f}  "
            f"max={tensor.max().item():.4f}"
        )

        # Check bounds (gates should be in (0,1))
        in_bounds = (tensor >= 0).all() and (tensor <= 1).all()
        lines.append(f"    in [0,1]: {'yes' if in_bounds else 'NO — out of expected range!'}")

        # Saturation analysis
        near_zero = (tensor < 0.01).float().mean().item() * 100
        near_one = (tensor > 0.99).float().mean().item() * 100
        lines.append(f"    saturation: {near_zero:.1f}% near 0, {near_one:.1f}% near 1")

        # Per-timestep view for first few dims
        lines.append(f"    Timestep values (first {display_dims} dims):")
        header = "    t  │ " + " ".join(f"d{d:<3d}" for d in range(display_dims))
        lines.append(header)
        lines.append("    " + "─" * (7 + display_dims * 6))

        for t in range(display_steps):
            vals = [f"{tensor[t, d].item():.3f}" for d in range(display_dims)]
            lines.append(f"    {t:3d} │ " + " ".join(vals))

        if seq_len > max_timesteps:
            lines.append(f"    ... ({seq_len - max_timesteps} more timesteps)")

        lines.append("")

    lines.append("╚" + "═" * 24 + "╝")

    result = "\n".join(lines)
    out.write(result + "\n")
    return result
