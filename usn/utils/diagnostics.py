"""Diagnostics utilities for USN models.

Provides functions to inspect gradient statistics, activation statistics,
and state health for debugging training issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class GradientStats:
    """Summary statistics for model gradients.

    Attributes:
        total_norm: Global gradient L2 norm across all parameters.
        per_layer: Per-parameter statistics.
        has_nan: Whether any gradient contains NaN.
        has_inf: Whether any gradient contains Inf.
        num_zero_grad: Number of parameters with all-zero gradients.
    """

    total_norm: float
    per_layer: dict[str, dict[str, float]]
    has_nan: bool
    has_inf: bool
    num_zero_grad: int

    def __repr__(self) -> str:
        status = "HEALTHY"
        if self.has_nan:
            status = "NaN DETECTED"
        elif self.has_inf:
            status = "Inf DETECTED"
        elif self.num_zero_grad > 0:
            status = f"WARNING: {self.num_zero_grad} zero grads"

        lines = [
            f"GradientStats (status={status}):",
            f"  total_norm={self.total_norm:.6f}",
            f"  has_nan={self.has_nan}, has_inf={self.has_inf}",
            f"  zero_grad_params={self.num_zero_grad}",
        ]
        # Show top 5 largest gradients
        sorted_layers = sorted(
            self.per_layer.items(),
            key=lambda x: x[1].get("norm", 0.0),
            reverse=True,
        )
        lines.append("  Top 5 largest gradient norms:")
        for name, stats in sorted_layers[:5]:
            lines.append(
                f"    {name}: norm={stats['norm']:.6f} "
                f"mean={stats['mean']:.6f} max={stats['max']:.6f}"
            )
        return "\n".join(lines)


@dataclass
class ActivationStats:
    """Summary statistics for model activations.

    Attributes:
        per_layer: Per-layer activation statistics.
        has_nan: Whether any activation contains NaN.
        has_inf: Whether any activation contains Inf.
    """

    per_layer: dict[str, dict[str, float]]
    has_nan: bool
    has_inf: bool

    def __repr__(self) -> str:
        status = "HEALTHY"
        if self.has_nan:
            status = "NaN DETECTED"
        elif self.has_inf:
            status = "Inf DETECTED"

        lines = [
            f"ActivationStats (status={status}):",
            f"  layers monitored: {len(self.per_layer)}",
        ]
        for name, stats in self.per_layer.items():
            lines.append(
                f"  {name}: mean={stats['mean']:.4f} "
                f"std={stats['std']:.4f} "
                f"max_abs={stats['max_abs']:.4f}"
            )
        return "\n".join(lines)


@dataclass
class StateHealthReport:
    """Health report for model state.

    Attributes:
        healthy: Overall health status.
        issues: List of identified issues.
        layer_reports: Per-layer state statistics.
    """

    healthy: bool
    issues: list[str]
    layer_reports: list[dict[str, Any]]

    def __repr__(self) -> str:
        status = "HEALTHY" if self.healthy else "UNHEALTHY"
        lines = [f"StateHealthReport (status={status}):"]
        if self.issues:
            lines.append("  Issues:")
            for issue in self.issues:
                lines.append(f"    ⚠ {issue}")
        for i, report in enumerate(self.layer_reports):
            lines.append(f"  Layer {i}:")
            lines.append(
                f"    semantic: norm={report['semantic_norm']:.4f} max={report['semantic_max']:.4f}"
            )
            lines.append(
                f"    relational: fro_norm={report['relational_norm']:.4f} "
                f"max={report['relational_max']:.4f}"
            )
        return "\n".join(lines)


def gradient_stats(model: nn.Module) -> GradientStats:
    """Compute gradient statistics for all model parameters.

    Call after loss.backward() and before optimizer.step() to inspect
    gradient health.

    Args:
        model: The model whose gradients to inspect.

    Returns:
        GradientStats with per-parameter and aggregate information.
    """
    per_layer: dict[str, dict[str, float]] = {}
    total_norm_sq = 0.0
    has_nan = False
    has_inf = False
    num_zero_grad = 0

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad.detach().float()
        grad_norm = grad.norm().item()
        grad_mean = grad.mean().item()
        grad_std = grad.std().item() if grad.numel() > 1 else 0.0
        grad_max = grad.abs().max().item()

        total_norm_sq += grad_norm**2

        if torch.isnan(grad).any():
            has_nan = True
        if torch.isinf(grad).any():
            has_inf = True
        if grad_norm == 0.0:
            num_zero_grad += 1

        per_layer[name] = {
            "norm": grad_norm,
            "mean": grad_mean,
            "std": grad_std,
            "max": grad_max,
            "numel": param.numel(),
        }

    total_norm = total_norm_sq**0.5

    return GradientStats(
        total_norm=total_norm,
        per_layer=per_layer,
        has_nan=has_nan,
        has_inf=has_inf,
        num_zero_grad=num_zero_grad,
    )


def activation_stats(
    model: nn.Module,
    input_ids: Tensor,
) -> ActivationStats:
    """Compute activation statistics by hooking into model layers.

    Runs a forward pass with hooks to capture intermediate activations
    and compute their statistics.

    Args:
        model: The model to monitor.
        input_ids: Input tensor (batch, seq_len) for the forward pass.

    Returns:
        ActivationStats with per-layer activation information.
    """
    stats: dict[str, dict[str, float]] = {}
    has_nan = False
    has_inf = False
    hooks: list[torch.utils.hooks.RemovableHook] = []

    def make_hook(name: str):
        def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
            nonlocal has_nan, has_inf
            # Handle tuple outputs (take first tensor)
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output

            if not isinstance(out, Tensor):
                return

            out_float = out.detach().float()
            if torch.isnan(out_float).any():
                has_nan = True
            if torch.isinf(out_float).any():
                has_inf = True

            stats[name] = {
                "mean": out_float.mean().item(),
                "std": out_float.std().item() if out_float.numel() > 1 else 0.0,
                "max_abs": out_float.abs().max().item(),
                "min": out_float.min().item(),
                "max": out_float.max().item(),
            }

        return hook_fn

    # Register hooks on key layers
    for name, module in model.named_modules():
        # Hook into blocks and their submodules
        if any(
            key in name
            for key in [
                "block",
                "norm",
                "input_proj",
                "temporal_mix",
                "exp_gate",
                "selective_write",
                "state_update",
                "state_readout",
                "channel_mix",
                "embedding",
                "output_head",
                "final_norm",
            ]
        ):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Run forward pass
    model.eval()
    with torch.no_grad():
        model(input_ids)

    # Remove hooks
    for h in hooks:
        h.remove()

    return ActivationStats(
        per_layer=stats,
        has_nan=has_nan,
        has_inf=has_inf,
    )


def check_state_health(
    model: nn.Module,
    state_norm_threshold: float = 1000.0,
    relational_norm_threshold: float = 500.0,
) -> StateHealthReport:
    """Check the health of the model's current state.

    Inspects cached or computed state for signs of divergence such as
    exploding norms, NaN values, or imbalanced layers.

    Args:
        model: A USN model. Must have a get_state() method or a
            _cached_state attribute returning a ModelState.
        state_norm_threshold: Alert if semantic state norm exceeds this.
        relational_norm_threshold: Alert if relational Frobenius norm
            exceeds this.

    Returns:
        StateHealthReport with health status and per-layer details.
    """
    issues: list[str] = []
    layer_reports: list[dict[str, Any]] = []
    healthy = True

    # Get state from model
    state = None
    if hasattr(model, "get_state"):
        state = model.get_state()
    elif hasattr(model, "_cached_state"):
        state = model._cached_state

    if state is None:
        # Try to generate initial state for basic check
        if hasattr(model, "get_initial_state"):
            state = model.get_initial_state(batch_size=1)
            issues.append("No cached state — showing initial state health.")
        else:
            return StateHealthReport(
                healthy=True,
                issues=["No state available to inspect."],
                layer_reports=[],
            )

    for i, layer_state in enumerate(state.layers):
        semantic = layer_state.semantic.detach().float()
        relational = layer_state.relational.detach().float()

        # Handle batched tensors
        if semantic.dim() == 2:
            semantic = semantic[0]
        if relational.dim() == 3:
            relational = relational[0]

        s_norm = semantic.norm().item()
        s_max = semantic.abs().max().item()
        s_mean = semantic.mean().item()
        r_norm = relational.norm().item()
        r_max = relational.abs().max().item()
        r_mean = relational.mean().item()

        report = {
            "semantic_norm": s_norm,
            "semantic_max": s_max,
            "semantic_mean": s_mean,
            "semantic_has_nan": bool(torch.isnan(semantic).any()),
            "semantic_has_inf": bool(torch.isinf(semantic).any()),
            "relational_norm": r_norm,
            "relational_max": r_max,
            "relational_mean": r_mean,
            "relational_has_nan": bool(torch.isnan(relational).any()),
            "relational_has_inf": bool(torch.isinf(relational).any()),
        }
        layer_reports.append(report)

        # Check for issues
        if report["semantic_has_nan"] or report["relational_has_nan"]:
            issues.append(f"Layer {i}: NaN detected in state!")
            healthy = False
        if report["semantic_has_inf"] or report["relational_has_inf"]:
            issues.append(f"Layer {i}: Inf detected in state!")
            healthy = False
        if s_norm > state_norm_threshold:
            issues.append(
                f"Layer {i}: Semantic norm ({s_norm:.1f}) exceeds "
                f"threshold ({state_norm_threshold})"
            )
            healthy = False
        if r_norm > relational_norm_threshold:
            issues.append(
                f"Layer {i}: Relational norm ({r_norm:.1f}) exceeds "
                f"threshold ({relational_norm_threshold})"
            )
            healthy = False

    return StateHealthReport(
        healthy=healthy,
        issues=issues,
        layer_reports=layer_reports,
    )
