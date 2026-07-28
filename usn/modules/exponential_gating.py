"""Exponential Gating module for USN architecture.

Computes bounded decay factors that control state memory persistence.
Uses exp(-softplus(·)) to guarantee output strictly in (0, 1).

Equations:
    λ_t = exp(-softplus(W_λ x_t + b_λ))  ∈ (0, 1) for semantic state
    ρ_t = exp(-softplus(W_ρ x_t + b_ρ))  ∈ (0, 1) for relational state

Objective: Bounded decay for state memory control.
Complexity: O(d_s + 1) per timestep.
Constraints: Output strictly in (0,1), numerically stable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from usn.core.base import USNModule


class ExponentialGating(USNModule):
    """Exponential decay gating: λ_t = exp(-softplus(W x + b)).

    Produces bounded decay factors for both semantic state (vector)
    and relational state (scalar). The exp(-softplus(·)) construction
    mathematically guarantees outputs in (0, 1) for all inputs.

    The bias is initialized so that initial decay values are in
    [0.9, 0.99] for stable early training (long initial memory).

    Args:
        d_model: Input dimension.
        d_s: Semantic state dimension (output dimension for λ_t).
    """

    def __init__(self, d_model: int, d_s: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_s = d_s

        # Semantic decay: λ_t ∈ R^{d_s}
        self.semantic_proj = nn.Linear(d_model, d_s)

        # Relational decay: ρ_t ∈ R^{1} (scalar, broadcast over k×k)
        self.relational_proj = nn.Linear(d_model, 1)

        self.reset_parameters()

    @property
    def objective(self) -> str:
        return "Bounded decay for state memory control"

    @property
    def complexity(self) -> str:
        return "O(d_s + 1) per timestep"

    @property
    def constraints(self) -> list[str]:
        return [
            "λ_t strictly in (0, 1) by exp(-softplus(·)) construction",
            "ρ_t strictly in (0, 1) by same construction",
            "Numerically stable: softplus clamped for large inputs",
            "Initial λ in [0.9, 0.99] via bias initialization",
        ]

    def reset_parameters(self) -> None:
        """Initialize for stable training.

        W_λ, W_ρ: Xavier uniform
        b_λ, b_ρ: Uniform so that initial λ = exp(-softplus(b)) ∈ [0.9, 0.99]

        Derivation:
            exp(-softplus(b)) ∈ [0.9, 0.99]
            softplus(b) ∈ [-ln(0.99), -ln(0.9)] = [0.01005, 0.10536]
            b = ln(exp(target) - 1) → b ∈ [-4.595, -2.197]
        """
        nn.init.xavier_uniform_(self.semantic_proj.weight)
        nn.init.xavier_uniform_(self.relational_proj.weight)

        # Initialize bias so exp(-softplus(b)) ∈ [0.9, 0.99]
        # b ∈ [ln(exp(0.01005) - 1), ln(exp(0.10536) - 1)] ≈ [-4.595, -2.197]
        nn.init.uniform_(self.semantic_proj.bias, -4.595, -2.197)
        nn.init.uniform_(self.relational_proj.bias, -4.595, -2.197)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute decay factors.

        Args:
            x: Input tensor (batch_size, seq_len, d_model).

        Returns:
            Tuple of:
                lambda_t: Semantic decay (batch, seq, d_s), values in (0, 1).
                rho_t: Relational decay (batch, seq, 1), values in (0, 1).
        """
        # Semantic decay: λ_t = exp(-softplus(W_λ x + b_λ))
        semantic_pre = self.semantic_proj(x)
        # Numerically stable: clamp pre-activation for strict (0, 1) in float32
        # Upper clamp at 20: softplus(20) ≈ 20, exp(-20) ≈ 2e-9 > 0
        # Lower clamp at -15: softplus(-15) ≈ 3e-7, exp(-3e-7) < 1.0
        semantic_pre = semantic_pre.clamp(min=-15.0, max=20.0)
        lambda_t = torch.exp(-F.softplus(semantic_pre))

        # Relational decay: ρ_t = exp(-softplus(W_ρ x + b_ρ))
        relational_pre = self.relational_proj(x)
        relational_pre = relational_pre.clamp(min=-15.0, max=20.0)
        rho_t = torch.exp(-F.softplus(relational_pre))

        return lambda_t, rho_t
