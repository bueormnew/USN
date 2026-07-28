"""Local Temporal Mixing module for USN architecture.

Blends the current timestep representation with the previous timestep
using a learned sigmoid gate, capturing immediate temporal context.

Equations:
    α_t = σ(W_α x_t + b_α)
    m_t = α_t ⊙ u_t + (1 - α_t) ⊙ u_{t-1}

Objective: Local temporal context blending with one-step lookback.
Complexity: O(d_model) per timestep (element-wise operations + one projection).
Constraints: Causal (one-step lookback only), never accesses future values.
"""

import torch
import torch.nn as nn
from torch import Tensor

from usn.core.base import USNModule


class TemporalMixing(USNModule):
    """Local temporal mixing with learned gate.

    Blends current projected input u_t with previous step u_{t-1}
    using a content-dependent gate α_t = σ(W_α x_t + b_α).

    During training, all positions are computed in parallel using
    shifted tensors. During inference, u_{t-1} is cached for
    single-step computation.

    Args:
        d_model: Model dimension.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.gate_proj = nn.Linear(d_model, d_model)
        self.u_prev_init = nn.Parameter(torch.zeros(d_model))
        self._cached_u_prev: Tensor | None = None
        self.reset_parameters()

    @property
    def objective(self) -> str:
        return "Local temporal context blending with one-step lookback"

    @property
    def complexity(self) -> str:
        return "O(d_model) per timestep"

    @property
    def constraints(self) -> list[str]:
        return [
            "Causal: m_t depends only on x_t, u_t, and u_{t-1}",
            "One-step lookback only (never accesses future values)",
            "O(1) additional memory for u_{t-1} cache in inference",
        ]

    def reset_parameters(self) -> None:
        """Xavier uniform for W_α, zeros for b_α."""
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.zeros_(self.u_prev_init)

    def forward(self, x: Tensor, u: Tensor, u_prev: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Apply temporal mixing.

        Args:
            x: Raw input for gate computation (batch, seq, d_model).
            u: Projected input from InputProjection (batch, seq, d_model).
            u_prev: Previous step u_{t-1} for inference (batch, 1, d_model),
                    or None for training (uses shifted tensor).

        Returns:
            Tuple of:
                m: Temporally mixed representation (batch, seq, d_model).
                u_last: Last u value for caching (batch, 1, d_model).
        """
        batch_size, seq_len, _ = u.shape

        # Compute gate: α_t = σ(W_α x_t + b_α)
        alpha = torch.sigmoid(self.gate_proj(x))

        # Build u_{t-1} sequence
        if u_prev is not None:
            # Inference: u_prev is the cached previous value
            u_shifted = torch.cat([u_prev, u[:, :-1, :]], dim=1)
        else:
            # Training: shift u right by 1, prepend learned initial
            init = self.u_prev_init.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            u_shifted = torch.cat([init, u[:, :-1, :]], dim=1)

        # m_t = α_t ⊙ u_t + (1 - α_t) ⊙ u_{t-1}
        m = alpha * u + (1.0 - alpha) * u_shifted

        # Return last u for caching in inference
        u_last = u[:, -1:, :]

        return m, u_last
