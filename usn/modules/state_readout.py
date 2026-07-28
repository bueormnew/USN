"""State Readout module for USN architecture.

Extracts information from both semantic and relational state subspaces,
combines them, and applies a confidence gate for controlled output.

Equations:
    z_t = W_s s_t + W_r vec(R_t)
    c_t = σ(W_c m_t + b_c)
    o_t = c_t ⊙ z_t

Objective: Extract and gate state information.
Complexity: O(d_model × d_s + d_model × k²) per timestep.
Constraints: c_t bounded in (0,1).
"""

import torch.nn as nn
from torch import Tensor

from usn.core.base import USNModule


class StateReadout(USNModule):
    """State readout with confidence gating.

    Combines information from both the semantic state vector s_t and
    the relational state matrix R_t into a single d_model-dimensional
    representation, then applies a learned confidence gate to control
    how much state information flows to downstream modules.

    The confidence gate c_t ∈ (0, 1) prevents noisy or stale state
    information from corrupting the output.

    Args:
        d_model: Model dimension (output dimension).
        d_s: Semantic state dimension.
        k: Relational state matrix size (R_t is k × k).
    """

    def __init__(self, d_model: int, d_s: int, k: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_s = d_s
        self.k = k

        # W_s ∈ R^{d_model × d_s}: project semantic state to model dim
        self.semantic_proj = nn.Linear(d_s, d_model, bias=False)

        # W_r ∈ R^{d_model × k²}: project vectorized relational state
        self.relational_proj = nn.Linear(k * k, d_model, bias=False)

        # W_c ∈ R^{d_model × d_model}, b_c ∈ R^{d_model}: confidence gate
        self.confidence_gate = nn.Linear(d_model, d_model)

        self.reset_parameters()

    @property
    def objective(self) -> str:
        return "Extract and gate state information"

    @property
    def complexity(self) -> str:
        return "O(d_model × d_s + d_model × k²) per timestep"

    @property
    def constraints(self) -> list[str]:
        return [
            "c_t bounded in (0, 1) via sigmoid activation",
            "Combines both semantic and relational state subspaces",
            "vec(R_t) flattens relational matrix to k²-dimensional vector",
        ]

    def reset_parameters(self) -> None:
        """Xavier uniform for projection weights, zeros for confidence bias."""
        nn.init.xavier_uniform_(self.semantic_proj.weight)
        nn.init.xavier_uniform_(self.relational_proj.weight)
        nn.init.xavier_uniform_(self.confidence_gate.weight)
        nn.init.zeros_(self.confidence_gate.bias)

    def forward(self, s: Tensor, R: Tensor, m: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute gated state readout.

        Args:
            s: Semantic state (batch, seq, d_s).
            R: Relational state (batch, seq, k, k).
            m: Temporal mix output (batch, seq, d_model).

        Returns:
            Tuple of:
                o_t: Gated output (batch, seq, d_model).
                c_t: Confidence gate values (batch, seq, d_model), in (0, 1).
                z_t: Raw state readout (batch, seq, d_model).
        """
        batch, seq, k1, k2 = R.shape

        # vec(R_t): flatten (batch, seq, k, k) → (batch, seq, k²)
        R_vec = R.reshape(batch, seq, k1 * k2)

        # z_t = W_s s_t + W_r vec(R_t)
        z_t = self.semantic_proj(s) + self.relational_proj(R_vec)

        # c_t = σ(W_c m_t + b_c)
        c_t = self.confidence_gate(m).sigmoid()

        # o_t = c_t ⊙ z_t
        o_t = c_t * z_t

        return o_t, c_t, z_t
