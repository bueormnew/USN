"""Selective Writing module for USN architecture.

Computes content-dependent write gates that control what information
enters the unified persistent state. Uses both current input and
previous state to determine writing intensity.

Equation: g_t = σ(W_g m_t + U_g read(S_{t-1}) + b_g)
Where: read(S) = concat(s, vec(R))

Objective: Content-dependent filtering of state writes.
Complexity: O(d_s × d_model + d_s × (d_s + k²)) per timestep.
Constraints: g_t bounded in (0,1) via sigmoid.
"""

import torch
import torch.nn as nn
from torch import Tensor

from usn.core.base import USNModule
from usn.core.types import UnifiedState


class SelectiveWriting(USNModule):
    """Content-dependent write gate for unified state.

    Computes a write gate g_t that controls how much of the new
    information is written into the semantic state. The gate is
    conditioned on both the current temporal mix m_t and a read
    from the previous state S_{t-1}.

    The read operation concatenates the semantic state s_{t-1}
    with the vectorized relational state vec(R_{t-1}), forming
    a combined representation of dimension d_s + k².

    Equations:
        read(S_{t-1}) = concat(s_{t-1}, vec(R_{t-1}))
        g_t = σ(W_g m_t + U_g read(S_{t-1}) + b_g)

    Args:
        d_model: Model dimension (input dimension of m_t).
        d_s: Semantic state dimension (output dimension of g_t).
        k: Relational state dimension (R ∈ R^{k×k}).
    """

    def __init__(self, d_model: int, d_s: int, k: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_s = d_s
        self.k = k
        self.d_read = d_s + k * k

        # W_g ∈ R^{d_s × d_model}: projects temporal mix m_t
        self.gate_input_proj = nn.Linear(d_model, d_s, bias=False)

        # U_g ∈ R^{d_s × (d_s + k²)}: projects read(state)
        self.gate_state_proj = nn.Linear(self.d_read, d_s, bias=False)

        # b_g ∈ R^{d_s}: bias term
        self.gate_bias = nn.Parameter(torch.zeros(d_s))

        self.reset_parameters()

    @property
    def objective(self) -> str:
        return "Content-dependent filtering of state writes"

    @property
    def complexity(self) -> str:
        return "O(d_s × d_model + d_s × (d_s + k²)) per timestep"

    @property
    def constraints(self) -> list[str]:
        return [
            "g_t bounded in (0, 1) via sigmoid activation",
            "Uses only past state S_{t-1} (causal)",
            "read(S) combines both semantic and relational subspaces",
        ]

    def reset_parameters(self) -> None:
        """Xavier uniform for projections, zeros for bias."""
        nn.init.xavier_uniform_(self.gate_input_proj.weight)
        nn.init.xavier_uniform_(self.gate_state_proj.weight)
        nn.init.zeros_(self.gate_bias)

    def read_state(self, state: UnifiedState) -> Tensor:
        """Extract read vector from previous state.

        Concatenates the semantic state s_{t-1} with the flattened
        relational state vec(R_{t-1}) to form a combined read vector
        of dimension d_s + k².

        Args:
            state: Previous UnifiedState with:
                - semantic: (batch, d_s)
                - relational: (batch, k, k)

        Returns:
            Read vector of shape (batch, d_s + k²).
        """
        # semantic: (batch, d_s)
        s = state.semantic

        # relational: (batch, k, k) → (batch, k²)
        R_flat = state.relational.flatten(start_dim=1)

        # concat: (batch, d_s + k²)
        return torch.cat([s, R_flat], dim=-1)

    def forward(self, m: Tensor, prev_state: UnifiedState) -> Tensor:
        """Compute write gate g_t.

        During training (full sequence), the state read uses the initial
        state expanded to all positions (simplified approach — full
        parallel scan integration happens in StateUpdate).

        During inference (single step), reads from actual previous state.

        Args:
            m: Temporal mix (batch, seq, d_model).
            prev_state: Previous unified state with:
                - semantic: (batch, d_s)
                - relational: (batch, k, k)

        Returns:
            g_t: Write gate values (batch, seq, d_s) in (0, 1).
        """
        batch_size, seq_len, _ = m.shape

        # Compute input contribution: W_g m_t → (batch, seq, d_s)
        gate_from_input = self.gate_input_proj(m)

        # Read from previous state: (batch, d_read)
        state_read = self.read_state(prev_state)

        # Project state read: U_g read(S_{t-1}) → (batch, d_s)
        gate_from_state = self.gate_state_proj(state_read)

        # Expand state contribution to all positions: (batch, 1, d_s)
        # For training: same initial state applies to all positions
        # For inference (seq_len=1): naturally broadcasts
        gate_from_state = gate_from_state.unsqueeze(1)

        # Compute gate: g_t = σ(W_g m_t + U_g read(S_{t-1}) + b_g)
        g_t = torch.sigmoid(gate_from_input + gate_from_state + self.gate_bias)

        return g_t
