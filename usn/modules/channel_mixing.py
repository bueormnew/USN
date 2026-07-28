"""Channel Mixing (MLP) module for USN architecture.

Provides inter-channel interaction through a feedforward network
with a residual connection from the temporal mix. The input to
the MLP is the gated state readout (c_t ⊙ z_t), and the residual
comes from m_t.

Equation: y_t = m_t + W_2 φ(W_1(c_t ⊙ z_t))

Objective: Inter-channel feature mixing with residual.
Complexity: O(d_model × d_ff) per timestep.
Constraints: Residual connection mandatory, activation configurable.
"""

import torch.nn as nn
from torch import Tensor

from usn.core.activations import get_activation
from usn.core.base import USNModule


class ChannelMixing(USNModule):
    """Feedforward network with residual from temporal mix.

    Args:
        d_model: Model dimension.
        d_ff: Feedforward intermediate dimension (typically 4 × d_model).
        activation: Activation function name ("gelu", "silu", "relu").
        dropout: Dropout rate on MLP output.
    """

    def __init__(
        self, d_model: int, d_ff: int, activation: str = "gelu", dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    @property
    def objective(self) -> str:
        return "Inter-channel feature mixing with residual"

    @property
    def complexity(self) -> str:
        return "O(d_model × d_ff) per timestep"

    @property
    def constraints(self) -> list[str]:
        return [
            "Residual connection from m_t is mandatory",
            "Input to MLP is gated readout (c_t ⊙ z_t), not raw state",
            "Activation function is configurable",
        ]

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)

    def forward(self, m: Tensor, c: Tensor, z: Tensor) -> Tensor:
        """Apply channel mixing MLP with residual.

        Args:
            m: Temporal mix output (batch, seq, d_model) — used as residual.
            c: Confidence gate values (batch, seq, d_model).
            z: State readout (batch, seq, d_model).

        Returns:
            y: Block output (batch, seq, d_model).
        """
        # Input to MLP is gated readout: c_t ⊙ z_t
        mlp_input = c * z
        hidden = self.up_proj(mlp_input)
        hidden = self.activation(hidden)
        output = self.down_proj(hidden)
        output = self.dropout(output)
        # Residual from m_t
        return m + output
