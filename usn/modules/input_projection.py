"""Input Projection module for USN architecture.

Implements the first stage of the USN block: a linear transformation
that maps token embeddings to internal representations.

Equation: u_t = W_u x_t + b_u

Objective: Linear transformation of input embeddings to internal representation.
Complexity: O(d_model²) per timestep.
Constraints: No temporal dependency — operates independently per position.
"""

import torch.nn as nn
from torch import Tensor

from usn.core.base import USNModule


class InputProjection(USNModule):
    """Linear input projection: u_t = W_u x_t + b_u.

    Transforms token embeddings into the internal representation space
    used by subsequent modules. Operates independently per timestep
    position (no temporal dependency).

    Args:
        d_model: Model dimension (both input and output).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(d_model, d_model)
        self.reset_parameters()

    @property
    def objective(self) -> str:
        return "Linear transformation of input embeddings to internal representation"

    @property
    def complexity(self) -> str:
        return "O(d_model²) per timestep"

    @property
    def constraints(self) -> list[str]:
        return [
            "No temporal dependency",
            "Operates independently on each timestep position",
            "Input and output have same dimension d_model",
        ]

    def reset_parameters(self) -> None:
        """Xavier uniform for W_u, zeros for b_u."""
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply input projection.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Projected tensor u_t of shape (batch_size, seq_len, d_model).
        """
        return self.linear(x)
