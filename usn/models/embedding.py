"""Token Embedding and Output Head for USN models.

TokenEmbedding maps discrete token IDs to continuous vectors.
OutputHead projects hidden states to vocabulary logits.
These form the input and output layers of the complete USN model.
"""

import math

import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    """Learned token embeddings with optional scaling and dropout.

    Maps integer token IDs to d_model-dimensional continuous vectors.
    Optionally scales embeddings by √d_model and applies dropout.

    Args:
        vocab_size: Size of the token vocabulary.
        d_model: Embedding dimension.
        scale: Whether to scale embeddings by √d_model.
        dropout: Dropout rate applied after embedding lookup.
    """

    def __init__(
        self, vocab_size: int, d_model: int, scale: bool = False, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale_factor = math.sqrt(d_model) if scale else 1.0
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embedding with N(0, 0.02)."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    @property
    def weight(self) -> nn.Parameter:
        """Access embedding weight matrix (for weight tying)."""
        return self.embedding.weight

    @weight.setter
    def weight(self, value: nn.Parameter) -> None:
        """Set embedding weight matrix (for weight tying)."""
        self.embedding.weight = value

    def tie_weights(self, parameter: nn.Parameter) -> None:
        """Tie embedding weights to an external parameter.

        Args:
            parameter: The nn.Parameter to share (e.g., OutputHead.weight).
        """
        self.embedding.weight = parameter

    def forward(self, token_ids: Tensor) -> Tensor:
        """Look up token embeddings.

        Args:
            token_ids: Integer token IDs (batch, seq_len).

        Returns:
            Embedded vectors (batch, seq_len, d_model).
        """
        x = self.embedding(token_ids)
        x = x * self.scale_factor
        return self.dropout(x)


class OutputHead(nn.Module):
    """Linear projection to vocabulary logits (no softmax).

    Projects the final hidden state to vocabulary-sized logits
    for next-token prediction. Does NOT apply softmax — that
    is handled by the loss function.

    Args:
        d_model: Input dimension.
        vocab_size: Output dimension (vocabulary size).
        bias: Whether to include bias (default: False).
    """

    def __init__(self, d_model: int, vocab_size: int, bias: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size, bias=bias)
        self._init_weights()

    def _init_weights(self, num_layers: int = 12) -> None:
        """Initialize with N(0, 0.02 / √(2 * num_layers))."""
        std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.linear.weight, mean=0.0, std=std)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    @property
    def weight(self) -> nn.Parameter:
        """Access output weight matrix (for weight tying)."""
        return self.linear.weight

    @weight.setter
    def weight(self, value: nn.Parameter) -> None:
        """Set output weight matrix (for weight tying)."""
        self.linear.weight = value

    def tie_weights(self, parameter: nn.Parameter) -> None:
        """Tie output weights to an external parameter.

        Args:
            parameter: The nn.Parameter to share (e.g., TokenEmbedding.weight).
        """
        self.linear.weight = parameter

    def forward(self, hidden: Tensor) -> Tensor:
        """Project hidden states to logits.

        Args:
            hidden: Hidden state (batch, seq, d_model).

        Returns:
            Logits (batch, seq, vocab_size). No softmax applied.
        """
        return self.linear(hidden)
