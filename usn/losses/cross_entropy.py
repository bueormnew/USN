"""Cross-entropy loss for next-token prediction.

Numerically stable implementation using log_softmax (not softmax + log separately).
Supports label smoothing and padding mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class USNCrossEntropyLoss(nn.Module):
    """Numerically stable cross-entropy for autoregressive LM training.

    Uses F.cross_entropy which internally computes log_softmax + nll_loss
    in a single numerically stable operation.

    Args:
        label_smoothing: Label smoothing factor (default: 0.0).
        ignore_index: Token ID to ignore in loss (default: -100).
    """

    def __init__(self, label_smoothing: float = 0.0, ignore_index: int = -100) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: (batch, seq, vocab_size) — raw logits (no softmax).
            targets: (batch, seq) — target token IDs.
            mask: (batch, seq) — True for valid positions (optional).

        Returns:
            Scalar loss (mean over valid tokens).
        """
        batch, seq, vocab = logits.shape

        # If mask provided, set masked positions to ignore_index
        if mask is not None:
            targets = targets.clone()
            targets[~mask] = self.ignore_index

        # Reshape for F.cross_entropy: (N, C) and (N,)
        logits_flat = logits.reshape(-1, vocab)
        targets_flat = targets.reshape(-1)

        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )
        return loss


def compute_perplexity(loss: Tensor) -> Tensor:
    """Compute perplexity from cross-entropy loss.

    perplexity = exp(loss)

    Args:
        loss: Scalar cross-entropy loss.

    Returns:
        Scalar perplexity value.
    """
    return torch.exp(loss)
