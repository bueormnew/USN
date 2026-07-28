"""Core type definitions for the USN architecture.

Defines the fundamental data structures used throughout the library:
- UnifiedState: persistent state for a single layer (semantic + relational)
- ModelState: full state across all layers
- BlockOutput: output from a single USN block
- GenerationOutput: output from autoregressive generation
- AffineTransition: parameters for associative scan
"""

from typing import NamedTuple, Optional

import torch
from torch import Tensor


class UnifiedState(NamedTuple):
    """Persistent state for a single USN layer.

    Attributes:
        semantic: s_t ∈ R^{batch × d_s} - semantic state vector
        relational: R_t ∈ R^{batch × k × k} - relational state matrix
        u_prev: u_{t} ∈ R^{batch × 1 × d_model} - last projected input for temporal mixing
    """

    semantic: Tensor
    relational: Tensor
    u_prev: Optional[Tensor] = None


class ModelState(NamedTuple):
    """Full model state across all layers.

    Attributes:
        layers: Tuple of UnifiedState, one per layer.
    """

    layers: tuple[UnifiedState, ...]


class BlockOutput(NamedTuple):
    """Output from a single USN block.

    Attributes:
        hidden: y_t tensor of shape (batch, seq, d_model)
        state: Updated UnifiedState for this layer
    """

    hidden: Tensor
    state: UnifiedState


class GenerationOutput(NamedTuple):
    """Output from autoregressive generation.

    Attributes:
        token_ids: Generated token IDs (batch, generated_len)
        log_probs: Log-probabilities for each generated token (optional)
        final_state: Final model state after generation
    """

    token_ids: Tensor
    log_probs: Tensor | None
    final_state: ModelState


class AffineTransition(NamedTuple):
    """Affine map parameters for associative scan.

    The state transition S_t = A_t S_{t-1} + b_t is represented as
    separate components for semantic and relational subspaces.

    Attributes:
        A_semantic: λ_t decay factors (batch, seq, d_s)
        b_semantic: g_t ⊙ B_s m_t additive term (batch, seq, d_s)
        A_relational: ρ_t decay factor (batch, seq, 1) or scalar
        b_relational: (B_r m_t)(C_r m_t)^T outer product (batch, seq, k, k)
    """

    A_semantic: Tensor
    b_semantic: Tensor
    A_relational: Tensor
    b_relational: Tensor
