"""Complete USN Model: Embedding → N × Block → Norm → Output Head.

Achieves O(n) training complexity via parallel scan.
Achieves O(1) inference memory via constant-size state.
Contains NO attention mechanism or quadratic operations.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from usn.config.model_config import USNConfig
from usn.core.types import ModelState, UnifiedState
from usn.layers.block import USNBlock
from usn.layers.norm import create_norm
from usn.models.embedding import OutputHead, TokenEmbedding

logger = logging.getLogger(__name__)


class USNModel(nn.Module):
    """Complete USN model: Embedding → N × Block → Norm → Output Head.

    The USN architecture replaces attention mechanisms with a unified
    persistent state partitioned into semantic (vector) and relational
    (matrix) subspaces. Training achieves O(n) complexity via associative
    parallel scan; inference uses O(1) memory via constant-size state.

    Args:
        config: USNConfig specifying all model hyperparameters.
    """

    def __init__(self, config: USNConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding layer
        self.embedding = TokenEmbedding(
            config.vocab_size,
            config.d_model,
            scale=config.scale_embeddings,
            dropout=config.embedding_dropout,
        )

        # N stacked USN blocks
        self.blocks = nn.ModuleList(
            [USNBlock(config, layer_idx=i) for i in range(config.num_layers)]
        )

        # Final normalization
        self.final_norm = create_norm(config.norm_type, config.d_model, config.norm_eps)

        # Output head (logits projection)
        self.output_head = OutputHead(config.d_model, config.vocab_size, bias=False)

        # Weight tying: embedding and output head share weights
        if config.tie_weights:
            self.output_head.tie_weights(self.embedding.weight)

        # Gradient checkpointing flag
        self._gradient_checkpointing = False

        # Cached state for stateful inference
        self._cached_state: ModelState | None = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply weight initialization scheme based on config."""
        # The individual modules handle their own init (Xavier for projections,
        # Normal(0, 0.02) for embeddings). Here we apply any global adjustments.
        # Output head gets scaled init if not tied
        if not self.config.tie_weights:
            std = 0.02 / (2 * self.config.num_layers) ** 0.5
            nn.init.normal_(self.output_head.linear.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: Tensor,
        initial_state: ModelState | None = None,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, ModelState]:
        """Forward pass through the complete model.

        Args:
            input_ids: Token IDs (batch, seq_len).
            initial_state: Optional initial state for all layers.
                If None, zero-initialized states are used.
            padding_mask: Boolean mask (batch, seq_len) where True
                indicates valid (non-padding) positions. Currently
                reserved for future use.

        Returns:
            logits: Vocabulary logits (batch, seq_len, vocab_size).
            final_state: Updated ModelState for all layers.
        """
        # Embed tokens
        hidden = self.embedding(input_ids)  # (batch, seq, d_model)

        # Process through blocks
        final_states: list[UnifiedState] = []
        for i, block in enumerate(self.blocks):
            layer_state = initial_state.layers[i] if initial_state is not None else None

            if self._gradient_checkpointing and self.training:
                block_out = checkpoint(block, hidden, layer_state, use_reentrant=False)
            else:
                block_out = block(hidden, layer_state)

            hidden = block_out.hidden
            final_states.append(block_out.state)

        # Final normalization
        hidden = self.final_norm(hidden)

        # Project to vocabulary logits
        logits = self.output_head(hidden)  # (batch, seq, vocab_size)

        model_state = ModelState(layers=tuple(final_states))
        return logits, model_state

    # ──────────────────────────────────────────────
    # State management
    # ──────────────────────────────────────────────

    def get_state(self) -> ModelState | None:
        """Get the current cached model state.

        Returns:
            The last cached ModelState, or None if no state is cached.
        """
        return self._cached_state

    def set_state(self, state: ModelState) -> None:
        """Set the cached model state.

        Args:
            state: ModelState to cache for subsequent forward passes.
        """
        self._cached_state = state

    def get_initial_state(
        self, batch_size: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> ModelState:
        """Create zero-initialized state for all layers.

        Args:
            batch_size: Batch size for state tensors.
            device: Target device.
            dtype: Data type for state tensors.

        Returns:
            ModelState with zero-initialized states for all layers.
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        layers = tuple(
            UnifiedState(
                semantic=torch.zeros(batch_size, self.config.d_s, device=device, dtype=dtype),
                relational=torch.zeros(
                    batch_size, self.config.k, self.config.k, device=device, dtype=dtype
                ),
                u_prev=None,
            )
            for _ in range(self.config.num_layers)
        )
        return ModelState(layers=layers)

    def reset_state(self) -> None:
        """Clear any cached state (resets to None for fresh inference)."""
        self._cached_state = None

    # ──────────────────────────────────────────────
    # Gradient checkpointing
    # ──────────────────────────────────────────────

    def enable_gradient_checkpointing(self, level: str = "per_block") -> None:
        """Enable gradient checkpointing for memory-efficient training.

        Args:
            level: Checkpointing granularity. Currently supports "per_block".
        """
        self._gradient_checkpointing = True
        logger.info(f"Gradient checkpointing enabled (level={level})")

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    # ──────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        """Total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def state_size_per_layer(self) -> int:
        """Number of state floats per layer: d_s + k²."""
        return self.config.d_s + self.config.k**2

    @property
    def total_state_size(self) -> int:
        """Total state floats across all layers: num_layers × (d_s + k²)."""
        return self.config.num_layers * self.state_size_per_layer

    # ──────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────

    def summary(self) -> str:
        """Generate a human-readable model summary.

        Returns:
            Multi-line string with architecture details.
        """
        lines = [
            "USN Model Summary",
            "=" * 50,
            f"  Layers:             {self.config.num_layers}",
            f"  d_model:            {self.config.d_model}",
            f"  d_s (semantic):     {self.config.d_s}",
            f"  k (relational):     {self.config.k}",
            f"  d_ff:               {self.config.d_ff}",
            f"  Vocab size:         {self.config.vocab_size}",
            f"  Max seq len:        {self.config.max_seq_len}",
            f"  Norm type:          {self.config.norm_type}",
            f"  Activation:         {self.config.activation}",
            f"  Tie weights:        {self.config.tie_weights}",
            "-" * 50,
            f"  Total parameters:   {self.num_parameters:,}",
            f"  Trainable params:   {self.num_trainable_parameters:,}",
            f"  State/layer:        {self.state_size_per_layer:,} floats",
            f"  Total state:        {self.total_state_size:,} floats",
            "=" * 50,
        ]
        return "\n".join(lines)
