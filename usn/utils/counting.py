"""Parameter counting, memory estimation, and FLOPs estimation utilities.

Provides functions to analyze USN model size, memory requirements, and
computational cost for capacity planning and performance budgeting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch.nn as nn

if TYPE_CHECKING:
    from usn.config.model_config import USNConfig


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count the total number of parameters in a model.

    Args:
        model: A PyTorch module to count parameters for.
        trainable_only: If True, count only parameters that require grad.

    Returns:
        Total number of scalar parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def estimate_memory(
    config: USNConfig,
    mode: Literal["inference", "training"] = "inference",
    batch_size: int = 1,
    seq_len: int | None = None,
    dtype_bytes: int = 4,
) -> dict[str, int]:
    """Estimate memory usage for a USN model in bytes.

    Calculates approximate memory for model parameters, state, activations,
    and optimizer state based on the configuration.

    Args:
        config: Model configuration.
        mode: Either "inference" or "training".
        batch_size: Batch size for activation memory estimation.
        seq_len: Sequence length (defaults to config.max_seq_len).
        dtype_bytes: Bytes per parameter (4 for fp32, 2 for fp16/bf16).

    Returns:
        Dictionary with memory estimates in bytes:
            - parameters: Memory for model weights.
            - state: Memory for persistent state (per batch).
            - activations: Estimated activation memory (training only).
            - optimizer: Optimizer state memory (training only, assumes AdamW).
            - total: Sum of all components.
    """
    if seq_len is None:
        seq_len = config.max_seq_len

    # Parameter count estimate (approximate formula based on USN architecture)
    # Per block: input_proj + temporal_mix + exp_gate + selective_write + state_update + state_readout + channel_mix
    d = config.d_model
    d_s = config.d_s
    k = config.k
    d_ff = config.d_ff
    n_layers = config.num_layers
    vocab = config.vocab_size

    # Per-block parameter estimate
    input_proj_params = d * d + d  # W_u, b_u
    temporal_mix_params = d * d + d + d  # W_alpha, b_alpha, u_prev_init
    exp_gate_params = d * d_s + d_s + d * 1 + 1  # W_lambda, b_lambda, W_rho, b_rho
    selective_write_params = d * d_s + (d_s + k * k) * d_s + d_s  # W_g, U_g, b_g
    state_update_params = d * d_s + d * k + d * k  # B_s, B_r, C_r
    state_readout_params = d_s * d + (k * k) * d + d * d + d  # W_s, W_r, W_c, b_c
    channel_mix_params = d * d_ff + d_ff + d_ff * d + d  # W_1, b_1, W_2, b_2
    norm_params = d  # gamma only for RMSNorm

    block_params = (
        input_proj_params
        + temporal_mix_params
        + exp_gate_params
        + selective_write_params
        + state_update_params
        + state_readout_params
        + channel_mix_params
        + norm_params
    )

    # Embedding + output head + final norm
    embedding_params = vocab * d
    output_head_params = 0 if config.tie_weights else vocab * d
    final_norm_params = d

    total_params = (
        n_layers * block_params + embedding_params + output_head_params + final_norm_params
    )
    param_memory = total_params * dtype_bytes

    # State memory: per layer = d_s (semantic) + k*k (relational), per batch element
    state_per_layer = (d_s + k * k) * dtype_bytes
    state_memory = n_layers * state_per_layer * batch_size

    # Activation memory (training only, rough estimate)
    # Each block stores intermediate activations for backward pass
    if mode == "training":
        # Approximate: each position stores ~6*d_model activations per block
        activations_per_block = batch_size * seq_len * d * 6 * dtype_bytes
        activation_memory = n_layers * activations_per_block
        # AdamW stores 2 extra copies (first + second moments)
        optimizer_memory = total_params * dtype_bytes * 2
    else:
        activation_memory = 0
        optimizer_memory = 0

    total = param_memory + state_memory + activation_memory + optimizer_memory

    return {
        "parameters": param_memory,
        "state": state_memory,
        "activations": activation_memory,
        "optimizer": optimizer_memory,
        "total": total,
    }


def estimate_flops(
    config: USNConfig,
    seq_len: int | None = None,
    batch_size: int = 1,
) -> dict[str, int]:
    """Estimate FLOPs for a single forward pass of a USN model.

    Provides approximate floating-point operation counts broken down
    by component. FLOPs here count multiply-add as 2 operations.

    Args:
        config: Model configuration.
        seq_len: Sequence length (defaults to config.max_seq_len).
        batch_size: Batch size.

    Returns:
        Dictionary with FLOPs estimates:
            - embedding: Token embedding lookup (negligible).
            - per_block: FLOPs for a single block.
            - all_blocks: FLOPs for all blocks combined.
            - output_head: Final projection to vocab.
            - total: Sum of all components.
    """
    if seq_len is None:
        seq_len = config.max_seq_len

    d = config.d_model
    d_s = config.d_s
    k = config.k
    d_ff = config.d_ff
    n_layers = config.num_layers
    vocab = config.vocab_size
    tokens = batch_size * seq_len

    # Embedding: lookup, essentially free in FLOPs (just memory access)
    embedding_flops = 0

    # Per-block FLOPs (multiply-adds counted as 2 ops each)
    input_proj_flops = tokens * 2 * d * d  # W_u x_t
    temporal_mix_flops = tokens * (2 * d * d + 3 * d)  # gate proj + blend
    exp_gate_flops = tokens * (2 * d * d_s + 2 * d * 1 + 2 * d_s)  # projections + softplus + exp
    selective_write_flops = tokens * (
        2 * d * d_s + 2 * (d_s + k * k) * d_s + d_s
    )  # gate computation
    state_update_flops = tokens * (
        2 * d * d_s + 2 * d * k * 2 + k * k + 2 * d_s
    )  # projections + outer product + decay
    state_readout_flops = tokens * (
        2 * d_s * d + 2 * (k * k) * d + 2 * d * d + d
    )  # readout + confidence gate
    channel_mix_flops = tokens * (
        2 * d * d_ff + 2 * d_ff * d + d
    )  # up + down projections + residual

    block_flops = (
        input_proj_flops
        + temporal_mix_flops
        + exp_gate_flops
        + selective_write_flops
        + state_update_flops
        + state_readout_flops
        + channel_mix_flops
    )

    all_blocks_flops = n_layers * block_flops

    # Output head: linear projection to vocab
    output_head_flops = tokens * 2 * d * vocab

    total_flops = embedding_flops + all_blocks_flops + output_head_flops

    return {
        "embedding": embedding_flops,
        "per_block": block_flops,
        "all_blocks": all_blocks_flops,
        "output_head": output_head_flops,
        "total": total_flops,
    }
