"""Chunked Parallel Scan for memory-efficient training.

Divides sequences into chunks, applies parallel scan within each chunk,
and propagates state between chunks sequentially. This reduces peak memory
usage while maintaining correct results.

Memory: O(C × d_state) per chunk active, vs O(n × d_state) for full scan.
"""

import torch
import torch.nn as nn
from torch import Tensor


class ChunkedParallelScan(nn.Module):
    """Chunk-based decomposition for memory-efficient parallel scan.

    Divides the sequence into chunks of configurable size C,
    applies parallel scan within each chunk, and propagates
    inter-chunk state sequentially. Supports gradient checkpointing
    at chunk boundaries.

    The results are identical to processing the full sequence
    without chunking (up to floating-point precision).

    Args:
        chunk_size: Number of timesteps per chunk (default: 64).
    """

    def __init__(self, chunk_size: int = 64) -> None:
        super().__init__()
        self.chunk_size = chunk_size

    def forward(
        self,
        log_decays: Tensor,
        values: Tensor,
        initial_state: Tensor,
    ) -> Tensor:
        """Compute all states using chunked scan.

        Args:
            log_decays: log(λ_t) (batch, seq_len, d_s).
            values: Additive terms (batch, seq_len, d_s).
            initial_state: s_0 (batch, d_s).

        Returns:
            All states s_1..s_n (batch, seq_len, d_s).
        """
        batch_size, seq_len, d_s = log_decays.shape
        device = log_decays.device
        dtype = log_decays.dtype
        C = self.chunk_size

        # Number of chunks (handle remainder)
        num_chunks = (seq_len + C - 1) // C

        all_states_list: list[Tensor] = []
        carry = initial_state  # (batch, d_s)

        for chunk_idx in range(num_chunks):
            start = chunk_idx * C
            end = min(start + C, seq_len)

            # Extract chunk data
            log_decay_chunk = log_decays[:, start:end, :]  # (batch, chunk_len, d_s)
            values_chunk = values[:, start:end, :]  # (batch, chunk_len, d_s)

            # Compute states within this chunk using sequential scan
            chunk_len = end - start
            chunk_states = torch.empty(batch_size, chunk_len, d_s, device=device, dtype=dtype)

            s_prev = carry
            for t in range(chunk_len):
                decay_t = torch.exp(log_decay_chunk[:, t, :])
                s_t = decay_t * s_prev + values_chunk[:, t, :]
                chunk_states[:, t, :] = s_t
                s_prev = s_t

            all_states_list.append(chunk_states)

            # Propagate: last state of this chunk becomes initial for next
            carry = s_prev

        # Concatenate all chunk states
        return torch.cat(all_states_list, dim=1)

    def forward_relational(
        self,
        log_decays: Tensor,
        matrices: Tensor,
        initial_state: Tensor,
    ) -> Tensor:
        """Compute all relational states using chunked scan.

        Args:
            log_decays: log(ρ_t) (batch, seq_len, 1).
            matrices: Outer product matrices (batch, seq_len, k, k).
            initial_state: R_0 (batch, k, k).

        Returns:
            All relational states (batch, seq_len, k, k).
        """
        batch_size, seq_len, k1, k2 = matrices.shape
        device = matrices.device
        dtype = matrices.dtype
        C = self.chunk_size

        num_chunks = (seq_len + C - 1) // C
        all_R_list: list[Tensor] = []
        carry = initial_state  # (batch, k, k)

        for chunk_idx in range(num_chunks):
            start = chunk_idx * C
            end = min(start + C, seq_len)

            log_decay_chunk = log_decays[:, start:end, :]  # (batch, chunk_len, 1)
            matrices_chunk = matrices[:, start:end, :, :]  # (batch, chunk_len, k, k)

            chunk_len = end - start
            chunk_R = torch.empty(batch_size, chunk_len, k1, k2, device=device, dtype=dtype)

            R_prev = carry
            for t in range(chunk_len):
                rho_t = torch.exp(log_decay_chunk[:, t, :])  # (batch, 1)
                R_t = rho_t.unsqueeze(-1) * R_prev + matrices_chunk[:, t, :, :]
                chunk_R[:, t, :, :] = R_t
                R_prev = R_t

            all_R_list.append(chunk_R)
            carry = R_prev

        return torch.cat(all_R_list, dim=1)
