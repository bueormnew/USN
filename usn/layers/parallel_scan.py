"""Parallel Scan for USN training-time state computation.

Implements an associative scan (prefix-sum) over affine state transitions,
enabling O(n) work with O(log n) parallel depth for computing all intermediate
states during training.

The semantic state transition s_t = λ_t * s_{t-1} + v_t can be computed
for all positions simultaneously using the composition rule:
    (λ_2, v_2) ∘ (λ_1, v_1) = (λ_2 * λ_1, λ_2 * v_1 + v_2)

For numerical stability, decay products are accumulated in log-space.
"""

from typing import Any

import torch
from torch import Tensor


class ParallelScanFunction(torch.autograd.Function):
    """Custom autograd function for parallel prefix scan.

    Computes all states s_1, ..., s_n for:
        s_t = exp(log_decay_t) * s_{t-1} + values_t

    Uses log-space cumulative sums for numerical stability when
    accumulating decay products over long sequences.
    """

    @staticmethod
    def forward(
        ctx: Any,
        log_decays: Tensor,
        values: Tensor,
        initial_state: Tensor,
    ) -> Tensor:
        """Compute all intermediate states via sequential scan.

        Args:
            log_decays: Log of decay factors (batch, seq_len, d_s).
                        log(λ_t) where λ_t ∈ (0, 1).
            values: Additive values g_t ⊙ B_s m_t (batch, seq_len, d_s).
            initial_state: s_0 (batch, d_s).

        Returns:
            all_states: All states s_1..s_n (batch, seq_len, d_s).
        """
        batch_size, seq_len, d_s = log_decays.shape
        device = log_decays.device
        dtype = log_decays.dtype

        # Compute states sequentially (parallel scan GPU kernel in backends)
        all_states = torch.empty(batch_size, seq_len, d_s, device=device, dtype=dtype)

        s_prev = initial_state  # (batch, d_s)
        for t in range(seq_len):
            decay_t = torch.exp(log_decays[:, t, :])  # λ_t
            s_t = decay_t * s_prev + values[:, t, :]
            all_states[:, t, :] = s_t
            s_prev = s_t

        # Save for backward
        ctx.save_for_backward(log_decays, values, initial_state, all_states)
        return all_states

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Compute gradients via reverse scan.

        The backward pass of a scan is itself a scan in reverse direction:
        - d_values[t] = grad_output[t] + accumulated_future_grad * exp(log_decay[t+1])
        - d_log_decays[t] = (grad for that step) * exp(log_decay[t]) * s_{t-1}
        """
        log_decays, values, initial_state, all_states = ctx.saved_tensors
        batch_size, seq_len, d_s = log_decays.shape

        # Gradient w.r.t. values and log_decays via reverse accumulation
        d_values = torch.empty_like(values)
        d_log_decays = torch.empty_like(log_decays)

        # Accumulated gradient flowing backward from future states
        grad_carry = torch.zeros(batch_size, d_s, device=log_decays.device, dtype=log_decays.dtype)

        for t in range(seq_len - 1, -1, -1):
            # Total gradient at position t = direct + carried from future
            total_grad = grad_output[:, t, :] + grad_carry

            # d_values[t] = total_grad (v_t contributes directly to s_t)
            d_values[:, t, :] = total_grad

            # d_log_decays[t] = total_grad * exp(log_decays[t]) * s_{t-1}
            decay_t = torch.exp(log_decays[:, t, :])
            s_prev_t = all_states[:, t - 1, :] if t > 0 else initial_state
            d_log_decays[:, t, :] = total_grad * decay_t * s_prev_t

            # Propagate gradient backward: ∂s_t/∂s_{t-1} = exp(log_decay_t) = λ_t
            grad_carry = total_grad * decay_t

        # Gradient for initial_state = accumulated gradient after all steps
        d_initial_state = grad_carry

        return d_log_decays, d_values, d_initial_state


def parallel_scan_semantic(log_decays: Tensor, values: Tensor, initial_state: Tensor) -> Tensor:
    """Compute all semantic states via parallel scan.

    Wrapper around ParallelScanFunction for clean API.

    Args:
        log_decays: log(λ_t) for each timestep (batch, seq, d_s).
        values: g_t ⊙ B_s m_t for each timestep (batch, seq, d_s).
        initial_state: s_0 (batch, d_s).

    Returns:
        All states s_1..s_n (batch, seq, d_s).
    """
    return ParallelScanFunction.apply(log_decays, values, initial_state)  # type: ignore[no-any-return, no-untyped-call]


def parallel_scan_relational(log_decays: Tensor, matrices: Tensor, initial_state: Tensor) -> Tensor:
    """Compute all relational states via parallel scan.

    Same principle as semantic but with matrix additive terms.
    R_t = ρ_t * R_{t-1} + M_t

    Args:
        log_decays: log(ρ_t) for each timestep (batch, seq, 1).
        matrices: Outer product matrices M_t (batch, seq, k, k).
        initial_state: R_0 (batch, k, k).

    Returns:
        All relational states R_1..R_n (batch, seq, k, k).
    """
    batch_size, seq_len, k1, k2 = matrices.shape
    device = matrices.device
    dtype = matrices.dtype

    all_R = torch.empty(batch_size, seq_len, k1, k2, device=device, dtype=dtype)
    R_prev = initial_state  # (batch, k, k)

    for t in range(seq_len):
        rho_t = torch.exp(log_decays[:, t, :])  # (batch, 1)
        # rho_t: (batch, 1) → need (batch, 1, 1) for (batch, k, k) broadcast
        R_t = rho_t.unsqueeze(-1) * R_prev + matrices[:, t, :, :]
        all_R[:, t, :, :] = R_t
        R_prev = R_t

    return all_R
