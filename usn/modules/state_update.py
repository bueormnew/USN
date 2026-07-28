"""State Update module for USN architecture.

Applies the affine associative state transition to both semantic
and relational subspaces of the unified persistent state.

Equations:
    s_t = λ_t ⊙ s_{t-1} + g_t ⊙ (B_s m_t)
    R_t = ρ_t R_{t-1} + (B_r m_t)(C_r m_t)^T

Objective: Unified persistent memory update.
Complexity: O(d_s + k²) per timestep.
Constraints: Bounded state (affine, associative).
"""

import torch
import torch.nn as nn
from torch import Tensor

from usn.core.base import USNModule
from usn.core.types import UnifiedState


class StateUpdate(USNModule):
    """Unified state transition for semantic and relational subspaces.

    Computes the affine, associative state update:
        s_t = λ_t ⊙ s_{t-1} + g_t ⊙ (B_s m_t)    — semantic
        R_t = ρ_t R_{t-1} + (B_r m_t)(C_r m_t)^T  — relational (outer product)

    The transition is affine (S_t = A_t S_{t-1} + b_t) and associative,
    enabling parallel scan during training. State remains bounded because
    λ_t, ρ_t ∈ (0, 1) act as decay factors.

    Args:
        d_model: Input dimension (dimension of m_t).
        d_s: Semantic state dimension.
        k: Relational state dimension (R_t is k×k).
    """

    def __init__(self, d_model: int, d_s: int, k: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_s = d_s
        self.k = k

        # Semantic write projection: B_s ∈ R^{d_s × d_model} (no bias)
        self.B_s = nn.Linear(d_model, d_s, bias=False)

        # Relational left projection: B_r ∈ R^{k × d_model} (no bias)
        self.B_r = nn.Linear(d_model, k, bias=False)

        # Relational right projection: C_r ∈ R^{k × d_model} (no bias)
        self.C_r = nn.Linear(d_model, k, bias=False)

        self.reset_parameters()

    @property
    def objective(self) -> str:
        return "Unified persistent memory update"

    @property
    def complexity(self) -> str:
        return "O(d_s + k²) per timestep"

    @property
    def constraints(self) -> list[str]:
        return [
            "State bounded by affine transition with λ_t, ρ_t ∈ (0, 1)",
            "Transition is associative: compose(T_a, compose(T_b, T_c)) == compose(compose(T_a, T_b), T_c)",
            "s_0 = 0, R_0 = 0 unless initial state is provided",
            "Relational update uses outer product of two k-dim vectors",
        ]

    def reset_parameters(self) -> None:
        """Xavier uniform initialization for all projection weights."""
        nn.init.xavier_uniform_(self.B_s.weight)
        nn.init.xavier_uniform_(self.B_r.weight)
        nn.init.xavier_uniform_(self.C_r.weight)

    def _zero_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> UnifiedState:
        """Create zero initial state.

        Args:
            batch_size: Batch dimension.
            device: Target device.
            dtype: Target dtype.

        Returns:
            UnifiedState with s_0 = 0 and R_0 = 0.
        """
        s = torch.zeros(batch_size, self.d_s, device=device, dtype=dtype)
        R = torch.zeros(batch_size, self.k, self.k, device=device, dtype=dtype)
        return UnifiedState(semantic=s, relational=R)

    def forward_sequential(
        self,
        m: Tensor,
        lambda_t: Tensor,
        rho_t: Tensor,
        g_t: Tensor,
        prev_state: UnifiedState | None = None,
    ) -> UnifiedState:
        """Single-step state update for inference.

        Computes one timestep of the state transition. Expects inputs
        with seq_len=1.

        Args:
            m: Mixed input (batch, 1, d_model).
            lambda_t: Semantic decay (batch, 1, d_s), values in (0, 1).
            rho_t: Relational decay (batch, 1, 1), values in (0, 1).
            g_t: Write gate (batch, 1, d_s), values in (0, 1).
            prev_state: Previous UnifiedState, or None for zero init.

        Returns:
            Updated UnifiedState (semantic: (batch, d_s), relational: (batch, k, k)).
        """
        batch_size = m.shape[0]
        device = m.device
        dtype = m.dtype

        if prev_state is None:
            prev_state = self._zero_state(batch_size, device, dtype)

        # Squeeze the sequence dimension (seq_len=1)
        m_t = m[:, 0, :]  # (batch, d_model)
        lam = lambda_t[:, 0, :]  # (batch, d_s)
        rho = rho_t[:, 0, :]  # (batch, 1)
        g = g_t[:, 0, :]  # (batch, d_s)

        # Semantic state update: s_t = λ_t ⊙ s_{t-1} + g_t ⊙ (B_s m_t)
        write_semantic = self.B_s(m_t)  # (batch, d_s)
        s_t = lam * prev_state.semantic + g * write_semantic

        # Relational state update: R_t = ρ_t R_{t-1} + (B_r m_t)(C_r m_t)^T
        left = self.B_r(m_t)  # (batch, k)
        right = self.C_r(m_t)  # (batch, k)
        # Outer product: (batch, k, 1) × (batch, 1, k) → (batch, k, k)
        outer = left.unsqueeze(-1) * right.unsqueeze(-2)
        # ρ_t is (batch, 1), expand to (batch, 1, 1) for broadcasting with (batch, k, k)
        R_t = rho.unsqueeze(-1) * prev_state.relational + outer

        return UnifiedState(semantic=s_t, relational=R_t)

    def forward_parallel(
        self,
        m: Tensor,
        lambda_t: Tensor,
        rho_t: Tensor,
        g_t: Tensor,
        initial_state: UnifiedState | None = None,
    ) -> tuple[Tensor, Tensor, UnifiedState]:
        """Full-sequence state update for training.

        Computes all intermediate states across the sequence using a
        sequential loop. (Parallel scan integration in task 7.1.)

        Args:
            m: Mixed input (batch, seq_len, d_model).
            lambda_t: Semantic decay (batch, seq_len, d_s), values in (0, 1).
            rho_t: Relational decay (batch, seq_len, 1), values in (0, 1).
            g_t: Write gate (batch, seq_len, d_s), values in (0, 1).
            initial_state: Initial UnifiedState, or None for zero init.

        Returns:
            Tuple of:
                all_s: All semantic states (batch, seq_len, d_s).
                all_R: All relational states (batch, seq_len, k, k).
                final_state: UnifiedState at the last timestep.
        """
        batch_size, seq_len, _ = m.shape
        device = m.device
        dtype = m.dtype

        if initial_state is None:
            initial_state = self._zero_state(batch_size, device, dtype)

        # Pre-compute projections for the full sequence
        write_semantic = self.B_s(m)  # (batch, seq_len, d_s)
        left = self.B_r(m)  # (batch, seq_len, k)
        right = self.C_r(m)  # (batch, seq_len, k)

        # Outer products for all timesteps: (batch, seq_len, k, k)
        outer_products = left.unsqueeze(-1) * right.unsqueeze(-2)

        # Allocate output tensors
        all_s = torch.empty(batch_size, seq_len, self.d_s, device=device, dtype=dtype)
        all_R = torch.empty(batch_size, seq_len, self.k, self.k, device=device, dtype=dtype)

        # Sequential loop (parallel scan integration comes in task 7.1)
        s_prev = initial_state.semantic  # (batch, d_s)
        R_prev = initial_state.relational  # (batch, k, k)

        for t in range(seq_len):
            # Semantic: s_t = λ_t ⊙ s_{t-1} + g_t ⊙ (B_s m_t)
            s_t = lambda_t[:, t, :] * s_prev + g_t[:, t, :] * write_semantic[:, t, :]

            # Relational: R_t = ρ_t R_{t-1} + outer_t
            # rho_t[:, t, :] is (batch, 1), needs (batch, 1, 1) for (batch, k, k) broadcast
            R_t = rho_t[:, t, :].unsqueeze(-1) * R_prev + outer_products[:, t, :, :]

            all_s[:, t, :] = s_t
            all_R[:, t, :, :] = R_t

            s_prev = s_t
            R_prev = R_t

        final_state = UnifiedState(semantic=s_prev, relational=R_prev)
        return all_s, all_R, final_state
