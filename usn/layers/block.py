"""USN Block layer - complete state processing cycle.

Composes all submodules in the exact order specified by the paper:
Norm → InputProjection → TemporalMixing → ExponentialGating →
SelectiveWriting → StateUpdate → StateReadout → ChannelMixing

With pre-norm residual: output = x + block(norm(x))
"""

import torch
import torch.nn as nn
from torch import Tensor

from usn.config.model_config import USNConfig
from usn.core.types import BlockOutput, UnifiedState
from usn.layers.norm import create_norm
from usn.modules.channel_mixing import ChannelMixing
from usn.modules.exponential_gating import ExponentialGating
from usn.modules.input_projection import InputProjection
from usn.modules.selective_writing import SelectiveWriting
from usn.modules.state_readout import StateReadout
from usn.modules.state_update import StateUpdate
from usn.modules.temporal_mixing import TemporalMixing


class USNBlock(nn.Module):
    """Complete USN processing block.

    Applies submodules in exact order per paper specification:
    1. Normalization (pre-norm)
    2. Input Projection
    3. Temporal Mixing
    4. Exponential Gating
    5. Selective Writing
    6. State Update (parallel scan for training, sequential for inference)
    7. State Readout
    8. Channel Mixing

    Block-level residual: output = x + dropout(block_output)

    Args:
        config: Model configuration.
        layer_idx: Index of this block in the model stack.
    """

    def __init__(self, config: USNConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # 1. Normalization (pre-norm)
        self.norm = create_norm(config.norm_type, config.d_model, config.norm_eps)

        # 2. Input Projection
        self.input_proj = InputProjection(config.d_model)

        # 3. Temporal Mixing
        self.temporal_mix = TemporalMixing(config.d_model)

        # 4. Exponential Gating
        self.exp_gate = ExponentialGating(config.d_model, config.d_s)

        # 5. Selective Writing
        self.selective_write = SelectiveWriting(config.d_model, config.d_s, config.k)

        # 6. State Update
        self.state_update = StateUpdate(config.d_model, config.d_s, config.k)

        # 7. State Readout
        self.state_readout = StateReadout(config.d_model, config.d_s, config.k)

        # 8. Channel Mixing
        self.channel_mix = ChannelMixing(
            config.d_model, config.d_ff, config.activation, config.dropout
        )

        # Residual dropout
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self,
        x: Tensor,
        state: UnifiedState | None = None,
    ) -> BlockOutput:
        """Process input through all submodules.

        Args:
            x: Input tensor (batch, seq, d_model).
            state: Previous layer state (includes u_prev for temporal mixing).

        Returns:
            BlockOutput with hidden (batch, seq, d_model) and updated state.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Initialize state if not provided
        if state is None:
            state = UnifiedState(
                semantic=torch.zeros(batch_size, self.config.d_s, device=device, dtype=dtype),
                relational=torch.zeros(
                    batch_size, self.config.k, self.config.k, device=device, dtype=dtype
                ),
                u_prev=None,
            )

        # 1. Pre-norm
        x_norm = self.norm(x)

        # 2. Input Projection: u_t = W_u norm(x) + b_u
        u = self.input_proj(x_norm)

        # 3. Temporal Mixing: m_t = α_t ⊙ u_t + (1-α_t) ⊙ u_{t-1}
        # Use u_prev from state (for inference continuity)
        m, u_last = self.temporal_mix(x_norm, u, state.u_prev)

        # 4. Exponential Gating: λ_t, ρ_t ∈ (0, 1)
        lambda_t, rho_t = self.exp_gate(x_norm)

        # 5+6. Selective Writing + State Update (FUSED per paper)
        # The paper specifies: g_t = σ(W_g m_t + U_g read(S_{t-1}) + b_g)
        # This means g_t at position t depends on state S_{t-1} from the
        # PREVIOUS timestep. Therefore, g_t computation MUST be interleaved
        # with the state update loop — it cannot be computed independently
        # for all positions in advance.
        all_s, all_R, final_state = self._fused_write_and_update(m, lambda_t, rho_t, state, u_last)

        # 7. State Readout: o_t = c_t ⊙ z_t
        o_t, c_t, z_t = self.state_readout(all_s, all_R, m)

        # 8. Channel Mixing: y_t = m_t + W_2 φ(W_1(c_t ⊙ z_t))
        y = self.channel_mix(m, c_t, z_t)

        # Block-level residual: output = x + dropout(y)
        output = x + self.residual_dropout(y)

        return BlockOutput(hidden=output, state=final_state)

    def _fused_write_and_update(
        self,
        m: Tensor,
        lambda_t: Tensor,
        rho_t: Tensor,
        initial_state: UnifiedState,
        u_last: Tensor,
    ) -> tuple[Tensor, Tensor, UnifiedState]:
        """Fused selective writing + state update as per paper.

        The paper requires g_t to depend on S_{t-1}. This means we must
        compute the write gate and state update together in a sequential
        loop, where each step reads from the actual previous state.

        This is the CORRECT implementation per the paper's pseudocode:
            For t = 1..n:
                g_t = σ(W_g m_t + U_g read(S_{t-1}) + b_g)
                s_t = λ_t ⊙ s_{t-1} + g_t ⊙ (B_s m_t)
                R_t = ρ_t R_{t-1} + (B_r m_t)(C_r m_t)^T

        Args:
            m: Temporal mix (batch, seq, d_model).
            lambda_t: Semantic decay (batch, seq, d_s).
            rho_t: Relational decay (batch, seq, 1).
            initial_state: Initial state S_0.

        Returns:
            all_s: All semantic states (batch, seq, d_s).
            all_R: All relational states (batch, seq, k, k).
            final_state: Last UnifiedState.
        """
        batch_size, seq_len, _ = m.shape
        device = m.device
        dtype = m.dtype
        d_s = self.config.d_s
        k = self.config.k

        # Pre-compute projections (these don't depend on state)
        write_semantic = self.state_update.B_s(m)  # (batch, seq, d_s)
        left = self.state_update.B_r(m)  # (batch, seq, k)
        right = self.state_update.C_r(m)  # (batch, seq, k)
        outer_products = left.unsqueeze(-1) * right.unsqueeze(-2)  # (batch, seq, k, k)

        # Pre-compute the input-dependent part of the gate: W_g m_t
        gate_from_input = self.selective_write.gate_input_proj(m)  # (batch, seq, d_s)

        # Allocate output
        all_s = torch.empty(batch_size, seq_len, d_s, device=device, dtype=dtype)
        all_R = torch.empty(batch_size, seq_len, k, k, device=device, dtype=dtype)

        # Sequential loop: compute g_t from S_{t-1}, then update state
        s_prev = initial_state.semantic  # (batch, d_s)
        R_prev = initial_state.relational  # (batch, k, k)

        for t in range(seq_len):
            # ── Selective Writing: g_t depends on S_{t-1} ──
            # read(S_{t-1}) = concat(s_{t-1}, vec(R_{t-1}))
            R_flat = R_prev.flatten(start_dim=1)  # (batch, k²)
            state_read = torch.cat([s_prev, R_flat], dim=-1)  # (batch, d_s + k²)

            # U_g read(S_{t-1})
            gate_from_state = self.selective_write.gate_state_proj(state_read)  # (batch, d_s)

            # g_t = σ(W_g m_t + U_g read(S_{t-1}) + b_g)
            g_t = torch.sigmoid(
                gate_from_input[:, t, :] + gate_from_state + self.selective_write.gate_bias
            )  # (batch, d_s)

            # ── State Update ──
            # s_t = λ_t ⊙ s_{t-1} + g_t ⊙ (B_s m_t)
            s_t = lambda_t[:, t, :] * s_prev + g_t * write_semantic[:, t, :]

            # R_t = ρ_t R_{t-1} + (B_r m_t)(C_r m_t)^T
            R_t = rho_t[:, t, :].unsqueeze(-1) * R_prev + outer_products[:, t, :, :]

            all_s[:, t, :] = s_t
            all_R[:, t, :, :] = R_t

            s_prev = s_t
            R_prev = R_t

        final_state = UnifiedState(semantic=s_prev, relational=R_prev, u_prev=u_last)
        return all_s, all_R, final_state
