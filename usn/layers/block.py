"""USN Block layer - complete state processing cycle.

Composes all submodules in the exact order specified by the paper:
Norm → InputProjection → TemporalMixing → ExponentialGating →
SelectiveWriting → StateUpdate → StateReadout → ChannelMixing

With pre-norm residual: output = x + block(norm(x))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _recurrence_loop(
    gate_from_input: Tensor,  # (batch, seq, d_s)
    write_semantic: Tensor,   # (batch, seq, d_s)
    outer_products: Tensor,   # (batch, seq, k, k)
    lambda_t: Tensor,         # (batch, seq, d_s)
    rho_t: Tensor,            # (batch, seq, 1)
    s_init: Tensor,           # (batch, d_s)
    R_init: Tensor,           # (batch, k, k)
    U_g_weight: Tensor,       # (d_s, d_s + k²)
    b_g: Tensor,              # (d_s,)
    seq_len: int,
    d_s: int,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Core sequential recurrence — designed to be compiled by torch.compile.

    This function contains ONLY tensor operations (no Python objects, no
    module calls, no NamedTuples) so torch.compile can fully trace and
    optimize it into a single CUDA graph.
    """
    batch = s_init.shape[0]
    device = s_init.device
    dtype = s_init.dtype

    all_s = torch.empty(batch, seq_len, d_s, device=device, dtype=dtype)
    all_R = torch.empty(batch, seq_len, k, k, device=device, dtype=dtype)

    s = s_init
    R = R_init

    for t in range(seq_len):
        # Selective writing: g_t = σ(W_g m_t + U_g [s; vec(R)] + b_g)
        R_flat = R.reshape(batch, k * k)
        state_read = torch.cat([s, R_flat], dim=-1)
        gate_from_state = F.linear(state_read, U_g_weight)
        g_t = torch.sigmoid(gate_from_input[:, t, :] + gate_from_state + b_g)

        # State update
        s = lambda_t[:, t, :] * s + g_t * write_semantic[:, t, :]
        R = rho_t[:, t, :].unsqueeze(-1) * R + outer_products[:, t, :, :]

        all_s[:, t, :] = s
        all_R[:, t, :, :] = R

    return all_s, all_R


# Try to compile the recurrence for massive speedup on GPU
# Only compile on CUDA — on CPU the eager loop is fine
_compiled_recurrence = _recurrence_loop
if torch.cuda.is_available():
    try:
        _compiled_recurrence = torch.compile(
            _recurrence_loop, mode="default", fullgraph=False
        )
    except Exception:
        _compiled_recurrence = _recurrence_loop


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

        Uses torch.compile to JIT-compile the sequential recurrence into
        an efficient CUDA kernel, eliminating Python loop overhead.

        Per paper pseudocode:
            For t = 1..n:
                g_t = σ(W_g m_t + U_g read(S_{t-1}) + b_g)
                s_t = λ_t ⊙ s_{t-1} + g_t ⊙ (B_s m_t)
                R_t = ρ_t R_{t-1} + (B_r m_t)(C_r m_t)^T
        """
        batch_size, seq_len, _ = m.shape
        d_s = self.config.d_s
        k = self.config.k

        # Pre-compute ALL input-dependent projections in parallel (batched GEMMs)
        write_semantic = self.state_update.B_s(m)      # (batch, seq, d_s)
        left = self.state_update.B_r(m)                # (batch, seq, k)
        right = self.state_update.C_r(m)               # (batch, seq, k)
        outer_products = left.unsqueeze(-1) * right.unsqueeze(-2)  # (batch, seq, k, k)
        gate_from_input = self.selective_write.gate_input_proj(m)   # (batch, seq, d_s)

        # Run the compiled sequential recurrence
        all_s, all_R = _compiled_recurrence(
            gate_from_input, write_semantic, outer_products,
            lambda_t, rho_t,
            initial_state.semantic, initial_state.relational,
            self.selective_write.gate_state_proj.weight,
            self.selective_write.gate_bias,
            seq_len, d_s, k,
        )

        s_final = all_s[:, -1, :]
        R_final = all_R[:, -1, :, :]
        final_state = UnifiedState(semantic=s_final, relational=R_final, u_prev=u_last)
        return all_s, all_R, final_state
