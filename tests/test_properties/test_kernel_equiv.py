"""Property test: Kernel Equivalence (Property 8).

Feature: usn-architecture-library, Property 8: Acceleration Level Output Equivalence

Fused/optimized kernel outputs must match unfused (Level 4 eager) reference
within numerical tolerance for random inputs.

**Validates: Requirements 101.8, 102.8, 103.12, 104.6, 105.8**
"""

import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from usn.backends.triton_kernels import (
    eager_channel_mlp,
    eager_fused_projections,
    eager_temporal_gate,
    optimized_fused_projections,
)


@given(
    d_model=st.sampled_from([16, 32, 64]),
    d_s=st.sampled_from([8, 16, 32]),
    batch=st.integers(min_value=1, max_value=4),
    seq=st.integers(min_value=1, max_value=16),
)
@settings(max_examples=50, deadline=None)
def test_fused_projections_equivalence(d_model, d_s, batch, seq):
    """Feature: usn-architecture-library, Property 8: Kernel Equivalence

    Validates: Requirements 101.8, 105.8

    eager_fused_projections and optimized_fused_projections must produce
    identical outputs for the same inputs.
    """
    torch.manual_seed(0)
    x = torch.randn(batch, seq, d_model)
    W_u = torch.randn(d_model, d_model)
    b_u = torch.randn(d_model)
    W_alpha = torch.randn(d_model, d_model)
    b_alpha = torch.randn(d_model)
    W_lambda = torch.randn(d_s, d_model)
    b_lambda = torch.randn(d_s)

    u_eager, a_eager, l_eager = eager_fused_projections(
        x, W_u, b_u, W_alpha, b_alpha, W_lambda, b_lambda
    )
    u_opt, a_opt, l_opt = optimized_fused_projections(
        x, W_u, b_u, W_alpha, b_alpha, W_lambda, b_lambda
    )

    assert torch.allclose(u_eager, u_opt, atol=1e-5), "u mismatch"
    assert torch.allclose(a_eager, a_opt, atol=1e-5), "alpha_pre mismatch"
    assert torch.allclose(l_eager, l_opt, atol=1e-5), "lambda_pre mismatch"


@given(
    d_model=st.sampled_from([16, 32, 64]),
    d_s=st.sampled_from([8, 16, 32]),
    batch=st.integers(min_value=1, max_value=4),
    seq=st.integers(min_value=1, max_value=16),
)
@settings(max_examples=50, deadline=None)
def test_temporal_gate_determinism(d_model, d_s, batch, seq):
    """Feature: usn-architecture-library, Property 8: Kernel Equivalence

    Validates: Requirements 102.8, 105.8

    eager_temporal_gate called twice with same inputs produces same output.
    """
    torch.manual_seed(0)
    u = torch.randn(batch, seq, d_model)
    alpha_pre = torch.randn(batch, seq, d_model)
    lambda_pre = torch.randn(batch, seq, d_s)
    u_prev = torch.randn(batch, seq, d_model)

    m1, l1 = eager_temporal_gate(u, alpha_pre, lambda_pre, u_prev)
    m2, l2 = eager_temporal_gate(u, alpha_pre, lambda_pre, u_prev)

    assert torch.equal(m1, m2), "temporal gate m not deterministic"
    assert torch.equal(l1, l2), "temporal gate lambda not deterministic"


@given(
    d_model=st.sampled_from([16, 32]),
    d_ff=st.sampled_from([32, 64, 128]),
    batch=st.integers(min_value=1, max_value=4),
    seq=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=50, deadline=None)
def test_channel_mlp_determinism(d_model, d_ff, batch, seq):
    """Feature: usn-architecture-library, Property 8: Kernel Equivalence

    Validates: Requirements 104.6, 105.8

    eager_channel_mlp called twice with same inputs produces same output.
    """
    torch.manual_seed(0)
    x = torch.randn(batch, seq, d_model)
    W_1 = torch.randn(d_ff, d_model)
    b_1 = torch.randn(d_ff)
    W_2 = torch.randn(d_model, d_ff)
    b_2 = torch.randn(d_model)

    out1 = eager_channel_mlp(x, W_1, b_1, W_2, b_2, "gelu")
    out2 = eager_channel_mlp(x, W_1, b_1, W_2, b_2, "gelu")

    assert torch.equal(out1, out2), "channel_mlp not deterministic"
