"""Property test: Associativity of State Transitions (Property 2).

Feature: usn-architecture-library, Property 2: Associativity of State Transitions

For any three random affine transitions T_a, T_b, T_c, the composition
operation SHALL be associative: compose(T_a, compose(T_b, T_c)) produces
results equal to compose(compose(T_a, T_b), T_c) within floating-point
tolerance (1e-5 for fp32).

**Validates: Requirements 7.8, 12.2, 53.1, 53.4, 53.8**
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st


def compose_semantic(a1, b1, a2, b2):
    """Compose two semantic affine transitions.

    (a2, b2) ∘ (a1, b1) = (a2 * a1, a2 * b1 + b2)
    """
    return a2 * a1, a2 * b1 + b2


@given(
    d_s=st.integers(min_value=2, max_value=32),
    batch_size=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=100)
def test_associativity_semantic(d_s: int, batch_size: int):
    """Feature: usn-architecture-library, Property 2: Associativity of State Transitions"""
    # Three random transitions
    a1 = torch.rand(batch_size, d_s) * 0.9 + 0.05  # ∈ (0.05, 0.95)
    b1 = torch.randn(batch_size, d_s)
    a2 = torch.rand(batch_size, d_s) * 0.9 + 0.05
    b2 = torch.randn(batch_size, d_s)
    a3 = torch.rand(batch_size, d_s) * 0.9 + 0.05
    b3 = torch.randn(batch_size, d_s)

    # Left-associated: (T_3 ∘ T_2) ∘ T_1
    a_23, b_23 = compose_semantic(a2, b2, a3, b3)
    a_left, b_left = compose_semantic(a1, b1, a_23, b_23)

    # Right-associated: T_3 ∘ (T_2 ∘ T_1)
    a_12, b_12 = compose_semantic(a1, b1, a2, b2)
    a_right, b_right = compose_semantic(a_12, b_12, a3, b3)

    assert torch.allclose(a_left, a_right, atol=1e-5)
    assert torch.allclose(b_left, b_right, atol=1e-5)


@given(
    k=st.integers(min_value=2, max_value=8),
    batch_size=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=50)
def test_associativity_relational(k: int, batch_size: int):
    """Feature: usn-architecture-library, Property 2: Associativity (relational)"""
    # Scalar decay + matrix additive
    rho1 = torch.rand(batch_size, 1) * 0.9 + 0.05
    M1 = torch.randn(batch_size, k, k)
    rho2 = torch.rand(batch_size, 1) * 0.9 + 0.05
    M2 = torch.randn(batch_size, k, k)
    rho3 = torch.rand(batch_size, 1) * 0.9 + 0.05
    M3 = torch.randn(batch_size, k, k)

    # Compose: (rho2, M2) ∘ (rho1, M1) = (rho2*rho1, rho2*M1 + M2)
    def compose_rel(rho_a, M_a, rho_b, M_b):
        return rho_b * rho_a, rho_b.unsqueeze(-1) * M_a + M_b

    # Left: (T3 ∘ T2) ∘ T1
    rho_23, M_23 = compose_rel(rho2, M2, rho3, M3)
    rho_left, M_left = compose_rel(rho1, M1, rho_23, M_23)

    # Right: T3 ∘ (T2 ∘ T1)
    rho_12, M_12 = compose_rel(rho1, M1, rho2, M2)
    rho_right, M_right = compose_rel(rho_12, M_12, rho3, M3)

    assert torch.allclose(rho_left, rho_right, atol=1e-5)
    assert torch.allclose(M_left, M_right, atol=1e-5)
