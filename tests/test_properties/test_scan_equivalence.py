"""Property test: Parallel Scan Equivalence to Sequential Recurrence (Property 3).

Feature: usn-architecture-library, Property 3: Parallel Scan Equivalence

For any valid input sequence of length n (1 ≤ n ≤ 1024), random initial state,
and random transition parameters (λ, v), the parallel scan SHALL produce output
states identical to sequential recurrence computation within tolerance (1e-5 fp32).

**Validates: Requirements 12.4, 12.5, 13.4, 53.2, 53.3, 53.5**
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from usn.layers.chunked_scan import ChunkedParallelScan
from usn.layers.parallel_scan import parallel_scan_relational, parallel_scan_semantic


def sequential_scan_semantic(log_decays, values, s0):
    """Reference sequential implementation."""
    batch, seq_len, d_s = log_decays.shape
    states = torch.empty_like(values)
    s_prev = s0
    for t in range(seq_len):
        s_t = torch.exp(log_decays[:, t, :]) * s_prev + values[:, t, :]
        states[:, t, :] = s_t
        s_prev = s_t
    return states


@given(
    seq_len=st.integers(min_value=1, max_value=128),
    d_s=st.integers(min_value=2, max_value=32),
    batch=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=100)
def test_parallel_scan_matches_sequential(seq_len: int, d_s: int, batch: int):
    """Feature: usn-architecture-library, Property 3: Parallel Scan Equivalence

    **Validates: Requirements 12.4, 12.5, 13.4, 53.2, 53.3, 53.5**
    """
    # Random inputs
    log_decays = torch.randn(batch, seq_len, d_s) * 0.5 - 1.0  # mostly negative
    values = torch.randn(batch, seq_len, d_s) * 0.1
    s0 = torch.randn(batch, d_s) * 0.1

    # Parallel scan
    result = parallel_scan_semantic(log_decays, values, s0)

    # Sequential reference
    expected = sequential_scan_semantic(log_decays, values, s0)

    assert torch.allclose(result, expected, atol=1e-5), (
        f"Mismatch at seq_len={seq_len}, d_s={d_s}: "
        f"max diff={torch.abs(result - expected).max().item()}"
    )


@given(
    seq_len=st.integers(min_value=1, max_value=64),
    d_s=st.integers(min_value=2, max_value=16),
    chunk_size=st.integers(min_value=2, max_value=32),
)
@settings(max_examples=50)
def test_chunked_scan_matches_full(seq_len: int, d_s: int, chunk_size: int):
    """Feature: usn-architecture-library, Property 3: Chunked scan equivalence

    **Validates: Requirements 12.4, 12.5, 13.4, 53.2, 53.3, 53.5**
    """
    batch = 2
    log_decays = torch.randn(batch, seq_len, d_s) * 0.5 - 1.0
    values = torch.randn(batch, seq_len, d_s) * 0.1
    s0 = torch.randn(batch, d_s) * 0.1

    full = parallel_scan_semantic(log_decays, values, s0)
    chunked = ChunkedParallelScan(chunk_size=chunk_size)
    result = chunked(log_decays, values, s0)

    assert torch.allclose(full, result, atol=1e-5), (
        f"Mismatch at seq_len={seq_len}, d_s={d_s}, chunk_size={chunk_size}: "
        f"max diff={torch.abs(full - result).max().item()}"
    )
