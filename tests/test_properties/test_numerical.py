"""Property 10: Log-Space Numerical Stability."""

import torch

from usn.layers.parallel_scan import parallel_scan_semantic


def test_long_sequence_no_nan():
    """Validates: Requirements 10 - Log-space scan remains finite over long sequences."""
    batch, seq_len, d_s = 1, 10000, 8
    log_decays = torch.full((batch, seq_len, d_s), -5.0)  # exp(-5) ≈ 0.007
    values = torch.randn(batch, seq_len, d_s) * 0.01
    s0 = torch.randn(batch, d_s)
    result = parallel_scan_semantic(log_decays, values, s0)
    assert torch.isfinite(result).all()
    assert not torch.isnan(result).any()
