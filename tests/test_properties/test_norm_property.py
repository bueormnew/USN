"""Property 11: RMSNorm Output Scale."""

import torch

from usn.layers.norm import RMSNorm


def test_rmsnorm_output_rms_approx_one():
    """Validates: Requirements 11 - RMSNorm output has approximately unit RMS."""
    norm = RMSNorm(64)
    x = torch.randn(4, 16, 64) * 10.0
    out = norm(x)
    rms = out.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)
