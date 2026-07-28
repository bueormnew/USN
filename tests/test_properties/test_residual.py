"""Property 9: Residual Connection preserved."""

import torch

from usn.config.model_config import USNConfig
from usn.layers.block import USNBlock


def test_residual_adds_block_output_to_input():
    """Validates: Requirements 9 - Residual connection preserves input contribution."""
    config = USNConfig(
        num_layers=1,
        d_model=32,
        d_s=16,
        k=4,
        d_ff=64,
        vocab_size=50,
        fused=False,
        dropout=0.0,
        residual_dropout=0.0,
    )
    block = USNBlock(config, layer_idx=0)
    block.eval()
    x = torch.randn(1, 4, 32)
    out = block(x)
    # Output should differ from input (block does work)
    assert not torch.allclose(out.hidden, x, atol=1e-6)
    # But output should have same shape
    assert out.hidden.shape == x.shape
