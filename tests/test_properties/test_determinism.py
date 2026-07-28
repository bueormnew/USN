"""Property 13: Deterministic Initialization."""

import torch

from usn.config.model_config import USNConfig
from usn.models.usn_model import USNModel


def test_same_seed_same_params():
    """Validates: Requirements 13 - Same seed produces identical parameters."""
    config = USNConfig(
        num_layers=2,
        d_model=32,
        d_s=16,
        k=4,
        d_ff=64,
        vocab_size=50,
        fused=False,
    )
    torch.manual_seed(42)
    m1 = USNModel(config)
    torch.manual_seed(42)
    m2 = USNModel(config)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)
