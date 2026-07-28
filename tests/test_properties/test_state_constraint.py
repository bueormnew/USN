"""Property 12: State Norm Constraint Enforcement."""

import torch

from usn.training.stability import clip_state_norm


def test_clip_enforces_max_norm():
    """Validates: Requirements 12 - clip_state_norm enforces maximum L2 norm."""
    state = torch.randn(1, 64) * 100  # large norm
    clipped = clip_state_norm(state, max_norm=10.0)
    assert torch.norm(clipped.float()).item() <= 10.0 + 1e-5
