"""Property test: Strict Causality (Property 4).

Feature: usn-architecture-library, Property 4: Strict Causality

For any input sequence and position t, modifying input at j > t does NOT
change the output at position t. Verified via Jacobian: ∂output[t]/∂input[j] = 0
for all j > t.

**Validates: Requirements 28.1–28.5, 42.1–42.10**
"""

import pytest
import torch

from usn.config.model_config import USNConfig
from usn.models.usn_model import USNModel


@pytest.fixture
def causal_config():
    """Minimal config for causality testing."""
    return USNConfig(
        num_layers=2, d_model=32, d_s=16, k=4, d_ff=64,
        vocab_size=50, max_seq_len=16, dropout=0.0,
        embedding_dropout=0.0, residual_dropout=0.0, fused=False,
    )


@pytest.mark.parametrize("seq_len", [4, 8])
@pytest.mark.parametrize("t", [0, 1, 2])
def test_causality_future_modification(causal_config, seq_len, t):
    """Feature: usn-architecture-library, Property 4: Strict Causality

    Validates: Requirements 28.1–28.5, 42.1–42.10

    Modifying any input position j > t should not change output at position t.
    """
    if t >= seq_len:
        pytest.skip("t must be < seq_len")

    model = USNModel(causal_config)
    model.eval()

    torch.manual_seed(42)
    input_ids = torch.randint(0, causal_config.vocab_size, (1, seq_len))

    with torch.no_grad():
        logits_orig, _ = model(input_ids)
        output_at_t = logits_orig[0, t].clone()

    # Modify every position j > t and verify output[t] unchanged
    for j in range(t + 1, seq_len):
        modified = input_ids.clone()
        modified[0, j] = (input_ids[0, j] + 1) % causal_config.vocab_size

        with torch.no_grad():
            logits_mod, _ = model(modified)

        assert torch.allclose(logits_mod[0, t], output_at_t, atol=1e-6), (
            f"Output at t={t} changed when modifying position j={j}"
        )
