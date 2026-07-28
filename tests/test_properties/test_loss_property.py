"""Property 14: Cross-Entropy Loss Non-Negativity."""
import torch
from usn.losses.cross_entropy import USNCrossEntropyLoss


def test_loss_is_non_negative():
    """Validates: Requirements 14 - Cross-entropy loss is always non-negative."""
    loss_fn = USNCrossEntropyLoss()
    logits = torch.randn(2, 8, 50)
    targets = torch.randint(0, 50, (2, 8))
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0.0


def test_perfect_prediction_low_loss():
    """Validates: Requirements 14 - Perfect predictions yield near-zero loss."""
    loss_fn = USNCrossEntropyLoss()
    # Create logits where target token has very high score
    logits = torch.full((1, 4, 10), -10.0)
    targets = torch.zeros(1, 4, dtype=torch.long)
    logits[0, :, 0] = 100.0  # Token 0 is the target
    loss = loss_fn(logits, targets)
    assert loss.item() < 0.01  # Should be very close to 0
