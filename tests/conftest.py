"""Shared test fixtures for the USN test suite.

Provides common configurations, models, and sample data for use
across all test modules. Import fixtures by name in test functions.
"""

import pytest
import torch

from usn.config.model_config import USNConfig
from usn.config.training_config import USNTrainingConfig
from usn.config.generation_config import USNGenerationConfig


@pytest.fixture
def tiny_config() -> USNConfig:
    """Minimal config for fast testing (2 layers, d_model=32)."""
    return USNConfig(
        num_layers=2,
        d_model=32,
        d_s=16,
        k=4,
        d_ff=64,
        vocab_size=100,
        max_seq_len=32,
        dropout=0.0,
        embedding_dropout=0.0,
        residual_dropout=0.0,
        tie_weights=True,
        fused=False,
    )


@pytest.fixture
def tiny_model(tiny_config):
    """Tiny USN model for fast testing."""
    from usn.models.usn_model import USNModel
    model = USNModel(tiny_config)
    model.eval()
    return model


@pytest.fixture
def sample_batch(tiny_config):
    """Sample training batch for testing."""
    batch_size = 2
    seq_len = 8
    return {
        "input_ids": torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len)),
        "targets": torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len)),
        "padding_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
    }


@pytest.fixture
def training_config() -> USNTrainingConfig:
    """Minimal training config for testing."""
    return USNTrainingConfig(
        learning_rate=1e-3,
        batch_size=4,
        max_steps=10,
        warmup_steps=2,
        mixed_precision="none",
        gradient_accumulation_steps=1,
        eval_interval=5,
        checkpoint_interval=0,
        log_interval=5,
        early_stopping_patience=0,
    )


@pytest.fixture
def generation_config() -> USNGenerationConfig:
    """Default generation config for testing."""
    return USNGenerationConfig(
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        max_new_tokens=16,
    )


@pytest.fixture
def device():
    """Best available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
