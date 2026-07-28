"""End-to-end integration tests for the USN library.

Tests the full workflow: create → train → save → load → generate.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from usn.config import USNConfig, USNTrainingConfig
from usn.datasets import MathDataset
from usn.inference import USNGenerator
from usn.models import USNModel, create_model
from usn.serialization import USNReader, USNWriter
from usn.training import USNTrainer


@pytest.mark.integration
def test_create_train_save_load_generate():
    """Full workflow: create model → train → save → load → generate."""
    # Create
    config = USNConfig(
        num_layers=2,
        d_model=32,
        d_s=16,
        k=4,
        d_ff=64,
        vocab_size=19,
        max_seq_len=16,
        tie_weights=True,
        fused=False,
    )
    model = USNModel(config)

    # Train
    dataset = MathDataset(num_samples=100, max_digits=1)
    train_config = USNTrainingConfig(
        learning_rate=3e-3,
        batch_size=16,
        max_steps=10,
        warmup_steps=2,
        mixed_precision="none",
        log_interval=5,
        eval_interval=0,
        checkpoint_interval=0,
    )
    trainer = USNTrainer(model, dataset, train_config)
    result = trainer.train()
    assert result["total_steps"] == 10

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.usn"
        writer = USNWriter()
        writer.save(str(path), model, config=config)
        assert path.exists()

        # Load
        reader = USNReader()
        data = reader.load(str(path))
        loaded_model = USNModel(data["config"])
        weights = {
            k: v for k, v in data["weights"].items() if not k.startswith("__buffer__.")
        }
        loaded_model.load_state_dict(weights, strict=False)

        # Generate
        loaded_model.eval()
        gen = USNGenerator(loaded_model, dataset.tokenizer)
        output = gen.generate("1+1=", max_new_tokens=3, temperature=0)
        assert output.token_ids.shape[0] == 1


@pytest.mark.integration
def test_create_model_factory():
    """Test the create_model factory with preset."""
    model = create_model("tiny", device="cpu")
    assert model.num_parameters > 0
    # Forward pass
    ids = torch.randint(0, 1000, (1, 4))
    logits, state = model(ids)
    assert logits.shape == (1, 4, 1000)


@pytest.mark.integration
def test_streaming_generation():
    """Test streaming generation produces valid tokens."""
    config = USNConfig(
        num_layers=2,
        d_model=32,
        d_s=16,
        k=4,
        d_ff=64,
        vocab_size=19,
        max_seq_len=16,
        fused=False,
    )
    model = USNModel(config)
    model.eval()
    dataset = MathDataset(num_samples=10, max_digits=1)
    gen = USNGenerator(model, dataset.tokenizer)

    tokens = list(gen.stream("3+", max_new_tokens=5, temperature=1.0))
    # Should yield tuples of (text, id, log_prob)
    for text, tok_id, log_prob in tokens:
        assert isinstance(text, str)
        assert isinstance(tok_id, int)
        assert isinstance(log_prob, float)
        assert log_prob <= 0.0
