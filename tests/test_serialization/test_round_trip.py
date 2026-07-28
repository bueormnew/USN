"""Test save/load round-trip with a tiny model produces identical outputs."""

import tempfile
from pathlib import Path

import pytest
import torch

from usn.config import USNConfig
from usn.models.usn_model import USNModel
from usn.serialization.reader import USNReader
from usn.serialization.writer import USNWriter


class TestSerializationRoundTrip:
    """Verify that saving and loading a model produces identical forward outputs."""

    @pytest.fixture
    def tiny_cfg(self):
        return USNConfig(
            num_layers=2,
            d_model=32,
            d_s=16,
            k=4,
            d_ff=64,
            vocab_size=50,
            max_seq_len=16,
            tie_weights=False,
            fused=False,
        )

    def test_round_trip_produces_identical_output(self, tiny_cfg):
        """Save a tiny model, load weights back, verify logits match."""
        torch.manual_seed(42)
        model = USNModel(tiny_cfg)
        model.eval()

        input_ids = torch.randint(0, tiny_cfg.vocab_size, (1, 8))

        with torch.no_grad():
            original_logits, _ = model(input_ids)

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.usn"
            writer = USNWriter()
            writer.save(str(path), model, config=tiny_cfg)

            # Load back
            reader = USNReader()
            data = reader.load(str(path))

            assert "config" in data
            assert "weights" in data
            assert data["config"] == tiny_cfg

            # Rebuild model with loaded weights
            loaded_model = USNModel(data["config"])
            state_dict = loaded_model.state_dict()
            # Map loaded weights back
            for name, tensor in data["weights"].items():
                if name.startswith("__buffer__."):
                    continue
                if name in state_dict:
                    state_dict[name] = tensor
            loaded_model.load_state_dict(state_dict)
            loaded_model.eval()

            with torch.no_grad():
                loaded_logits, _ = loaded_model(input_ids)

        assert torch.allclose(original_logits, loaded_logits, atol=1e-6), (
            "Loaded model output differs from original"
        )
