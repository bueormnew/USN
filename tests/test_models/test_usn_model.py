"""Unit tests for USNModel: forward/backward, state management, architecture."""

import torch
import torch.nn as nn
import pytest

from usn.config.model_config import USNConfig
from usn.models.usn_model import USNModel


@pytest.fixture
def model_config():
    return USNConfig(
        num_layers=2, d_model=32, d_s=16, k=4, d_ff=64,
        vocab_size=100, max_seq_len=32, dropout=0.0,
        embedding_dropout=0.0, residual_dropout=0.0,
        tie_weights=True, fused=False,
    )


@pytest.fixture
def model(model_config):
    m = USNModel(model_config)
    m.eval()
    return m


class TestForwardBackward:
    def test_forward_output_shape(self, model, model_config):
        """Forward pass produces (batch, seq, vocab_size) logits."""
        for batch, seq in [(1, 4), (2, 8), (3, 16)]:
            input_ids = torch.randint(0, model_config.vocab_size, (batch, seq))
            logits, state = model(input_ids)
            assert logits.shape == (batch, seq, model_config.vocab_size)

    def test_backward_succeeds(self, model_config):
        """Backward pass completes without error."""
        model = USNModel(model_config)
        model.train()
        input_ids = torch.randint(0, model_config.vocab_size, (2, 8))
        logits, _ = model(input_ids)
        loss = logits.sum()
        loss.backward()  # Should not raise
        # Verify gradients exist
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestParameterCount:
    def test_parameter_count_positive(self, model):
        """Model has a reasonable positive parameter count."""
        assert model.num_parameters > 0
        assert model.num_trainable_parameters > 0
        assert model.num_trainable_parameters <= model.num_parameters

    def test_state_size(self, model, model_config):
        """State size matches config: num_layers × (d_s + k²)."""
        expected_per_layer = model_config.d_s + model_config.k ** 2
        assert model.state_size_per_layer == expected_per_layer
        assert model.total_state_size == model_config.num_layers * expected_per_layer


class TestStateManagement:
    def test_get_set_state(self, model, model_config):
        """get_state/set_state round-trip works."""
        assert model.get_state() is None  # Initially no cached state

        # Run forward to get a state
        input_ids = torch.randint(0, model_config.vocab_size, (2, 4))
        _, state = model(input_ids)
        model.set_state(state)

        retrieved = model.get_state()
        assert retrieved is not None
        assert len(retrieved.layers) == model_config.num_layers

    def test_reset_state(self, model, model_config):
        """reset_state clears cached state."""
        input_ids = torch.randint(0, model_config.vocab_size, (2, 4))
        _, state = model(input_ids)
        model.set_state(state)
        assert model.get_state() is not None

        model.reset_state()
        assert model.get_state() is None

    def test_get_initial_state(self, model, model_config):
        """get_initial_state produces zero-init state with correct shapes."""
        state = model.get_initial_state(batch_size=3)
        assert len(state.layers) == model_config.num_layers
        for layer_state in state.layers:
            assert layer_state.semantic.shape == (3, model_config.d_s)
            assert layer_state.relational.shape == (3, model_config.k, model_config.k)
            assert (layer_state.semantic == 0).all()
            assert (layer_state.relational == 0).all()


class TestArchitectureConstraints:
    def test_no_attention_ops(self, model):
        """Model contains no nn.MultiheadAttention modules."""
        for name, module in model.named_modules():
            assert not isinstance(module, nn.MultiheadAttention), (
                f"Found attention at {name}"
            )

    def test_weight_tying(self, model_config):
        """When tie_weights=True, embedding and output head share weights."""
        model = USNModel(model_config)
        emb_weight = model.embedding.embedding.weight
        out_weight = model.output_head.linear.weight
        # They should be the same tensor (data_ptr match)
        assert emb_weight.data_ptr() == out_weight.data_ptr()

    def test_no_weight_tying(self):
        """When tie_weights=False, weights are independent."""
        config = USNConfig(
            num_layers=2, d_model=32, d_s=16, k=4, d_ff=64,
            vocab_size=100, max_seq_len=32, dropout=0.0,
            embedding_dropout=0.0, residual_dropout=0.0,
            tie_weights=False, fused=False,
        )
        model = USNModel(config)
        emb_weight = model.embedding.embedding.weight
        out_weight = model.output_head.linear.weight
        assert emb_weight.data_ptr() != out_weight.data_ptr()
