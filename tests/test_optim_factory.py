"""Tests for OptimizerFactory and parameter group separation."""

import pytest
import torch
import torch.nn as nn

from usn.config.training_config import USNTrainingConfig
from usn.optim.factory import OptimizerFactory


class SimpleModel(nn.Module):
    """Minimal model to test parameter grouping."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(100, 32)
        self.linear = nn.Linear(32, 64)  # weight (2D) + bias (1D)
        self.norm = nn.LayerNorm(64)  # weight + bias (1D each)
        self.output = nn.Linear(64, 100)  # weight (2D) + bias (1D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.linear(x)
        x = self.norm(x)
        return self.output(x)


class TestGetParameterGroups:
    """Tests for parameter group separation logic."""

    def test_separates_decay_and_no_decay(self) -> None:
        model = SimpleModel()
        groups = OptimizerFactory.get_parameter_groups(model, weight_decay=0.1)

        assert len(groups) == 2
        decay_group = groups[0]
        no_decay_group = groups[1]

        assert decay_group["weight_decay"] == 0.1
        assert no_decay_group["weight_decay"] == 0.0

    def test_2d_weights_in_decay_group(self) -> None:
        model = SimpleModel()
        groups = OptimizerFactory.get_parameter_groups(model, weight_decay=0.1)

        decay_params = groups[0]["params"]
        # linear.weight and output.weight are 2D and not norm/embedding
        decay_shapes = [p.shape for p in decay_params]
        # All decay params should be 2D
        assert all(len(s) >= 2 for s in decay_shapes)

    def test_biases_in_no_decay_group(self) -> None:
        model = SimpleModel()
        groups = OptimizerFactory.get_parameter_groups(model, weight_decay=0.1)

        no_decay_params = groups[1]["params"]
        # Check that 1D params (biases) are in no-decay
        one_d_params = [p for p in no_decay_params if p.dim() == 1]
        assert len(one_d_params) > 0

    def test_norm_params_in_no_decay(self) -> None:
        model = SimpleModel()
        groups = OptimizerFactory.get_parameter_groups(model, weight_decay=0.1)

        no_decay_params = set(id(p) for p in groups[1]["params"])
        # norm.weight should be in no-decay despite being a "weight"
        assert id(model.norm.weight) in no_decay_params

    def test_embedding_in_no_decay(self) -> None:
        model = SimpleModel()
        groups = OptimizerFactory.get_parameter_groups(model, weight_decay=0.1)

        no_decay_params = set(id(p) for p in groups[1]["params"])
        # embedding.weight is 2D but should be in no-decay
        assert id(model.embedding.weight) in no_decay_params

    def test_all_params_accounted_for(self) -> None:
        model = SimpleModel()
        groups = OptimizerFactory.get_parameter_groups(model, weight_decay=0.1)

        total_in_groups = len(groups[0]["params"]) + len(groups[1]["params"])
        total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert total_in_groups == total_trainable

    def test_frozen_params_excluded(self) -> None:
        model = SimpleModel()
        # Freeze the embedding
        model.embedding.weight.requires_grad = False
        groups = OptimizerFactory.get_parameter_groups(model, weight_decay=0.1)

        all_param_ids = {id(p) for p in groups[0]["params"] + groups[1]["params"]}
        assert id(model.embedding.weight) not in all_param_ids


class TestOptimizerFactoryCreate:
    """Tests for optimizer creation."""

    def test_default_adamw(self) -> None:
        model = SimpleModel()
        config = USNTrainingConfig()
        optimizer = OptimizerFactory.create(model, config)

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_adamw_hyperparams(self) -> None:
        model = SimpleModel()
        config = USNTrainingConfig(
            learning_rate=3e-4,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
            weight_decay=0.1,
        )
        optimizer = OptimizerFactory.create(model, config)

        # Check LR is set on parameter groups
        for group in optimizer.param_groups:
            assert group["lr"] == 3e-4
            assert group["betas"] == (0.9, 0.95)
            assert group["eps"] == 1e-8

    def test_adam_optimizer(self) -> None:
        model = SimpleModel()
        config = USNTrainingConfig(optimizer="adam")
        optimizer = OptimizerFactory.create(model, config)

        assert isinstance(optimizer, torch.optim.Adam)

    def test_sgd_optimizer(self) -> None:
        model = SimpleModel()
        config = USNTrainingConfig(optimizer="sgd")
        optimizer = OptimizerFactory.create(model, config)

        assert isinstance(optimizer, torch.optim.SGD)

    def test_unsupported_optimizer_raises(self) -> None:
        model = SimpleModel()
        # We need to bypass the Literal type check in training config
        # by using object.__setattr__ on frozen dataclass
        config = USNTrainingConfig.__new__(USNTrainingConfig)
        object.__setattr__(config, "optimizer", "unsupported")
        object.__setattr__(config, "learning_rate", 3e-4)
        object.__setattr__(config, "weight_decay", 0.1)
        object.__setattr__(config, "adam_beta1", 0.9)
        object.__setattr__(config, "adam_beta2", 0.95)
        object.__setattr__(config, "adam_eps", 1e-8)

        with pytest.raises(ValueError, match="Unsupported optimizer"):
            OptimizerFactory.create(model, config)

    def test_weight_decay_separation_in_optimizer(self) -> None:
        model = SimpleModel()
        config = USNTrainingConfig(weight_decay=0.1)
        optimizer = OptimizerFactory.create(model, config)

        # Should have two param groups
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["weight_decay"] == 0.1
        assert optimizer.param_groups[1]["weight_decay"] == 0.0


class TestCustomOptimizerRegistration:
    """Tests for custom optimizer registration."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        OptimizerFactory._registry.clear()

    def test_register_custom_optimizer(self) -> None:
        OptimizerFactory.register("custom", torch.optim.Adagrad)
        assert "custom" in OptimizerFactory._registry

    def test_create_with_registered_optimizer(self) -> None:
        OptimizerFactory.register("adagrad", torch.optim.Adagrad)
        model = SimpleModel()

        config = USNTrainingConfig.__new__(USNTrainingConfig)
        object.__setattr__(config, "optimizer", "adagrad")
        object.__setattr__(config, "learning_rate", 3e-4)
        object.__setattr__(config, "weight_decay", 0.1)
        object.__setattr__(config, "adam_beta1", 0.9)
        object.__setattr__(config, "adam_beta2", 0.95)
        object.__setattr__(config, "adam_eps", 1e-8)

        optimizer = OptimizerFactory.create(model, config)
        assert isinstance(optimizer, torch.optim.Adagrad)

    def test_unregister_optimizer(self) -> None:
        OptimizerFactory.register("custom", torch.optim.Adagrad)
        OptimizerFactory.unregister("custom")
        assert "custom" not in OptimizerFactory._registry
