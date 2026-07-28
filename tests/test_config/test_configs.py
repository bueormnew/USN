"""Tests for USNConfig validation, presets, and serialization round-trip."""

import pytest

from usn.config import USNConfig, USNGenerationConfig, USNTrainingConfig
from usn.exceptions import InvalidParameterError


class TestUSNConfigValidation:
    """Test that invalid configs raise appropriate errors."""

    def test_d_s_exceeds_d_model_raises(self):
        with pytest.raises(InvalidParameterError, match="d_s"):
            USNConfig(d_model=64, d_s=128)

    def test_k_less_than_one_raises(self):
        with pytest.raises(InvalidParameterError, match="k"):
            USNConfig(k=0)

    def test_d_ff_less_than_d_model_raises(self):
        with pytest.raises(InvalidParameterError, match="d_ff"):
            USNConfig(d_model=768, d_ff=256)

    def test_negative_dropout_raises(self):
        with pytest.raises(InvalidParameterError, match="dropout"):
            USNConfig(dropout=-0.1)

    def test_valid_config_does_not_raise(self):
        cfg = USNConfig(num_layers=4, d_model=64, d_s=32, k=4, d_ff=128)
        assert cfg.num_layers == 4


class TestUSNConfigPresets:
    """Test that presets return valid USNConfig instances."""

    @pytest.mark.parametrize("name", ["tiny", "micro", "mini", "small", "base"])
    def test_from_preset_returns_usn_config(self, name):
        cfg = USNConfig.from_preset(name)
        assert isinstance(cfg, USNConfig)
        assert cfg.num_layers >= 1
        assert cfg.d_s <= cfg.d_model

    def test_from_preset_invalid_raises(self):
        with pytest.raises(InvalidParameterError):
            USNConfig.from_preset("nonexistent")


class TestUSNConfigSerialization:
    """Test JSON/YAML round-trip preserves all fields."""

    def test_json_round_trip(self):
        original = USNConfig.tiny()
        restored = USNConfig.from_json(original.to_json())
        assert original == restored

    def test_yaml_round_trip(self):
        original = USNConfig(num_layers=6, d_model=128, d_s=64, k=8, d_ff=256)
        restored = USNConfig.from_yaml(original.to_yaml())
        assert original == restored

    def test_from_dict_ignores_unknown_keys(self):
        d = {"num_layers": 4, "d_model": 64, "d_s": 32, "k": 4, "d_ff": 128, "unknown": 999}
        cfg = USNConfig.from_dict(d)
        assert cfg.num_layers == 4


class TestUSNTrainingConfig:
    """Test training config validation."""

    def test_negative_learning_rate_raises(self):
        with pytest.raises(InvalidParameterError):
            USNTrainingConfig(learning_rate=-1.0)

    def test_valid_training_config(self):
        cfg = USNTrainingConfig(learning_rate=1e-3, batch_size=8, max_steps=100)
        assert cfg.batch_size == 8


class TestUSNGenerationConfig:
    """Test generation config instantiation."""

    def test_default_generation_config(self):
        cfg = USNGenerationConfig()
        assert cfg.temperature == 1.0
        assert cfg.max_new_tokens == 256
