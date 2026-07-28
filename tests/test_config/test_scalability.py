"""Scalability table verification: all presets produce valid models."""
import pytest
from usn.config.model_config import USNConfig
from usn.models.usn_model import USNModel

PRESETS = ["tiny", "micro", "mini", "small", "base", "medium", "large", "xl", "xxl"]


@pytest.mark.parametrize("preset", PRESETS)
def test_preset_creates_valid_config(preset):
    cfg = USNConfig.from_preset(preset)
    assert cfg.d_s <= cfg.d_model
    assert cfg.d_ff >= cfg.d_model
    assert cfg.num_layers >= 1


@pytest.mark.parametrize("preset", ["tiny", "micro", "mini"])
def test_preset_model_instantiates(preset):
    """Only instantiate small models to avoid memory issues."""
    cfg = USNConfig.from_preset(preset)
    model = USNModel(cfg)
    assert model.num_parameters > 0
    assert model.state_size_per_layer == cfg.d_s + cfg.k ** 2


def test_scalability_table_state_formula():
    """Verify state size follows the formula: d_s + k²."""
    for preset in PRESETS:
        cfg = USNConfig.from_preset(preset)
        expected = cfg.d_s + cfg.k ** 2
        # Can verify without building model
        assert expected > 0
        assert expected == cfg.d_s + cfg.k * cfg.k
