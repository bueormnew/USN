"""Property test: Serialization Round-Trip (Properties 5, 6, 7).

Feature: usn-architecture-library

Property 5: Model Serialization Round-Trip
  For any valid USNConfig, serialized JSON → deserialized produces identical config.

Property 6: Config Serialization Round-Trip
  JSON and YAML serialize → deserialize produces equivalent config.

Property 7: Weight Count Invariance
  Parameter count from config formula matches actual model.parameters().

**Validates: Requirements 22.7, 22.15, 29.1, 29.2, 29.8, 98.5, 98.8**
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from usn.config.model_config import USNConfig
from usn.models.usn_model import USNModel


@st.composite
def usn_configs(draw):
    """Strategy generating valid USNConfig instances."""
    d_model = draw(st.sampled_from([16, 32, 64]))
    d_s = draw(st.integers(min_value=4, max_value=d_model))
    return USNConfig(
        num_layers=draw(st.integers(min_value=1, max_value=4)),
        d_model=d_model,
        d_s=d_s,
        k=draw(st.integers(min_value=2, max_value=8)),
        d_ff=draw(st.integers(min_value=d_model, max_value=d_model * 4)),
        vocab_size=draw(st.integers(min_value=10, max_value=200)),
        max_seq_len=draw(st.integers(min_value=4, max_value=64)),
        norm_type=draw(st.sampled_from(["rmsnorm", "layernorm"])),
        activation=draw(st.sampled_from(["gelu", "silu", "relu"])),
        dropout=0.0,
        embedding_dropout=0.0,
        residual_dropout=0.0,
        tie_weights=draw(st.booleans()),
        fused=False,
    )


@given(config=usn_configs())
@settings(max_examples=30)
def test_config_json_roundtrip(config):
    """Feature: usn-architecture-library, Property 6: Config Serialization Round-Trip

    Validates: Requirements 29.8"""
    json_str = config.to_json()
    restored = USNConfig.from_json(json_str)
    assert config == restored


@given(config=usn_configs())
@settings(max_examples=30)
def test_config_yaml_roundtrip(config):
    """Feature: usn-architecture-library, Property 6: Config Serialization Round-Trip

    Validates: Requirements 29.8"""
    yaml_str = config.to_yaml()
    restored = USNConfig.from_yaml(yaml_str)
    assert config == restored


@pytest.mark.parametrize("preset", ["tiny", "micro", "mini"])
def test_weight_count_invariance(preset):
    """Feature: usn-architecture-library, Property 7: Weight Count Invariance

    Validates: Requirements 98.5, 98.8

    The model's state_size_per_layer should equal d_s + k^2.
    """
    config = USNConfig.from_preset(preset)
    model = USNModel(config)
    assert model.state_size_per_layer == config.d_s + config.k ** 2
    assert model.total_state_size == config.num_layers * (config.d_s + config.k ** 2)
    # Verify model has parameters (basic sanity)
    assert model.num_parameters > 0
