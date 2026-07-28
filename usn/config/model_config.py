"""USN Model Configuration.

Immutable dataclass-based configuration for USN architecture models.
Provides validation, preset configurations for various model sizes,
and serialization to/from JSON and YAML.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Literal

from usn.exceptions import InvalidParameterError


@dataclass(frozen=True)
class USNConfig:
    """Immutable model configuration with full validation.

    All parameters are validated on creation. Use preset class methods
    (e.g., USNConfig.small()) to create standard configurations, or
    instantiate directly with custom parameters.
    """

    # Architecture
    num_layers: int = 12
    d_model: int = 768
    d_s: int = 512
    k: int = 16
    d_ff: int = 3072
    vocab_size: int = 50257
    max_seq_len: int = 2048

    # Normalization and activation
    norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    norm_eps: float = 1e-6
    activation: Literal["gelu", "silu", "relu"] = "gelu"

    # Regularization
    dropout: float = 0.0
    embedding_dropout: float = 0.0
    residual_dropout: float = 0.0

    # Architecture options
    tie_weights: bool = True
    scale_embeddings: bool = False
    init_method: Literal["xavier", "normal", "kaiming"] = "xavier"

    # Performance
    chunk_size: int = 64
    fused: bool = True

    def __post_init__(self) -> None:
        """Validate all parameters on creation."""
        # Integer range checks
        if self.num_layers < 1:
            raise InvalidParameterError("num_layers", self.num_layers, ">= 1")
        if self.d_model < 4:
            raise InvalidParameterError("d_model", self.d_model, ">= 4")
        if self.d_s < 1:
            raise InvalidParameterError("d_s", self.d_s, ">= 1")
        if self.d_s > self.d_model:
            raise InvalidParameterError("d_s", self.d_s, f"<= d_model ({self.d_model})")
        if self.k < 1:
            raise InvalidParameterError("k", self.k, ">= 1")
        if self.d_ff < self.d_model:
            raise InvalidParameterError("d_ff", self.d_ff, f">= d_model ({self.d_model})")
        if self.vocab_size < 2:
            raise InvalidParameterError("vocab_size", self.vocab_size, ">= 2")
        if self.max_seq_len < 1:
            raise InvalidParameterError("max_seq_len", self.max_seq_len, ">= 1")
        if self.chunk_size < 1:
            raise InvalidParameterError("chunk_size", self.chunk_size, ">= 1")

        # Dropout range checks
        for name in ("dropout", "embedding_dropout", "residual_dropout"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise InvalidParameterError(name, value, "in [0.0, 1.0]")

        # Literal/enum checks
        if self.norm_type not in ("rmsnorm", "layernorm"):
            raise InvalidParameterError(
                "norm_type", self.norm_type, "one of ('rmsnorm', 'layernorm')"
            )
        if self.activation not in ("gelu", "silu", "relu"):
            raise InvalidParameterError(
                "activation", self.activation, "one of ('gelu', 'silu', 'relu')"
            )
        if self.init_method not in ("xavier", "normal", "kaiming"):
            raise InvalidParameterError(
                "init_method", self.init_method, "one of ('xavier', 'normal', 'kaiming')"
            )

        # Warning for large relational state
        if self.k * self.k > 10 * self.d_model:
            warnings.warn(
                f"k²={self.k**2} > 10×d_model={10 * self.d_model}. "
                f"Large relational state may cause memory issues.",
                UserWarning,
                stacklevel=2,
            )

    # ──────────────────────────────────────────────
    # Preset configurations
    # ──────────────────────────────────────────────

    @classmethod
    def tiny(cls) -> USNConfig:
        """~2M parameter configuration for rapid prototyping."""
        return cls(
            num_layers=4,
            d_model=128,
            d_s=64,
            k=8,
            d_ff=512,
            vocab_size=1000,
        )

    @classmethod
    def micro(cls) -> USNConfig:
        """~5M parameter configuration for testing."""
        return cls(
            num_layers=6,
            d_model=192,
            d_s=128,
            k=8,
            d_ff=768,
            vocab_size=1000,
        )

    @classmethod
    def mini(cls) -> USNConfig:
        """~15M parameter configuration."""
        return cls(
            num_layers=8,
            d_model=384,
            d_s=256,
            k=12,
            d_ff=1536,
        )

    @classmethod
    def small(cls) -> USNConfig:
        """~125M parameter configuration."""
        return cls(
            num_layers=12,
            d_model=768,
            d_s=512,
            k=16,
            d_ff=3072,
        )

    @classmethod
    def base(cls) -> USNConfig:
        """~350M parameter configuration."""
        return cls(
            num_layers=24,
            d_model=1024,
            d_s=768,
            k=24,
            d_ff=4096,
        )

    @classmethod
    def medium(cls) -> USNConfig:
        """~750M parameter configuration."""
        return cls(
            num_layers=32,
            d_model=1280,
            d_s=1024,
            k=32,
            d_ff=5120,
        )

    @classmethod
    def large(cls) -> USNConfig:
        """~1.3B parameter configuration."""
        return cls(
            num_layers=36,
            d_model=1536,
            d_s=1024,
            k=32,
            d_ff=6144,
        )

    @classmethod
    def xl(cls) -> USNConfig:
        """~2.7B parameter configuration."""
        return cls(
            num_layers=48,
            d_model=2048,
            d_s=1536,
            k=48,
            d_ff=8192,
        )

    @classmethod
    def xxl(cls) -> USNConfig:
        """~6.7B parameter configuration."""
        return cls(
            num_layers=64,
            d_model=2560,
            d_s=2048,
            k=48,
            d_ff=10240,
        )

    @classmethod
    def from_preset(cls, name: str) -> USNConfig:
        """Create a configuration from a named preset.

        Args:
            name: Preset name (tiny, micro, mini, small, base, medium, large, xl, xxl).

        Returns:
            USNConfig for the named preset.

        Raises:
            InvalidParameterError: If the preset name is not recognized.
        """
        presets = {
            "tiny": cls.tiny,
            "micro": cls.micro,
            "mini": cls.mini,
            "small": cls.small,
            "base": cls.base,
            "medium": cls.medium,
            "large": cls.large,
            "xl": cls.xl,
            "xxl": cls.xxl,
        }
        if name not in presets:
            raise InvalidParameterError("name", name, f"one of {tuple(presets.keys())}")
        return presets[name]()

    # ──────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize configuration to a JSON string."""
        return json.dumps(asdict(self), indent=2)

    def to_yaml(self) -> str:
        """Serialize configuration to a YAML string."""
        import yaml

        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_json(cls, json_str: str) -> USNConfig:
        """Deserialize configuration from a JSON string.

        Args:
            json_str: JSON string representation of a USNConfig.

        Returns:
            A new USNConfig instance.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> USNConfig:
        """Deserialize configuration from a YAML string.

        Args:
            yaml_str: YAML string representation of a USNConfig.

        Returns:
            A new USNConfig instance.
        """
        import yaml

        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: dict) -> USNConfig:
        """Create a USNConfig from a dictionary.

        Only keys that match valid USNConfig fields are used;
        unknown keys are ignored.

        Args:
            d: Dictionary with configuration parameters.

        Returns:
            A new USNConfig instance.
        """
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
