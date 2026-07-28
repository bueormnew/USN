"""USN - Unified State Network Architecture Library.

A production-grade Python package implementing the Unified State Network (USN)
architecture for autoregressive sequence modeling with O(n) training complexity
via associative parallel scan and O(1) inference memory via constant-size state.

Quick Start:
    >>> import usn
    >>> model = usn.create_model("tiny")
    >>> print(model.summary())

Author: BUEORM
License: MIT
"""

__version__ = "0.1.0"
__author__ = "BUEORM"

# Core classes
# Backends
from usn.backends import AccelerationLevel, AccelerationManager, DeviceDetector
from usn.config import USNConfig, USNGenerationConfig, USNTrainingConfig
from usn.inference import USNGenerator
from usn.models import create_model
from usn.models.usn_model import USNModel
from usn.serialization.export import export_model
from usn.serialization.reader import USNReader

# Serialization
from usn.serialization.writer import USNWriter
from usn.training import USNTrainer
from usn.utils.counting import count_parameters, estimate_flops, estimate_memory

# Utilities
from usn.utils.seed import set_seed


# High-level API functions
def save(model, path: str, **kwargs) -> None:
    """Save a model to .usn format."""
    writer = USNWriter()
    config = getattr(model, "config", None)
    writer.save(path, model, config=config, **kwargs)


def load(path: str, map_location=None):
    """Load a model from .usn format."""
    reader = USNReader()
    data = reader.load(path, map_location=map_location)
    config = data.get("config")
    if config is None:
        raise ValueError("No config found in .usn file")
    model = USNModel(config)
    weights = data.get("weights", {})
    if weights:
        # Filter buffer keys for state_dict loading
        state_dict = {k: v for k, v in weights.items() if not k.startswith("__buffer__.")}
        model.load_state_dict(state_dict, strict=False)
    return model


def export(model, format: str, path: str, **kwargs) -> None:
    """Export model to standard format (onnx, safetensors, state_dict, torchscript)."""
    config = getattr(model, "config", None)
    export_model(model, format, path, config=config, **kwargs)


def generate(model, prompt: str, max_tokens: int = 256, tokenizer=None, **kwargs) -> str:
    """Generate text from a model (convenience function)."""
    if tokenizer is None:
        raise ValueError("tokenizer is required for generate()")
    gen = USNGenerator(model, tokenizer)
    output = gen.generate(prompt, max_new_tokens=max_tokens, **kwargs)
    return tokenizer.decode(output.token_ids[0].tolist())


def train(model, dataset, config=None, **kwargs):
    """Train a model (convenience function)."""
    if config is None:
        config = USNTrainingConfig()
    trainer = USNTrainer(model, dataset, config, **kwargs)
    return trainer.train()


def summary(model) -> str:
    """Get model summary string."""
    if hasattr(model, "summary"):
        return model.summary()
    return f"Model with {count_parameters(model):,} parameters"


def from_pretrained(path_or_id: str):
    """Load a pretrained model from path."""
    return load(path_or_id)


def device_info() -> dict:
    """Get available device information."""
    return DeviceDetector.detect()


def set_acceleration_level(level: int) -> None:
    """Set the acceleration level manually."""
    AccelerationManager.set_level(level)


def benchmark_acceleration() -> dict:
    """Compare throughput across acceleration levels."""
    return {"current_level": AccelerationManager.get_level().name}


__all__ = [
    # Version
    "__version__",
    "__author__",
    # Core classes
    "USNModel",
    "USNConfig",
    "USNTrainingConfig",
    "USNGenerationConfig",
    "USNTrainer",
    "USNGenerator",
    # Factory
    "create_model",
    # High-level API
    "save",
    "load",
    "export",
    "generate",
    "train",
    "summary",
    "from_pretrained",
    # Utilities
    "set_seed",
    "count_parameters",
    "estimate_memory",
    "estimate_flops",
    "device_info",
    "set_acceleration_level",
    "benchmark_acceleration",
    # Backend
    "AccelerationLevel",
    "AccelerationManager",
    "DeviceDetector",
]
