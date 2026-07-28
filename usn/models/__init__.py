"""USN model classes: complete models, embeddings, and output heads.

Provides USNModel for creating complete USN architecture models,
and TokenEmbedding/OutputHead as building blocks.
"""

import logging
from typing import Optional

import torch

from usn.config.model_config import USNConfig
from usn.models.embedding import OutputHead, TokenEmbedding
from usn.models.usn_model import USNModel

logger = logging.getLogger(__name__)

__all__ = [
    "USNModel",
    "TokenEmbedding",
    "OutputHead",
    "create_model",
]


def create_model(
    config: USNConfig | str | None = None,
    device: str = "auto",
    **kwargs,
) -> USNModel:
    """Factory function for creating USN models.

    Args:
        config: USNConfig instance, preset name (e.g., "tiny", "base"),
                or None for default (small) config. Can also pass keyword
                args to override config fields.
        device: Target device ("auto", "cpu", "cuda", "cuda:0", etc.).
                "auto" selects the best available device.
        **kwargs: Override config fields when config is a USNConfig or preset.

    Returns:
        Initialized USNModel on the specified device.

    Examples:
        >>> model = create_model("tiny")
        >>> model = create_model(USNConfig(num_layers=4, d_model=128))
        >>> model = create_model("small", device="cuda")
    """
    # Resolve config
    if config is None:
        cfg = USNConfig.small()
    elif isinstance(config, str):
        cfg = USNConfig.from_preset(config)
    elif isinstance(config, USNConfig):
        cfg = config
    else:
        raise TypeError(f"config must be USNConfig, str, or None, got {type(config)}")

    # Apply overrides if any
    if kwargs:
        from dataclasses import asdict, fields

        d = asdict(cfg)
        valid_keys = {f.name for f in fields(USNConfig)}
        for k, v in kwargs.items():
            if k not in valid_keys:
                raise ValueError(f"Unknown config field: '{k}'")
            d[k] = v
        cfg = USNConfig.from_dict(d)

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            resolved_device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved_device = torch.device("mps")
        else:
            resolved_device = torch.device("cpu")
    else:
        resolved_device = torch.device(device)

    # Create and move model
    model = USNModel(cfg)
    model = model.to(resolved_device)

    logger.info(f"Created USNModel: {model.num_parameters:,} params on {resolved_device}")
    return model
