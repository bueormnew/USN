"""Optimizer factory with correct parameter group separation.

Creates optimizers with weight decay properly applied only to weight
matrices (2D parameters), excluding biases, normalization parameters,
and embedding weights.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from usn.config.training_config import USNTrainingConfig


class OptimizerFactory:
    """Creates optimizers with correct parameter group separation.

    The factory separates parameters into two groups:
    - Decay group: 2D weight matrices (excluding norm and embedding layers)
    - No-decay group: biases (1D), normalization params, embedding weights

    This follows the standard practice of not applying weight decay to
    parameters that don't benefit from regularization.
    """

    # Registry for custom optimizers
    _registry: dict[str, type[torch.optim.Optimizer]] = {}

    @staticmethod
    def create(
        model: nn.Module,
        config: USNTrainingConfig,
    ) -> torch.optim.Optimizer:
        """Create an optimizer with proper weight decay separation.

        Args:
            model: The model whose parameters will be optimized.
            config: Training configuration specifying optimizer type
                and hyperparameters.

        Returns:
            Configured optimizer instance.

        Raises:
            ValueError: If the optimizer type is not supported.
        """
        param_groups = OptimizerFactory.get_parameter_groups(model, config.weight_decay)

        optimizer_type = config.optimizer.lower()

        if optimizer_type == "adamw":
            return torch.optim.AdamW(
                param_groups,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_eps,
                weight_decay=config.weight_decay,
            )
        elif optimizer_type == "adam":
            return torch.optim.Adam(
                param_groups,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_eps,
                weight_decay=config.weight_decay,
            )
        elif optimizer_type == "sgd":
            return torch.optim.SGD(
                param_groups,
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay,
            )
        elif optimizer_type in OptimizerFactory._registry:
            optimizer_cls = OptimizerFactory._registry[optimizer_type]
            return optimizer_cls(  # type: ignore[call-arg]
                param_groups,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            supported = ["adamw", "adam", "sgd"] + list(OptimizerFactory._registry.keys())
            raise ValueError(f"Unsupported optimizer: '{config.optimizer}'. Supported: {supported}")

    @staticmethod
    def get_parameter_groups(
        model: nn.Module,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        """Separate parameters into decay and no-decay groups.

        Decay group: 2D weight parameters that are NOT part of
        normalization layers or embedding layers.

        No-decay group: all biases (1D), normalization parameters,
        and embedding weights.

        Args:
            model: Model whose parameters to group.
            weight_decay: Weight decay value for the decay group.

        Returns:
            List of two parameter group dicts suitable for optimizer
            construction: [decay_group, no_decay_group].
        """
        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # No decay for: biases (1D), norm params, embedding weights
            if param.dim() < 2:
                # 1D params: biases, norm scale/shift
                no_decay_params.append(param)
            elif "norm" in name.lower():
                # Normalization layer weights
                no_decay_params.append(param)
            elif "embedding" in name.lower():
                # Embedding weights
                no_decay_params.append(param)
            else:
                # 2D+ weight matrices get decay
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return param_groups

    @classmethod
    def register(cls, name: str, optimizer_cls: type[torch.optim.Optimizer]) -> None:
        """Register a custom optimizer type.

        Args:
            name: Name to register the optimizer under (lowercase).
            optimizer_cls: Optimizer class to register.
        """
        cls._registry[name.lower()] = optimizer_cls

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a custom optimizer from the registry.

        Args:
            name: Name of the optimizer to remove.
        """
        cls._registry.pop(name.lower(), None)
