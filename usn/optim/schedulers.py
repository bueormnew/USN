"""Learning rate scheduler implementations.

Provides configurable LR schedules for USN training, all implementing
the SchedulerInterface from usn.core.interfaces.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from usn.core.interfaces import SchedulerInterface

if TYPE_CHECKING:
    from usn.config.training_config import USNTrainingConfig


class CosineAnnealingScheduler(SchedulerInterface):
    """Cosine annealing from max_lr to min_lr over total_steps.

    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * step / total_steps))
    """

    def __init__(self, max_lr: float, min_lr: float, total_steps: int) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps

    def get_lr(self, step: int) -> float:
        """Get learning rate for a given training step."""
        if step >= self.total_steps:
            return self.min_lr
        progress = step / self.total_steps
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize scheduler state for checkpointing."""
        return {
            "type": "cosine_annealing",
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "total_steps": self.total_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scheduler state from checkpoint."""
        self.max_lr = state["max_lr"]
        self.min_lr = state["min_lr"]
        self.total_steps = state["total_steps"]


class LinearWarmupScheduler(SchedulerInterface):
    """Linear warmup from 0 to max_lr over warmup_steps.

    lr = max_lr * step / warmup_steps  (for step < warmup_steps)
    lr = max_lr                        (for step >= warmup_steps)
    """

    def __init__(self, max_lr: float, warmup_steps: int) -> None:
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps

    def get_lr(self, step: int) -> float:
        """Get learning rate for a given training step."""
        if self.warmup_steps == 0:
            return self.max_lr
        if step >= self.warmup_steps:
            return self.max_lr
        return self.max_lr * step / self.warmup_steps

    def state_dict(self) -> dict[str, Any]:
        """Serialize scheduler state for checkpointing."""
        return {
            "type": "linear_warmup",
            "max_lr": self.max_lr,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scheduler state from checkpoint."""
        self.max_lr = state["max_lr"]
        self.warmup_steps = state["warmup_steps"]


class WarmupCosineScheduler(SchedulerInterface):
    """Linear warmup followed by cosine annealing (DEFAULT scheduler).

    Phase 1 (step < warmup_steps):
        lr = min_lr + (max_lr - min_lr) * step / warmup_steps

    Phase 2 (step >= warmup_steps):
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
    """

    def __init__(self, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step: int) -> float:
        """Get learning rate for a given training step."""
        if step < self.warmup_steps:
            # Linear warmup from min_lr to max_lr
            if self.warmup_steps == 0:
                return self.max_lr
            return self.min_lr + (self.max_lr - self.min_lr) * step / self.warmup_steps

        # Cosine annealing from max_lr to min_lr
        if step >= self.total_steps:
            return self.min_lr

        decay_steps = self.total_steps - self.warmup_steps
        if decay_steps == 0:
            return self.min_lr

        progress = (step - self.warmup_steps) / decay_steps
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize scheduler state for checkpointing."""
        return {
            "type": "warmup_cosine",
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scheduler state from checkpoint."""
        self.max_lr = state["max_lr"]
        self.min_lr = state["min_lr"]
        self.warmup_steps = state["warmup_steps"]
        self.total_steps = state["total_steps"]


class CosineWarmRestartsScheduler(SchedulerInterface):
    """Cosine annealing with warm restarts (SGDR-style).

    The learning rate follows a cosine curve within each restart period,
    resetting to max_lr at the beginning of each new period.

    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * t_i / restart_period))

    where t_i = step % restart_period
    """

    def __init__(self, max_lr: float, min_lr: float, restart_period: int) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.restart_period = restart_period

    def get_lr(self, step: int) -> float:
        """Get learning rate for a given training step."""
        if self.restart_period == 0:
            return self.min_lr
        t_i = step % self.restart_period
        progress = t_i / self.restart_period
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize scheduler state for checkpointing."""
        return {
            "type": "cosine_warm_restarts",
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "restart_period": self.restart_period,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scheduler state from checkpoint."""
        self.max_lr = state["max_lr"]
        self.min_lr = state["min_lr"]
        self.restart_period = state["restart_period"]


class ConstantScheduler(SchedulerInterface):
    """Constant learning rate (no decay or warmup).

    lr = lr (constant for all steps)
    """

    def __init__(self, lr: float) -> None:
        self.lr = lr

    def get_lr(self, step: int) -> float:
        """Get learning rate for a given training step."""
        return self.lr

    def state_dict(self) -> dict[str, Any]:
        """Serialize scheduler state for checkpointing."""
        return {
            "type": "constant",
            "lr": self.lr,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scheduler state from checkpoint."""
        self.lr = state["lr"]


def create_scheduler(config: USNTrainingConfig) -> SchedulerInterface:
    """Create a learning rate scheduler from training configuration.

    Maps config.scheduler_type to the appropriate scheduler class:
      - "cosine" → WarmupCosineScheduler (default)
      - "linear" → LinearWarmupScheduler
      - "constant" → ConstantScheduler
      - "cosine_restarts" → CosineWarmRestartsScheduler

    Args:
        config: Training configuration with scheduler parameters.

    Returns:
        Scheduler implementing SchedulerInterface.

    Raises:
        ValueError: If scheduler_type is not recognized.
    """
    scheduler_type = config.scheduler_type

    if scheduler_type == "cosine":
        return WarmupCosineScheduler(
            max_lr=config.learning_rate,
            min_lr=config.min_lr,
            warmup_steps=config.warmup_steps,
            total_steps=config.max_steps,
        )
    elif scheduler_type == "linear":
        return LinearWarmupScheduler(
            max_lr=config.learning_rate,
            warmup_steps=config.warmup_steps,
        )
    elif scheduler_type == "constant":
        return ConstantScheduler(lr=config.learning_rate)
    elif scheduler_type == "cosine_restarts":
        return CosineWarmRestartsScheduler(
            max_lr=config.learning_rate,
            min_lr=config.min_lr,
            restart_period=config.cosine_restart_period,
        )
    else:
        raise ValueError(
            f"Unknown scheduler_type: '{scheduler_type}'. "
            f"Supported: 'cosine', 'linear', 'constant', 'cosine_restarts'"
        )
