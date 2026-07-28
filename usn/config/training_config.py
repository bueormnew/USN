"""Training configuration for the USN trainer."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class USNTrainingConfig:
    """Training hyperparameters for USNTrainer.

    All parameters are validated on creation. Use with USNTrainer
    to configure the training loop.
    """

    # Optimizer
    learning_rate: float = 3e-4
    batch_size: int = 32
    max_steps: int = 100_000
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Precision
    mixed_precision: Literal["none", "fp16", "bf16"] = "bf16"
    gradient_accumulation_steps: int = 1

    # Schedule
    scheduler_type: Literal["cosine", "linear", "constant", "cosine_restarts"] = "cosine"
    min_lr: float = 1e-5
    cosine_restart_period: int = 10000

    # Evaluation and checkpointing
    eval_interval: int = 500
    checkpoint_interval: int = 1000
    max_checkpoints: int = 5
    early_stopping_patience: int = 0  # 0 = disabled
    early_stopping_min_delta: float = 1e-4

    # Distributed
    distributed_strategy: Literal["none", "ddp", "fsdp"] = "none"

    # Curriculum
    sequence_curriculum: bool = False
    curriculum_start_len: int = 128
    curriculum_end_len: int = 2048
    curriculum_warmup_steps: int = 10_000
    curriculum_schedule: Literal["linear", "step", "exponential"] = "linear"

    # Stability
    stability_mode: bool = False
    nan_skip_batch: bool = True
    loss_spike_threshold: float = 5.0
    state_max_norm: float = 1000.0

    # Logging
    log_interval: int = 10
    log_format: Literal["console", "json", "tensorboard", "wandb"] = "console"

    # EMA
    use_ema: bool = False
    ema_decay: float = 0.9999

    # Gradient checkpointing
    gradient_checkpointing: bool = False
    checkpointing_level: Literal["none", "per_block", "per_chunk"] = "none"

    def __post_init__(self) -> None:
        """Validate all training parameters."""
        from usn.exceptions import InvalidParameterError

        errors: list[tuple[str, object, str]] = []
        if self.learning_rate <= 0:
            errors.append(("learning_rate", self.learning_rate, "> 0"))
        if self.batch_size < 1:
            errors.append(("batch_size", self.batch_size, ">= 1"))
        if self.max_steps < 1:
            errors.append(("max_steps", self.max_steps, ">= 1"))
        if self.warmup_steps < 0:
            errors.append(("warmup_steps", self.warmup_steps, ">= 0"))
        if self.weight_decay < 0:
            errors.append(("weight_decay", self.weight_decay, ">= 0"))
        if self.grad_clip <= 0:
            errors.append(("grad_clip", self.grad_clip, "> 0"))
        if self.gradient_accumulation_steps < 1:
            errors.append(
                (
                    "gradient_accumulation_steps",
                    self.gradient_accumulation_steps,
                    ">= 1",
                )
            )
        if not 0 <= self.min_lr <= self.learning_rate:
            errors.append(
                (
                    "min_lr",
                    self.min_lr,
                    f"in [0, {self.learning_rate}]",
                )
            )
        if errors:
            msg = "; ".join(f"{n}={v} (expected {r})" for n, v, r in errors)
            raise InvalidParameterError("training_config", msg, "see individual constraints")
