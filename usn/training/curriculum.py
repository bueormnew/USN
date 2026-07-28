"""Sequence length curriculum for gradual difficulty increase.

Implements curriculum learning where training starts with shorter sequences
and gradually increases to the maximum length, helping the model learn
local patterns before long-range dependencies.
"""

from __future__ import annotations

import math
from typing import Any, Literal


class CurriculumScheduler:
    """Sequence length curriculum: short → long over training.

    Starts at start_len and gradually increases to end_len over
    warmup_steps using the specified schedule.

    Args:
        start_len: Initial (minimum) sequence length.
        end_len: Final (maximum) sequence length.
        warmup_steps: Steps over which to ramp from start to end.
        schedule: Ramp schedule ("linear", "step", "exponential").
    """

    def __init__(
        self,
        start_len: int = 128,
        end_len: int = 2048,
        warmup_steps: int = 10000,
        schedule: Literal["linear", "step", "exponential"] = "linear",
    ) -> None:
        self.start_len = start_len
        self.end_len = end_len
        self.warmup_steps = warmup_steps
        self.schedule = schedule

    def get_seq_len(self, step: int) -> int:
        """Get the current maximum sequence length for a training step.

        Args:
            step: Current training step (0-indexed).

        Returns:
            Maximum sequence length allowed at this step.
        """
        if step >= self.warmup_steps:
            return self.end_len

        progress = step / max(self.warmup_steps, 1)

        if self.schedule == "linear":
            length = self.start_len + (self.end_len - self.start_len) * progress
        elif self.schedule == "step":
            # Step increases at 25%, 50%, 75%
            if progress < 0.25:
                length = self.start_len
            elif progress < 0.5:
                length = self.start_len + (self.end_len - self.start_len) * 0.33
            elif progress < 0.75:
                length = self.start_len + (self.end_len - self.start_len) * 0.66
            else:
                length = self.end_len
        elif self.schedule == "exponential":
            ratio = self.end_len / max(self.start_len, 1)
            length = self.start_len * (ratio**progress)
        else:
            length = self.end_len

        return int(min(length, self.end_len))

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state for checkpointing."""
        return {
            "start_len": self.start_len,
            "end_len": self.end_len,
            "warmup_steps": self.warmup_steps,
            "schedule": self.schedule,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scheduler state from a checkpoint.

        Args:
            state: State dict previously returned by state_dict().
        """
        self.start_len = state["start_len"]
        self.end_len = state["end_len"]
        self.warmup_steps = state["warmup_steps"]
        self.schedule = state["schedule"]
