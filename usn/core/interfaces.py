"""Abstract interfaces for pluggable components.

Defines the contracts that tokenizers, loss functions, and schedulers
must implement to be used with the USN training/inference system.
"""

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor


class TokenizerInterface(ABC):
    """Interface for all tokenizer implementations.

    Any tokenizer used with USN must implement encode/decode
    and expose vocabulary metadata.
    """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        ...

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        """ID of the padding token."""
        ...

    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        """ID of the beginning-of-sequence token."""
        ...

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """ID of the end-of-sequence token."""
        ...

    @property
    def unk_token_id(self) -> int:
        """ID of the unknown token. Default: 3."""
        return 3


class LossInterface(ABC):
    """Interface for registrable loss functions."""

    @abstractmethod
    def forward(self, logits: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
        """Compute loss.

        Args:
            logits: (batch, seq, vocab_size) predicted logits
            targets: (batch, seq) target token IDs
            mask: (batch, seq) True for valid positions

        Returns:
            Scalar loss tensor.
        """
        ...


class SchedulerInterface(ABC):
    """Interface for learning rate scheduler implementations."""

    @abstractmethod
    def get_lr(self, step: int) -> float:
        """Get learning rate for a given training step."""
        ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Serialize scheduler state for checkpointing."""
        ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scheduler state from checkpoint."""
        ...
