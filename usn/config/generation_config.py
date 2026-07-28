"""Generation configuration for the USN generator."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class USNGenerationConfig:
    """Generation/inference hyperparameters.

    Controls decoding strategy, sampling, and output behavior
    for autoregressive text generation.
    """

    temperature: float = 1.0
    top_k: int = 0  # 0 = disabled
    top_p: float = 1.0  # 1.0 = disabled (no nucleus sampling)
    beam_width: int = 1  # 1 = no beam search (greedy or sampling)
    max_new_tokens: int = 256
    repetition_penalty: float = 1.0  # 1.0 = no penalty
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    no_repeat_ngram_size: int = 0  # 0 = disabled
    length_penalty: float = 1.0  # For beam search
    stop_tokens: tuple[int, ...] = ()
    streaming: bool = False

    def __post_init__(self) -> None:
        """Validate generation parameters."""
        from usn.exceptions import InvalidParameterError

        errors: list[tuple[str, object, str]] = []
        if self.temperature < 0:
            errors.append(("temperature", self.temperature, ">= 0"))
        if self.top_k < 0:
            errors.append(("top_k", self.top_k, ">= 0"))
        if not 0 < self.top_p <= 1.0:
            errors.append(("top_p", self.top_p, "in (0, 1]"))
        if self.beam_width < 1:
            errors.append(("beam_width", self.beam_width, ">= 1"))
        if self.max_new_tokens < 1:
            errors.append(("max_new_tokens", self.max_new_tokens, ">= 1"))
        if self.repetition_penalty < 1.0:
            errors.append(
                (
                    "repetition_penalty",
                    self.repetition_penalty,
                    ">= 1.0",
                )
            )
        if errors:
            msg = "; ".join(f"{n}={v} (expected {r})" for n, v, r in errors)
            raise InvalidParameterError("generation_config", msg, "see individual constraints")
