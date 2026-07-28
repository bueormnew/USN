"""USN Library Exception Hierarchy.

All custom exceptions for the USN (Unified State Network) library.
Provides descriptive error messages for configuration, shape, serialization,
training, and generation errors.
"""


class USNError(Exception):
    """Base exception for all USN library errors."""

    pass


class ConfigError(USNError):
    """Invalid configuration parameters."""

    pass


class InvalidParameterError(ConfigError):
    """A specific parameter has an invalid value or type."""

    def __init__(self, param_name: str, value: object, valid_range: str = "") -> None:
        self.param_name = param_name
        self.value = value
        self.valid_range = valid_range
        msg = f"Invalid parameter '{param_name}': got {value!r}"
        if valid_range:
            msg += f", expected {valid_range}"
        super().__init__(msg)


class IncompatibleConfigError(ConfigError):
    """Cross-parameter constraint violation."""

    def __init__(self, message: str, params: dict[str, object] | None = None) -> None:
        self.params = params or {}
        super().__init__(message)


class ShapeError(USNError):
    """Tensor shape mismatch."""

    def __init__(
        self, expected: tuple[int, ...], actual: tuple[int, ...], context: str = ""
    ) -> None:
        self.expected = expected
        self.actual = actual
        msg = f"Shape mismatch: expected {expected}, got {actual}"
        if context:
            msg += f" in {context}"
        super().__init__(msg)


class IntegrityError(USNError):
    """File integrity check failed (checksum mismatch, corruption)."""

    pass


class VersionError(USNError):
    """Incompatible format version."""

    def __init__(self, file_version: int, library_version: int) -> None:
        self.file_version = file_version
        self.library_version = library_version
        super().__init__(
            f"Format version mismatch: file is v{file_version}, "
            f"library supports up to v{library_version}. "
            f"Please upgrade the USN library."
        )


class TrainingError(USNError):
    """Error during training."""

    pass


class NaNDetectedError(TrainingError):
    """NaN or Inf detected during training."""

    def __init__(self, layer: int | None = None, module: str = "", step: int | None = None) -> None:
        self.layer = layer
        self.module = module
        self.step = step
        parts = ["NaN/Inf detected during training"]
        if layer is not None:
            parts.append(f"layer={layer}")
        if module:
            parts.append(f"module={module}")
        if step is not None:
            parts.append(f"step={step}")
        msg = ", ".join(parts)
        msg += ". Try reducing learning rate or enabling gradient clipping."
        super().__init__(msg)


class DivergenceError(TrainingError):
    """Training loss has diverged (exploded)."""

    pass


class OOMError(TrainingError):
    """Out of memory during training with suggestions."""

    def __init__(self, message: str = "") -> None:
        suggestions = (
            "Suggestions: reduce batch_size, enable gradient_checkpointing, "
            "use mixed_precision='bf16', or reduce max_seq_len."
        )
        msg = message or "CUDA out of memory during training."
        super().__init__(f"{msg} {suggestions}")


class GenerationError(USNError):
    """Error during text generation."""

    pass


class InvalidPromptError(GenerationError):
    """Invalid or empty prompt for generation."""

    pass


class DecodingError(GenerationError):
    """Decoding strategy failure."""

    pass
