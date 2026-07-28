"""Core module: base classes, type definitions, and interfaces.

This package contains the fundamental building blocks used across
the entire USN library.
"""

from usn.core.activations import get_activation, list_activations, register_activation
from usn.core.base import USNModule
from usn.core.interfaces import LossInterface, SchedulerInterface, TokenizerInterface
from usn.core.types import (
    AffineTransition,
    BlockOutput,
    GenerationOutput,
    ModelState,
    UnifiedState,
)

__all__ = [
    "USNModule",
    "TokenizerInterface",
    "LossInterface",
    "SchedulerInterface",
    "UnifiedState",
    "ModelState",
    "BlockOutput",
    "GenerationOutput",
    "AffineTransition",
    "get_activation",
    "register_activation",
    "list_activations",
]
