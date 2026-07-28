"""Base classes for all USN modules.

Provides USNModule as the abstract base for all architecture submodules,
enforcing documentation of objective, complexity, and constraints.
"""

from abc import ABC, abstractmethod

import torch.nn as nn


class USNModule(nn.Module, ABC):
    """Base class for all USN submodules.

    Every USN module must document its:
    - objective: what the module does
    - complexity: computational cost per timestep
    - constraints: invariants that must hold

    Subclasses should call super().__init__() and implement
    the abstract properties and forward().
    """

    @property
    @abstractmethod
    def objective(self) -> str:
        """One-line description of what this module does."""
        ...

    @property
    @abstractmethod
    def complexity(self) -> str:
        """Big-O complexity per timestep."""
        ...

    @property
    @abstractmethod
    def constraints(self) -> list[str]:
        """List of invariants/constraints this module must satisfy."""
        ...

    def reset_parameters(self) -> None:
        """Re-initialize all parameters to their initial values.

        Default implementation re-initializes all Linear layers
        with Xavier uniform and zeros for biases.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
