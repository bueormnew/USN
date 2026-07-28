"""Type stubs for the USN public API."""

from typing import Any

import torch.nn as nn
from torch import Tensor

from usn.backends.acceleration import AccelerationLevel as AccelerationLevel
from usn.backends.acceleration import AccelerationManager as AccelerationManager
from usn.backends.detection import DeviceDetector as DeviceDetector
from usn.config.generation_config import USNGenerationConfig as USNGenerationConfig
from usn.config.model_config import USNConfig as USNConfig
from usn.config.training_config import USNTrainingConfig as USNTrainingConfig
from usn.inference.generator import USNGenerator as USNGenerator
from usn.models import create_model as create_model
from usn.models.usn_model import USNModel as USNModel
from usn.training.trainer import USNTrainer as USNTrainer
from usn.utils.counting import count_parameters as count_parameters
from usn.utils.counting import estimate_flops as estimate_flops
from usn.utils.counting import estimate_memory as estimate_memory
from usn.utils.seed import set_seed as set_seed

__version__: str
__author__: str

def save(model: nn.Module, path: str, **kwargs: Any) -> None: ...
def load(path: str, map_location: str | None = None) -> USNModel: ...
def export(model: nn.Module, format: str, path: str, **kwargs: Any) -> None: ...
def generate(
    model: nn.Module, prompt: str, max_tokens: int = 256, tokenizer: Any = None, **kwargs: Any
) -> str: ...
def train(
    model: nn.Module, dataset: Any, config: USNTrainingConfig | None = None, **kwargs: Any
) -> dict[str, Any]: ...
def summary(model: nn.Module) -> str: ...
def from_pretrained(path_or_id: str) -> USNModel: ...
def device_info() -> dict[str, Any]: ...
def set_acceleration_level(level: int) -> None: ...
def benchmark_acceleration() -> dict[str, Any]: ...
