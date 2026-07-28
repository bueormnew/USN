"""Backend acceleration: device detection, Triton kernels, and fallbacks.

Public API:
    - DeviceDetector: Hardware auto-detection (detect(), best_device())
    - AccelerationLevel: Enum of 4 acceleration tiers
    - AccelerationManager: Manages acceleration level with graceful fallback

Importing this package triggers kernel registration at all acceleration levels:
    - triton_kernels: Registers Level 1 (TRITON), Level 3 (AUTOGRAD), Level 4 (EAGER)
    - fallbacks: Registers Level 2 (COMPILE) via torch.compile wrappers
"""

# Import fallbacks to register Level 2 (COMPILE) kernels
import usn.backends.fallbacks as _fallbacks  # noqa: F401

# Import triton_kernels to register Level 1, 3, 4 kernels
import usn.backends.triton_kernels as _triton_kernels  # noqa: F401
from usn.backends.acceleration import AccelerationLevel, AccelerationManager
from usn.backends.detection import DeviceDetector

__all__ = [
    "AccelerationLevel",
    "AccelerationManager",
    "DeviceDetector",
]
