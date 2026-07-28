"""Hardware device detection for the USN library.

Provides automatic detection of available compute hardware (CUDA GPUs, Apple MPS,
CPU) and selection of the optimal device for model execution.
"""

from __future__ import annotations

from typing import Any

import torch


class DeviceDetector:
    """Auto-detect available hardware and capabilities.

    Provides static methods for querying the runtime environment's compute
    resources. Used by AccelerationManager to determine the best acceleration
    level and by model initialization to select the target device.
    """

    @staticmethod
    def detect() -> dict[str, Any]:
        """Detect available hardware and return a capabilities dictionary.

        Returns:
            Dictionary containing:
                - device: str — best available device name ("cuda", "mps", "cpu")
                - cuda_available: bool — whether CUDA is available
                - mps_available: bool — whether Apple MPS is available
                - gpu_name: str | None — name of the CUDA GPU (if available)
                - gpu_memory: int | None — total GPU memory in bytes (if available)
                - compute_capability: tuple[int, int] | None — CUDA compute cap
                - cuda_version: str | None — CUDA toolkit version string
                - device_count: int — number of CUDA devices
        """
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        gpu_name: str | None = None
        gpu_memory: int | None = None
        compute_capability: tuple[int, int] | None = None
        cuda_version: str | None = None
        device_count = 0

        if cuda_available:
            device_count = torch.cuda.device_count()
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = None
            try:
                props = torch.cuda.get_device_properties(0)
                gpu_memory = props.total_memory
                compute_capability = (props.major, props.minor)
            except Exception:
                pass
            cuda_version = torch.version.cuda

        # Determine best device string
        if cuda_available:
            device = "cuda"
        elif mps_available:
            device = "mps"
        else:
            device = "cpu"

        return {
            "device": device,
            "cuda_available": cuda_available,
            "mps_available": mps_available,
            "gpu_name": gpu_name,
            "gpu_memory": gpu_memory,
            "compute_capability": compute_capability,
            "cuda_version": cuda_version,
            "device_count": device_count,
        }

    @staticmethod
    def best_device() -> torch.device:
        """Return the best available torch.device.

        Priority: CUDA > MPS > CPU.

        Returns:
            torch.device for the best available hardware.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
