"""Distributed training support for USN models.

Provides utilities for DDP and FSDP distributed training with
proper gradient synchronization, rank-based operations, and
graceful fallback when distributed is unavailable.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """Helper for distributed training setup and teardown.

    Supports DDP (DistributedDataParallel) and FSDP (FullyShardedDataParallel)
    strategies. Falls back to single-device training with a warning when
    distributed backend is not available.
    """

    @staticmethod
    def setup(strategy: str, model: nn.Module, device_id: int | None = None) -> nn.Module:
        """Set up distributed training wrapper around model.

        Args:
            strategy: "ddp" or "fsdp".
            model: Model to wrap.
            device_id: Local device ID for this process.

        Returns:
            Wrapped model (DDP or FSDP), or original model if distributed unavailable.
        """
        if not torch.distributed.is_available():
            logger.warning("torch.distributed not available, using single-device training")
            return model

        if not torch.distributed.is_initialized():
            # Try to initialize from environment
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            try:
                torch.distributed.init_process_group(backend=backend)
            except Exception as e:
                logger.warning(f"Could not initialize distributed: {e}. Using single-device.")
                return model

        if strategy == "ddp":
            if device_id is not None:
                model = model.to(device_id)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device_id] if device_id is not None else None
            )
            logger.info(f"DDP initialized on device {device_id}")
        elif strategy == "fsdp":
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                model = FSDP(model)
                logger.info("FSDP initialized")
            except ImportError:
                logger.warning("FSDP not available, falling back to DDP")
                model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            logger.warning(f"Unknown strategy '{strategy}', using model as-is")

        return model

    @staticmethod
    def cleanup() -> None:
        """Clean up distributed process group."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    @staticmethod
    def is_main_process() -> bool:
        """Check if this is the main (rank 0) process."""
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    @staticmethod
    def get_rank() -> int:
        """Get current process rank (0 if not distributed)."""
        if not torch.distributed.is_initialized():
            return 0
        return torch.distributed.get_rank()

    @staticmethod
    def get_world_size() -> int:
        """Get total number of processes (1 if not distributed)."""
        if not torch.distributed.is_initialized():
            return 1
        return torch.distributed.get_world_size()
