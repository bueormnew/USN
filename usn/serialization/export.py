"""Model export to standard formats (ONNX, SafeTensors, state_dict, TorchScript).

Provides export utilities for deploying USN models in production
environments that don't use the USN library directly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_model(
    model: nn.Module,
    format: str,
    path: str,
    config: Any | None = None,
    **kwargs,
) -> None:
    """Export a USN model to the specified format.

    Args:
        model: The USN model to export.
        format: Target format ("onnx", "safetensors", "state_dict", "torchscript").
        path: Output file path.
        config: Model configuration (included as metadata where supported).
        **kwargs: Format-specific options.

    Raises:
        ValueError: If format is not supported.
        ImportError: If required export library is not installed.
    """
    format_lower = format.lower()

    if format_lower == "state_dict":
        _export_state_dict(model, path)
    elif format_lower == "safetensors":
        _export_safetensors(model, path, config)
    elif format_lower == "onnx":
        _export_onnx(model, path, config, **kwargs)
    elif format_lower == "torchscript":
        _export_torchscript(model, path)
    else:
        raise ValueError(
            f"Unsupported export format: '{format}'. "
            f"Supported: 'onnx', 'safetensors', 'state_dict', 'torchscript'"
        )


def _export_state_dict(model: nn.Module, path: str) -> None:
    """Export as PyTorch state_dict."""
    torch.save(model.state_dict(), path)
    logger.info(f"Exported state_dict to {path}")


def _export_safetensors(model: nn.Module, path: str, config: Any | None = None) -> None:
    """Export as SafeTensors format."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError(
            "safetensors package required for SafeTensors export. "
            "Install with: pip install safetensors"
        )

    tensors = {name: param.data.contiguous() for name, param in model.named_parameters()}

    metadata = {}
    if config is not None:
        config_json = (
            config.to_json() if hasattr(config, "to_json") else json.dumps(config, default=str)
        )
        metadata["config"] = config_json

    save_file(tensors, path, metadata=metadata)
    logger.info(f"Exported SafeTensors to {path}")


def _export_onnx(model: nn.Module, path: str, config: Any | None = None, **kwargs) -> None:
    """Export as ONNX format."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise ImportError(
            "onnx package required for ONNX export. Install with: pip install onnx onnxruntime"
        )

    opset_version = kwargs.get("opset_version", 17)

    model.eval()
    # Create dummy input based on config
    batch_size = 1
    seq_len = kwargs.get("seq_len", 16)
    vocab_size = getattr(config, "vocab_size", 1000) if config else 1000

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Export with dynamic axes
    torch.onnx.export(
        model,
        (dummy_input,),
        path,
        opset_version=opset_version,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
    )
    logger.info(f"Exported ONNX to {path} (opset {opset_version})")


def _export_torchscript(model: nn.Module, path: str) -> None:
    """Export as TorchScript format."""
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(path)
    logger.info(f"Exported TorchScript to {path}")
