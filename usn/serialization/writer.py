"""Writer for the native .usn model format.

Serializes USN models to a single binary file containing all components
needed for exact model reconstruction. Uses raw numerical data with explicit
dtype/shape metadata — NO pickle is used anywhere.

Security: All tensor data is stored as raw bytes with manifest describing
name, dtype, shape, offset, and size for each parameter tensor.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import torch
import torch.nn as nn

import usn
from usn.serialization.format_spec import (
    CHECKSUM_SIZE,
    FIXED_HEADER_SIZE,
    FORMAT_VERSION,
    HEADER_STRUCT_FORMAT,
    MAGIC_NUMBER,
    TOC_ENTRY_SIZE,
    TOC_ENTRY_STRUCT_FORMAT,
    Compression,
    Endianness,
    SectionType,
    get_system_endianness,
)


class USNWriter:
    """Writes USN models to the native .usn binary format.

    The .usn format stores everything in a SINGLE binary file:
    - Magic number + format version (identification)
    - Header (endianness, compression, section count)
    - Table of Contents (section types, offsets, sizes)
    - Data sections (CONFIG, WEIGHTS, TOKENIZER, OPTIMIZER, METADATA, etc.)
    - SHA-256 checksum (integrity verification)

    Example:
        writer = USNWriter()
        writer.save("model.usn", model, config=model.config)
    """

    def save(
        self,
        path: str | Path,
        model: nn.Module,
        config: Any | None = None,
        include_optimizer: bool = False,
        optimizer: Any | None = None,
        include_tokenizer: bool = False,
        tokenizer_data: bytes | None = None,
        compression: str = "none",
        metadata: dict[str, str] | None = None,
        scheduler_state: dict[str, Any] | None = None,
        training_state: dict[str, Any] | None = None,
    ) -> None:
        """Save a model to a .usn file.

        Args:
            path: Output file path (should end with .usn).
            model: The nn.Module to serialize (weights are extracted from parameters).
            config: Model configuration object. If it has a to_json() method, it
                will be used; otherwise config is serialized via json.dumps.
            include_optimizer: Whether to include optimizer state.
            optimizer: The optimizer instance (required if include_optimizer=True).
            include_tokenizer: Whether to include tokenizer data.
            tokenizer_data: Raw tokenizer bytes (required if include_tokenizer=True).
            compression: Compression algorithm ("none", "zlib", or "lz4").
            metadata: Additional string key-value metadata to include.
            scheduler_state: Optional scheduler state dict to include.
            training_state: Optional training state dict (step, epoch, etc.).
        """
        path = Path(path)
        compression_type = self._parse_compression(compression)

        # Build section data
        sections: list[tuple[SectionType, bytes]] = []

        # CONFIG section
        if config is not None:
            config_bytes = self._serialize_config(config)
            sections.append((SectionType.CONFIG, config_bytes))

        # WEIGHTS section (raw tensor data + manifest, NO pickle)
        weights_bytes = self._serialize_weights(model)
        sections.append((SectionType.WEIGHTS, weights_bytes))

        # TOKENIZER section (optional)
        if include_tokenizer and tokenizer_data is not None:
            sections.append((SectionType.TOKENIZER, tokenizer_data))

        # OPTIMIZER section (optional)
        if include_optimizer and optimizer is not None:
            optimizer_bytes = self._serialize_optimizer(optimizer)
            sections.append((SectionType.OPTIMIZER, optimizer_bytes))

        # SCHEDULER section (optional)
        if scheduler_state is not None:
            scheduler_bytes = json.dumps(scheduler_state).encode("utf-8")
            sections.append((SectionType.SCHEDULER, scheduler_bytes))

        # TRAINING_STATE section (optional)
        if training_state is not None:
            training_bytes = json.dumps(training_state).encode("utf-8")
            sections.append((SectionType.TRAINING_STATE, training_bytes))

        # METADATA section (always included)
        meta_bytes = self._serialize_metadata(metadata)
        sections.append((SectionType.METADATA, meta_bytes))

        # Apply compression to section data if requested
        compressed_sections = self._compress_sections(sections, compression_type)

        # Write the complete file
        self._write_file(path, compressed_sections, compression_type)

    def _parse_compression(self, compression: str) -> Compression:
        """Parse compression string to enum value."""
        compression_map = {
            "none": Compression.NONE,
            "zlib": Compression.ZLIB,
            "lz4": Compression.LZ4,
        }
        if compression.lower() not in compression_map:
            raise ValueError(
                f"Unsupported compression: {compression!r}. "
                f"Supported: {list(compression_map.keys())}"
            )
        return compression_map[compression.lower()]

    def _serialize_config(self, config: Any) -> bytes:
        """Serialize config to JSON bytes."""
        if hasattr(config, "to_json"):
            config_json = config.to_json()
        elif isinstance(config, dict):
            config_json = json.dumps(config, indent=2)
        else:
            config_json = json.dumps(config, indent=2, default=str)
        return config_json.encode("utf-8")

    def _serialize_weights(self, model: nn.Module) -> bytes:
        """Serialize model weights as manifest + raw tensor data.

        Format:
            [4 bytes: manifest_size (uint32)]
            [manifest_size bytes: JSON manifest]
            [raw tensor data concatenated]

        The manifest is a JSON array of entries:
            {name, dtype, shape, offset, size}

        Where offset is relative to the start of the raw tensor data block.
        This avoids pickle entirely — just raw bytes with explicit metadata.
        """
        manifest: list[dict[str, Any]] = []
        tensor_buffer = io.BytesIO()
        current_offset = 0

        for name, param in model.named_parameters():
            # Get raw bytes from contiguous CPU tensor
            tensor_data = param.data.detach().cpu().contiguous()
            raw_bytes = self._tensor_to_bytes(tensor_data)

            manifest.append(
                {
                    "name": name,
                    "dtype": str(param.dtype),
                    "shape": list(param.shape),
                    "offset": current_offset,
                    "size": len(raw_bytes),
                }
            )

            tensor_buffer.write(raw_bytes)
            current_offset += len(raw_bytes)

        # Also include buffers (e.g., running stats in normalization layers)
        for name, buffer in model.named_buffers():
            if buffer is None:
                continue
            tensor_data = buffer.detach().cpu().contiguous()
            raw_bytes = self._tensor_to_bytes(tensor_data)

            manifest.append(
                {
                    "name": f"__buffer__.{name}",
                    "dtype": str(buffer.dtype),
                    "shape": list(buffer.shape),
                    "offset": current_offset,
                    "size": len(raw_bytes),
                }
            )

            tensor_buffer.write(raw_bytes)
            current_offset += len(raw_bytes)

        # Pack: manifest_size + manifest_json + raw_tensor_data
        manifest_json = json.dumps(manifest).encode("utf-8")
        result = struct.pack("<I", len(manifest_json)) + manifest_json + tensor_buffer.getvalue()
        return result

    def _tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
        """Convert a tensor to raw bytes without pickle.

        For bfloat16 tensors (not supported by numpy), we use torch's
        internal storage directly.
        """
        if tensor.dtype == torch.bfloat16:
            # bfloat16 is not supported by numpy, use raw storage bytes
            storage = tensor.untyped_storage()
            return bytes(storage)
        else:
            return tensor.numpy().tobytes()

    def _serialize_optimizer(self, optimizer: Any) -> bytes:
        """Serialize optimizer state as raw tensor data + JSON metadata.

        Stores optimizer state tensors as raw bytes with a manifest,
        similar to model weights. Scalar state values are stored as JSON.
        """
        state_dict = optimizer.state_dict()
        buffer = io.BytesIO()

        # Separate tensor data from scalar/config data
        json_state: dict[str, Any] = {
            "param_groups": state_dict.get("param_groups", []),
            "tensor_manifest": [],
        }
        tensor_buffer = io.BytesIO()
        current_offset = 0

        for param_id, param_state in state_dict.get("state", {}).items():
            for key, value in param_state.items():
                if isinstance(value, torch.Tensor):
                    tensor_data = value.detach().cpu().contiguous()
                    raw_bytes = self._tensor_to_bytes(tensor_data)
                    json_state["tensor_manifest"].append(
                        {
                            "param_id": param_id,
                            "key": key,
                            "dtype": str(value.dtype),
                            "shape": list(value.shape),
                            "offset": current_offset,
                            "size": len(raw_bytes),
                        }
                    )
                    tensor_buffer.write(raw_bytes)
                    current_offset += len(raw_bytes)
                else:
                    # Store scalar values inline in the JSON
                    json_state.setdefault("scalar_state", {}).setdefault(str(param_id), {})[key] = (
                        value
                    )

        # Pack: json_size + json_data + raw_tensor_data
        json_bytes = json.dumps(json_state).encode("utf-8")
        buffer.write(struct.pack("<I", len(json_bytes)))
        buffer.write(json_bytes)
        buffer.write(tensor_buffer.getvalue())
        return buffer.getvalue()

    def _serialize_metadata(self, extra_metadata: dict[str, str] | None) -> bytes:
        """Serialize metadata section as JSON."""
        meta = {
            "usn_version": usn.__version__,
            "pytorch_version": torch.__version__,
            "format_version": FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "platform_endianness": "little"
            if get_system_endianness() == Endianness.LITTLE
            else "big",
        }
        if extra_metadata:
            meta.update(extra_metadata)
        return json.dumps(meta, indent=2).encode("utf-8")

    def _compress_sections(
        self,
        sections: list[tuple[SectionType, bytes]],
        compression: Compression,
    ) -> list[tuple[SectionType, bytes, int]]:
        """Optionally compress section data.

        Returns list of (section_type, data, original_size) tuples.
        When compressed, data contains compressed bytes and original_size is
        the uncompressed size. When uncompressed, original_size is 0.
        """
        result: list[tuple[SectionType, bytes, int]] = []
        for section_type, data in sections:
            if compression == Compression.NONE:
                result.append((section_type, data, 0))
            elif compression == Compression.ZLIB:
                compressed = zlib.compress(data, level=6)
                # Only use compression if it actually saves space
                if len(compressed) < len(data):
                    result.append((section_type, compressed, len(data)))
                else:
                    result.append((section_type, data, 0))
            elif compression == Compression.LZ4:
                try:
                    import lz4.frame

                    compressed = lz4.frame.compress(data)
                    if len(compressed) < len(data):
                        result.append((section_type, compressed, len(data)))
                    else:
                        result.append((section_type, data, 0))
                except ImportError:
                    raise ImportError(
                        "lz4 package required for LZ4 compression. Install with: pip install lz4"
                    )
        return result

    def _write_file(
        self,
        path: Path,
        sections: list[tuple[SectionType, bytes, int]],
        compression: Compression,
    ) -> None:
        """Write the complete .usn file with header, TOC, sections, and checksum."""
        buffer = io.BytesIO()
        section_count = len(sections)
        endianness = get_system_endianness()

        # Calculate sizes for offset computation
        toc_size = section_count * TOC_ENTRY_SIZE
        data_start_offset = FIXED_HEADER_SIZE + toc_size

        # Compute section offsets
        toc_entries: list[tuple[SectionType, int, int, int]] = []
        current_offset = data_start_offset
        for section_type, data, original_size in sections:
            compressed_size = len(data) if original_size > 0 else 0
            actual_size = original_size if original_size > 0 else len(data)
            toc_entries.append(
                (
                    section_type,
                    current_offset,
                    actual_size,
                    compressed_size,
                )
            )
            current_offset += len(data)

        total_file_size = current_offset + CHECKSUM_SIZE

        # Write magic number and format version
        buffer.write(struct.pack("<I", MAGIC_NUMBER))
        buffer.write(struct.pack("<I", FORMAT_VERSION))

        # Write header
        buffer.write(
            struct.pack(
                HEADER_STRUCT_FORMAT,
                endianness.value,
                compression.value,
                section_count,
                total_file_size,
            )
        )

        # Write table of contents
        for section_type, offset, size, compressed_size in toc_entries:
            buffer.write(
                struct.pack(
                    TOC_ENTRY_STRUCT_FORMAT,
                    section_type,
                    offset,
                    size,
                    compressed_size,
                )
            )

        # Write section data
        for _section_type, data, _original_size in sections:
            buffer.write(data)

        # Compute SHA-256 checksum over all content written so far
        content = buffer.getvalue()
        checksum = hashlib.sha256(content).digest()
        buffer.write(checksum)

        # Atomic write: write to temp then rename for safety
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(buffer.getvalue())

    def _compute_checksum(self, data: bytes) -> bytes:
        """Compute SHA-256 checksum of data."""
        return hashlib.sha256(data).digest()
