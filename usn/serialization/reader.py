"""Reader for the native .usn model format.

Deserializes .usn binary files back into Python objects (config, weights,
metadata, etc.). Verifies SHA-256 checksum before loading any data.
Uses raw bytes + manifest to reconstruct tensors — NO pickle is used anywhere.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from usn.config.model_config import USNConfig
from usn.serialization.format_spec import (
    CHECKSUM_SIZE,
    FIXED_HEADER_SIZE,
    FORMAT_VERSION,
    HEADER_STRUCT_FORMAT,
    MAGIC_NUMBER,
    MAX_FILE_SIZE,
    TOC_ENTRY_SIZE,
    TOC_ENTRY_STRUCT_FORMAT,
    Compression,
    SectionType,
)


class USNReader:
    """Reads .usn binary files and reconstructs model components.

    Supports:
    - Full loading (all sections)
    - Partial loading (config-only, metadata-only, specific sections)
    - Checksum verification before any data is used
    - Compression handling (zlib, lz4)
    - map_location for device placement of tensors

    Example:
        reader = USNReader()
        data = reader.load("model.usn")
        config = data["config"]
        weights = data["weights"]
    """

    def load(
        self,
        path: str | Path,
        map_location: str | torch.device | None = None,
        sections: list[str] | None = None,
    ) -> dict[str, Any]:
        """Load a .usn file and return all (or selected) components.

        Args:
            path: Path to the .usn file.
            map_location: Device to map tensors to (e.g., "cpu", "cuda:0").
            sections: Optional list of section names to load. If None, loads all.
                Valid names: "config", "weights", "tokenizer", "optimizer",
                "metadata", "scheduler", "training_state".

        Returns:
            Dictionary with keys corresponding to loaded sections.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If magic number or format version is invalid.
            IntegrityError: If checksum verification fails.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_data = path.read_bytes()

        if len(file_data) > MAX_FILE_SIZE:
            raise ValueError(
                f"File size ({len(file_data)} bytes) exceeds maximum "
                f"allowed size ({MAX_FILE_SIZE} bytes)"
            )

        # Verify checksum first (integrity check before any parsing)
        self._verify_checksum(file_data)

        # Parse header and TOC
        header, toc_entries = self._parse_header(file_data)
        compression = Compression(header["compression"])

        # Determine which sections to load
        requested_types = self._resolve_sections(sections)

        # Load each requested section
        result: dict[str, Any] = {}
        for section_type, offset, size, compressed_size in toc_entries:
            sec_type = SectionType(section_type)
            if requested_types is not None and sec_type not in requested_types:
                continue

            # Extract raw section bytes
            if compressed_size > 0:
                raw = file_data[offset : offset + compressed_size]
                section_data = self._decompress(raw, compression, size)
            else:
                section_data = file_data[offset : offset + size]

            # Parse section based on type
            key, value = self._parse_section(sec_type, section_data, map_location)
            result[key] = value

        return result

    def load_config_only(self, path: str | Path) -> USNConfig:
        """Load only the config section from a .usn file.

        Args:
            path: Path to the .usn file.

        Returns:
            The reconstructed USNConfig object.

        Raises:
            ValueError: If the file has no CONFIG section.
        """
        data = self.load(path, sections=["config"])
        if "config" not in data:
            raise ValueError(f"No CONFIG section found in {path}")
        return data["config"]

    def load_metadata_only(self, path: str | Path) -> dict[str, Any]:
        """Load only the metadata section from a .usn file.

        Args:
            path: Path to the .usn file.

        Returns:
            Metadata dictionary.

        Raises:
            ValueError: If the file has no METADATA section.
        """
        data = self.load(path, sections=["metadata"])
        if "metadata" not in data:
            raise ValueError(f"No METADATA section found in {path}")
        return data["metadata"]

    def _verify_checksum(self, file_data: bytes) -> None:
        """Verify SHA-256 checksum of the file content.

        The last 32 bytes of the file contain the checksum of everything
        before those 32 bytes.
        """
        from usn.exceptions import IntegrityError

        if len(file_data) < CHECKSUM_SIZE + FIXED_HEADER_SIZE:
            raise ValueError("File is too small to be a valid .usn file")

        content = file_data[:-CHECKSUM_SIZE]
        stored_checksum = file_data[-CHECKSUM_SIZE:]
        computed_checksum = hashlib.sha256(content).digest()

        if computed_checksum != stored_checksum:
            raise IntegrityError(
                "SHA-256 checksum verification failed. The file may be corrupted or tampered with."
            )

    def _parse_header(
        self, file_data: bytes
    ) -> tuple[dict[str, Any], list[tuple[int, int, int, int]]]:
        """Parse magic number, format version, header, and TOC.

        Returns:
            Tuple of (header_dict, list of TOC entries).
            Each TOC entry is (section_type, offset, size, compressed_size).
        """
        offset = 0

        # Magic number (4 bytes)
        magic = struct.unpack_from("<I", file_data, offset)[0]
        offset += 4
        if magic != MAGIC_NUMBER:
            raise ValueError(
                f"Invalid magic number: 0x{magic:08X}. "
                f"Expected 0x{MAGIC_NUMBER:08X} ('USNF'). "
                "This is not a valid .usn file."
            )

        # Format version (4 bytes)
        version = struct.unpack_from("<I", file_data, offset)[0]
        offset += 4
        if version > FORMAT_VERSION:
            raise ValueError(
                f"Unsupported format version: {version}. "
                f"Maximum supported version: {FORMAT_VERSION}. "
                "Please update the USN library."
            )

        # Header fields
        endianness, compression, section_count, total_file_size = struct.unpack_from(
            HEADER_STRUCT_FORMAT, file_data, offset
        )
        offset += struct.calcsize(HEADER_STRUCT_FORMAT)

        header = {
            "version": version,
            "endianness": endianness,
            "compression": compression,
            "section_count": section_count,
            "total_file_size": total_file_size,
        }

        # Table of Contents
        toc_entries: list[tuple[int, int, int, int]] = []
        for _ in range(section_count):
            section_type, sec_offset, size, compressed_size = struct.unpack_from(
                TOC_ENTRY_STRUCT_FORMAT, file_data, offset
            )
            toc_entries.append((section_type, sec_offset, size, compressed_size))
            offset += TOC_ENTRY_SIZE

        return header, toc_entries

    def _resolve_sections(self, sections: list[str] | None) -> set[SectionType] | None:
        """Convert section name strings to SectionType enum values."""
        if sections is None:
            return None

        name_to_type = {
            "config": SectionType.CONFIG,
            "weights": SectionType.WEIGHTS,
            "tokenizer": SectionType.TOKENIZER,
            "optimizer": SectionType.OPTIMIZER,
            "metadata": SectionType.METADATA,
            "scheduler": SectionType.SCHEDULER,
            "training_state": SectionType.TRAINING_STATE,
        }

        result = set()
        for name in sections:
            name_lower = name.lower()
            if name_lower not in name_to_type:
                raise ValueError(
                    f"Unknown section name: {name!r}. Valid sections: {list(name_to_type.keys())}"
                )
            result.add(name_to_type[name_lower])

        return result

    def _decompress(self, data: bytes, compression: Compression, original_size: int) -> bytes:
        """Decompress section data using the specified algorithm."""
        if compression == Compression.ZLIB:
            decompressed = zlib.decompress(data)
        elif compression == Compression.LZ4:
            try:
                import lz4.frame  # type: ignore[import-untyped]

                decompressed = lz4.frame.decompress(data)
            except ImportError:
                raise ImportError(
                    "lz4 package required for LZ4 decompression. Install with: pip install lz4"
                )
        else:
            return data

        if len(decompressed) != original_size:
            raise ValueError(
                f"Decompressed size ({len(decompressed)}) does not match "
                f"expected size ({original_size})"
            )
        return decompressed

    def _parse_section(
        self,
        section_type: SectionType,
        data: bytes,
        map_location: str | torch.device | None,
    ) -> tuple[str, Any]:
        """Parse a section's raw bytes into the appropriate Python object."""
        if section_type == SectionType.CONFIG:
            config_json = data.decode("utf-8")
            config = USNConfig.from_json(config_json)
            return "config", config

        elif section_type == SectionType.WEIGHTS:
            weights = self._parse_weights(data, map_location)
            return "weights", weights

        elif section_type == SectionType.TOKENIZER:
            return "tokenizer", data

        elif section_type == SectionType.OPTIMIZER:
            optimizer_state = self._parse_optimizer(data, map_location)
            return "optimizer", optimizer_state

        elif section_type == SectionType.METADATA:
            metadata = json.loads(data.decode("utf-8"))
            return "metadata", metadata

        elif section_type == SectionType.SCHEDULER:
            scheduler_state = json.loads(data.decode("utf-8"))
            return "scheduler", scheduler_state

        elif section_type == SectionType.TRAINING_STATE:
            training_state = json.loads(data.decode("utf-8"))
            return "training_state", training_state

        else:
            # Unknown section type — store raw bytes
            return f"unknown_{section_type}", data

    def _parse_weights(
        self, data: bytes, map_location: str | torch.device | None
    ) -> dict[str, torch.Tensor]:
        """Parse WEIGHTS section: manifest + raw tensor data → dict of tensors.

        Format:
            [4 bytes: manifest_size (uint32)]
            [manifest_size bytes: JSON manifest]
            [raw tensor data concatenated]
        """
        # Read manifest size
        manifest_size = struct.unpack_from("<I", data, 0)[0]
        offset = 4

        # Read manifest JSON
        manifest_json = data[offset : offset + manifest_size].decode("utf-8")
        manifest = json.loads(manifest_json)
        offset += manifest_size

        # Raw tensor data starts after the manifest
        tensor_data_start = offset

        # Reconstruct tensors
        weights: dict[str, torch.Tensor] = {}
        device = torch.device(map_location) if map_location is not None else torch.device("cpu")

        for entry in manifest:
            name: str = entry["name"]
            dtype_str: str = entry["dtype"]
            shape: list[int] = entry["shape"]
            tensor_offset: int = entry["offset"]
            tensor_size: int = entry["size"]

            # Extract raw bytes for this tensor
            start = tensor_data_start + tensor_offset
            end = start + tensor_size
            raw_bytes = data[start:end]

            # Reconstruct tensor
            tensor = self._bytes_to_tensor(raw_bytes, dtype_str, shape)
            tensor = tensor.to(device)
            weights[name] = tensor

        return weights

    def _bytes_to_tensor(self, raw_bytes: bytes, dtype_str: str, shape: list[int]) -> torch.Tensor:
        """Convert raw bytes + dtype + shape into a torch.Tensor.

        Handles bfloat16 specially since numpy doesn't support it.
        """
        if dtype_str == "torch.bfloat16":
            # bfloat16 is not supported by numpy; use torch.frombuffer
            tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.bfloat16).reshape(shape)
            return tensor.clone()

        # Map dtype string to torch dtype
        dtype_map = {
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
            "torch.float16": torch.float16,
            "torch.int8": torch.int8,
            "torch.int16": torch.int16,
            "torch.int32": torch.int32,
            "torch.int64": torch.int64,
            "torch.uint8": torch.uint8,
            "torch.bool": torch.bool,
        }

        torch_dtype = dtype_map.get(dtype_str)
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

        # Map to numpy dtype for reconstruction
        np_dtype_map = {
            "torch.float32": np.float32,
            "torch.float64": np.float64,
            "torch.float16": np.float16,
            "torch.int8": np.int8,
            "torch.int16": np.int16,
            "torch.int32": np.int32,
            "torch.int64": np.int64,
            "torch.uint8": np.uint8,
            "torch.bool": np.bool_,
        }

        np_dtype = np_dtype_map[dtype_str]
        array = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape)
        tensor = torch.from_numpy(array.copy())
        return tensor

    def _parse_optimizer(
        self, data: bytes, map_location: str | torch.device | None
    ) -> dict[str, Any]:
        """Parse OPTIMIZER section: JSON metadata + raw tensor data.

        Format:
            [4 bytes: json_size (uint32)]
            [json_size bytes: JSON with param_groups, tensor_manifest, scalar_state]
            [raw tensor data concatenated]
        """
        # Read JSON size
        json_size = struct.unpack_from("<I", data, 0)[0]
        offset = 4

        # Read JSON metadata
        json_bytes = data[offset : offset + json_size]
        json_state = json.loads(json_bytes.decode("utf-8"))
        offset += json_size

        # Raw tensor data starts here
        tensor_data_start = offset

        device = torch.device(map_location) if map_location is not None else torch.device("cpu")

        # Reconstruct optimizer state
        state: dict[str, Any] = {
            "param_groups": json_state.get("param_groups", []),
            "state": {},
        }

        # Reconstruct tensor entries
        for entry in json_state.get("tensor_manifest", []):
            param_id = str(entry["param_id"])
            key = entry["key"]
            dtype_str = entry["dtype"]
            shape = entry["shape"]
            tensor_offset = entry["offset"]
            tensor_size = entry["size"]

            start = tensor_data_start + tensor_offset
            end = start + tensor_size
            raw_bytes = data[start:end]

            tensor = self._bytes_to_tensor(raw_bytes, dtype_str, shape)
            tensor = tensor.to(device)

            if param_id not in state["state"]:
                state["state"][param_id] = {}
            state["state"][param_id][key] = tensor

        # Restore scalar state values
        for param_id, scalars in json_state.get("scalar_state", {}).items():
            if param_id not in state["state"]:
                state["state"][param_id] = {}
            state["state"][param_id].update(scalars)

        return state
