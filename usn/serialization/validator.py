"""Format validation for .usn files.

Provides integrity and compatibility checks without loading full model data.
Designed for quick pre-flight validation before committing to a full load.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

from usn.serialization.format_spec import (
    CHECKSUM_SIZE,
    FIXED_HEADER_SIZE,
    FORMAT_VERSION,
    HEADER_STRUCT_FORMAT,
    MAGIC_NUMBER,
    TOC_ENTRY_SIZE,
    TOC_ENTRY_STRUCT_FORMAT,
    SectionType,
)


class FormatValidator:
    """Validates .usn file integrity and compatibility.

    Provides lightweight validation methods that don't require loading
    the entire file into memory for tensor reconstruction.

    Example:
        validator = FormatValidator()
        if validator.verify_checksum("model.usn"):
            version = validator.verify_format_version("model.usn")
            print(f"Valid .usn file, format version {version}")
    """

    def verify_checksum(self, path: str | Path) -> bool:
        """Verify the SHA-256 checksum of a .usn file.

        Reads the file and compares the stored checksum (last 32 bytes)
        against a computed checksum of all preceding content.

        Args:
            path: Path to the .usn file.

        Returns:
            True if checksum matches, False otherwise.
        """
        path = Path(path)
        if not path.exists():
            return False

        file_data = path.read_bytes()
        if len(file_data) < CHECKSUM_SIZE + FIXED_HEADER_SIZE:
            return False

        content = file_data[:-CHECKSUM_SIZE]
        stored_checksum = file_data[-CHECKSUM_SIZE:]
        computed_checksum = hashlib.sha256(content).digest()

        return computed_checksum == stored_checksum

    def verify_format_version(self, path: str | Path) -> int:
        """Read and verify the format version of a .usn file.

        Checks that the magic number is valid and returns the format version.
        Raises an error if the version is unsupported.

        Args:
            path: Path to the .usn file.

        Returns:
            The format version number (integer).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If magic number is invalid or version is unsupported.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "rb") as f:
            header_bytes = f.read(8)  # magic (4) + version (4)

        if len(header_bytes) < 8:
            raise ValueError("File is too small to be a valid .usn file")

        magic = struct.unpack_from("<I", header_bytes, 0)[0]
        if magic != MAGIC_NUMBER:
            raise ValueError(
                f"Invalid magic number: 0x{magic:08X}. "
                f"Expected 0x{MAGIC_NUMBER:08X} ('USNF'). "
                "This is not a valid .usn file."
            )

        version: int = struct.unpack_from("<I", header_bytes, 4)[0]
        if version > FORMAT_VERSION:
            raise ValueError(
                f"Unsupported format version: {version}. "
                f"Maximum supported version: {FORMAT_VERSION}. "
                "Please update the USN library."
            )

        return version

    def verify_weights_match_config(
        self, weights_manifest: list[dict[str, Any]], config: Any
    ) -> bool:
        """Verify that a weights manifest is compatible with a model config.

        Checks that expected parameter names/shapes from the config are
        present in the weights manifest.

        This is a basic structural check — it verifies that the total number
        of weight tensors (excluding buffers) and their shapes are consistent
        with what the config would produce.

        Args:
            weights_manifest: List of manifest entries, each with keys
                "name", "dtype", "shape", "offset", "size".
            config: A USNConfig instance (or object with num_layers, d_model,
                d_s, k, d_ff, vocab_size attributes).

        Returns:
            True if the weights are structurally compatible with the config,
            False otherwise.
        """
        if not weights_manifest:
            return False

        # Extract parameter names (exclude buffers prefixed with __buffer__)
        param_names = [
            entry["name"]
            for entry in weights_manifest
            if not entry["name"].startswith("__buffer__.")
        ]

        if not param_names:
            return False

        # Basic structural checks:
        # 1. Check that embedding dimension matches d_model
        for entry in weights_manifest:
            if "embedding" in entry["name"] and "weight" in entry["name"]:
                shape = entry["shape"]
                if len(shape) == 2:
                    # Embedding weight should be (vocab_size, d_model)
                    if hasattr(config, "vocab_size") and shape[0] != config.vocab_size:
                        return False
                    if hasattr(config, "d_model") and shape[1] != config.d_model:
                        return False
                break

        # 2. Check that we have parameters for each expected layer
        if hasattr(config, "num_layers"):
            layer_indices = set()
            for name in param_names:
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if part == "blocks" and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            layer_indices.add(layer_idx)
                        except ValueError:
                            pass
            # If we found layer indices, verify count matches
            if layer_indices:
                expected_layers = set(range(config.num_layers))
                if layer_indices != expected_layers:
                    return False

        # 3. Verify d_model appears in linear layer shapes
        if hasattr(config, "d_model"):
            found_d_model = False
            for entry in weights_manifest:
                shape = entry["shape"]
                if config.d_model in shape:
                    found_d_model = True
                    break
            if not found_d_model:
                return False

        return True
