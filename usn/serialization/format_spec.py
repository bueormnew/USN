"""
.usn Binary Format Specification v1

Native binary format for USN model serialization.
NO pickle is used — weights are stored as raw numerical data with explicit dtype/shape.

Layout:
┌─────────────────────────────────┐
│ Magic Number (4 bytes): 0x55534E46 ("USNF")
│ Format Version (4 bytes): uint32
├─────────────────────────────────┤
│ Header (variable):
│   - endianness: uint8 (0=little, 1=big)
│   - compression: uint8 (0=none, 1=zlib, 2=lz4)
│   - section_count: uint32
│   - total_file_size: uint64
├─────────────────────────────────┤
│ Table of Contents (section_count × entry):
│   - section_type: uint16
│   - offset: uint64
│   - size: uint64
│   - compressed_size: uint64 (0 if uncompressed)
├─────────────────────────────────┤
│ Section: CONFIG (type=0x01)
│   JSON-encoded USNConfig
├─────────────────────────────────┤
│ Section: WEIGHTS (type=0x02)
│   Tensor manifest: [{name, dtype, shape, offset, size}]
│   Raw tensor data (contiguous, platform-independent)
├─────────────────────────────────┤
│ Section: TOKENIZER (type=0x03) [optional]
│   Tokenizer type + vocabulary + merges
├─────────────────────────────────┤
│ Section: OPTIMIZER (type=0x04) [optional]
│   Optimizer state tensors (raw numerical data)
├─────────────────────────────────┤
│ Section: METADATA (type=0x05)
│   JSON: {usn_version, pytorch_version, format_version, date, ...}
├─────────────────────────────────┤
│ Section: SCHEDULER (type=0x06) [optional]
│   Scheduler state as JSON
├─────────────────────────────────┤
│ Section: TRAINING_STATE (type=0x07) [optional]
│   Training state as JSON (step, epoch, best_loss, etc.)
├─────────────────────────────────┤
│ SHA-256 Checksum (32 bytes)
└─────────────────────────────────┘
"""

from __future__ import annotations

import sys
from enum import IntEnum

# Magic number: "USNF" in ASCII (0x55 0x53 0x4E 0x46)
MAGIC_NUMBER: int = 0x55534E46

# Current format version
FORMAT_VERSION: int = 1

# SHA-256 checksum size in bytes
CHECKSUM_SIZE: int = 32

# Header struct format (after magic + version):
#   endianness (uint8) + compression (uint8) + section_count (uint32) + total_file_size (uint64)
HEADER_STRUCT_FORMAT: str = "<BBIq"

# Table of contents entry format:
#   section_type (uint16) + offset (uint64) + size (uint64) + compressed_size (uint64)
TOC_ENTRY_STRUCT_FORMAT: str = "<HQQq"

# Size of a single TOC entry in bytes (2 + 8 + 8 + 8 = 26)
TOC_ENTRY_SIZE: int = 26

# Size of the fixed header portion (magic + version + header fields)
# 4 (magic) + 4 (version) + 1 (endianness) + 1 (compression) + 4 (section_count) + 8 (total_size) = 22
FIXED_HEADER_SIZE: int = 22


class SectionType(IntEnum):
    """Section identifiers for the .usn format."""

    CONFIG = 0x01
    WEIGHTS = 0x02
    TOKENIZER = 0x03
    OPTIMIZER = 0x04
    METADATA = 0x05
    SCHEDULER = 0x06
    TRAINING_STATE = 0x07


class Compression(IntEnum):
    """Compression algorithms supported by the .usn format."""

    NONE = 0
    ZLIB = 1
    LZ4 = 2


class Endianness(IntEnum):
    """Byte order specification."""

    LITTLE = 0
    BIG = 1


def get_system_endianness() -> Endianness:
    """Return the endianness of the current system."""
    if sys.byteorder == "little":
        return Endianness.LITTLE
    return Endianness.BIG


# Mapping from torch dtype strings to numpy-compatible dtype identifiers
DTYPE_MAP: dict[str, str] = {
    "torch.float32": "float32",
    "torch.float64": "float64",
    "torch.float16": "float16",
    "torch.bfloat16": "bfloat16",
    "torch.int8": "int8",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "torch.uint8": "uint8",
    "torch.bool": "bool",
}

# Maximum file size for security (10 GB)
MAX_FILE_SIZE: int = 10 * 1024 * 1024 * 1024
