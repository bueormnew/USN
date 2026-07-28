"""USN layers: blocks, normalization, and parallel scan implementations.

This package contains the composed layers that build on individual modules:
- USNBlock: Complete processing block (all 8 submodules in order)
- ParallelScan: Associative scan for training-time parallelization
- ChunkedParallelScan: Memory-efficient chunked scan
- RMSNorm/USNLayerNorm: Normalization layers
"""

from usn.layers.block import USNBlock
from usn.layers.chunked_scan import ChunkedParallelScan
from usn.layers.norm import RMSNorm, USNLayerNorm, create_norm
from usn.layers.parallel_scan import (
    ParallelScanFunction,
    parallel_scan_relational,
    parallel_scan_semantic,
)

__all__ = [
    "USNBlock",
    "ChunkedParallelScan",
    "ParallelScanFunction",
    "RMSNorm",
    "USNLayerNorm",
    "create_norm",
    "parallel_scan_relational",
    "parallel_scan_semantic",
]
