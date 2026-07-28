# USN - Unified State Network Architecture Library

A production-grade Python package implementing the Unified State Network (USN) architecture for autoregressive sequence modeling.

USN replaces attention mechanisms with a unified persistent state partitioned into semantic (vector) and relational (matrix) subspaces, achieving **O(n) training complexity** via associative parallel scan and **O(1) inference memory** via constant-size state.

## Features

- Novel state-space architecture with no attention mechanism
- O(n) training via parallel associative scan
- O(1) inference memory via constant-size persistent state
- 4-level acceleration hierarchy (Triton → torch.compile → custom autograd → eager)
- Single-file `.usn` serialization format
- CLI interface for training, generation, and benchmarking

## Installation

```bash
pip install USN
```

For development:

```bash
pip install USN[dev]
```

For CUDA acceleration:

```bash
pip install USN[cuda]
```

For all optional dependencies:

```bash
pip install USN[all]
```

## Quick Start

```python
import usn

# Create a model from a preset configuration
config = usn.USNConfig.tiny()
model = usn.create_model(config)

# Print model summary
print(model.summary())
```

## License

MIT License - Copyright (c) 2024 BUEORM
