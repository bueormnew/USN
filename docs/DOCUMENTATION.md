# USN Architecture Library — Documentation

## 1. Architecture Overview

The **Unified State Network (USN)** is a novel autoregressive sequence modeling architecture that replaces attention mechanisms with a unified persistent state. Key properties:

- **O(n) training complexity** via associative parallel scan
- **O(1) inference memory** — constant-size state, no KV cache
- **No attention mechanism** or quadratic operations anywhere

### Block Diagram

A complete USN model is structured as:

```
Token Embedding → N × USN Block → Final Norm → Output Head → Logits
```

Each USN block processes input through 8 sequential stages:

```
x_t → [RMSNorm] → [Input Projection] → [Temporal Mixing] → [Exponential Gating]
    → [Selective Writing] → [State Update] → [State Readout] → [Channel Mixing] → y_t
    └─────────────────────── + Residual ───────────────────────────────────────────┘
```

### Core Equations

| Stage | Equation |
|-------|----------|
| Input Projection | u_t = W_u x_t + b_u |
| Temporal Mixing | α_t = σ(W_α x_t + b_α); m_t = α_t ⊙ u_t + (1-α_t) ⊙ u_{t-1} |
| Exponential Gating | λ_t = exp(-softplus(W_λ x_t + b_λ)) ∈ (0,1) |
| Relational Decay | ρ_t = exp(-softplus(W_ρ x_t + b_ρ)) ∈ (0,1) |
| Write Gate | g_t = σ(W_g m_t + U_g read(S_{t-1}) + b_g) |
| Semantic Update | s_t = λ_t ⊙ s_{t-1} + g_t ⊙ (B_s m_t) |
| Relational Update | R_t = ρ_t R_{t-1} + (B_r m_t)(C_r m_t)^T |
| State Readout | z_t = W_s s_t + W_r vec(R_t); c_t = σ(W_c m_t + b_c); o_t = c_t ⊙ z_t |
| Channel Mixing | y_t = m_t + W_2 φ(W_1(c_t ⊙ z_t)) |

### Data Flow

- **Training**: Full sequences processed in parallel using associative scan (prefix-sum) over affine state transitions.
- **Inference**: Token-by-token generation updating only the fixed-size state S = (s, R) per layer.

The unified state has two subspaces:
- **Semantic state** s_t ∈ R^{d_s} — feature-level information (vector)
- **Relational state** R_t ∈ R^{k×k} — entity-entity interactions (matrix)

---

## 2. Installation and Quick Start

### Installation

```bash
# From PyPI (when published)
pip install usn

# From source
git clone <repo-url>
cd USN
pip install -e ".[dev]"
```

### Dependencies

- Python >= 3.10
- PyTorch >= 2.1
- NumPy, PyYAML, tqdm

Optional: `triton` (GPU kernels), `safetensors`, `onnx`, `wandb`

### Quick Start

```python
import usn

# Create a model from a preset
model = usn.create_model("tiny")
print(usn.summary(model))

# Save and load
usn.save(model, "my_model.usn")
model = usn.load("my_model.usn")

# Training (with a dataset)
from usn import USNTrainingConfig
config = USNTrainingConfig(max_steps=1000, batch_size=8)
result = usn.train(model, dataset, config)

# Generation (with a tokenizer)
text = usn.generate(model, "Hello world", max_tokens=50, tokenizer=tokenizer)
```

---

## 3. API Reference

### `usn.create_model(config_or_preset, **kwargs)`

Factory function to instantiate a USNModel.

```python
# From preset name
model = usn.create_model("small")

# From config object
from usn import USNConfig
config = USNConfig(num_layers=8, d_model=512, d_s=256, k=12, d_ff=2048)
model = usn.create_model(config)
```

### `usn.train(model, dataset, config=None, **kwargs)`

High-level training function. Returns training result with loss history.

```python
from usn import USNTrainingConfig
config = USNTrainingConfig(learning_rate=3e-4, max_steps=50000)
result = usn.train(model, train_dataset, config)
```

### `usn.generate(model, prompt, max_tokens=256, tokenizer=None, **kwargs)`

Generate text from a prompt. Requires a tokenizer.

```python
output = usn.generate(model, "Once upon a time", max_tokens=100, tokenizer=tok)
```

### `usn.save(model, path, **kwargs)`

Save model to `.usn` format (binary, no pickle, SHA-256 verified).

```python
usn.save(model, "checkpoints/model.usn")
```

### `usn.load(path, map_location=None)`

Load a model from `.usn` format.

```python
model = usn.load("checkpoints/model.usn")
```

### `usn.export(model, format, path, **kwargs)`

Export to standard formats.

```python
usn.export(model, "safetensors", "model.safetensors")
usn.export(model, "onnx", "model.onnx")
usn.export(model, "state_dict", "weights.pt")
usn.export(model, "torchscript", "model.pt")
```

### `usn.summary(model)`

Print architecture summary with parameter counts and memory estimates.

### `usn.from_pretrained(path)`

Alias for `usn.load()` — loads a pretrained model from path.

### Utility Functions

| Function | Description |
|----------|-------------|
| `usn.set_seed(seed)` | Set all random seeds (Python, NumPy, PyTorch, CUDA) |
| `usn.count_parameters(model)` | Total parameter count |
| `usn.estimate_memory(config)` | Memory estimate for a config |
| `usn.estimate_flops(config, seq_len)` | FLOPs estimate |
| `usn.device_info()` | Detected hardware information |
| `usn.set_acceleration_level(level)` | Override kernel acceleration level |

---

## 4. Configuration

### USNConfig — Model Architecture

```python
from usn import USNConfig

config = USNConfig(
    num_layers=12,       # Number of USN blocks
    d_model=768,         # Model hidden dimension
    d_s=512,             # Semantic state dimension
    k=16,                # Relational state dimension (R ∈ R^{k×k})
    d_ff=3072,           # Feedforward intermediate dimension
    vocab_size=50257,    # Vocabulary size
    max_seq_len=2048,    # Maximum sequence length
    norm_type="rmsnorm", # "rmsnorm" or "layernorm"
    activation="gelu",   # "gelu", "silu", or "relu"
    dropout=0.0,         # Dropout rate
    tie_weights=True,    # Tie embedding and output weights
    chunk_size=64,       # Parallel scan chunk size
    fused=True,          # Use fused kernels if available
)
```

**Presets** (class methods):

| Preset | Params | Layers | d_model | d_s | k |
|--------|--------|--------|---------|-----|---|
| `tiny` | ~2M | 4 | 128 | 64 | 8 |
| `micro` | ~5M | 6 | 192 | 128 | 8 |
| `mini` | ~15M | 8 | 384 | 256 | 12 |
| `small` | ~125M | 12 | 768 | 512 | 16 |
| `base` | ~350M | 24 | 1024 | 768 | 24 |
| `medium` | ~750M | 32 | 1280 | 1024 | 32 |
| `large` | ~1.3B | 36 | 1536 | 1024 | 32 |
| `xl` | ~2.7B | 48 | 2048 | 1536 | 48 |
| `xxl` | ~6.7B | 64 | 2560 | 2048 | 48 |

```python
config = USNConfig.from_preset("small")
```

### USNTrainingConfig — Training Hyperparameters

```python
from usn import USNTrainingConfig

config = USNTrainingConfig(
    learning_rate=3e-4,
    batch_size=32,
    max_steps=100_000,
    warmup_steps=2000,
    weight_decay=0.1,
    grad_clip=1.0,
    mixed_precision="bf16",          # "none", "fp16", "bf16"
    gradient_accumulation_steps=1,
    scheduler_type="cosine",         # "cosine", "linear", "constant", "cosine_restarts"
    eval_interval=500,
    checkpoint_interval=1000,
    early_stopping_patience=0,       # 0 = disabled
    distributed_strategy="none",     # "none", "ddp", "fsdp"
    sequence_curriculum=False,       # Enable curriculum learning
    stability_mode=False,            # NaN/spike detection
)
```

### USNGenerationConfig — Inference Parameters

```python
from usn import USNGenerationConfig

config = USNGenerationConfig(
    temperature=1.0,
    top_k=0,              # 0 = disabled
    top_p=1.0,            # 1.0 = no nucleus sampling
    beam_width=1,         # 1 = greedy/sampling, >1 = beam search
    max_new_tokens=256,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
    streaming=False,
)
```

---

## 5. Training Guide

### Basic Training

```python
import usn
from usn import USNConfig, USNTrainingConfig, USNTrainer
from usn.datasets import USNDataset

# Setup
config = USNConfig.small()
model = usn.create_model(config)

train_dataset = USNDataset("train.txt", tokenizer=tokenizer, max_seq_len=512)
val_dataset = USNDataset("val.txt", tokenizer=tokenizer, max_seq_len=512)

training_config = USNTrainingConfig(
    learning_rate=3e-4,
    batch_size=32,
    max_steps=50000,
    mixed_precision="bf16",
)

# Train
trainer = USNTrainer(model, train_dataset, training_config, val_dataset=val_dataset)
result = trainer.train()
```

### Distributed Training

```python
config = USNTrainingConfig(distributed_strategy="ddp")
trainer = USNTrainer(model, dataset, config)
trainer.train()
```

Launch with:
```bash
torchrun --nproc_per_node=4 train_script.py
```

FSDP for large models:
```python
config = USNTrainingConfig(distributed_strategy="fsdp")
```

### Curriculum Learning

Gradually increase sequence length during training:

```python
config = USNTrainingConfig(
    sequence_curriculum=True,
    curriculum_start_len=128,
    curriculum_end_len=2048,
    curriculum_warmup_steps=10000,
    curriculum_schedule="linear",
)
```

### Training Stability

Enable stability monitoring to detect and handle NaN/divergence:

```python
config = USNTrainingConfig(
    stability_mode=True,
    nan_skip_batch=True,
    loss_spike_threshold=5.0,
    state_max_norm=1000.0,
)
```

### Checkpointing and Resume

```python
# Save/load handled automatically by trainer at checkpoint_interval
# Resume from checkpoint:
trainer = USNTrainer(model, dataset, config)
trainer.load_checkpoint("checkpoints/step_10000.usn")
trainer.train()  # Continues from step 10000
```

---

## 6. Inference Guide

### Basic Generation

```python
from usn import USNGenerator, USNGenerationConfig

gen_config = USNGenerationConfig(temperature=0.8, top_p=0.95, max_new_tokens=200)
generator = USNGenerator(model, tokenizer, gen_config)

output = generator.generate("The quick brown fox")
text = tokenizer.decode(output.token_ids[0].tolist())
```

### Streaming

```python
for token_text, token_id, log_prob in generator.stream("Tell me a story"):
    print(token_text, end="", flush=True)
```

Async streaming for web frameworks:
```python
async for token_text, token_id, log_prob in generator.astream("Hello"):
    await websocket.send(token_text)
```

### Beam Search

```python
config = USNGenerationConfig(beam_width=5, length_penalty=0.6)
generator = USNGenerator(model, tokenizer, config)
output = generator.generate("Translate: hello")
```

### Repetition Control

```python
config = USNGenerationConfig(
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    frequency_penalty=0.5,
)
```

### Batch Generation

```python
prompts = ["Question 1:", "Question 2:", "Question 3:"]
output = generator.generate(prompts, max_new_tokens=100)
# output.token_ids shape: (3, generated_len)
```

---

## 7. .usn Format Specification

The native `.usn` binary format stores everything needed to reconstruct a model in a single file with **no pickle** dependency.

### File Layout

```
┌─────────────────────────────────────┐
│ Magic Number: 0x55534E46 (4 bytes)  │  "USNF"
│ Format Version (4 bytes)            │  uint32
├─────────────────────────────────────┤
│ Header:                             │
│   endianness (uint8)                │
│   compression (uint8): none/zlib/lz4│
│   section_count (uint32)            │
│   total_file_size (uint64)          │
├─────────────────────────────────────┤
│ Table of Contents                   │
│   Per entry: type, offset, size     │
├─────────────────────────────────────┤
│ Section: CONFIG (JSON USNConfig)    │
│ Section: WEIGHTS (raw tensors)      │
│ Section: TOKENIZER (optional)       │
│ Section: OPTIMIZER (optional)       │
│ Section: METADATA (JSON)            │
│ Section: SCHEDULER (optional)       │
│ Section: TRAINING_STATE (optional)  │
├─────────────────────────────────────┤
│ SHA-256 Checksum (32 bytes)         │
└─────────────────────────────────────┘
```

### Design Choices

- **No pickle**: All tensors stored as raw bytes with explicit dtype/shape metadata
- **Single file**: Everything needed to reload a model is in one `.usn` file
- **Integrity**: SHA-256 checksum verifies file integrity on load
- **Compression**: Optional zlib or lz4 compression per section
- **Partial loading**: Can read just config or metadata without loading weights

### Usage

```python
from usn.serialization import USNWriter, USNReader

# Write
writer = USNWriter()
writer.save("model.usn", model, config=model.config)

# Read
reader = USNReader()
data = reader.load("model.usn")
# data["config"], data["weights"], data["metadata"]
```

---

## 8. Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_backends.py -v

# By marker
pytest tests/ -m "not slow"
pytest tests/ -m "gpu"

# With coverage
pytest tests/ --cov=usn --cov-report=html
```

### Test Structure

```
tests/
├── conftest.py                    # Shared fixtures (tiny_config, tiny_model)
├── test_modules/                  # Per-module unit tests
│   ├── test_input_projection.py
│   ├── test_temporal_mixing.py
│   ├── test_exponential_gating.py
│   └── ...
├── test_layers/                   # Block, scan, norm tests
├── test_models/                   # Full model tests
├── test_config/                   # Config validation tests
├── test_serialization/            # Save/load round-trip tests
├── test_properties/               # Property-based tests (hypothesis)
│   ├── test_gate_bounds.py        # All gates ∈ (0,1)
│   ├── test_associativity.py      # Scan composition law
│   ├── test_scan_equivalence.py   # Parallel == sequential
│   ├── test_causality.py          # No future information leaks
│   └── test_state_boundedness.py  # State never explodes
└── test_backends.py               # Acceleration fallback tests
```

### Property-Based Tests

The library uses [hypothesis](https://hypothesis.readthedocs.io/) for property-based testing. Key properties verified:

1. **Gate Boundedness**: All gates (λ, ρ, g, α, c) remain strictly in (0, 1) for any input
2. **Associativity**: Transition composition is associative within floating-point tolerance
3. **Scan Equivalence**: Parallel scan produces identical results to sequential recurrence
4. **Causality**: No operation leaks future information
5. **State Boundedness**: State norms remain finite under arbitrary inputs

```bash
# Run property tests
pytest tests/test_properties/ -v --hypothesis-seed=42
```

---

## 9. Benchmarks

### Running Benchmarks

```bash
# Via CLI
usn benchmark --model model.usn --all

# Programmatically
import usn
model = usn.create_model("small")
results = usn.benchmark_acceleration()
```

### What's Measured

| Benchmark | Description |
|-----------|-------------|
| Forward throughput | Tokens/sec in training forward pass |
| Inference latency | Time per generated token |
| Memory usage | Peak GPU memory per batch |
| Scan performance | Parallel scan vs sequential recurrence speed |
| Acceleration levels | Throughput at each kernel level (Triton → eager) |

### Benchmarks Directory

Place custom benchmarks in `benchmarks/`. The library provides utilities:

```python
from usn.utils.timing import timer, memory_tracker
from usn.utils.profiling import profile_forward, profile_backward

with timer("forward_pass"):
    output = model(input_ids)

with memory_tracker() as mem:
    output = model(input_ids)
print(f"Peak memory: {mem.peak_mb:.1f} MB")
```

---

## 10. Design Rationale

### Why No Attention?

Attention is O(n²) in sequence length. USN replaces it with a fixed-size state updated via O(1) operations per token, yielding O(n) total training cost and O(1) inference memory. The unified state captures both local context (temporal mixing) and long-range dependencies (state persistence with learned decay).

### Why exp(-softplus(·)) for Gating?

The construction `exp(-softplus(x))` guarantees output strictly in (0, 1) by mathematical identity:
- `softplus(x) = ln(1 + exp(x))` is always positive
- Negation makes it negative
- `exp(negative)` is strictly in (0, 1)

This is more numerically stable than alternatives like `sigmoid` for decay rates, avoids the vanishing gradient problem at extremes, and provides smooth gradients everywhere.

### Why Separate Semantic + Relational State?

- **Semantic vector** (s_t ∈ R^{d_s}): Captures feature-level information efficiently in O(d_s) memory
- **Relational matrix** (R_t ∈ R^{k×k}): Captures entity-entity interactions via outer products in O(k²) memory

Together they provide richer state representation than either alone, at controllable memory cost.

### Why Pre-Norm Architecture?

Normalization before each block (rather than after) improves gradient flow and training stability for deep stacks. This is empirically validated in transformer literature and applies equally to USN.

### Why RMSNorm over LayerNorm?

RMSNorm skips mean subtraction, making it cheaper to compute while being empirically equivalent for deep networks. The paper specifies it as the preferred normalization.

### Why Affine Associative Transitions?

The state update `S_t = A_t S_{t-1} + b_t` is both affine and associative under composition:
```
(A_2, b_2) ∘ (A_1, b_1) = (A_2·A_1, A_2·b_1 + b_2)
```
This enables parallel prefix-sum computation during training (O(log n) depth) while maintaining sequential O(1) inference.

### Why Log-Space Decay Accumulation?

Computing `Σ log(λ_i)` instead of `Π λ_i` prevents numerical underflow when accumulating many decay factors across long sequences. Essential for sequences >100 tokens.

### Why Outer Product for Relational State?

The update `(B_r m_t)(C_r m_t)^T` captures bilinear interactions between two projections of the input, encoding relational structure in O(k²) rather than O(d_model²). The rank-1 outer product is computationally cheap yet expressively rich.

### Why Weight Tying?

Tying embedding and output weights (E = W_out^T) reduces parameters significantly for large vocabularies and enforces consistency between input and output token representations.

### Why Confidence Gate on Readout?

The gate c_t = σ(W_c m_t + b_c) controls how much state information flows to output. This prevents noise in stale or partially-updated state entries from corrupting predictions.

---

## 11. Future Work

- **Longer context**: Extend chunk decomposition for 100K+ sequences
- **Multimodal**: Adapt state architecture for vision and audio inputs
- **Sparse state**: Explore sparse relational matrices for larger k
- **Quantization**: INT8/INT4 inference for deployment on edge devices
- **Model parallelism**: Tensor-parallel for training XXL configurations
- **Retrieval augmentation**: Use state as retrieval query for external memory
- **Continual learning**: Exploit state persistence for online adaptation
- **Hardware-specific kernels**: CUDA kernels for non-Triton GPUs, Metal for Apple Silicon
- **Mixture of states**: Multiple state banks with routing for capacity scaling
- **Formal verification**: Prove boundedness and convergence properties mathematically
