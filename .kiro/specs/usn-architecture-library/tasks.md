# Implementation Plan: USN Architecture Library

## Overview

Complete implementation of the Unified State Network (USN) Architecture Library — a production-grade Python package implementing a novel autoregressive sequence modeling architecture with O(n) training complexity via associative parallel scan and O(1) inference memory. The implementation follows a phased approach: project structure → core types → modules → layers → model → training → inference → testing → benchmarks → documentation.

## Tasks

- [x] 1. Project Structure and Packaging
  - [x] 1.1 Create project directory structure and packaging files
    - Create pyproject.toml with package name "USN", license "MIT", author "BUEORM", Python >=3.10, all dependencies, build system, CLI entry points
    - Create setup.py for backward compatibility
    - Create all package directories: usn/, usn/core/, usn/modules/, usn/layers/, usn/models/, usn/training/, usn/datasets/, usn/tokenizers/, usn/serialization/, usn/utils/, usn/optim/, usn/losses/, usn/config/, usn/backends/, usn/cli/, usn/inference/
    - Create support directories: tests/, benchmarks/, scripts/, notebooks/, docs/, examples/
    - Create LICENSE (MIT, BUEORM), README.md, CHANGELOG.md, MANIFEST.in, .gitignore
    - Add __init__.py stubs to all packages
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.12, 1.13_

  - [x] 1.2 Create exception hierarchy and base error classes
    - Implement usn/exceptions.py with: USNError, ConfigError, ShapeError, IntegrityError, VersionError, TrainingError, GenerationError
    - Add sub-exceptions: InvalidParameterError, IncompatibleConfigError, NaNDetectedError, DivergenceError, OOMError, InvalidPromptError, DecodingError
    - _Requirements: 89.4, 54.1_

- [x] 2. Core Types, Interfaces, and Base Classes
  - [x] 2.1 Implement core type definitions
    - Create usn/core/types.py with NamedTuples: UnifiedState(semantic, relational), ModelState(layers), BlockOutput(hidden, state), GenerationOutput(token_ids, log_probs, final_state), AffineTransition(A_semantic, b_semantic, A_relational, b_relational)
    - All types fully typed with torch.Tensor annotations
    - _Requirements: 7.3, 7.4, 54.1_

  - [x] 2.2 Implement base module class and interfaces
    - Create usn/core/base.py with USNModule(nn.Module, ABC) defining abstract properties: objective, complexity, constraints, and reset_parameters method
    - Create usn/core/interfaces.py with TokenizerInterface(ABC), LossInterface(ABC), SchedulerInterface(ABC)
    - Create usn/core/__init__.py exporting all public symbols
    - _Requirements: 54.1, 92.4, 92.6, 25.1_

  - [x] 2.3 Implement activation function registry and utilities
    - Create usn/core/activations.py with get_activation(name) factory, register_activation(name, fn) for extensibility
    - Support "gelu", "silu", "relu" using PyTorch built-in implementations
    - _Requirements: 65.1, 65.2, 65.3, 65.4, 65.6, 65.7, 92.1_

- [x] 3. Configuration System
  - [x] 3.1 Implement USNConfig with validation and presets
    - Create usn/config/model_config.py with frozen dataclass USNConfig: num_layers, d_model, d_s, k, d_ff, vocab_size, max_seq_len, norm_type, norm_eps, activation, dropout, embedding_dropout, residual_dropout, tie_weights, scale_embeddings, init_method, chunk_size, fused
    - Implement __post_init__ validation: d_s <= d_model, k >= 1, d_ff >= d_model, all positive, cross-parameter constraints
    - Implement preset class methods: tiny(), micro(), mini(), small(), base(), medium(), large(), xl(), xxl(), from_preset(name)
    - Implement serialization: to_json(), to_yaml(), from_json(), from_yaml(), from_dict()
    - _Requirements: 24.1, 24.4, 24.5, 24.6, 24.7, 24.8, 24.9, 24.10, 24.11, 24.12, 50.1–50.12, 93.1–93.7_

  - [x] 3.2 Implement USNTrainingConfig and USNGenerationConfig
    - Create USNTrainingConfig frozen dataclass with all training hyperparameters: learning_rate, batch_size, max_steps, warmup_steps, weight_decay, grad_clip, mixed_precision, gradient_accumulation_steps, scheduler_type, eval/checkpoint intervals, early_stopping, distributed_strategy, curriculum, stability, logging
    - Create USNGenerationConfig frozen dataclass: temperature, top_k, top_p, beam_width, max_new_tokens, repetition_penalty, frequency_penalty, presence_penalty, no_repeat_ngram_size, length_penalty, stop_tokens, streaming
    - Implement validation for both configs
    - Create usn/config/__init__.py exporting all configs
    - _Requirements: 24.2, 24.3, 24.4, 24.5_

- [x] 4. Normalization Layers
  - [x] 4.1 Implement RMSNorm, LayerNorm, and norm factory
    - Create usn/layers/norm.py with RMSNorm (y = x / RMS(x) × γ), LayerNorm (standard), and create_norm(norm_type, d_model, eps) factory
    - Support float32, float16, bfloat16 computation
    - Initialize γ=1, β=0 (LayerNorm only)
    - Default eps=1e-6
    - _Requirements: 15.1, 15.2, 15.5, 15.6, 15.7, 15.8, 15.9_

- [x] 5. USN Modules Implementation
  - [x] 5.1 Implement InputProjection module
    - Create usn/modules/input_projection.py with InputProjection(USNModule): linear W_u x_t + b_u
    - Xavier uniform init for W_u, zeros for b_u
    - Input/output shape: (batch, seq, d_model)
    - Implement objective, complexity, constraints properties
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

  - [x] 5.2 Implement TemporalMixing module
    - Create usn/modules/temporal_mixing.py: α_t = σ(W_α x_t + b_α), m_t = α_t ⊙ u_t + (1-α_t) ⊙ u_{t-1}
    - Learned initial u_{-1} parameter, caching for inference mode
    - Training: parallel via shifted tensors; Inference: single-step with cached u_prev
    - Xavier init for W_α, zeros for b_α
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 4.11_

  - [x] 5.3 Implement ExponentialGating module
    - Create usn/modules/exponential_gating.py: λ_t = exp(-softplus(W_λ x_t + b_λ)), ρ_t = exp(-softplus(W_ρ x_t + b_ρ))
    - Guarantee λ_t, ρ_t ∈ (0, 1) by construction
    - Initialize b_λ so initial λ ∈ [0.9, 0.99]; numerically stable softplus (clamp for x > 20)
    - Separate projections: W_λ ∈ R^{d_s × d_model}, W_ρ ∈ R^{1 × d_model}
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11, 95.1–95.7_

  - [x] 5.4 Implement SelectiveWriting module
    - Create usn/modules/selective_writing.py: g_t = σ(W_g m_t + U_g read(S_{t-1}) + b_g)
    - Implement read_state: combine semantic + vectorized relational projections
    - W_g ∈ R^{d_s × d_model}, U_g ∈ R^{d_s × d_read}, b_g ∈ R^{d_s}
    - Output g_t ∈ (0,1) shape (batch, seq, d_s)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 6.11, 6.12, 56.1–56.7_

  - [x] 5.5 Implement StateUpdate module
    - Create usn/modules/state_update.py: s_t = λ_t ⊙ s_{t-1} + g_t ⊙ (B_s m_t), R_t = ρ_t R_{t-1} + (B_r m_t)(C_r m_t)^T
    - B_s ∈ R^{d_s × d_model}, B_r ∈ R^{k × d_model}, C_r ∈ R^{k × d_model}
    - Implement forward_sequential (inference) and forward_parallel (training) methods
    - Initialize s_0=0, R_0=0; support passing initial state
    - Outer product via torch.bmm on unsqueezed vectors for relational update
    - _Requirements: 7.1–7.15, 55.1–55.9_

  - [x] 5.6 Implement StateReadout module
    - Create usn/modules/state_readout.py: z_t = W_s s_t + W_r vec(R_t), c_t = σ(W_c m_t + b_c), o_t = c_t ⊙ z_t
    - W_s ∈ R^{d_model × d_s}, W_r ∈ R^{d_model × k²}, W_c ∈ R^{d_model × d_model}
    - Vectorize R_t into k² vector before projection
    - Return tuple (o_t, c_t, z_t) for downstream use
    - _Requirements: 8.1–8.10_

  - [x] 5.7 Implement ChannelMixing module
    - Create usn/modules/channel_mixing.py: y_t = m_t + W_2 φ(W_1(c_t ⊙ z_t))
    - W_1 ∈ R^{d_ff × d_model} (up-projection), W_2 ∈ R^{d_model × d_ff} (down-projection)
    - Configurable activation φ (gelu default), dropout on MLP output
    - Residual connection from m_t is mandatory
    - _Requirements: 9.1–9.10, 64.2_

  - [x] 5.8 Create modules package __init__.py with all exports
    - Export all 7 modules from usn/modules/__init__.py
    - Verify no circular imports
    - _Requirements: 54.2, 54.15, 54.16_

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all module implementations are correct; run quick shape/gradient checks, ask the user if questions arise.

- [x] 7. Parallel Scan and Layer Assembly
  - [x] 7.1 Implement ParallelScan custom autograd function
    - Create usn/layers/parallel_scan.py with ParallelScan(torch.autograd.Function)
    - Forward: log-space associative scan computing all states s_1..s_n from log_decay, values, initial_state
    - Backward: reverse scan for gradient computation (custom backward for memory efficiency)
    - Composition rule: (log_a2, b2) ∘ (log_a1, b1) = (log_a2 + log_a1, exp(log_a2) × b1 + b2)
    - Log-space for numerical stability; produce identical results to sequential recurrence
    - _Requirements: 12.1–12.13_

  - [x] 7.2 Implement ChunkedParallelScan module
    - Create ChunkedParallelScan(nn.Module) in same file or usn/layers/chunked_scan.py
    - Divide sequence into chunks of size C, parallel scan within each, sequential propagation between chunks
    - Support non-divisible sequence lengths (tail chunk handling)
    - Configurable chunk_size (default 64); memory O(C × d_state + n/C × d_state)
    - Compatible with gradient checkpointing at chunk boundaries
    - _Requirements: 13.1–13.9_

  - [x] 7.3 Implement USNBlock layer
    - Create usn/layers/block.py with USNBlock(nn.Module) composing all submodules in exact order:
      1. Normalization (pre-norm), 2. InputProjection, 3. TemporalMixing, 4. ExponentialGating, 5. SelectiveWriting, 6. StateUpdate, 7. StateReadout, 8. ChannelMixing
    - Block-level residual: output = x + dropout(block_output)
    - Support training (parallel scan) and inference (sequential) modes
    - Expose all submodules as named attributes
    - Manage caches/buffers when switching train/eval modes
    - _Requirements: 10.1–10.12, 64.1, 64.3, 64.4, 64.5_

  - [x] 7.4 Create layers package __init__.py
    - Export USNBlock, ParallelScan, ChunkedParallelScan, RMSNorm, LayerNorm, create_norm
    - _Requirements: 54.3, 54.15_

- [x] 8. Model Assembly
  - [x] 8.1 Implement TokenEmbedding and OutputHead
    - Create usn/models/embedding.py with TokenEmbedding: learned E ∈ R^{vocab_size × d_model}, optional √d_model scaling, embedding dropout
    - Create OutputHead: linear W_out to vocab logits (no softmax), optional bias (default False)
    - Support weight tying: output_head.weight = embedding.weight
    - Normal(0, 0.02) init for embeddings; Normal(0, 0.02/√(2*num_layers)) for output head
    - _Requirements: 58.1–58.8, 59.1–59.7, 16.3, 16.6_

  - [x] 8.2 Implement USNModel class
    - Create usn/models/usn_model.py: Embedding → N × USNBlock → FinalNorm → OutputHead
    - Accept USNConfig; apply weight tying if configured; _init_weights() applying full initialization scheme
    - forward(input_ids, initial_state, padding_mask) → (logits, final_state)
    - Properties: num_parameters, state_size_per_layer, total_state_size
    - Methods: get_state(), set_state(), reset_state(), summary(), enable_gradient_checkpointing()
    - O(n) training via parallel scan, O(1) inference memory, no attention anywhere
    - _Requirements: 11.1–11.15, 16.1–16.10, 51.1–51.7, 52.1–52.9, 60.1–60.7, 69.1–69.7, 98.1–98.8_

  - [x] 8.3 Implement model factory function and package init
    - Create usn/models/__init__.py exporting USNModel, TokenEmbedding, OutputHead
    - Wire up usn.create_model(config, device) factory in models or top-level
    - Log model creation with parameter count, state size, memory estimate
    - _Requirements: 2.6, 2.3, 54.4_

- [x] 9. Checkpoint - Ensure all tests pass
  - Verify USNModel instantiates correctly with tiny/micro configs, forward pass produces correct shapes, backward pass succeeds. Ask the user if questions arise.

- [x] 10. Backend Acceleration System
  - [x] 10.1 Implement DeviceDetector and AccelerationManager
    - Create usn/backends/detection.py: detect() returning hardware info, best_device()
    - Create usn/backends/acceleration.py: AccelerationLevel enum (TRITON=1, COMPILE=2, AUTOGRAD=3, EAGER=4), AccelerationManager with detect_best_level(), set_level(), get_level(), get_kernel()
    - Auto-detect at import; graceful fallback; manual override support
    - _Requirements: 105.1–105.12_

  - [x] 10.2 Implement Triton fused projections kernel
    - Create usn/backends/triton_kernels.py with fused_projections: combines W_u, W_α, W_λ into single GEMM reading x once
    - Concatenate weights into W_gate; produce three output slices
    - Support both training and inference; verified numerically identical to unfused
    - _Requirements: 101.1–101.10, 14.1, 63.1_

  - [x] 10.3 Implement Triton fused temporal+gate kernel
    - Add fused_temporal_gate: sigmoid + interpolation + exp(-softplus) entirely in registers
    - Zero intermediate VRAM materializations; write only m_t and λ_t
    - Numerically stable softplus (clamp for x > 20)
    - _Requirements: 102.1–102.10, 14.2, 63.2_

  - [x] 10.4 Implement Triton fused state core kernel
    - Add fused_state_core: entire intra-chunk state update + readout in SRAM
    - State variables (s, R) live in shared memory for chunk duration
    - Sequential loop through C timesteps within SRAM; write only output and final state to VRAM
    - Custom backward pass with gradient checkpointing from chunk boundaries
    - _Requirements: 103.1–103.14, 14.3, 63.3, 63.4_

  - [x] 10.5 Implement Triton fused channel MLP kernel
    - Add fused_channel_mlp: tiled W_1 → activation → W_2 avoiding full d_ff materialization
    - Support GELU and SiLU activations in fused path
    - Process in tiles of TILE_F along d_ff dimension
    - _Requirements: 104.1–104.7, 14.5, 63.5_

  - [x] 10.6 Implement PyTorch fallback implementations for all kernels
    - Create usn/backends/fallbacks.py with eager-mode equivalents for each fused kernel
    - Level 2 (torch.compile hints), Level 3 (custom autograd), Level 4 (standard eager)
    - All fallbacks produce identical results to Triton kernels
    - Create usn/backends/__init__.py exporting public API
    - _Requirements: 105.3–105.9, 14.6, 63.6, 63.7_

- [x] 11. Optimizer and Loss Functions
  - [x] 11.1 Implement OptimizerFactory and parameter groups
    - Create usn/optim/factory.py: OptimizerFactory.create(model, config) creating AdamW with proper weight decay separation
    - get_parameter_groups: separate decay (weights) vs no-decay (biases, norms, embeddings)
    - Support Adam, SGD with momentum, custom optimizer registration
    - _Requirements: 19.1, 19.2, 19.8, 19.9, 19.10_

  - [x] 11.2 Implement LR schedulers
    - Create usn/optim/schedulers.py: CosineAnnealingScheduler, LinearWarmupScheduler, WarmupCosineScheduler (default), CosineWarmRestartsScheduler, ConstantScheduler
    - All implement SchedulerInterface: get_lr(step), state_dict(), load_state_dict()
    - Create create_scheduler(config) factory
    - Create usn/optim/__init__.py
    - _Requirements: 19.3, 19.4, 19.5, 19.6, 19.7, 92.2_

  - [x] 11.3 Implement cross-entropy loss and perplexity
    - Create usn/losses/cross_entropy.py: USNCrossEntropyLoss(nn.Module) with label_smoothing, ignore_index
    - Use log_softmax formulation for numerical stability
    - Support padding mask; average over valid tokens
    - Implement compute_perplexity(loss) = exp(loss)
    - Create usn/losses/__init__.py
    - _Requirements: 20.1–20.8, 92.3_

- [x] 12. Tokenizer System
  - [x] 12.1 Implement tokenizer implementations
    - Create usn/tokenizers/char_tokenizer.py: CharTokenizer implementing TokenizerInterface, supports special tokens (PAD, BOS, EOS, UNK)
    - Create usn/tokenizers/word_tokenizer.py: WordTokenizer with configurable vocabulary
    - Create usn/tokenizers/bpe_tokenizer.py: BPETokenizer wrapping HuggingFace tokenizers library, supports from_pretrained, train, save, load, batch encode/decode
    - Create usn/tokenizers/__init__.py
    - _Requirements: 25.1–25.10, 54.7_

- [x] 13. Dataset System
  - [x] 13.1 Implement USNDataset and streaming variant
    - Create usn/datasets/usn_dataset.py: USNDataset(torch.utils.data.Dataset) supporting text, json, jsonl, csv, huggingface sources
    - Tokenize text; create causal LM pairs (input=tokens[:-1], target=tokens[1:])
    - Configurable max_seq_len with truncation and padding
    - Create StreamingUSNDataset(IterableDataset) for large corpora with shuffle buffer
    - _Requirements: 26.1–26.10, 54.6_

  - [x] 13.2 Implement collate function and MathDataset
    - Create usn_collate_fn: pad sequences, create padding masks
    - Create usn/datasets/math_dataset.py: MathDataset generating "5+3=8", "12*7=84" etc. with CharTokenizer
    - Support configurable operations, max_digits, train/val/test splits
    - Create usn/datasets/__init__.py
    - _Requirements: 26.7, 26.11, 32.2_

- [x] 14. Serialization System
  - [x] 14.1 Implement .usn format specification and writer
    - Create usn/serialization/format_spec.py: MAGIC_NUMBER, FORMAT_VERSION, SectionType enum
    - Create usn/serialization/writer.py: USNWriter.save() writing header, TOC, config (JSON), weights (raw tensors with manifest), tokenizer, optimizer, metadata, SHA-256 checksum
    - NO pickle usage; raw numerical data with explicit dtype/shape
    - Support optional compression (zlib/lz4)
    - _Requirements: 22.1–22.15, 91.1–91.7_

  - [x] 14.2 Implement .usn reader and validator
    - Create usn/serialization/reader.py: USNReader.load() with checksum verification, partial loading (config-only, metadata-only, weights-only), map_location support
    - Create usn/serialization/validator.py: FormatValidator with verify_checksum, verify_format_version, verify_weights_match_config
    - Create usn/serialization/migration.py: FormatMigrator with migration registry for version evolution
    - Create usn/serialization/__init__.py
    - _Requirements: 22.6, 22.9, 22.11, 22.12, 22.13, 66.1–66.8, 87.1–87.6_

  - [x] 14.3 Implement model export formats
    - Create usn/serialization/export.py: export to ONNX (with tracing), SafeTensors, state_dict, TorchScript
    - Verify output equivalence post-export
    - Support format-specific options (opset version, dynamic axes)
    - _Requirements: 23.1–23.8_

- [x] 15. Checkpoint - Ensure all tests pass
  - Verify serialization round-trip works (save → load → forward produces identical output). Ask the user if questions arise.

- [x] 16. Training System
  - [x] 16.1 Implement USNTrainer core training loop
    - Create usn/training/trainer.py: USNTrainer accepting model, train_dataset, training_config, val_dataset, tokenizer, optimizer, scheduler
    - Implement train(): forward → loss → backward → grad_clip → optimizer_step → scheduler_step → logging
    - Implement train_step(batch): single step with mixed precision (autocast + GradScaler for fp16, autocast only for bf16)
    - Gradient accumulation over configurable micro-batches
    - Teacher forcing; keep normalization/loss in fp32
    - _Requirements: 17.1–17.18, 57.1–57.10_

  - [x] 16.2 Implement evaluation, checkpointing, and early stopping
    - Implement evaluate(): validation loss + perplexity on val set
    - Implement save_checkpoint/load_checkpoint/resume: full state (model, optimizer, scheduler, step, epoch, loss_history, random_states, grad_scaler, config, curriculum, dataloader)
    - Keep N most recent checkpoints; save best model by val loss
    - Early stopping with configurable patience and min_delta
    - _Requirements: 17.10–17.14, 61.1–61.13, 70.1–70.7, 82.1–82.7, 85.1–85.7_

  - [x] 16.3 Implement curriculum scheduler and training stability
    - Create usn/training/curriculum.py: CurriculumScheduler (start_len → end_len over warmup_steps, linear/log schedule)
    - Implement stability features: NaN detection in forward pass (configurable stability_mode), loss spike detection (>5x threshold), state magnitude monitoring with emergency clipping, nan_skip_batch
    - Implement diagnose() method for training diagnostics
    - _Requirements: 17.15, 106.1–106.4_

  - [x] 16.4 Implement distributed training support
    - Create usn/training/distributed.py: DistributedTrainer with setup(strategy, model), cleanup(), is_main_process()
    - Support DDP and FSDP strategies via torch.distributed with NCCL backend
    - Correct gradient synchronization, rank-based checkpointing (rank 0 saves), data partitioning
    - Fallback to single-device with warning if distributed unavailable
    - Create usn/training/__init__.py
    - _Requirements: 18.1–18.10, 62.1–62.6_

- [x] 17. Inference and Generation System
  - [x] 17.1 Implement USNGenerator with decode strategies
    - Create usn/inference/generator.py: USNGenerator(model, tokenizer, config)
    - Implement generate(prompt, max_new_tokens): prefill state from prompt, then autoregressive decode
    - Implement decode strategies: _greedy_decode, _sample_with_temperature, _top_k_filter, _top_p_filter
    - Combined strategies: temperature + top-k + top-p
    - O(1) memory w.r.t. generated length; causality guaranteed
    - _Requirements: 21.1–21.16, 52.1–52.9_

  - [x] 17.2 Implement beam search and repetition penalty
    - Implement _beam_search: maintain beam_width hypotheses, length penalty, early stopping per beam
    - Implement _apply_repetition_penalty: divide logits of previously generated tokens by penalty factor
    - Support n-gram blocking, frequency penalty, presence penalty
    - Correct state management for beam expansion (duplicate state per beam)
    - _Requirements: 83.1–83.7, 84.1–84.8_

  - [x] 17.3 Implement streaming and batch generation
    - Implement stream(prompt) → Iterator yielding (token_text, token_id, log_prob) one at a time
    - Implement astream(prompt) → AsyncIterator for async web frameworks
    - Support batch generation: multiple prompts processed simultaneously with early stopping per sequence
    - Support cancellation without resource leaks
    - Create usn/inference/__init__.py
    - _Requirements: 21.9, 21.10, 71.1–71.6, 72.1–72.7, 73.1–73.7_

- [x] 18. Utility Functions
  - [x] 18.1 Implement counting, seed, and timing utilities
    - Create usn/utils/counting.py: count_parameters(model), estimate_memory(config, mode), estimate_flops(config, seq_len)
    - Create usn/utils/seed.py: set_seed(seed) setting Python, NumPy, PyTorch, CUDA seeds
    - Create usn/utils/timing.py: timer(name) context manager, memory_tracker() context manager
    - _Requirements: 30.4, 98.1–98.8_

  - [x] 18.2 Implement profiling, visualization, and diagnostics
    - Create usn/utils/profiling.py: profile_forward, profile_backward, profile_memory
    - Create usn/utils/visualization.py: visualize_state, visualize_gates
    - Create usn/utils/diagnostics.py: gradient_stats, activation_stats, check_state_health
    - Create usn/utils/__init__.py exporting all utilities
    - _Requirements: 74.1–74.7_

- [x] 19. CLI System
  - [x] 19.1 Implement CLI commands
    - Create usn/cli/main.py using click: cli group with train, generate, benchmark, info, export, validate commands
    - train: --config (YAML path), --verbose/--quiet
    - generate: --model, --prompt, --max-tokens, --temperature, --top-k, --top-p
    - benchmark: --model, --all flag
    - info: --model (display model summary)
    - export: --model, --format (onnx/safetensors/state_dict/torchscript), --output
    - validate: --model (verify .usn file integrity)
    - Create usn/cli/__init__.py; wire entry point in pyproject.toml
    - _Requirements: 54.14_

- [x] 20. Public API and Top-Level Package
  - [x] 20.1 Implement top-level usn/__init__.py and public API
    - Export core classes: USNModel, USNConfig, USNTrainer, USNGenerator, USNTrainingConfig, USNGenerationConfig
    - Implement factory functions: create_model, train, generate, save, load, export, from_pretrained, summary, benchmark, set_seed, device_info, set_acceleration_level, benchmark_acceleration
    - Set __version__ = "0.1.0", __author__ = "BUEORM"
    - Define __all__ in every __init__.py
    - Verify all imports work: `from usn import USNModel, USNConfig` and `from usn.modules import InputProjection`
    - _Requirements: 2.1–2.16, 99.1–99.8, 89.1–89.7_

- [x] 21. Checkpoint - Ensure all tests pass
  - Full integration: create_model → forward → train → save → load → generate works end-to-end. Ask the user if questions arise.

- [x] 22. Code Quality Rewrite
  - [x] 22.1 Add type stubs and enforce type safety
    - Create .pyi type stub files for all public API surfaces
    - Run mypy strict on entire codebase; fix all type errors
    - Ensure all public functions have complete type annotations
    - _Requirements: 2.15, 100.3_

  - [x] 22.2 Add comprehensive docstrings and lint cleanup
    - Add Google-style docstrings to all public classes and functions
    - Run ruff lint + format on entire codebase; fix all warnings
    - Remove all TODO/FIXME/HACK comments; replace with implementations
    - Verify no circular imports; verify no placeholder/stub functions remain
    - _Requirements: 100.4, 100.13, 100.14, 100.15_

- [x] 23. Paper Validation
  - [x] 23.1 Create PAPER_VALIDATION.md checklist
    - Document every equation, architectural choice, and stability mechanism from the paper
    - Mark each as ✅ Implemented with reference to implementing file/class
    - Verify 100% coverage of paper specifications
    - Document design rationale for each decision (why RMSNorm, why pre-norm, why exp(-softplus), why outer product, etc.)
    - _Requirements: 96.1–96.10, 100.6_

- [x] 24. Checkpoint - Ensure all tests pass
  - Verify code quality: mypy passes, ruff passes, all docstrings present. Ask the user if questions arise.

- [x] 25. Unit Tests - Core and Modules
  - [x] 25.1 Implement test fixtures and conftest
    - Create tests/conftest.py with shared fixtures: tiny_config (4 layers, d_model=64, d_s=32, k=4), tiny_model, sample_batch, device fixture
    - Set up pytest markers: slow, gpu, integration, property
    - _Requirements: 27.9_

  - [x] 25.2 Implement module unit tests
    - Create tests/test_modules/test_input_projection.py: shape verification, gradient flow, Xavier init check
    - Create tests/test_modules/test_temporal_mixing.py: shape, causality (shifted tensor), gate bounds, cache behavior
    - Create tests/test_modules/test_exponential_gating.py: output ∈ (0,1), numerical stability with extreme inputs, init range
    - Create tests/test_modules/test_selective_writing.py: gate bounds, state read correctness, shape
    - Create tests/test_modules/test_state_update.py: sequential vs parallel equivalence, shape, bounds
    - Create tests/test_modules/test_state_readout.py: confidence gate bounds, shape, vectorization
    - Create tests/test_modules/test_channel_mixing.py: residual structure, activation, shape
    - Parametrize across multiple configs (varying d_model, batch_size, seq_len)
    - _Requirements: 27.1–27.8, 27.11_

  - [x] 25.3 Implement layer and model unit tests
    - Create tests/test_layers/test_block.py: exact ordering, residual connection, mode switching, shape
    - Create tests/test_layers/test_parallel_scan.py: equivalence to sequential, gradient flow, log-space stability
    - Create tests/test_layers/test_chunk_decomposition.py: equivalence to full scan, chunk boundary handling
    - Create tests/test_layers/test_norm.py: RMSNorm scale property, LayerNorm correctness
    - Create tests/test_models/test_usn_model.py: full forward/backward shapes, parameter count, no attention ops
    - Create tests/test_models/test_embedding.py: weight tying, scaling
    - _Requirements: 27.1–27.8, 27.10_

  - [x] 25.4 Implement config, serialization, and utility tests
    - Create tests/test_config/test_usn_config.py: validation, serialization round-trip, presets
    - Create tests/test_config/test_presets.py: parameter counts for all presets
    - Create tests/test_serialization/test_round_trip.py: save/load identity
    - Create tests/test_serialization/test_format_validator.py: corruption detection, version handling
    - Create tests/test_utils/test_utilities.py: count_parameters, seed determinism
    - Create tests/test_cli/test_commands.py: CLI smoke tests
    - _Requirements: 29.1–29.9, 30.1–30.6, 31.3, 31.4_

- [x] 26. Property-Based Tests
  - [x] 26.1 Property test: Gate and Decay Boundedness (Property 1)
    - Create tests/test_properties/test_gate_bounds.py
    - **Property 1: Gate and Decay Boundedness**
    - For any input tensor (including extreme ±1e6), verify λ_t, ρ_t, g_t, α_t, c_t ∈ (0, 1) strictly
    - Use hypothesis with @settings(max_examples=100)
    - **Validates: Requirements 5.1, 5.2, 6.1, 6.7, 8.9, 40.1–40.4, 95.4**

  - [x] 26.2 Property test: Associativity of State Transitions (Property 2)
    - Create tests/test_properties/test_associativity.py
    - **Property 2: Associativity of State Transitions**
    - For any three random affine transitions, verify compose(T_a, compose(T_b, T_c)) ≈ compose(compose(T_a, T_b), T_c) within 1e-5
    - **Validates: Requirements 7.8, 12.2, 53.1, 53.4, 53.8**

  - [x] 26.3 Property test: Parallel Scan Equivalence (Property 3)
    - Create tests/test_properties/test_scan_equivalence.py
    - **Property 3: Parallel Scan Equivalence to Sequential Recurrence**
    - For any valid sequence (1 ≤ n ≤ 1024), verify parallel scan matches sequential within tolerance (1e-5 fp32, 1e-3 fp16)
    - **Validates: Requirements 12.10, 13.4, 28.3, 30.5, 53.5, 79.1**

  - [x] 26.4 Property test: Strict Causality (Property 4)
    - Create tests/test_properties/test_causality_prop.py
    - **Property 4: Strict Causality**
    - For any input sequence and position t, verify modifying input at j > t does not change output at t
    - Jacobian-based verification: ∂output[t]/∂input[j] = 0 for j > t
    - **Validates: Requirements 28.1–28.5, 42.1–42.10**

  - [x] 26.5 Property test: Serialization Round-Trip (Property 5)
    - Create tests/test_properties/test_serialization_rt.py
    - **Property 5: Model Serialization Round-Trip**
    - For any valid USNModel with random weights, save → load produces identical forward output (torch.equal)
    - **Validates: Requirements 22.7, 22.15, 29.1, 29.2**

  - [x] 26.6 Property test: Config Serialization Round-Trip (Property 6) and Weight Invariance (Property 7)
    - Add to tests/test_properties/test_serialization_rt.py
    - **Property 6: Config Serialization Round-Trip** — JSON/YAML serialize → deserialize produces equivalent config
    - **Property 7: Weight Count Invariance** — parameter count from config formula matches actual model.parameters()
    - **Validates: Requirements 29.8, 98.5, 98.8**

  - [x] 26.7 Property test: Kernel Equivalence (Property 8)
    - Create tests/test_properties/test_kernel_equiv.py
    - **Property 8: Acceleration Level Output Equivalence**
    - Fused kernel outputs match unfused reference (Level 4) within tolerance for random inputs
    - **Validates: Requirements 101.8, 102.8, 103.12, 104.6, 105.8**

  - [x] 26.8 Property test: Residual Connection (Property 9)
    - Create tests/test_properties/test_residual.py
    - **Property 9: Residual Connection Preservation**
    - For zero-initialized block weights, block output equals input (identity through residual)
    - **Validates: Requirements 64.1, 64.3**

  - [x] 26.9 Property test: Log-Space Numerical Stability (Property 10)
    - Create tests/test_properties/test_numerical.py
    - **Property 10: Log-Space Numerical Stability**
    - For decay sequences up to 10,000 in length with λ_t close to 0 (1e-6), cumulative log-space computation produces finite non-NaN results
    - **Validates: Requirements 5.10, 12.12, 40.5, 95.5**

  - [x] 26.10 Property test: RMSNorm Output Scale (Property 11)
    - Create tests/test_properties/test_norm_property.py
    - **Property 11: RMSNorm Output Scale**
    - For any non-zero input, RMS of output (before gain) ≈ 1.0 within 1e-4
    - **Validates: Requirements 15.1, 15.5**

  - [x] 26.11 Property test: State Norm Constraint (Property 12)
    - Create tests/test_properties/test_state_constraint.py
    - **Property 12: State Norm Constraint Enforcement**
    - After clipping, ‖s_t‖ ≤ max_state_norm guaranteed
    - **Validates: Requirements 106.3, 107.2, 107.3**

  - [x] 26.12 Property test: Deterministic Initialization (Property 13)
    - Create tests/test_properties/test_determinism.py
    - **Property 13: Deterministic Initialization**
    - Two models with same seed and config have identical parameters (torch.equal for every tensor)
    - **Validates: Requirements 30.2, 30.1**

  - [x] 26.13 Property test: Cross-Entropy Loss Non-Negativity (Property 14)
    - Create tests/test_properties/test_loss_property.py
    - **Property 14: Cross-Entropy Loss Non-Negativity**
    - For any valid logits and targets, loss ≥ 0; perfect predictions → loss approaches 0
    - **Validates: Requirements 20.1, 20.7**

- [x] 27. Scalability Table Verification
  - [x] 27.1 Implement scalability table test and preset verification
    - Create tests/test_config/test_scalability.py: instantiate all 12+ preset configs (Tiny through XXL)
    - Verify parameter count within 1% of expected for each
    - Verify state_size_per_layer = d_s + k² for each
    - Verify linear scaling (no quadratic blowup)
    - Document table in tests or generate formatted output
    - _Requirements: 31.1–31.7, 86.7_

- [x] 28. Micro-Model Training Validation
  - [x] 28.1 Implement micro-model end-to-end training test
    - Create tests/test_integration/test_micro_model.py: define Micro config (~2M params)
    - Train on MathDataset for enough steps to demonstrate learning
    - Verify: loss decreases monotonically, gradients non-zero and bounded, model generates correct arithmetic answers
    - Complete in <5 min GPU / <15 min CPU
    - Log: initial loss, final loss, reduction ratio, sample generations, throughput
    - Test checkpoint save → load → resume with continued improvement
    - _Requirements: 32.1–32.10_

- [x] 29. Integration Tests
  - [x] 29.1 Implement end-to-end integration tests
    - Create tests/test_integration/test_end_to_end.py: create → train → save → load → generate full workflow
    - Create tests/test_integration/test_export.py: export to ONNX/SafeTensors → reload → verify output matches
    - Create tests/test_inference/test_generator.py: all decode strategies, streaming, batch generation
    - Create tests/test_inference/test_causality.py: Jacobian-based causality verification
    - Create tests/test_inference/test_state_management.py: O(1) memory, get/set state, branching
    - Create tests/test_training/test_trainer.py: training loop convergence, checkpointing
    - Create tests/test_training/test_curriculum.py: schedule monotonicity
    - All marked with appropriate pytest markers
    - _Requirements: 94.1–94.8, 28.1–28.7, 88.1–88.9_

- [x] 30. Checkpoint - Ensure all tests pass
  - Run full test suite: pytest tests/ -m "not slow and not gpu" --cov=usn. Verify ≥95% coverage. Ask the user if questions arise.

- [x] 31. Benchmarking System
  - [x] 31.1 Implement benchmark suite
    - Create benchmarks/benchmark_forward.py: forward pass latency at seq_len 128, 256, 512, 1024, 2048, 4096, 8192 (prove O(n) linearity)
    - Create benchmarks/benchmark_inference.py: generation throughput (tokens/sec), time-to-first-token, memory at generation lengths 100–10000 (prove O(1) memory)
    - Create benchmarks/benchmark_scan.py: parallel scan speedup over sequential
    - Create benchmarks/benchmark_kernels.py: acceleration level comparison (Triton vs compile vs eager)
    - Report mean, std, min, max over 10+ measurement runs; document hardware specs
    - Produce JSON output for automated analysis
    - _Requirements: 33.1–33.8, 86.1–86.6, 97.1–97.8_

- [x] 32. Documentation
  - [x] 32.1 Create comprehensive documentation
    - Create docs/ structure: getting_started.md, api_reference.md, architecture.md, training_guide.md, inference_guide.md, configuration.md, benchmarks.md, design_rationale.md, contributing.md
    - Include: installation instructions, quick-start examples, full API reference, architecture diagrams, scalability table, design rationale (Req 96)
    - Create examples/: quick_start.py, training_example.py, generation_example.py, custom_dataset.py, distributed_training.py
    - Update README.md with installation, quick-start, architecture overview, API summary, links
    - _Requirements: 1.4, 96.1–96.10, 100.7_

  - [x] 32.2 Create CI/CD configuration and build scripts
    - Create .github/workflows/ci.yml: lint (ruff), type check (mypy), fast tests, slow tests (optional), build, publish
    - Create Makefile with targets: test, lint, format, typecheck, build, publish, docs, clean
    - Create pre-commit config: ruff, mypy, pytest fast
    - Verify: python -m build produces sdist and wheel; pip install from wheel works
    - _Requirements: 90.1–90.7, 100.11_

- [x] 33. Final Checkpoint - Ensure all tests pass
  - Run full test suite including property tests and integration tests. Verify 95%+ coverage, mypy strict passes, ruff clean. Ensure all examples run. Ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The implementation uses Python throughout as specified in the design document
- All Triton kernels have PyTorch fallback implementations (Level 4 always works)
- The dependency graph ensures no orphaned code: each phase builds on the previous

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "2.2", "2.3"] },
    { "id": 2, "tasks": ["3.1", "3.2"] },
    { "id": 3, "tasks": ["4.1", "5.1", "5.2", "5.3"] },
    { "id": 4, "tasks": ["5.4", "5.5", "5.6", "5.7", "5.8"] },
    { "id": 5, "tasks": ["7.1", "7.2"] },
    { "id": 6, "tasks": ["7.3", "7.4"] },
    { "id": 7, "tasks": ["8.1"] },
    { "id": 8, "tasks": ["8.2", "8.3"] },
    { "id": 9, "tasks": ["10.1", "11.1", "11.2", "11.3", "12.1"] },
    { "id": 10, "tasks": ["10.2", "10.3", "10.4", "10.5", "13.1"] },
    { "id": 11, "tasks": ["10.6", "13.2", "14.1"] },
    { "id": 12, "tasks": ["14.2", "14.3"] },
    { "id": 13, "tasks": ["16.1", "17.1"] },
    { "id": 14, "tasks": ["16.2", "16.3", "17.2", "17.3"] },
    { "id": 15, "tasks": ["16.4", "18.1", "18.2"] },
    { "id": 16, "tasks": ["19.1", "20.1"] },
    { "id": 17, "tasks": ["22.1", "22.2"] },
    { "id": 18, "tasks": ["23.1"] },
    { "id": 19, "tasks": ["25.1"] },
    { "id": 20, "tasks": ["25.2", "25.3"] },
    { "id": 21, "tasks": ["25.4", "26.1", "26.2", "26.3"] },
    { "id": 22, "tasks": ["26.4", "26.5", "26.6", "26.7"] },
    { "id": 23, "tasks": ["26.8", "26.9", "26.10", "26.11", "26.12", "26.13"] },
    { "id": 24, "tasks": ["27.1", "28.1"] },
    { "id": 25, "tasks": ["29.1"] },
    { "id": 26, "tasks": ["31.1"] },
    { "id": 27, "tasks": ["32.1", "32.2"] }
  ]
}
```
