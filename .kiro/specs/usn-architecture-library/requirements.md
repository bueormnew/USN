# Requirements Document

## Introduction

This document specifies the complete, exhaustive requirements for a professional, PyPI-distributable Python library that implements 100% of the Unified State Network (USN) architecture as described in the original technical paper. The USN is a novel autoregressive sequence modeling architecture featuring a single unified persistent state partitioned into semantic and relational subspaces, affine associative state transitions enabling parallel scan during training, linear O(n) complexity with respect to sequence length, constant inference memory, and no attention mechanism or quadratic operations. This library shall provide a production-grade implementation covering architecture core, training system, inference system, serialization, benchmarks, testing, documentation, CLI tools, and public API design comparable in quality and usability to TensorFlow, PyTorch, and HuggingFace Transformers.

## Glossary

- **USN**: Unified State Network — the architecture described in the technical paper
- **Library**: The complete Python package distributed via PyPI implementing USN
- **Semantic_State**: The vector subspace s_t ∈ R^{d_s} of the unified persistent state
- **Relational_State**: The matrix subspace R_t ∈ R^{k×k} of the unified persistent state
- **Unified_State**: The combined persistent state S_t = (s_t, R_t) maintained across timesteps
- **State_Transition**: The affine, associative update rule applied to the Unified_State at each timestep
- **Associative_Scan**: A parallel prefix-sum algorithm exploiting the associative property of State_Transition for efficient training
- **Chunk_Decomposition**: GPU-efficient partitioning of sequences into chunks for parallelized state computation
- **Input_Projection**: The linear transformation u_t = W_u x_t + b_u mapping token embeddings to internal representation
- **Temporal_Mixing**: The local temporal blending operation α_t = σ(W_α x_t + b_α), m_t = α_t ⊙ u_t + (1-α_t) ⊙ u_{t-1}
- **Exponential_Gating**: The decay mechanism λ_t = exp(-softplus(W_λ x_t + b_λ)) constraining state memory
- **Selective_Writing**: The gated write mechanism g_t = σ(W_g m_t + U_g read(S_{t-1}) + b_g), Δ_t = W_Δ m_t + b_Δ
- **State_Update**: The unified update rule s_t = λ_t ⊙ s_{t-1} + g_t ⊙ (B_s m_t), R_t = ρ_t R_{t-1} + (B_r m_t)(C_r m_t)^T
- **State_Readout**: The output extraction z_t = W_s s_t + W_r vec(R_t), c_t = σ(W_c m_t + b_c), o_t = c_t ⊙ z_t
- **Channel_Mixing**: The MLP block y_t = m_t + W_2 φ(W_1(c_t ⊙ z_t)) providing inter-channel interaction
- **Confidence_Gate**: The gate c_t = σ(W_c m_t + b_c) controlling readout contribution
- **Decay_Factor**: λ_t ∈ (0,1) ensuring bounded state dynamics and preventing state explosion
- **Write_Gate**: g_t ∈ (0,1) controlling selective writing of new information into state
- **Relational_Decay**: ρ_t ∈ (0,1) the decay factor for the relational state matrix
- **Block**: A single USN processing block containing all submodules in defined order
- **Model**: A complete USN model composed of N stacked Blocks with embedding and output layers
- **Token_Embedding**: The learned mapping from discrete tokens to continuous vectors
- **Output_Head**: The final linear projection from model hidden state to vocabulary logits
- **Parallel_Scan**: Training-time parallelization strategy using associative scan over state transitions
- **Autoregressive_Generation**: Token-by-token generation where each token depends only on previous tokens
- **Causality**: The strict constraint that no operation may access future information
- **USN_Format**: The native .usn serialization format storing all model data in a single file
- **Checkpoint**: A training state snapshot including model weights, optimizer state, and training metadata
- **Mixed_Precision**: Training with reduced precision (BF16/FP16) for efficiency while maintaining accuracy
- **Gradient_Accumulation**: Accumulating gradients over multiple mini-batches before updating parameters
- **Distributed_Training**: Training across multiple devices/nodes using data or model parallelism
- **CLI**: Command-line interface for model operations (train, generate, export, benchmark)
- **RMSNorm**: Root Mean Square Layer Normalization applied before input projection
- **LayerNorm**: Layer Normalization as an alternative to RMSNorm
- **AdamW**: Adam optimizer with decoupled weight decay, the default optimizer
- **Cross_Entropy_Loss**: The training objective for next-token prediction
- **Teacher_Forcing**: Training strategy where ground-truth tokens are provided as input at each step
- **Beam_Search**: Decoding strategy maintaining multiple candidate sequences
- **Top_K_Sampling**: Sampling from the top K most probable tokens
- **Top_P_Sampling**: Nucleus sampling from the smallest set of tokens whose cumulative probability exceeds P
- **Temperature_Scaling**: Dividing logits by a temperature parameter before softmax
- **Streaming_Inference**: Generating and yielding tokens one at a time without waiting for full sequence
- **Batch_Inference**: Processing multiple sequences simultaneously for throughput
- **Scalability_Table**: A reference table with 20+ model configurations from Tiny to XXL
- **Micro_Model**: A ~2M parameter model used for validation on synthetic tasks

## Requirements

### Requirement 1: Project Structure and Distribution

**User Story:** As a developer, I want the USN library to follow professional Python packaging standards with complete PyPI distribution support, so that I can install it via pip and integrate it into my projects seamlessly.

#### Acceptance Criteria

1. THE Library SHALL be distributed as a Python package named "usn" installable via `pip install usn`
2. THE Library SHALL include a valid pyproject.toml specifying package name "USN", license "MIT", author "BUEORM", Python version requirements, all dependencies, build system configuration, entry points for CLI, and project metadata
3. THE Library SHALL include a setup.py file for backward compatibility with older pip versions
4. THE Library SHALL include a README.md with installation instructions, quick-start examples, architecture overview, API summary, and links to full documentation
5. THE Library SHALL include a LICENSE file containing the full MIT license text attributed to BUEORM
6. THE Library SHALL organize source code into the following package directories: usn/, usn/core/, usn/modules/, usn/layers/, usn/models/, usn/training/, usn/datasets/, usn/tokenizers/, usn/serialization/, usn/utils/, usn/optim/, usn/losses/, usn/config/, usn/backends/, usn/cli/
7. THE Library SHALL include directories for: tests/, benchmarks/, scripts/, notebooks/, docs/, examples/
8. WHEN imported, THE Library SHALL expose a top-level `usn` namespace with version information accessible via `usn.__version__`
9. THE Library SHALL include a MANIFEST.in file specifying all non-Python files to include in source distributions
10. THE Library SHALL include a .gitignore file configured for Python projects
11. THE Library SHALL include a CHANGELOG.md documenting version history
12. THE Library SHALL declare all runtime dependencies with pinned minimum versions in pyproject.toml
13. THE Library SHALL declare optional dependency groups: [dev] for testing/linting, [docs] for documentation building, [cuda] for GPU acceleration, [all] for everything

### Requirement 2: Public API Design and Usability

**User Story:** As a machine learning practitioner, I want a simple, intuitive API comparable to TensorFlow/PyTorch/Transformers, so that I can create, train, evaluate, and deploy USN models with minimal boilerplate.

#### Acceptance Criteria

1. THE Library SHALL provide a `USNModel` class that can be instantiated with a configuration object or keyword arguments specifying all hyperparameters
2. THE Library SHALL provide a `USNConfig` class encapsulating all model hyperparameters with validated defaults and serialization support
3. WHEN `USNModel` is instantiated, THE Library SHALL display the total parameter count and memory estimate
4. THE Library SHALL provide a `USNTrainer` class that accepts a model, dataset, and training configuration to execute the full training loop
5. THE Library SHALL provide a `USNGenerator` class for autoregressive text generation with configurable decoding strategies
6. THE Library SHALL provide `usn.create_model(config)` as a factory function for model instantiation
7. THE Library SHALL provide `usn.train(model, dataset, config)` as a high-level training function
8. THE Library SHALL provide `usn.generate(model, prompt, max_tokens)` as a high-level generation function
9. THE Library SHALL provide `usn.save(model, path)` and `usn.load(path)` for model persistence in native .usn format
10. THE Library SHALL provide `usn.export(model, format, path)` supporting export to ONNX, SafeTensors, and raw PyTorch state_dict formats
11. THE Library SHALL provide `usn.summary(model)` displaying architecture visualization with layer names, shapes, parameter counts per layer, total parameters, and memory usage
12. THE Library SHALL provide `usn.benchmark(model, config)` for running performance benchmarks
13. WHEN any public API function receives invalid arguments, THE Library SHALL raise a descriptive exception specifying the invalid argument, expected type/range, and received value
14. THE Library SHALL maintain backward compatibility within major versions following semantic versioning
15. THE Library SHALL provide type stubs (.pyi files) for all public API surfaces enabling IDE autocompletion
16. THE Library SHALL provide `usn.from_pretrained(path_or_id)` for loading pretrained models from local paths or model registries

### Requirement 3: Input Projection Module

**User Story:** As a developer implementing USN, I want the input projection module to transform token embeddings into internal representations exactly as defined in the paper, so that downstream modules receive correctly shaped inputs.

#### Acceptance Criteria

1. THE Input_Projection module SHALL compute u_t = W_u x_t + b_u where W_u ∈ R^{d_model × d_model} and b_u ∈ R^{d_model}
2. THE Input_Projection module SHALL accept input tensors x_t of shape (batch_size, seq_len, d_model)
3. THE Input_Projection module SHALL produce output tensors u_t of shape (batch_size, seq_len, d_model)
4. THE Input_Projection module SHALL initialize W_u using Xavier uniform initialization
5. THE Input_Projection module SHALL initialize b_u to zeros
6. THE Input_Projection module SHALL support both single-step inference (seq_len=1) and full-sequence training modes
7. THE Input_Projection module SHALL maintain causality by operating independently on each timestep position
8. THE Input_Projection module SHALL be documented with: objective (linear transformation of input embeddings), inputs (x_t tensor shape and dtype), outputs (u_t tensor shape and dtype), complexity (O(d_model²) per timestep), justification (dimensionality adaptation), equations (u_t = W_u x_t + b_u), constraints (no temporal dependency)

### Requirement 4: Local Temporal Mixing Module

**User Story:** As a developer implementing USN, I want the temporal mixing module to blend current and previous timestep representations using a learned gate, so that the model captures local temporal context.

#### Acceptance Criteria

1. THE Temporal_Mixing module SHALL compute α_t = σ(W_α x_t + b_α) where σ is the sigmoid function, W_α ∈ R^{d_model × d_model}, b_α ∈ R^{d_model}
2. THE Temporal_Mixing module SHALL compute m_t = α_t ⊙ u_t + (1-α_t) ⊙ u_{t-1} where ⊙ denotes element-wise multiplication
3. THE Temporal_Mixing module SHALL accept inputs x_t and u_t of shape (batch_size, seq_len, d_model)
4. THE Temporal_Mixing module SHALL produce output m_t of shape (batch_size, seq_len, d_model)
5. WHEN processing the first timestep (t=0), THE Temporal_Mixing module SHALL use a learned initial state u_{-1} or zeros for u_{t-1}
6. THE Temporal_Mixing module SHALL maintain strict causality: m_t depends only on x_t, u_t, and u_{t-1} (never future values)
7. WHILE in training mode with full sequences, THE Temporal_Mixing module SHALL compute all positions in parallel using shifted tensors
8. WHILE in inference mode, THE Temporal_Mixing module SHALL cache u_{t-1} from the previous step for single-step computation
9. THE Temporal_Mixing module SHALL initialize W_α using Xavier uniform initialization
10. THE Temporal_Mixing module SHALL initialize b_α to zeros
11. THE Temporal_Mixing module SHALL be documented with: objective (local temporal context blending), inputs (x_t, u_t tensors), outputs (m_t tensor), complexity (O(d_model) per timestep), justification (captures immediate temporal dependencies without growing memory), equations (α_t = σ(W_α x_t + b_α), m_t = α_t ⊙ u_t + (1-α_t) ⊙ u_{t-1}), constraints (causal, one-step lookback only)

### Requirement 5: Exponential Gating/Decay Module

**User Story:** As a developer implementing USN, I want the exponential gating module to compute bounded decay factors that control state memory persistence, so that the model can learn appropriate forgetting rates.

#### Acceptance Criteria

1. THE Exponential_Gating module SHALL compute λ_t = exp(-softplus(W_λ x_t + b_λ)) where softplus(x) = ln(1 + exp(x))
2. THE Exponential_Gating module SHALL guarantee λ_t ∈ (0, 1) for all inputs due to the exp(-softplus(·)) construction
3. THE Exponential_Gating module SHALL accept input tensor x_t of shape (batch_size, seq_len, d_model)
4. THE Exponential_Gating module SHALL produce output tensor λ_t of shape (batch_size, seq_len, d_s) for the semantic state decay
5. THE Exponential_Gating module SHALL produce output tensor ρ_t of shape (batch_size, seq_len, 1) or scalar broadcast for the relational state decay
6. THE Exponential_Gating module SHALL use separate parameters W_λ ∈ R^{d_s × d_model} and b_λ ∈ R^{d_s} for semantic decay
7. THE Exponential_Gating module SHALL use separate parameters W_ρ ∈ R^{1 × d_model} and b_ρ ∈ R^{1} for relational decay (or per-element if specified)
8. THE Exponential_Gating module SHALL prevent state explosion by the mathematical guarantee that repeated multiplication by λ_t < 1 causes exponential decay of old state
9. THE Exponential_Gating module SHALL initialize b_λ such that initial λ_t values are in a moderate range (e.g., 0.9-0.99) for stable early training
10. THE Exponential_Gating module SHALL be numerically stable by using log-space computation when accumulating products of λ_t across timesteps
11. THE Exponential_Gating module SHALL be documented with: objective (bounded decay for state memory control), inputs (x_t), outputs (λ_t, ρ_t), complexity (O(d_s) per timestep), justification (prevents unbounded state growth, enables learned forgetting), equations (λ_t = exp(-softplus(W_λ x_t + b_λ))), constraints (output strictly in (0,1), numerically stable)

### Requirement 6: Selective Writing Module

**User Story:** As a developer implementing USN, I want the selective writing module to compute content-dependent write gates that control what information enters the unified state, so that the model can filter noise and write only relevant information.

#### Acceptance Criteria

1. THE Selective_Writing module SHALL compute g_t = σ(W_g m_t + U_g read(S_{t-1}) + b_g) where σ is the sigmoid function
2. THE Selective_Writing module SHALL compute Δ_t = W_Δ m_t + b_Δ as the write content vector
3. THE Selective_Writing module SHALL read from the previous state S_{t-1} using a read operation defined as read(S_{t-1}) = W_read_s s_{t-1} + W_read_r vec(R_{t-1}) or a simplified projection
4. THE Selective_Writing module SHALL accept inputs m_t of shape (batch_size, seq_len, d_model) and previous state S_{t-1}
5. THE Selective_Writing module SHALL produce write gate g_t of shape (batch_size, seq_len, d_s) with values in (0,1)
6. THE Selective_Writing module SHALL produce write content Δ_t of appropriate shape for state update
7. THE Selective_Writing module SHALL guarantee g_t ∈ (0, 1) via the sigmoid activation
8. WHEN state content is noise or irrelevant, THE Selective_Writing module SHALL produce g_t values near 0, effectively blocking state updates
9. WHEN state content is informative, THE Selective_Writing module SHALL produce g_t values near 1, allowing full state updates
10. THE Selective_Writing module SHALL use parameters W_g ∈ R^{d_s × d_model}, U_g ∈ R^{d_s × d_read}, b_g ∈ R^{d_s}
11. THE Selective_Writing module SHALL use parameters W_Δ ∈ R^{d_Δ × d_model}, b_Δ ∈ R^{d_Δ}
12. THE Selective_Writing module SHALL be documented with: objective (content-dependent filtering of state writes), inputs (m_t, S_{t-1}), outputs (g_t, Δ_t), complexity (O(d_s × d_model + d_s × d_read)), justification (noise filtering, selective memory), equations (g_t = σ(W_g m_t + U_g read(S_{t-1}) + b_g)), constraints (g_t bounded in (0,1))

### Requirement 7: Unified State Update Module

**User Story:** As a developer implementing USN, I want the state update module to apply the affine associative transition to both semantic and relational state subspaces exactly as defined in the paper, so that the model maintains a unified persistent memory.

#### Acceptance Criteria

1. THE State_Update module SHALL compute s_t = λ_t ⊙ s_{t-1} + g_t ⊙ (B_s m_t) for the semantic state
2. THE State_Update module SHALL compute R_t = ρ_t R_{t-1} + (B_r m_t)(C_r m_t)^T for the relational state
3. THE State_Update module SHALL maintain s_t ∈ R^{d_s} as the semantic state vector
4. THE State_Update module SHALL maintain R_t ∈ R^{k×k} as the relational state matrix
5. THE State_Update module SHALL use projection matrices B_s ∈ R^{d_s × d_model} for semantic write projection
6. THE State_Update module SHALL use projection matrices B_r ∈ R^{k × d_model} and C_r ∈ R^{k × d_model} for relational write projections
7. THE State_Update module SHALL ensure the state transition is affine: S_t = A_t S_{t-1} + b_t where A_t and b_t depend on input
8. THE State_Update module SHALL ensure the state transition is associative: composing transitions T_1 ∘ T_2 yields another valid transition of the same form
9. WHILE in training mode, THE State_Update module SHALL support parallel computation via associative scan across the sequence dimension
10. WHILE in inference mode, THE State_Update module SHALL compute state updates sequentially, one timestep at a time
11. THE State_Update module SHALL initialize s_0 = 0 (zero vector) and R_0 = 0 (zero matrix) at the start of each sequence unless a prior state is provided
12. THE State_Update module SHALL support passing initial state for continued generation or context extension
13. THE State_Update module SHALL never allow state values to grow unbounded (guaranteed by λ_t, ρ_t ∈ (0,1))
14. THE State_Update module SHALL compute the relational update (B_r m_t)(C_r m_t)^T as an outer product of two k-dimensional vectors, costing O(k²) per timestep
15. THE State_Update module SHALL be documented with: objective (unified persistent memory update), inputs (s_{t-1}, R_{t-1}, λ_t, ρ_t, g_t, m_t), outputs (s_t, R_t), complexity (O(d_s + k²) per timestep), justification (captures both feature-level and relational information), equations (s_t = λ_t ⊙ s_{t-1} + g_t ⊙ B_s m_t, R_t = ρ_t R_{t-1} + (B_r m_t)(C_r m_t)^T), constraints (bounded state, associative, affine)

### Requirement 8: State Readout Module

**User Story:** As a developer implementing USN, I want the state readout module to extract information from both state subspaces and combine them with a confidence gate, so that the model can use stored information for predictions.

#### Acceptance Criteria

1. THE State_Readout module SHALL compute z_t = W_s s_t + W_r vec(R_t) combining both state subspaces
2. THE State_Readout module SHALL compute c_t = σ(W_c m_t + b_c) as the confidence gate
3. THE State_Readout module SHALL compute o_t = c_t ⊙ z_t as the gated output
4. THE State_Readout module SHALL use W_s ∈ R^{d_model × d_s} to project semantic state to model dimension
5. THE State_Readout module SHALL use W_r ∈ R^{d_model × k²} to project vectorized relational state to model dimension
6. THE State_Readout module SHALL use W_c ∈ R^{d_model × d_model} and b_c ∈ R^{d_model} for confidence gate computation
7. THE State_Readout module SHALL vectorize R_t into a k²-dimensional vector before applying W_r
8. THE State_Readout module SHALL produce output o_t of shape (batch_size, seq_len, d_model)
9. THE Confidence_Gate c_t SHALL have values in (0, 1) via sigmoid activation
10. THE State_Readout module SHALL be documented with: objective (extract and gate state information), inputs (s_t, R_t, m_t), outputs (o_t, c_t, z_t), complexity (O(d_model × d_s + d_model × k²) per timestep), justification (controlled state readout prevents noise propagation), equations (z_t = W_s s_t + W_r vec(R_t), c_t = σ(W_c m_t + b_c), o_t = c_t ⊙ z_t), constraints (c_t bounded in (0,1))

### Requirement 9: Channel Mixing (MLP) Module

**User Story:** As a developer implementing USN, I want the channel mixing module to provide inter-channel interaction through a feedforward network with residual connection, so that the model can learn complex feature transformations.

#### Acceptance Criteria

1. THE Channel_Mixing module SHALL compute y_t = m_t + W_2 φ(W_1(c_t ⊙ z_t)) where φ is a nonlinear activation
2. THE Channel_Mixing module SHALL use W_1 ∈ R^{d_ff × d_model} as the up-projection matrix where d_ff is the feedforward intermediate dimension
3. THE Channel_Mixing module SHALL use W_2 ∈ R^{d_model × d_ff} as the down-projection matrix
4. THE Channel_Mixing module SHALL use GELU, SiLU/Swish, or configurable activation function φ
5. THE Channel_Mixing module SHALL include a residual connection from m_t to the output y_t
6. THE Channel_Mixing module SHALL accept configurable expansion ratio (default d_ff = 4 × d_model or as specified in paper)
7. THE Channel_Mixing module SHALL produce output y_t of shape (batch_size, seq_len, d_model)
8. THE Channel_Mixing module SHALL apply the gated state readout (c_t ⊙ z_t) as input to the MLP, NOT raw state
9. THE Channel_Mixing module SHALL maintain a short gradient path via the residual connection for training stability
10. THE Channel_Mixing module SHALL be documented with: objective (inter-channel feature mixing with residual), inputs (m_t, c_t, z_t), outputs (y_t), complexity (O(d_model × d_ff) per timestep), justification (nonlinear feature transformation, gradient highway via residual), equations (y_t = m_t + W_2 φ(W_1(c_t ⊙ z_t))), constraints (residual connection mandatory, activation configurable)

### Requirement 10: USN Block Assembly

**User Story:** As a developer implementing USN, I want a complete USN block that composes all submodules in the exact order specified by the paper, so that each block performs the full state-update-and-readout cycle.

#### Acceptance Criteria

1. THE Block SHALL apply submodules in this exact order: (1) Normalization, (2) Input_Projection, (3) Temporal_Mixing, (4) Exponential_Gating, (5) Selective_Writing, (6) State_Update, (7) State_Readout, (8) Channel_Mixing
2. THE Block SHALL apply RMSNorm or LayerNorm (configurable) before the Input_Projection as the first operation
3. THE Block SHALL accept input tensor x_t of shape (batch_size, seq_len, d_model) and previous state S_{t-1}
4. THE Block SHALL produce output tensor y_t of shape (batch_size, seq_len, d_model) and updated state S_t
5. THE Block SHALL include a block-level residual connection: output = x_t + y_t (pre-norm architecture)
6. THE Block SHALL pass state between internal submodules without copying or detaching gradients during training
7. THE Block SHALL expose individual submodules as named attributes for inspection and parameter access
8. THE Block SHALL support both training mode (full sequence parallel) and inference mode (single step sequential)
9. WHEN switching between training and inference modes, THE Block SHALL correctly manage internal caches and state buffers
10. THE Block SHALL not introduce any operations not specified in the paper
11. THE Block SHALL not omit any operations specified in the paper
12. THE Block SHALL be documented with: objective (complete state processing cycle), inputs (x_t, S_{t-1}), outputs (y_t, S_t), complexity (sum of all submodule complexities), justification (modular composition per paper specification), constraints (exact ordering, no additions, no removals)

### Requirement 11: Complete USN Model Assembly

**User Story:** As a developer, I want a complete USN model class that stacks N blocks with embedding and output layers, so that I can use the full architecture for sequence modeling tasks.

#### Acceptance Criteria

1. THE Model SHALL compose: Token_Embedding → N × Block → Final_Norm → Output_Head
2. THE Model SHALL use a learned token embedding matrix E ∈ R^{vocab_size × d_model}
3. THE Model SHALL apply a final normalization (RMSNorm or LayerNorm) after the last block
4. THE Model SHALL use an Output_Head linear projection W_out ∈ R^{vocab_size × d_model} mapping to vocabulary logits
5. THE Model SHALL optionally tie embedding weights (E = W_out^T) as a configurable option
6. THE Model SHALL accept a USNConfig object specifying: num_layers, d_model, d_s, k, d_ff, vocab_size, max_seq_len, dropout, norm_type, activation, tie_weights, and all other hyperparameters
7. THE Model SHALL maintain separate state for each layer, passing state[i] to block[i]
8. THE Model SHALL support variable sequence lengths up to max_seq_len
9. THE Model SHALL produce logits of shape (batch_size, seq_len, vocab_size) during training
10. THE Model SHALL produce next-token logits of shape (batch_size, 1, vocab_size) during single-step inference
11. THE Model SHALL report total parameter count, trainable parameter count, and memory estimate via a summary method
12. THE Model SHALL support gradient checkpointing for memory-efficient training of large models
13. THE Model SHALL use no attention mechanism and no quadratic-complexity operations anywhere in the forward pass
14. THE Model SHALL achieve O(n) time complexity with respect to sequence length n during training (with parallel scan)
15. THE Model SHALL achieve O(1) memory with respect to sequence length during inference (constant state size)

### Requirement 12: Parallel Scan Implementation

**User Story:** As a developer, I want the parallel scan (associative scan) algorithm implemented for training-time state computation, so that the model can be trained efficiently on long sequences using GPU parallelism.

#### Acceptance Criteria

1. THE Parallel_Scan module SHALL implement the associative scan (prefix-sum) algorithm over state transitions
2. THE Parallel_Scan module SHALL exploit the associative property: (T_a ∘ T_b) ∘ T_c = T_a ∘ (T_b ∘ T_c)
3. THE Parallel_Scan module SHALL compute all states S_1, S_2, ..., S_n in O(n) work and O(log n) parallel depth
4. THE Parallel_Scan module SHALL represent each transition as an affine map: T_t(S) = A_t S + b_t
5. THE Parallel_Scan module SHALL compose transitions via: (A_2, b_2) ∘ (A_1, b_1) = (A_2 A_1, A_2 b_1 + b_2)
6. THE Parallel_Scan module SHALL handle both semantic state (element-wise decay + additive) and relational state (scalar decay + outer product additive)
7. THE Parallel_Scan module SHALL support chunk-based decomposition: divide sequence into chunks, scan within chunks, propagate between chunks
8. THE Parallel_Scan module SHALL be implemented as a custom autograd function with explicit forward and backward passes for memory efficiency
9. THE Parallel_Scan module SHALL support configurable chunk sizes for tuning GPU occupancy
10. THE Parallel_Scan module SHALL produce identical results to the sequential recurrence (up to floating-point precision)
11. THE Parallel_Scan module SHALL be differentiable and support full backpropagation through the scan
12. THE Parallel_Scan module SHALL operate in log-space for decay accumulation to prevent numerical underflow
13. THE Parallel_Scan module SHALL be documented with: objective (parallel state computation for training), inputs (sequence of transition parameters A_t, b_t), outputs (sequence of states S_t), complexity (O(n) work, O(log n) depth), justification (enables training parallelism on GPUs), constraints (must match sequential output, differentiable, numerically stable)

### Requirement 13: Chunk-Based Decomposition

**User Story:** As a developer, I want chunk-based sequence decomposition for GPU-efficient training, so that the model can process long sequences without exceeding GPU memory while maximizing parallel utilization.

#### Acceptance Criteria

1. THE Chunk_Decomposition module SHALL divide input sequences of length n into chunks of configurable size C
2. THE Chunk_Decomposition module SHALL process within-chunk states using the parallel scan algorithm
3. THE Chunk_Decomposition module SHALL propagate inter-chunk state using sequential chunk boundary updates
4. THE Chunk_Decomposition module SHALL produce identical results to processing the full sequence without chunking (up to floating-point precision)
5. THE Chunk_Decomposition module SHALL support sequence lengths that are not evenly divisible by chunk size C via padding or tail handling
6. THE Chunk_Decomposition module SHALL allow chunk size C to be tuned for optimal GPU occupancy (default: 64 or 128)
7. THE Chunk_Decomposition module SHALL reduce peak memory from O(n × d_state) to O(C × d_state + n/C × d_state)
8. THE Chunk_Decomposition module SHALL be compatible with gradient checkpointing at chunk boundaries
9. THE Chunk_Decomposition module SHALL be documented with: objective (memory-efficient parallel training), inputs (full sequence transitions, chunk size C), outputs (all states S_1...S_n), complexity (O(n) work, O(C + n/C) memory), justification (GPU memory efficiency for long sequences), constraints (results match non-chunked computation)

### Requirement 14: Fuseable Kernel Design

**User Story:** As a developer targeting modern accelerators, I want the architecture designed for kernel fusion, so that operations can be combined into efficient GPU kernels minimizing memory bandwidth.

#### Acceptance Criteria

1. THE Library SHALL identify and document the following fuseable kernel groups: (a) projection+gates (Input_Projection + gate computations), (b) temporal_mix+activation (Temporal_Mixing + sigmoid), (c) selective_write+decay (Selective_Writing + Exponential_Gating), (d) state_read+confidence_gate (State_Readout + Confidence_Gate), (e) channel_MLP (W_1 + activation + W_2)
2. THE Library SHALL organize computation within each Block to maximize data locality for fused kernel execution
3. THE Library SHALL minimize intermediate tensor materializations between fuseable operations
4. THE Library SHALL provide a `fused=True/False` configuration option to enable/disable fused implementations
5. WHEN fused mode is enabled and a compatible backend is available, THE Library SHALL use fused kernel implementations
6. WHEN fused mode is disabled or no compatible backend is available, THE Library SHALL fall back to individual operation execution with identical results
7. THE Library SHALL support Triton-based custom kernels as the primary fusion backend
8. THE Library SHALL support PyTorch's torch.compile as an alternative fusion strategy
9. THE Library SHALL document memory bandwidth savings from fusion for each kernel group
10. THE Library SHALL be documented with: objective (minimize memory bandwidth, maximize arithmetic intensity), constraints (fused and unfused must produce identical results)

### Requirement 15: Normalization Layers

**User Story:** As a developer, I want configurable normalization layers (RMSNorm and LayerNorm) applied before input projection in each block, so that the model maintains stable activation magnitudes throughout depth.

#### Acceptance Criteria

1. THE Library SHALL implement RMSNorm computing: y = x / RMS(x) × γ where RMS(x) = √(mean(x²) + ε)
2. THE Library SHALL implement LayerNorm computing: y = (x - mean(x)) / √(var(x) + ε) × γ + β
3. THE Library SHALL apply normalization before Input_Projection in each Block (pre-norm architecture)
4. THE Library SHALL apply a final normalization after the last Block and before the Output_Head
5. THE Library SHALL use ε = 1e-6 as default epsilon for numerical stability (configurable)
6. THE Library SHALL support configurable norm type selection via USNConfig.norm_type ∈ {"rmsnorm", "layernorm"}
7. THE Library SHALL default to RMSNorm as specified in the paper
8. THE Library SHALL initialize γ (gain) to ones and β (bias, LayerNorm only) to zeros
9. THE Library SHALL support both float32 and float16/bfloat16 computation in normalization

### Requirement 16: Weight Initialization

**User Story:** As a developer, I want all model parameters initialized according to the paper's specifications for stable training, so that gradients flow properly from the first training step.

#### Acceptance Criteria

1. THE Library SHALL initialize all linear projection weight matrices (W_u, W_α, W_λ, W_g, U_g, W_Δ, B_s, B_r, C_r, W_s, W_r, W_c, W_1, W_2) using Xavier uniform initialization by default
2. THE Library SHALL initialize all bias terms to zeros by default
3. THE Library SHALL initialize the token embedding matrix using normal distribution with std = 0.02
4. THE Library SHALL initialize gate biases (b_α, b_g) such that initial gate values are in intermediate ranges (approximately 0.5) for balanced initial behavior
5. THE Library SHALL initialize decay bias b_λ such that initial λ_t values are in the range [0.9, 0.99] for long initial memory
6. THE Library SHALL initialize the Output_Head weights using normal distribution with std = 0.02 / √(2 × num_layers)
7. THE Library SHALL support configurable initialization schemes via USNConfig.init_method
8. THE Library SHALL provide a `reset_parameters()` method on the Model and each submodule to re-initialize all parameters
9. THE Library SHALL document the rationale for each initialization choice in terms of gradient flow and training stability
10. IF weight tying is enabled, THEN THE Library SHALL initialize the shared embedding/output matrix using the embedding initialization scheme

### Requirement 17: Training System - Core Training Loop

**User Story:** As a machine learning engineer, I want a complete training system with full backpropagation, teacher forcing, and all standard training features, so that I can train USN models from scratch on language modeling tasks.

#### Acceptance Criteria

1. THE USNTrainer SHALL implement a training loop performing: forward pass → loss computation → backward pass → gradient clipping → optimizer step → scheduler step → logging
2. THE USNTrainer SHALL use Cross_Entropy_Loss as the training objective for next-token prediction
3. THE USNTrainer SHALL implement Teacher_Forcing: at each timestep t, the input is the ground-truth token at position t (not the model's own prediction)
4. THE USNTrainer SHALL use AdamW as the default optimizer with configurable learning rate, β1, β2, weight decay, and epsilon
5. THE USNTrainer SHALL support gradient clipping by global norm with configurable max_grad_norm (default: 1.0)
6. THE USNTrainer SHALL support mixed precision training using torch.cuda.amp with BF16 or FP16 (configurable)
7. THE USNTrainer SHALL support gradient accumulation over configurable number of micro-batches before each optimizer step
8. THE USNTrainer SHALL support configurable learning rate schedulers: cosine annealing, linear warmup+decay, constant, cosine with warm restarts
9. THE USNTrainer SHALL support linear warmup over configurable number of steps before the main schedule begins
10. THE USNTrainer SHALL support checkpointing: save model weights, optimizer state, scheduler state, training step, epoch, loss history, and random states at configurable intervals
11. THE USNTrainer SHALL support resuming training from a checkpoint, restoring all state exactly
12. THE USNTrainer SHALL support optional early stopping based on validation loss with configurable patience
13. THE USNTrainer SHALL log training metrics (loss, learning rate, gradient norm, throughput tokens/sec) at configurable intervals
14. THE USNTrainer SHALL support evaluation on a validation set at configurable intervals, computing validation loss and perplexity
15. THE USNTrainer SHALL support configurable sequence length curriculum: starting with shorter sequences and increasing over training
16. THE USNTrainer SHALL guarantee that all training operations maintain causality (no future information leakage)
17. THE USNTrainer SHALL support training from scratch with random initialization
18. THE USNTrainer SHALL support fine-tuning from a pretrained model checkpoint

### Requirement 18: Distributed Training

**User Story:** As a machine learning engineer, I want distributed training support across multiple GPUs and nodes, so that I can train large USN models efficiently at scale.

#### Acceptance Criteria

1. THE Library SHALL support PyTorch DistributedDataParallel (DDP) for multi-GPU training on a single node
2. THE Library SHALL support PyTorch FSDP (Fully Sharded Data Parallel) for memory-efficient multi-GPU training
3. THE Library SHALL support multi-node distributed training via torch.distributed with NCCL backend
4. THE Library SHALL correctly synchronize gradients, batch normalization statistics (if any), and random states across devices
5. THE Library SHALL support configurable distributed training via USNTrainingConfig.distributed_strategy ∈ {"ddp", "fsdp", "none"}
6. THE Library SHALL partition data across workers with no overlap and proper shuffling
7. THE Library SHALL report aggregate training metrics (global loss, global throughput) across all workers
8. THE Library SHALL handle checkpoint saving/loading correctly in distributed settings (only rank 0 saves, all ranks load)
9. IF distributed backend is not available, THEN THE Library SHALL fall back to single-device training with a warning
10. THE Library SHALL support mixed precision in distributed mode without loss of numerical correctness

### Requirement 19: Optimizer and Scheduler

**User Story:** As a machine learning engineer, I want configurable optimizers and learning rate schedulers with sensible defaults, so that I can tune training dynamics for optimal convergence.

#### Acceptance Criteria

1. THE Library SHALL provide AdamW as the default optimizer with defaults: lr=3e-4, β1=0.9, β2=0.95, weight_decay=0.1, eps=1e-8
2. THE Library SHALL support alternative optimizers: Adam, SGD with momentum, and custom optimizer registration
3. THE Library SHALL provide a cosine annealing scheduler with configurable min_lr, max_lr, and total_steps
4. THE Library SHALL provide a linear warmup scheduler with configurable warmup_steps
5. THE Library SHALL provide a combined warmup + cosine decay scheduler as the default schedule
6. THE Library SHALL provide a constant learning rate scheduler option
7. THE Library SHALL provide a cosine with warm restarts scheduler with configurable restart period
8. THE Library SHALL allow separate learning rates for different parameter groups (e.g., embeddings vs. other parameters)
9. THE Library SHALL apply weight decay only to weight matrices, not to biases or normalization parameters
10. THE Library SHALL provide a `get_parameter_groups(model, weight_decay)` utility that correctly separates decayed and non-decayed parameters

### Requirement 20: Loss Functions

**User Story:** As a developer, I want properly implemented loss functions for language modeling training, so that the model receives correct gradient signals.

#### Acceptance Criteria

1. THE Library SHALL implement Cross_Entropy_Loss for next-token prediction: L = -Σ log P(x_{t+1} | x_1, ..., x_t)
2. THE Cross_Entropy_Loss SHALL compute loss only on valid positions (respecting padding masks and ignore_index)
3. THE Cross_Entropy_Loss SHALL support label smoothing with configurable smoothing factor (default: 0.0)
4. THE Cross_Entropy_Loss SHALL average loss over all valid tokens in the batch (mean reduction)
5. THE Library SHALL compute perplexity as exp(loss) for evaluation reporting
6. THE Library SHALL support auxiliary losses if needed (e.g., load balancing loss for future mixture-of-experts extensions)
7. THE Cross_Entropy_Loss SHALL be numerically stable using log-softmax formulation (not softmax + log separately)
8. THE Library SHALL place loss function implementations in the usn/losses/ directory

### Requirement 21: Inference System - Autoregressive Generation

**User Story:** As a user, I want to generate text autoregressively from trained USN models with multiple decoding strategies, so that I can use the model for language generation tasks.

#### Acceptance Criteria

1. THE USNGenerator SHALL generate tokens autoregressively: each new token is produced conditioned on all previous tokens via the persistent state
2. THE USNGenerator SHALL support greedy decoding: always select argmax of logits
3. THE USNGenerator SHALL support temperature scaling: divide logits by temperature T before softmax (T > 0)
4. THE USNGenerator SHALL support top-k sampling: zero out all logits below the k-th highest value, then sample from renormalized distribution
5. THE USNGenerator SHALL support top-p (nucleus) sampling: retain smallest set of tokens whose cumulative probability exceeds p, then sample from renormalized distribution
6. THE USNGenerator SHALL support beam search with configurable beam width, length penalty, and early stopping
7. THE USNGenerator SHALL support combined strategies: temperature + top-k, temperature + top-p, temperature + top-k + top-p
8. THE USNGenerator SHALL operate with O(1) memory with respect to generated sequence length (only state is maintained, not full context)
9. THE USNGenerator SHALL support streaming generation: yield tokens one at a time via a generator/iterator interface
10. THE USNGenerator SHALL support batch inference: generate for multiple prompts simultaneously
11. THE USNGenerator SHALL support configurable maximum generation length and stop tokens/sequences
12. THE USNGenerator SHALL guarantee causality: generated token at position t uses only information from positions 0..t-1
13. THE USNGenerator SHALL support repetition penalty to discourage repeated tokens/n-grams
14. THE USNGenerator SHALL support providing an initial state for continued generation
15. WHEN a stop token is generated, THE USNGenerator SHALL immediately stop generation for that sequence in the batch
16. THE USNGenerator SHALL support returning log-probabilities for each generated token alongside the tokens themselves

### Requirement 22: Native .usn Serialization Format

**User Story:** As a developer, I want a native .usn file format that stores everything needed to reconstruct the exact model in a single file, so that models are fully portable and self-contained.

#### Acceptance Criteria

1. THE .usn format SHALL store all model data in a SINGLE file with the .usn extension
2. THE .usn format SHALL contain: model weights (all parameter tensors), hyperparameters (full USNConfig), model version, metadata (creation date, training steps, author, description), tokenizer data, training state (optimizer state, scheduler state, step count), compatibility information (library version, PyTorch version), and a file integrity checksum
3. THE .usn format SHALL use a structured binary format with a header section, manifest/table of contents, and data sections
4. THE .usn format SHALL include a magic number and format version in the header for file identification
5. THE .usn format SHALL include a SHA-256 checksum for data integrity verification
6. WHEN loading a .usn file, THE Library SHALL verify the checksum and raise an error if integrity check fails
7. WHEN loading a .usn file, THE Library SHALL restore the exact original model with identical weights, configuration, and behavior
8. THE .usn format SHALL support optional compression (zlib/lz4) for reduced file size with a flag indicating compression state
9. THE .usn format SHALL be backward-compatible: newer library versions SHALL load older format versions
10. THE Library SHALL provide `usn.save(model, path, include_optimizer=True, include_tokenizer=True, metadata={})` for saving
11. THE Library SHALL provide `usn.load(path, map_location=None)` for loading, returning the fully reconstructed model
12. THE .usn format SHALL support partial loading: load only weights, only config, or only metadata without loading everything
13. THE .usn format SHALL store tensors in a platform-independent format (endianness specified in header)
14. THE Library SHALL implement the serialization in usn/serialization/ with: format specification, reader, writer, validator, and migration utilities
15. FOR ALL valid USNModel instances, saving then loading SHALL produce a model with identical forward pass outputs for the same input (round-trip property)

### Requirement 23: Model Export Formats

**User Story:** As a developer, I want to export USN models to standard formats like ONNX and SafeTensors, so that I can deploy models in production environments that don't use the USN library directly.

#### Acceptance Criteria

1. THE Library SHALL support export to ONNX format via `usn.export(model, "onnx", path)` with configurable opset version
2. THE Library SHALL support export to SafeTensors format via `usn.export(model, "safetensors", path)`
3. THE Library SHALL support export to raw PyTorch state_dict format via `usn.export(model, "state_dict", path)`
4. THE Library SHALL support export to PyTorch TorchScript format via `usn.export(model, "torchscript", path)`
5. WHEN exporting to ONNX, THE Library SHALL trace the model with representative inputs and verify output equivalence
6. WHEN exporting to any format, THE Library SHALL preserve numerical equivalence of outputs (verified against original model)
7. THE Library SHALL provide format-specific export options (e.g., ONNX opset version, dynamic axes, optimization level)
8. IF export to a format fails due to unsupported operations, THEN THE Library SHALL raise a descriptive error indicating which operations are unsupported

### Requirement 24: Configuration System

**User Story:** As a developer, I want a comprehensive configuration system that validates all hyperparameters and supports serialization, so that model configurations are always valid and reproducible.

#### Acceptance Criteria

1. THE USNConfig class SHALL define and validate all model hyperparameters: num_layers, d_model, d_s (semantic state dimension), k (relational state dimension), d_ff (feedforward dimension), vocab_size, max_seq_len, dropout, norm_type, activation, tie_weights, init_method, chunk_size, fused
2. THE USNConfig class SHALL define and validate all training hyperparameters via USNTrainingConfig: learning_rate, batch_size, max_steps, warmup_steps, weight_decay, grad_clip, mixed_precision, gradient_accumulation_steps, scheduler_type, eval_interval, checkpoint_interval, early_stopping_patience, distributed_strategy, sequence_curriculum
3. THE USNConfig class SHALL define and validate all generation hyperparameters via USNGenerationConfig: temperature, top_k, top_p, beam_width, max_new_tokens, repetition_penalty, stop_tokens, streaming
4. THE USNConfig class SHALL validate all parameters on instantiation and raise ValueError with descriptive messages for invalid combinations
5. THE USNConfig class SHALL provide sensible defaults for all parameters
6. THE USNConfig class SHALL support serialization to JSON and YAML formats
7. THE USNConfig class SHALL support deserialization from JSON and YAML formats
8. THE USNConfig class SHALL support creation from keyword arguments, dictionaries, or files
9. THE USNConfig class SHALL be immutable after creation (frozen dataclass or equivalent) to prevent accidental modification
10. THE USNConfig class SHALL validate cross-parameter constraints (e.g., d_s <= d_model, k² reasonably bounded)
11. THE USNConfig class SHALL support config inheritance/merging for partial overrides
12. THE Library SHALL provide predefined configurations for standard model sizes (tiny, small, base, large, xl)

### Requirement 25: Tokenizer Integration

**User Story:** As a developer, I want integrated tokenizer support that can be saved with the model and supports common tokenization schemes, so that text processing is seamless.

#### Acceptance Criteria

1. THE Library SHALL provide a tokenizer interface supporting: encode(text) → token_ids, decode(token_ids) → text, vocab_size property
2. THE Library SHALL support BPE (Byte Pair Encoding) tokenizer via integration with the `tokenizers` library (HuggingFace)
3. THE Library SHALL support character-level tokenizer for simple experiments
4. THE Library SHALL support word-level tokenizer with configurable vocabulary
5. THE Library SHALL support loading pretrained tokenizers from HuggingFace hub
6. THE Library SHALL support saving tokenizer data within the .usn model file
7. WHEN a .usn file containing tokenizer data is loaded, THE Library SHALL restore the exact tokenizer
8. THE Library SHALL support special tokens: [PAD], [BOS], [EOS], [UNK] with configurable token IDs
9. THE Library SHALL provide a `train_tokenizer(corpus, vocab_size, algorithm)` utility for training new tokenizers
10. THE Library SHALL support batch encoding and decoding for efficiency

### Requirement 26: Dataset Handling

**User Story:** As a machine learning engineer, I want dataset utilities for loading, preprocessing, and batching text data for USN training, so that I can feed data to the training loop efficiently.

#### Acceptance Criteria

1. THE Library SHALL provide a `USNDataset` class compatible with PyTorch DataLoader
2. THE USNDataset SHALL support loading text data from: plain text files, JSON/JSONL files, CSV files, HuggingFace datasets, and custom iterables
3. THE USNDataset SHALL tokenize text using the configured tokenizer
4. THE USNDataset SHALL create causal language modeling examples: input = tokens[:-1], target = tokens[1:]
5. THE USNDataset SHALL support configurable sequence length with proper truncation and padding
6. THE USNDataset SHALL support dynamic batching (grouping similar-length sequences to minimize padding)
7. THE USNDataset SHALL provide a collate function handling variable-length sequences with padding and attention/padding masks
8. THE USNDataset SHALL support streaming/iterable mode for datasets too large to fit in memory
9. THE USNDataset SHALL support shuffling with configurable buffer size for streaming mode
10. THE USNDataset SHALL support data preprocessing: filtering, deduplication, length filtering
11. THE Library SHALL provide a synthetic math dataset generator for validation (addition, multiplication, subtraction operations)
12. THE Library SHALL support curriculum learning: variable sequence length datasets that increase difficulty over training

### Requirement 27: Testing - Tensor and Dimension Verification

**User Story:** As a developer, I want comprehensive tests verifying tensor shapes, dtypes, and gradient flow through all modules, so that I can be confident the implementation is numerically correct.

#### Acceptance Criteria

1. THE test suite SHALL verify input/output tensor shapes for every module with multiple batch sizes and sequence lengths
2. THE test suite SHALL verify that all modules produce outputs of correct dtype (float32, float16, bfloat16 as configured)
3. THE test suite SHALL verify gradient flow: gradients are non-None and non-zero for all trainable parameters after a backward pass
4. THE test suite SHALL verify broadcasting correctness: operations work correctly with batch_size=1 and batch_size>1
5. THE test suite SHALL verify numerical stability: no NaN or Inf values in forward or backward pass under normal conditions
6. THE test suite SHALL verify that gate outputs (α_t, λ_t, g_t, c_t, ρ_t) are strictly within their defined bounds
7. THE test suite SHALL verify parameter count matches expected values for known configurations
8. THE test suite SHALL verify that model output logits have shape (batch_size, seq_len, vocab_size)
9. THE test suite SHALL use pytest as the test framework
10. THE test suite SHALL achieve minimum 95% code coverage across all modules
11. THE test suite SHALL include parametrized tests over multiple configurations (varying d_model, num_layers, seq_len, batch_size)

### Requirement 28: Testing - Causality Verification

**User Story:** As a developer, I want tests that prove the model never accesses future information, so that I can guarantee the causal property essential for autoregressive models.

#### Acceptance Criteria

1. THE test suite SHALL verify causality by checking that output at position t is invariant to changes in input at positions > t
2. THE test suite SHALL verify causality by computing Jacobian(output[t], input[j]) = 0 for all j > t
3. THE test suite SHALL verify that the parallel scan produces identical outputs to sequential recurrence for random inputs
4. THE test suite SHALL verify that during generation, modifying future tokens does not change current token predictions
5. THE test suite SHALL verify that each Block's state S_t depends only on inputs x_0, ..., x_t
6. THE test suite SHALL test causality for both training mode (parallel) and inference mode (sequential)
7. THE test suite SHALL verify that Temporal_Mixing uses only u_{t-1} and never u_{t+1} or later values

### Requirement 29: Testing - Serialization Round-Trip

**User Story:** As a developer, I want tests verifying that save/load produces bit-exact models, so that serialization is guaranteed lossless.

#### Acceptance Criteria

1. THE test suite SHALL verify that saving and loading a model produces identical parameter values (torch.equal for all parameters)
2. THE test suite SHALL verify that a loaded model produces identical forward pass outputs for the same input
3. THE test suite SHALL verify that the .usn format checksum validation detects file corruption
4. THE test suite SHALL verify that all metadata (config, version, author, training state) survives round-trip
5. THE test suite SHALL verify that tokenizer data survives round-trip and produces identical encodings
6. THE test suite SHALL verify backward compatibility by loading .usn files from older format versions
7. THE test suite SHALL verify that partial loading (config-only, weights-only) works correctly
8. FOR ALL valid USNConfig objects, serializing to JSON then deserializing SHALL produce an equivalent config (round-trip property)
9. FOR ALL valid USNModel instances with random weights, save then load SHALL produce a model where forward(input) yields identical output tensors (round-trip property)

### Requirement 30: Testing - Determinism and Reproducibility

**User Story:** As a researcher, I want deterministic execution when seeds are set, so that experiments are reproducible.

#### Acceptance Criteria

1. THE test suite SHALL verify that two runs with identical seeds, data, and configuration produce identical loss curves
2. THE test suite SHALL verify that model initialization is deterministic given the same seed
3. THE test suite SHALL verify that generation output is deterministic given the same seed and greedy/deterministic sampling
4. THE Library SHALL provide a `usn.set_seed(seed)` utility that sets random seeds for Python, NumPy, PyTorch, and CUDA
5. THE test suite SHALL verify that parallel scan and sequential recurrence produce identical outputs (up to float precision) given the same inputs
6. THE Library SHALL document known sources of non-determinism (e.g., CUDA atomics) and provide flags to enable full determinism where possible

### Requirement 31: Testing - Scalability Table

**User Story:** As a researcher, I want a comprehensive scalability table with 20+ model configurations showing expected resource usage, so that I can plan experiments and deployments.

#### Acceptance Criteria

1. THE Library SHALL define and document a scalability table with at least 20 model configurations named: Tiny, Micro, Mini, Small, Base, Medium, Large, XL, XXL, 2B, 4B, 7B, 13B, 30B, 65B, and additional intermediate sizes
2. THE scalability table SHALL specify for each configuration: num_layers, d_model, d_s, k, d_ff, total parameters, memory usage (fp32, fp16), approximate training FLOPs per token, state size per sequence, expected training cost (GPU-hours for 1B tokens), expected maximum context length
3. THE test suite SHALL verify that instantiating each configuration produces the expected parameter count (within 1% tolerance for rounding)
4. THE test suite SHALL verify that the state size matches d_s + k² for each configuration
5. THE scalability table SHALL be included in the documentation with formatted tables
6. THE Library SHALL provide `USNConfig.from_preset(name)` to instantiate any predefined configuration by name
7. THE scalability table SHALL demonstrate that USN achieves linear scaling of compute with respect to parameters (no quadratic blowup)

### Requirement 32: Testing - Micro-Model Training Validation

**User Story:** As a developer, I want a complete micro-model (~2M parameters) training run on synthetic math data that proves the implementation works end-to-end, so that correctness is validated empirically.

#### Acceptance Criteria

1. THE test suite SHALL define a Micro_Model configuration with approximately 2 million parameters
2. THE test suite SHALL provide a synthetic math dataset generating examples like: "5+5=10", "8*7=56", "12-3=9", "15+27=42"
3. THE test suite SHALL train the Micro_Model on the synthetic math dataset for enough steps to demonstrate learning
4. THE test suite SHALL verify that training loss decreases monotonically over the first N steps (convergence)
5. THE test suite SHALL verify that gradients are non-zero and within reasonable magnitude throughout training
6. THE test suite SHALL verify that the trained Micro_Model can generate correct answers for simple arithmetic problems seen during training
7. THE test suite SHALL verify that generation from the Micro_Model produces valid token sequences (no garbage output)
8. THE test suite SHALL complete the full micro-model validation (train + evaluate + generate) in under 5 minutes on a single GPU or under 15 minutes on CPU
9. THE test suite SHALL log and report: initial loss, final loss, loss reduction ratio, sample generations, training throughput (tokens/sec)
10. THE test suite SHALL verify that the Micro_Model checkpoint can be saved, loaded, and resumed with continued training showing improvement

### Requirement 33: Benchmarking System

**User Story:** As a developer, I want a comprehensive benchmarking system measuring speed, memory, and throughput across configurations, so that performance characteristics are documented and regressions can be detected.

#### Acceptance Criteria

1. THE Library SHALL provide benchmarks measuring: forward pass latency (ms) per sequence length, backward pass latency (ms) per sequence length, tokens per second throughput (training), tokens per second throughput (inference), peak memory usage (MB) during training, peak memory usage (MB) during inference, time to first token (ms) for generation, serialization save/load time (ms) by model size, batch throughput scaling (tokens/sec vs batch size)
2. THE benchmarks SHALL test multiple model sizes: at minimum Tiny, Small, Base, and Large configurations
3. THE benchmarks SHALL test multiple sequence lengths: 128, 256, 512, 1024, 2048, 4096, 8192
4. THE benchmarks SHALL test multiple batch sizes: 1, 2, 4, 8, 16, 32
5. THE benchmarks SHALL measure parallel scan vs sequential recurrence performance comparison
6. THE benchmarks SHALL report results in formatted tables with mean and standard deviation over multiple runs
7. THE benchmarks SHALL verify O(n) linear scaling of forward/backward pass time with sequence length
8. THE benchmarks SHALL verify O(1) constant memory usage during inference regardless of generated length
9. THE benchmarks SHALL be runnable via CLI: `usn benchmark --config <config> --device <device>`
10. THE benchmarks SHALL output results to JSON for automated comparison and regression detection
11. THE benchmarks SHALL include comparison metrics against theoretical peak (arithmetic intensity, memory bandwidth utilization)
12. THE Library SHALL store benchmark results in benchmarks/ directory with timestamp and hardware info

### Requirement 34: CLI Interface

**User Story:** As a developer, I want a command-line interface for common operations like training, generation, benchmarking, and model inspection, so that I can use the library without writing Python scripts.

#### Acceptance Criteria

1. THE CLI SHALL be accessible via the `usn` command after installation (entry point in pyproject.toml)
2. THE CLI SHALL provide subcommands: `usn train`, `usn generate`, `usn benchmark`, `usn info`, `usn export`, `usn validate`
3. WHEN `usn train --config <path>` is invoked, THE CLI SHALL execute a training run using the specified configuration file
4. WHEN `usn generate --model <path> --prompt <text>` is invoked, THE CLI SHALL generate text from the specified model
5. WHEN `usn benchmark --model <path>` is invoked, THE CLI SHALL run performance benchmarks on the specified model
6. WHEN `usn info --model <path>` is invoked, THE CLI SHALL display model architecture, parameter count, configuration, and metadata
7. WHEN `usn export --model <path> --format <fmt> --output <path>` is invoked, THE CLI SHALL export the model to the specified format
8. WHEN `usn validate --model <path>` is invoked, THE CLI SHALL run integrity checks on the .usn file (checksum, format version, completeness)
9. THE CLI SHALL provide `--help` documentation for all commands and options
10. THE CLI SHALL use argparse or click for argument parsing with proper validation
11. THE CLI SHALL support verbose/quiet output modes via `--verbose` and `--quiet` flags
12. IF any CLI command receives invalid arguments, THEN THE CLI SHALL display a helpful error message and exit with non-zero status

### Requirement 35: Backend System

**User Story:** As a developer, I want a backend abstraction supporting CPU, CUDA, and potentially other accelerators, so that the library works across different hardware environments.

#### Acceptance Criteria

1. THE Library SHALL automatically detect available hardware (CPU, CUDA GPUs, MPS) and select the best available device
2. THE Library SHALL support explicit device specification via `device` parameter in all relevant APIs
3. THE Library SHALL support moving models between devices via `model.to(device)`
4. THE Library SHALL provide device-specific optimizations: CUDA kernel fusion, CPU multithreading
5. THE Library SHALL provide a `usn.device_info()` utility reporting available devices, memory, and compute capability
6. WHEN CUDA is not available, THE Library SHALL fall back to CPU execution without errors
7. THE Library SHALL support Apple MPS backend for Mac GPU acceleration when available
8. THE Library SHALL organize backend-specific code in usn/backends/ with a clean abstraction layer
9. THE Library SHALL support torch.compile for JIT compilation and graph optimization on supported backends

### Requirement 36: Utilities Module

**User Story:** As a developer, I want utility functions for common operations like timing, memory measurement, visualization, and debugging, so that I can monitor and troubleshoot model behavior.

#### Acceptance Criteria

1. THE Library SHALL provide `usn.utils.count_parameters(model)` returning total, trainable, and non-trainable parameter counts
2. THE Library SHALL provide `usn.utils.estimate_memory(config)` returning estimated memory usage for training and inference
3. THE Library SHALL provide `usn.utils.estimate_flops(config, seq_len)` returning estimated FLOPs per forward pass
4. THE Library SHALL provide `usn.utils.set_seed(seed)` for deterministic execution
5. THE Library SHALL provide `usn.utils.timer()` context manager for timing code sections
6. THE Library SHALL provide `usn.utils.memory_tracker()` context manager for measuring peak memory
7. THE Library SHALL provide `usn.utils.gradient_stats(model)` returning per-parameter gradient norms and statistics
8. THE Library SHALL provide `usn.utils.activation_stats(model)` hook-based utility for monitoring activation magnitudes
9. THE Library SHALL provide `usn.utils.visualize_state(state)` for plotting/displaying the semantic and relational state values
10. THE Library SHALL provide `usn.utils.model_summary(model)` for formatted display of architecture details
11. THE Library SHALL provide logging utilities with configurable verbosity levels (DEBUG, INFO, WARNING, ERROR)
12. THE Library SHALL organize all utilities in usn/utils/ with clear submodule organization

### Requirement 37: Code Quality and Standards

**User Story:** As a developer maintaining the codebase, I want all code to meet strict quality standards with no duplication, full typing, and comprehensive documentation, so that the codebase is maintainable and professional.

#### Acceptance Criteria

1. THE Library SHALL follow PEP 8 style guidelines with max line length of 100 characters
2. THE Library SHALL provide complete type annotations for all public and private functions, methods, and class attributes
3. THE Library SHALL provide Google-style docstrings for all public classes, methods, and functions including: description, Args, Returns, Raises, Examples
4. THE Library SHALL contain no duplicate code: shared logic SHALL be extracted into utility functions
5. THE Library SHALL contain no dead code: unused imports, unreachable branches, or unused variables
6. THE Library SHALL maintain low coupling: modules SHALL communicate through well-defined interfaces
7. THE Library SHALL maintain high cohesion: each module SHALL have a single, well-defined responsibility
8. THE Library SHALL contain no circular imports between packages
9. THE Library SHALL pass mypy strict type checking with no errors
10. THE Library SHALL pass ruff or flake8 linting with no warnings
11. THE Library SHALL include a pre-commit configuration for automated code quality checks
12. THE Library SHALL use consistent naming conventions: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants

### Requirement 38: Documentation

**User Story:** As a user and contributor, I want a single extremely detailed Markdown documentation file covering every aspect of the library, so that I can understand, use, and extend the system completely.

#### Acceptance Criteria

1. THE Library SHALL include a comprehensive documentation file (docs/DOCUMENTATION.md) covering all of the following sections in full detail
2. THE documentation SHALL include an Architecture section: full USN architecture description, mathematical formulation of every module, block diagram, data flow, state transition equations, complexity analysis per module
3. THE documentation SHALL include a Project Structure section: every directory and file explained, dependency graph, module responsibilities, import structure
4. THE documentation SHALL include an API Reference section: every public class, method, and function documented with signatures, parameters, return types, exceptions, and examples
5. THE documentation SHALL include a Layers section: each USN layer/module documented with objective, inputs, outputs, equations, complexity, constraints, code examples
6. THE documentation SHALL include a Training section: training loop explanation, hyperparameter guide, curriculum strategy, distributed training setup, checkpoint management, troubleshooting
7. THE documentation SHALL include an Inference section: generation strategies explained, streaming usage, batch inference, performance tuning, deployment guide
8. THE documentation SHALL include a .usn Format section: format specification, binary layout, header structure, section types, versioning scheme, migration guide
9. THE documentation SHALL include a Tests section: test organization, running tests, coverage report interpretation, adding new tests
10. THE documentation SHALL include a Benchmarks section: benchmark methodology, hardware specifications, result tables, performance analysis, scaling behavior
11. THE documentation SHALL include a Results section: micro-model training results, convergence curves, sample generations, comparison tables
12. THE documentation SHALL include an Examples section: complete code examples for common tasks (create model, train, generate, save/load, fine-tune, export)
13. THE documentation SHALL include a Problems and Solutions section: known issues, workarounds, design decisions and rationale
14. THE documentation SHALL include a Future Work section: planned improvements, extension points, contribution guide
15. THE documentation SHALL include a Conclusions section: summary of achievements, key metrics, comparison with paper claims

### Requirement 39: Examples and Notebooks

**User Story:** As a new user, I want complete working examples and Jupyter notebooks demonstrating common workflows, so that I can learn to use the library quickly.

#### Acceptance Criteria

1. THE Library SHALL include examples/ directory with self-contained Python scripts for: model creation, training from scratch, text generation, model save/load, model export, fine-tuning, custom datasets, distributed training
2. THE Library SHALL include notebooks/ directory with Jupyter notebooks for: quick start tutorial, architecture visualization, training walkthrough, generation strategies comparison, benchmark reproduction
3. WHEN an example script is executed, THE script SHALL run to completion without errors given the documented prerequisites
4. THE examples SHALL include inline comments explaining each step
5. THE examples SHALL use the public API exclusively (no internal imports)
6. THE notebooks SHALL include expected outputs/visualizations inline

### Requirement 40: Stability and Numerical Safety

**User Story:** As a developer training deep models, I want built-in numerical stability guarantees preventing NaN, Inf, and gradient explosion, so that training runs are robust.

#### Acceptance Criteria

1. THE Library SHALL constrain λ_t ∈ (0,1) via the exp(-softplus(·)) construction, mathematically preventing state explosion
2. THE Library SHALL constrain g_t ∈ (0,1) via sigmoid, ensuring bounded state writes
3. THE Library SHALL constrain c_t ∈ (0,1) via sigmoid, ensuring bounded readout contribution
4. THE Library SHALL constrain ρ_t ∈ (0,1) via the same exp(-softplus(·)) construction for relational decay
5. THE Library SHALL use numerically stable log-space computation for accumulated decay products in parallel scan
6. THE Library SHALL use log-softmax (not separate softmax + log) for loss computation
7. THE Library SHALL support gradient clipping (default max_norm=1.0) to prevent gradient explosion
8. THE Library SHALL use epsilon values in all division and normalization operations to prevent division by zero
9. THE Library SHALL detect NaN/Inf in forward pass outputs during training and raise an informative error
10. THE Library SHALL provide an optional NaN/Inf detection hook that can be enabled for debugging
11. WHILE using mixed precision, THE Library SHALL use gradient scaling (GradScaler) to prevent underflow in FP16
12. THE Library SHALL initialize parameters such that initial activations and gradients are in reasonable magnitude ranges

### Requirement 41: Memory Efficiency

**User Story:** As a developer training on limited hardware, I want memory-efficient implementation techniques so that I can train the largest possible models.

#### Acceptance Criteria

1. THE Library SHALL support gradient checkpointing at block boundaries to trade compute for memory
2. THE Library SHALL support gradient checkpointing within the parallel scan to reduce scan memory from O(n) to O(√n)
3. THE Library SHALL minimize intermediate tensor materializations through in-place operations where safe
4. THE Library SHALL support activation offloading to CPU for extreme memory savings (optional)
5. THE Library SHALL report peak memory usage during training via `usn.utils.memory_tracker()`
6. THE Library SHALL support configurable precision per module (e.g., keep certain operations in fp32 while others use fp16)
7. THE Library SHALL document memory usage formulas for each model configuration in the scalability table
8. WHILE in inference mode, THE Library SHALL use O(1) memory with respect to generated sequence length (state-only, no growing KV cache)
9. THE Library SHALL support torch.no_grad() inference mode reducing memory by not storing activation for backward pass

### Requirement 42: Causality Guarantee Across All Operations

**User Story:** As a developer building an autoregressive model, I want absolute certainty that no operation in the entire forward pass accesses future information, so that the model is valid for autoregressive tasks.

#### Acceptance Criteria

1. THE Input_Projection SHALL operate independently per timestep (no temporal dependency)
2. THE Temporal_Mixing SHALL access only the immediately preceding timestep u_{t-1} (one-step lookback)
3. THE Exponential_Gating SHALL compute λ_t from only x_t at the current timestep
4. THE Selective_Writing SHALL compute g_t from m_t (current) and S_{t-1} (past state only)
5. THE State_Update SHALL compute S_t from S_{t-1} (past) and current inputs only
6. THE State_Readout SHALL read from S_t (current state, which depends only on past)
7. THE Channel_Mixing SHALL operate independently per timestep position
8. THE Parallel_Scan SHALL produce outputs equivalent to causal sequential computation
9. THE Model SHALL never use bidirectional operations, future-looking masks, or non-causal convolutions
10. THE Library SHALL include automated causality tests verifiable by Jacobian analysis (∂output_t/∂input_j = 0 for j > t)

### Requirement 43: Linear Complexity Guarantee

**User Story:** As a researcher, I want formal assurance that the model achieves O(n) time complexity with respect to sequence length during training, so that it can handle long sequences efficiently.

#### Acceptance Criteria

1. THE Model forward pass SHALL execute in O(n × d²) time where n is sequence length and d is model dimension (linear in n)
2. THE Parallel_Scan SHALL execute in O(n) work with O(log n) parallel depth, not O(n²)
3. THE Model SHALL contain no operation with quadratic O(n²) dependence on sequence length
4. THE Model SHALL contain no attention mechanism or equivalent dot-product computation between all pairs of positions
5. THE benchmarks SHALL empirically verify linear scaling by measuring wall-clock time at multiple sequence lengths and confirming linear trend
6. THE Library SHALL document the complexity of every module in terms of sequence length n and model dimensions

### Requirement 44: GPU/TPU Optimization

**User Story:** As a developer deploying on modern accelerators, I want the implementation optimized for GPU execution with vectorized operations and efficient memory patterns.

#### Acceptance Criteria

1. THE Library SHALL use batched matrix operations (torch.bmm, torch.einsum) instead of loops over batch/sequence dimensions
2. THE Library SHALL minimize CPU-GPU synchronization points during forward and backward passes
3. THE Library SHALL use contiguous memory layouts for tensors passed to CUDA kernels
4. THE Library SHALL support torch.compile for whole-model graph optimization
5. THE Library SHALL organize computations to maximize arithmetic intensity (compute per memory access)
6. THE Library SHALL support Triton custom kernels for fused operations on NVIDIA GPUs
7. THE Library SHALL avoid Python-level loops over sequence positions during training (use vectorized/scan operations)
8. THE Library SHALL support CUDA graphs for inference to eliminate kernel launch overhead
9. THE Library SHALL document hardware-specific performance tuning recommendations
10. WHEN running on GPU, THE Library SHALL keep all computation on device without unnecessary CPU transfers

### Requirement 45: Paper Validation Checklist

**User Story:** As the author, I want a systematic point-by-point comparison between the implementation and the paper, so that I can verify complete and faithful implementation.

#### Acceptance Criteria

1. THE Library SHALL include a PAPER_VALIDATION.md file with every claim, equation, and architectural decision from the paper listed as a checklist item
2. THE checklist SHALL mark each item as one of: ✅ Implemented, ⚠️ Partial, ❌ Missing
3. THE Library SHALL not be considered complete while any item is marked as ⚠️ Partial or ❌ Missing
4. THE checklist SHALL cover: all equations (numbered), all architectural components, all training procedures, all inference procedures, all stability mechanisms, all complexity claims, all design justifications
5. THE checklist SHALL include: state transition equations (s_t update, R_t update), gate computations (α_t, λ_t, g_t, c_t, ρ_t), projection matrices (W_u, B_s, B_r, C_r, W_s, W_r), MLP formulation, normalization placement, residual connections, parallel scan algorithm, chunk decomposition, initialization scheme
6. THE checklist SHALL reference the specific source code file and line number implementing each paper element
7. THE checklist SHALL be updated whenever code changes affect paper-specified behavior

### Requirement 46: Error Handling and Validation

**User Story:** As a developer, I want informative error messages when I misconfigure the model or provide invalid inputs, so that I can quickly diagnose and fix issues.

#### Acceptance Criteria

1. WHEN a USNConfig is created with invalid hyperparameters, THE Library SHALL raise ValueError with a message specifying: which parameter is invalid, what value was provided, what the valid range/type is
2. WHEN input tensors have incorrect shapes, THE Library SHALL raise a descriptive ShapeError or RuntimeError indicating expected vs actual shapes
3. WHEN a .usn file is corrupted or has invalid checksum, THE Library SHALL raise IntegrityError with details
4. WHEN a .usn file has incompatible format version, THE Library SHALL raise VersionError with migration instructions
5. WHEN CUDA out-of-memory occurs during training, THE Library SHALL provide suggestions (reduce batch size, enable gradient checkpointing, use mixed precision)
6. WHEN NaN is detected during training, THE Library SHALL provide debugging information (which layer, which parameter, step number)
7. IF a required dependency is missing, THEN THE Library SHALL raise ImportError with installation instructions
8. THE Library SHALL define custom exception classes in usn/exceptions.py: USNError (base), ConfigError, ShapeError, IntegrityError, VersionError, TrainingError, GenerationError
9. THE Library SHALL never produce bare exceptions or generic error messages without context

### Requirement 47: Logging and Monitoring

**User Story:** As a machine learning engineer, I want comprehensive logging and metric tracking during training, so that I can monitor progress and diagnose issues.

#### Acceptance Criteria

1. THE Library SHALL log training metrics at configurable intervals: step, loss, learning_rate, gradient_norm, tokens_per_second, memory_usage, epoch
2. THE Library SHALL log validation metrics: val_loss, val_perplexity, best_val_loss
3. THE Library SHALL support logging to: console (default), file, TensorBoard, Weights & Biases (optional integrations)
4. THE Library SHALL provide progress bars for training and evaluation loops (using tqdm or rich)
5. THE Library SHALL log model architecture summary at training start
6. THE Library SHALL log hardware information at training start (device, memory, compute capability)
7. THE Library SHALL use Python's standard logging module with configurable levels
8. THE Library SHALL support structured logging (JSON format) for machine-parseable output
9. THE Library SHALL never log sensitive information (file paths on disk are acceptable, but not tokens/raw data by default)

### Requirement 48: Dropout and Regularization

**User Story:** As a developer, I want configurable regularization mechanisms to prevent overfitting during training.

#### Acceptance Criteria

1. THE Library SHALL support dropout applied after the Channel_Mixing MLP with configurable rate (default: 0.0)
2. THE Library SHALL support dropout on the residual connection with configurable rate
3. THE Library SHALL support weight decay via AdamW (applied to weight matrices only, not biases or norms)
4. THE Library SHALL disable all dropout during inference/evaluation mode (model.eval())
5. THE Library SHALL support configurable embedding dropout (dropout on token embeddings)
6. THE Library SHALL ensure dropout does not break causality (standard dropout is position-independent, which is causal)

### Requirement 49: Positional Information Handling

**User Story:** As a developer, I want clarity on how positional information is handled in USN (which has no explicit position encoding due to its recurrent state), so that the implementation is faithful to the paper.

#### Acceptance Criteria

1. THE Library SHALL NOT include traditional positional encodings (sinusoidal or learned position embeddings) unless explicitly specified in the paper
2. THE Library SHALL rely on the recurrent state mechanism and temporal mixing to implicitly encode position information
3. THE Library SHALL document that position is encoded implicitly through state evolution (s_t carries information about all positions 0..t)
4. IF the paper specifies any explicit positional mechanism, THEN THE Library SHALL implement it exactly as specified
5. THE Library SHALL support optional positional embeddings as a configurable extension for experimentation (disabled by default)
6. THE Temporal_Mixing one-step lookback (u_{t-1}) SHALL serve as the primary local positional signal

### Requirement 50: Model Variants and Configurations

**User Story:** As a researcher, I want predefined model configurations spanning from tiny validation models to large-scale research models, with all hyperparameters pre-tuned.

#### Acceptance Criteria

1. THE Library SHALL provide USNConfig.tiny() with approximately 1-5M parameters suitable for unit tests
2. THE Library SHALL provide USNConfig.micro() with approximately 2M parameters suitable for synthetic validation
3. THE Library SHALL provide USNConfig.mini() with approximately 10-20M parameters suitable for small experiments
4. THE Library SHALL provide USNConfig.small() with approximately 50-125M parameters
5. THE Library SHALL provide USNConfig.base() with approximately 125-350M parameters
6. THE Library SHALL provide USNConfig.medium() with approximately 350-750M parameters
7. THE Library SHALL provide USNConfig.large() with approximately 750M-1.5B parameters
8. THE Library SHALL provide USNConfig.xl() with approximately 1.5-3B parameters
9. THE Library SHALL provide USNConfig.xxl() with approximately 3-7B parameters
10. THE Library SHALL provide additional configurations up to 65B parameters following consistent scaling rules
11. WHEN scaling model size, THE Library SHALL scale d_model, d_s, k, d_ff, and num_layers following established scaling laws
12. THE Library SHALL document the relationship between d_s, k, and d_model for each configuration (how state capacity scales with model dimension)

### Requirement 51: Gradient Checkpointing

**User Story:** As a developer training on limited GPU memory, I want gradient checkpointing support that trades compute for memory, so that I can train larger models or use longer sequences.

#### Acceptance Criteria

1. THE Library SHALL support gradient checkpointing at the Block level: recompute activations during backward pass instead of storing them
2. THE Library SHALL support gradient checkpointing within the parallel scan: checkpoint at chunk boundaries
3. THE Library SHALL provide configurable checkpointing granularity: none, per-block, per-chunk, custom
4. WHEN gradient checkpointing is enabled, THE Library SHALL produce numerically identical gradients to non-checkpointed training
5. THE Library SHALL report estimated memory savings from gradient checkpointing for each configuration
6. THE Library SHALL support selective checkpointing: only checkpoint specific layers (e.g., every other block)
7. THE Library SHALL enable gradient checkpointing via `model.enable_gradient_checkpointing()` or training config flag

### Requirement 52: State Management for Inference

**User Story:** As a developer deploying inference, I want efficient state management that enables constant-memory generation, context continuation, and state caching.

#### Acceptance Criteria

1. THE Library SHALL maintain model state as a fixed-size tuple (s ∈ R^{d_s}, R ∈ R^{k×k}) per layer during inference
2. THE Library SHALL never grow memory with generated sequence length (no KV cache, no growing buffers)
3. THE Library SHALL support caching and restoring state for "branching" generation (save state, generate multiple continuations)
4. THE Library SHALL support passing initial state to the generator for continued conversations/context
5. THE Library SHALL support resetting state to zeros for new independent sequences
6. THE Library SHALL support state serialization: save/load intermediate state for later continuation
7. THE Library SHALL ensure state is on the same device as the model to avoid CPU-GPU transfers during generation
8. THE Library SHALL provide `model.get_state()` and `model.set_state(state)` methods for state inspection and manipulation
9. THE total inference memory SHALL be O(num_layers × (d_s + k²)) regardless of sequence length generated

### Requirement 53: Associativity Verification

**User Story:** As a developer, I want formal verification that the state transition is truly associative, so that the parallel scan produces correct results.

#### Acceptance Criteria

1. THE test suite SHALL verify associativity: compose(T_a, compose(T_b, T_c)) produces identical results to compose(compose(T_a, T_b), T_c)
2. THE test suite SHALL verify that the semantic state transition (λ⊙s + g⊙Bm) is expressible as an affine map T(s) = As + b
3. THE test suite SHALL verify that the relational state transition (ρR + outer) is expressible as an affine map T(R) = ρR + b_R
4. THE test suite SHALL verify that composing two affine maps yields another affine map
5. THE test suite SHALL verify parallel scan output matches sequential for random inputs with sequence lengths 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
6. THE test suite SHALL test associativity with edge cases: λ=0, λ=1-ε, g=0, g=1-ε
7. THE test suite SHALL use randomized testing (property-based testing) to verify associativity over many random transition tuples
8. FOR ALL random affine transitions T_a, T_b, T_c, composition SHALL be associative: (T_a ∘ T_b) ∘ T_c = T_a ∘ (T_b ∘ T_c) within floating-point tolerance

### Requirement 54: Package Module Responsibilities

**User Story:** As a developer navigating the codebase, I want each package directory to have a clear, documented, non-overlapping responsibility.

#### Acceptance Criteria

1. THE usn/core/ package SHALL contain: base classes, abstract interfaces, type definitions, and fundamental data structures used across the library
2. THE usn/modules/ package SHALL contain: individual USN submodule implementations (InputProjection, TemporalMixing, ExponentialGating, SelectiveWriting, StateUpdate, StateReadout, ChannelMixing)
3. THE usn/layers/ package SHALL contain: composed layers (USNBlock, parallel scan, chunk decomposition) that combine multiple modules
4. THE usn/models/ package SHALL contain: complete model classes (USNModel) and model factory functions
5. THE usn/training/ package SHALL contain: trainer class, training loop, distributed training utilities, curriculum strategies
6. THE usn/datasets/ package SHALL contain: dataset classes, data loading utilities, synthetic data generators, collate functions
7. THE usn/tokenizers/ package SHALL contain: tokenizer interface, BPE/character/word tokenizer implementations, tokenizer training
8. THE usn/serialization/ package SHALL contain: .usn format specification, reader, writer, validator, migration utilities
9. THE usn/utils/ package SHALL contain: utility functions (counting, timing, memory, visualization, seeding, logging)
10. THE usn/optim/ package SHALL contain: optimizer configurations, parameter group utilities, scheduler implementations
11. THE usn/losses/ package SHALL contain: loss function implementations (cross-entropy with label smoothing, perplexity computation)
12. THE usn/config/ package SHALL contain: configuration classes (USNConfig, USNTrainingConfig, USNGenerationConfig), preset configurations, validation logic
13. THE usn/backends/ package SHALL contain: device detection, backend-specific optimizations, kernel fusion utilities, compile wrappers
14. THE usn/cli/ package SHALL contain: CLI entry point, command implementations, argument parsing
15. EACH package SHALL have an __init__.py exporting its public API
16. THE Library SHALL have no circular imports between any of these packages

### Requirement 55: Relational State Outer Product

**User Story:** As a developer, I want the relational state update implemented exactly as the outer product of two projected vectors, so that the model captures relational information between features.

#### Acceptance Criteria

1. THE relational state update SHALL compute (B_r m_t)(C_r m_t)^T as the outer product of two k-dimensional vectors
2. THE projection B_r ∈ R^{k × d_model} SHALL map m_t to a k-dimensional vector for the left factor
3. THE projection C_r ∈ R^{k × d_model} SHALL map m_t to a k-dimensional vector for the right factor
4. THE outer product SHALL produce a k×k matrix added to the decayed previous relational state
5. THE relational state R_t SHALL be a symmetric-capable (not necessarily symmetric) k×k matrix
6. THE vectorization vec(R_t) SHALL flatten the k×k matrix into a k²-dimensional vector for readout projection
7. THE Library SHALL implement the outer product efficiently using torch.outer or equivalent batched operation (torch.bmm on unsqueezed vectors)
8. THE k dimension SHALL be configurable independently of d_model and d_s
9. THE Library SHALL document that the relational state captures second-order interactions between projected features

### Requirement 56: State Read Operation

**User Story:** As a developer, I want the state read operation (used in selective writing gate) implemented as specified, extracting information from the previous unified state.

#### Acceptance Criteria

1. THE read operation SHALL combine information from both semantic and relational state: read(S_{t-1}) = f(s_{t-1}, R_{t-1})
2. THE read operation SHALL project the semantic state s_{t-1} via a learned linear projection
3. THE read operation SHALL project the vectorized relational state vec(R_{t-1}) via a learned linear projection
4. THE read operation SHALL combine both projections (concatenation or addition as specified in the paper)
5. THE read operation output SHALL be used as input to the write gate computation: g_t = σ(W_g m_t + U_g read(S_{t-1}) + b_g)
6. THE read operation SHALL be differentiable for gradient flow through the state
7. THE read operation SHALL be efficient: avoid materializing unnecessary intermediate tensors

### Requirement 57: Mixed Precision Training Details

**User Story:** As a developer, I want mixed precision training that maximizes throughput while maintaining training stability, with specific operations kept in higher precision.

#### Acceptance Criteria

1. THE Library SHALL support BF16 (bfloat16) as the primary mixed-precision format on compatible hardware
2. THE Library SHALL support FP16 (float16) as an alternative mixed-precision format
3. THE Library SHALL keep normalization layers (RMSNorm, LayerNorm) in FP32 for stability
4. THE Library SHALL keep loss computation in FP32 for numerical accuracy
5. THE Library SHALL keep softmax/log-softmax computations in FP32
6. THE Library SHALL use PyTorch's autocast context manager for automatic mixed precision
7. WHEN using FP16, THE Library SHALL use GradScaler for gradient scaling to prevent underflow
8. WHEN using BF16, THE Library SHALL NOT use GradScaler (BF16 has sufficient dynamic range)
9. THE Library SHALL support configurable per-operation precision overrides
10. THE Library SHALL document which operations are kept in FP32 and why

### Requirement 58: Embedding Layer

**User Story:** As a developer, I want a properly implemented token embedding layer with optional weight tying to the output head, supporting the full vocabulary.

#### Acceptance Criteria

1. THE Token_Embedding layer SHALL maintain a learnable embedding matrix E ∈ R^{vocab_size × d_model}
2. THE Token_Embedding layer SHALL map integer token IDs to d_model-dimensional continuous vectors
3. THE Token_Embedding layer SHALL support vocabulary sizes up to at least 256,000 tokens
4. THE Token_Embedding layer SHALL optionally scale embeddings by √d_model (configurable)
5. THE Token_Embedding layer SHALL support optional embedding dropout (applied after embedding lookup)
6. THE Token_Embedding layer SHALL support weight tying with the Output_Head when configured
7. WHEN weight tying is enabled, THE embedding matrix E and output projection W_out SHALL share the same parameter tensor
8. THE Token_Embedding layer SHALL handle padding token IDs correctly (embedding exists but can be masked)

### Requirement 59: Output Head

**User Story:** As a developer, I want the output head to project hidden states to vocabulary logits for next-token prediction.

#### Acceptance Criteria

1. THE Output_Head SHALL apply a linear projection W_out ∈ R^{vocab_size × d_model} to produce logits
2. THE Output_Head SHALL NOT apply softmax (logits are returned raw for loss computation)
3. THE Output_Head SHALL support weight tying with Token_Embedding (sharing W_out = E^T)
4. THE Output_Head SHALL produce output shape (batch_size, seq_len, vocab_size) during training
5. THE Output_Head SHALL produce output shape (batch_size, 1, vocab_size) during single-step inference
6. THE Output_Head SHALL optionally include a bias term (disabled by default)
7. THE Output_Head SHALL be applied after the final normalization layer

### Requirement 60: Sequence Length Handling

**User Story:** As a developer, I want the model to handle variable sequence lengths correctly during both training and inference.

#### Acceptance Criteria

1. THE Library SHALL support variable sequence lengths up to max_seq_len during training
2. THE Library SHALL support unlimited generation length during inference (state-based, no positional limit)
3. THE Library SHALL handle padding correctly: padded positions SHALL not contribute to loss or state updates
4. THE Library SHALL provide attention/padding masks to the loss function to exclude padded positions
5. WHEN sequence length exceeds max_seq_len during training, THE Library SHALL truncate or raise an informative error (configurable)
6. THE Library SHALL support efficient batching of sequences with different lengths via padding and masking
7. THE Library SHALL document that inference has no inherent sequence length limit (constant state regardless of position)

### Requirement 61: Training State Checkpointing Details

**User Story:** As a machine learning engineer, I want comprehensive checkpointing that saves absolutely everything needed to resume training exactly where it left off.

#### Acceptance Criteria

1. THE checkpoint SHALL save: model state_dict (all parameter tensors)
2. THE checkpoint SHALL save: optimizer state_dict (momentum buffers, adaptive learning rates)
3. THE checkpoint SHALL save: learning rate scheduler state
4. THE checkpoint SHALL save: current training step and epoch number
5. THE checkpoint SHALL save: current loss value and loss history
6. THE checkpoint SHALL save: random states (Python, NumPy, PyTorch, CUDA) for reproducibility
7. THE checkpoint SHALL save: gradient scaler state (for mixed precision)
8. THE checkpoint SHALL save: training configuration (USNTrainingConfig)
9. THE checkpoint SHALL save: data loader state (current position in dataset for resumption)
10. WHEN resuming from checkpoint, THE Library SHALL restore ALL saved state and continue training seamlessly
11. THE Library SHALL support periodic checkpointing at configurable step intervals
12. THE Library SHALL support keeping only the N most recent checkpoints to manage disk space
13. THE Library SHALL support saving "best model" checkpoint based on validation loss

### Requirement 62: Data Parallel Correctness

**User Story:** As a developer using distributed training, I want guarantees that data parallelism produces mathematically equivalent results to single-device training.

#### Acceptance Criteria

1. THE Library SHALL ensure that average loss over all devices equals the loss that would be computed on the full global batch
2. THE Library SHALL synchronize gradients correctly before optimizer steps
3. THE Library SHALL handle gradient accumulation correctly in distributed settings (divide by world_size × accumulation_steps)
4. THE Library SHALL ensure all devices have synchronized model parameters after each optimizer step
5. THE Library SHALL support different effective batch sizes: global_batch = per_device_batch × num_devices × accumulation_steps
6. THE Library SHALL seed data samplers with different seeds per rank but same base seed for reproducibility

### Requirement 63: Kernel Fusion Specifications

**User Story:** As a developer targeting maximum GPU throughput, I want specific kernel fusion groups that minimize memory bandwidth usage.

#### Acceptance Criteria

1. THE fused kernel group "projection_gates" SHALL combine: W_u x_t, σ(W_α x_t + b_α) into a single kernel reading x_t once
2. THE fused kernel group "temporal_mix_activation" SHALL combine: α_t ⊙ u_t + (1-α_t) ⊙ u_{t-1} with sigmoid gate computation in one pass
3. THE fused kernel group "selective_write_decay" SHALL combine: exp(-softplus(W_λ x_t + b_λ)), σ(W_g m_t + U_g read(S) + b_g) into a single kernel
4. THE fused kernel group "state_read_confidence" SHALL combine: W_s s_t + W_r vec(R_t), σ(W_c m_t + b_c), c_t ⊙ z_t into a single kernel
5. THE fused kernel group "channel_mlp" SHALL combine: W_1(input), activation φ, W_2(hidden) into a single fused operation
6. WHEN fused kernels are available, THE Library SHALL verify numerical equivalence against unfused reference implementation
7. THE Library SHALL fall back gracefully to unfused operations when Triton or custom kernels are unavailable
8. THE Library SHALL measure and report throughput improvement from fusion vs non-fusion

### Requirement 64: Residual Connection Architecture

**User Story:** As a developer, I want residual connections implemented exactly as specified to maintain gradient flow through deep models.

#### Acceptance Criteria

1. THE Block SHALL implement pre-norm residual architecture: output = x + Block(Norm(x))
2. THE Channel_Mixing SHALL include its own internal residual: y_t = m_t + MLP(c_t ⊙ z_t)
3. THE residual connections SHALL provide a short gradient path from output to input, preventing vanishing gradients in deep models
4. THE residual connections SHALL use addition (not concatenation) to maintain constant dimensionality
5. THE Library SHALL document that the dual residual (block-level + channel-mixing-level) ensures stable gradient flow as described in the paper

### Requirement 65: Activation Functions

**User Story:** As a developer, I want configurable activation functions for the channel mixing MLP with proper defaults.

#### Acceptance Criteria

1. THE Library SHALL support GELU activation function as default for Channel_Mixing
2. THE Library SHALL support SiLU/Swish activation function as an alternative
3. THE Library SHALL support ReLU activation function as a baseline option
4. THE Library SHALL support configuring the activation function via USNConfig.activation ∈ {"gelu", "silu", "relu"}
5. THE activation function SHALL be applied between W_1 and W_2 in the Channel_Mixing MLP: φ(W_1 x)
6. THE Library SHALL implement activation functions using PyTorch's built-in implementations for hardware optimization
7. THE Library SHALL support adding new activation functions via registration without modifying core code

### Requirement 66: Version Compatibility and Migration

**User Story:** As a maintainer, I want the library to handle version evolution gracefully with backward-compatible loading and clear migration paths.

#### Acceptance Criteria

1. THE .usn format SHALL include the library version that created the file
2. THE .usn format SHALL include a format version number independent of library version
3. WHEN loading a .usn file from an older format version, THE Library SHALL automatically migrate it to the current format
4. WHEN loading a .usn file from a newer format version, THE Library SHALL raise a clear error suggesting library upgrade
5. THE Library SHALL maintain a migration registry mapping format versions to migration functions
6. THE Library SHALL log a warning when auto-migrating from an older format version
7. THE USNConfig SHALL include a version field for tracking config schema changes
8. THE Library SHALL follow semantic versioning: MAJOR.MINOR.PATCH with clear upgrade guides for breaking changes

### Requirement 67: Thread Safety and Multiprocessing

**User Story:** As a developer deploying models in production, I want the library to be safe for concurrent use in multi-threaded and multi-process environments.

#### Acceptance Criteria

1. THE Library SHALL be safe for concurrent inference from multiple threads on the same model (read-only operations)
2. THE Library SHALL document that training operations are NOT thread-safe and must be serialized
3. THE Library SHALL support multiprocessing data loading via PyTorch DataLoader(num_workers>0)
4. THE Library SHALL ensure that model state is not corrupted by concurrent inference calls
5. IF state is modified during generation, THEN THE Library SHALL use independent state copies per thread/request
6. THE Library SHALL support spawning worker processes for distributed training without deadlocks

### Requirement 68: Dependency Management

**User Story:** As a developer, I want clear, minimal, and well-documented dependencies so that installation is straightforward and conflicts are avoided.

#### Acceptance Criteria

1. THE Library SHALL declare PyTorch (>=2.0) as the primary runtime dependency
2. THE Library SHALL declare numpy as a runtime dependency
3. THE Library SHALL declare tokenizers (HuggingFace) as a runtime dependency for tokenization
4. THE Library SHALL declare pyyaml for configuration file support
5. THE Library SHALL declare tqdm for progress bars
6. THE Library SHALL declare optional dependencies: triton (for fused kernels), tensorboard (for logging), wandb (for W&B logging), onnx/onnxruntime (for export), safetensors (for export), rich (for CLI formatting)
7. THE Library SHALL work with the minimum declared versions of all dependencies
8. THE Library SHALL pin minimum versions but allow newer compatible versions
9. THE Library SHALL NOT depend on any private, deprecated, or unmaintained packages
10. THE Library SHALL declare development dependencies: pytest, pytest-cov, mypy, ruff, pre-commit, sphinx (docs)

### Requirement 69: Initialization and Model Creation Workflow

**User Story:** As a developer, I want a clear, validated workflow for creating models that catches errors early and provides helpful guidance.

#### Acceptance Criteria

1. WHEN `USNModel(config)` is called, THE Library SHALL validate all config parameters, initialize all submodules, apply the initialization scheme, and report parameter count
2. WHEN `usn.create_model(config)` is called, THE Library SHALL create and return a fully initialized model on the specified device
3. WHEN `usn.create_model(preset="base")` is called with a preset name, THE Library SHALL create a model using the predefined configuration
4. THE Library SHALL log model creation with: parameter count, state size, estimated memory, device
5. IF creation fails due to insufficient memory, THEN THE Library SHALL suggest smaller configurations or memory-saving options
6. THE Library SHALL support creating a model and immediately moving it to a device: `usn.create_model(config, device="cuda")`
7. THE Library SHALL support lazy initialization for deferred device placement in distributed settings

### Requirement 70: Training Resume and Recovery

**User Story:** As a machine learning engineer, I want robust training resumption that handles crashes, interrupted training, and hardware failures gracefully.

#### Acceptance Criteria

1. WHEN `trainer.resume(checkpoint_path)` is called, THE Library SHALL restore all training state and continue seamlessly
2. THE Library SHALL verify checkpoint integrity (checksum) before resuming
3. IF a checkpoint is corrupted, THEN THE Library SHALL attempt to load the previous checkpoint and warn the user
4. THE Library SHALL support automatic checkpoint recovery: if training crashes, the most recent valid checkpoint enables resumption
5. THE Library SHALL log "Resuming training from step X" when resuming from checkpoint
6. THE Library SHALL verify config compatibility: the current config must match the checkpoint config (with allowed overrides like learning_rate)
7. THE Library SHALL support changing certain training parameters on resume (learning rate, batch size) while keeping others fixed (model architecture)

### Requirement 71: Batch Inference Efficiency

**User Story:** As a developer deploying for throughput, I want efficient batch inference that processes multiple sequences simultaneously.

#### Acceptance Criteria

1. THE Library SHALL support batch generation: multiple prompts processed in parallel
2. THE Library SHALL handle sequences of different lengths in a batch via padding and proper masking
3. THE Library SHALL support early stopping per sequence in a batch (when stop token is generated, that sequence is complete but others continue)
4. THE Library SHALL maximize GPU utilization by keeping all batch elements active until the longest sequence completes (or using dynamic batching)
5. THE Library SHALL report per-batch throughput metrics (tokens/second across all sequences)
6. THE Library SHALL support configurable maximum batch size for memory management

### Requirement 72: Streaming Generation

**User Story:** As a developer building interactive applications, I want streaming token generation that yields tokens one at a time with minimal latency.

#### Acceptance Criteria

1. THE USNGenerator SHALL support a streaming mode returning a Python generator/iterator yielding one token at a time
2. WHEN streaming, THE Library SHALL yield each token as soon as it is generated (no buffering of full sequence)
3. THE Library SHALL provide the token text (decoded), token ID, and log-probability for each yielded token
4. THE Library SHALL support async iteration (async generator) for integration with async web frameworks
5. THE Library SHALL support cancellation: stopping generation mid-stream without resource leaks
6. THE streaming interface SHALL maintain O(1) memory per generated token (state-based, no growing buffers)
7. THE Library SHALL measure and report time-to-first-token latency for streaming generation

### Requirement 73: Long Context Handling

**User Story:** As a researcher, I want the model to handle arbitrarily long contexts during inference by virtue of its constant-size state, with no degradation from sequence length.

#### Acceptance Criteria

1. THE Library SHALL process contexts of arbitrary length during inference by iterating through the input and updating state
2. THE Library SHALL maintain constant memory usage regardless of input context length
3. THE Library SHALL support "prefill" mode: process a long context to populate state, then generate from that state
4. THE Library SHALL measure prefill throughput (tokens/second for context processing) separately from generation throughput
5. THE Library SHALL document that context length is limited only by compute time, not memory
6. THE Library SHALL support chunked prefill: process long contexts in chunks to avoid peak memory spikes during prefill
7. THE Library SHALL verify that generation quality does not degrade with increasing context length (state captures relevant information)

### Requirement 74: Model Inspection and Debugging

**User Story:** As a researcher, I want tools to inspect model internals (state values, gate activations, gradient statistics) for understanding model behavior.

#### Acceptance Criteria

1. THE Library SHALL provide hooks to capture intermediate activations at any module during forward pass
2. THE Library SHALL provide `model.get_state()` returning the current state (s_t, R_t) for all layers
3. THE Library SHALL provide visualization utilities for: gate values (α, λ, g, c, ρ) over sequence positions, state norms over time, relational state matrix as heatmap
4. THE Library SHALL provide gradient statistics per parameter: norm, mean, max, min, fraction of zeros
5. THE Library SHALL provide activation statistics: norm, mean, max, min per layer per timestep
6. THE Library SHALL support registering custom hooks on any module for extensible debugging
7. THE Library SHALL provide a `model.named_parameters_with_info()` method returning parameter name, shape, requires_grad, device, dtype, and norm for each parameter

### Requirement 75: Fine-Tuning Support

**User Story:** As a machine learning engineer, I want to fine-tune pretrained USN models on new datasets with support for parameter-efficient methods.

#### Acceptance Criteria

1. THE Library SHALL support full fine-tuning: load pretrained model, continue training on new data with potentially different hyperparameters
2. THE Library SHALL support freezing specific layers: `model.freeze_layers([0, 1, 2])` to keep early layers fixed
3. THE Library SHALL support freezing specific module types: freeze all embeddings, freeze all normalization layers, etc.
4. THE Library SHALL support different learning rates for different parameter groups (e.g., lower LR for pretrained layers, higher for new head)
5. THE Library SHALL support adding a new output head for different vocabulary sizes
6. THE Library SHALL verify config compatibility when loading pretrained weights (d_model, num_layers must match)
7. IF fine-tuning requires extending vocabulary, THEN THE Library SHALL support embedding matrix expansion with random initialization for new tokens
8. THE Library SHALL log fine-tuning configuration: which layers are frozen, parameter count for trainable vs frozen parameters

### Requirement 76: Reproducibility Infrastructure

**User Story:** As a researcher, I want complete reproducibility guarantees with seed management, config logging, and environment tracking.

#### Acceptance Criteria

1. THE Library SHALL provide `usn.set_seed(seed)` setting: random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all
2. THE Library SHALL support enabling PyTorch deterministic mode: torch.use_deterministic_algorithms(True)
3. THE Library SHALL log the full training configuration (model + training + generation configs) at training start
4. THE Library SHALL log environment information: Python version, PyTorch version, CUDA version, GPU model, library version
5. THE Library SHALL save all random states in checkpoints for exact resumption
6. THE Library SHALL support exporting a complete "experiment card" with all information needed to reproduce results
7. THE Library SHALL document known sources of non-determinism and their mitigations

### Requirement 77: Performance Profiling Utilities

**User Story:** As a developer optimizing performance, I want built-in profiling utilities to identify bottlenecks in training and inference.

#### Acceptance Criteria

1. THE Library SHALL provide `usn.utils.profile_forward(model, input)` measuring per-module forward time
2. THE Library SHALL provide `usn.utils.profile_backward(model, input)` measuring per-module backward time
3. THE Library SHALL provide `usn.utils.profile_memory(model, input)` measuring per-module memory allocation
4. THE Library SHALL integrate with PyTorch's profiler (torch.profiler) for detailed GPU kernel analysis
5. THE Library SHALL support exporting profiling data to Chrome trace format for visualization
6. THE Library SHALL identify and report the top-5 time-consuming operations
7. THE Library SHALL measure and report arithmetic intensity (FLOPs / bytes transferred) for key operations
8. THE Library SHALL support profiling both training (forward + backward + optimizer) and inference (forward only) modes

### Requirement 78: Testing Infrastructure

**User Story:** As a developer contributing to the library, I want a well-organized test infrastructure that is easy to run, extend, and interpret.

#### Acceptance Criteria

1. THE test suite SHALL use pytest as the test framework with conftest.py for shared fixtures
2. THE test suite SHALL organize tests into directories mirroring source structure: tests/test_modules/, tests/test_layers/, tests/test_models/, tests/test_training/, tests/test_serialization/, tests/test_inference/, tests/test_utils/, tests/test_config/, tests/test_cli/
3. THE test suite SHALL provide fixtures for common test objects: tiny_config, tiny_model, sample_batch, sample_tokenizer
4. THE test suite SHALL support running subsets via markers: @pytest.mark.slow, @pytest.mark.gpu, @pytest.mark.integration
5. THE test suite SHALL complete fast tests (<5 min) without GPU for CI/CD pipelines
6. THE test suite SHALL include GPU-specific tests skipped when CUDA is not available
7. THE test suite SHALL generate coverage reports via pytest-cov
8. THE test suite SHALL include property-based tests using hypothesis for invariant verification
9. THE test suite SHALL be runnable via: `pytest tests/` (all), `pytest tests/test_modules/` (subset), `pytest -m "not slow"` (fast only)
10. THE test suite SHALL include integration tests that run mini end-to-end workflows (create → train → save → load → generate)

### Requirement 79: Numerical Precision Testing

**User Story:** As a developer, I want tests verifying numerical precision bounds, especially for mixed-precision operations and accumulations.

#### Acceptance Criteria

1. THE test suite SHALL verify that parallel scan output matches sequential output within absolute tolerance of 1e-5 (fp32) and 1e-3 (fp16/bf16)
2. THE test suite SHALL verify that mixed-precision training produces loss values within 1% of full-precision training
3. THE test suite SHALL verify that gate values remain strictly in (0, 1) and never hit exactly 0 or 1 (which would indicate saturation)
4. THE test suite SHALL verify that log-space decay accumulation prevents underflow for long sequences (1000+ steps)
5. THE test suite SHALL test numerical stability with extreme inputs: very large embeddings, very small embeddings, adversarial patterns
6. THE test suite SHALL verify that gradient magnitudes remain in reasonable ranges (not vanishing < 1e-8, not exploding > 1e3) for standard inputs
7. THE test suite SHALL test that repeated state updates with λ close to 1 do not cause float overflow over thousands of steps

### Requirement 80: Synthetic Math Dataset

**User Story:** As a developer validating the implementation, I want a built-in synthetic math dataset generator for quick training validation without external data.

#### Acceptance Criteria

1. THE Library SHALL provide a `MathDataset` class generating arithmetic problems: addition (a+b=c), subtraction (a-b=c), multiplication (a*b=c)
2. THE MathDataset SHALL generate problems with configurable digit count (1-digit, 2-digit, 3-digit, etc.)
3. THE MathDataset SHALL format problems as text strings: "5+3=8", "12*7=84", "100-42=58"
4. THE MathDataset SHALL generate a configurable number of unique problems
5. THE MathDataset SHALL provide train/validation/test splits with no overlap
6. THE MathDataset SHALL include a character-level tokenizer appropriate for math expressions
7. THE MathDataset SHALL be usable directly with USNTrainer without external dependencies
8. THE MathDataset SHALL serve as the validation dataset for the micro-model training test

### Requirement 81: Curriculum Learning Strategy

**User Story:** As a training engineer, I want curriculum learning support that gradually increases sequence length during training for improved convergence.

#### Acceptance Criteria

1. THE Library SHALL support sequence length curriculum: start with shorter sequences, gradually increase to max_seq_len
2. THE Library SHALL provide configurable curriculum schedules: linear increase, step increase, exponential increase
3. THE Library SHALL allow specifying curriculum parameters: start_seq_len, end_seq_len, warmup_steps_for_curriculum
4. THE Library SHALL ensure that batches at each curriculum stage contain only sequences of the current maximum length (padded or truncated)
5. THE Library SHALL log the current curriculum stage and sequence length during training
6. THE Library SHALL save curriculum state in checkpoints for proper resumption
7. THE Library SHALL support disabling curriculum (fixed sequence length throughout) as the default behavior

### Requirement 82: Early Stopping

**User Story:** As a training engineer, I want optional early stopping based on validation metrics to prevent overfitting and save compute.

#### Acceptance Criteria

1. THE Library SHALL support optional early stopping triggered when validation loss does not improve for `patience` consecutive evaluations
2. THE Library SHALL track the best validation loss observed during training
3. THE Library SHALL save the best model checkpoint automatically when validation loss improves
4. WHEN early stopping triggers, THE Library SHALL stop training, log the reason, and restore the best model weights
5. THE Library SHALL support configurable patience (number of evaluations without improvement before stopping)
6. THE Library SHALL support configurable minimum delta (minimum improvement to count as progress)
7. THE Library SHALL support disabling early stopping (default: disabled, train for max_steps)

### Requirement 83: Repetition Penalty for Generation

**User Story:** As a developer generating text, I want repetition penalty support to produce more diverse and natural-sounding outputs.

#### Acceptance Criteria

1. THE Library SHALL support repetition penalty: divide logits of previously generated tokens by a penalty factor > 1.0
2. THE Library SHALL support configurable repetition_penalty parameter (default: 1.0, no penalty)
3. THE Library SHALL track generated token history for penalty computation
4. THE Library SHALL support n-gram blocking: prevent repeating any n-gram that appeared previously
5. THE Library SHALL support frequency penalty: penalize tokens proportional to their frequency in generated text
6. THE Library SHALL support presence penalty: flat penalty for any token that appeared at all in generated text
7. THE repetition penalty SHALL not violate causality (only considers previously generated tokens)

### Requirement 84: Beam Search Implementation

**User Story:** As a developer wanting high-quality generation, I want beam search decoding that maintains multiple hypotheses for optimal output quality.

#### Acceptance Criteria

1. THE Library SHALL implement beam search maintaining `beam_width` active hypotheses at each generation step
2. THE Library SHALL support configurable beam_width (default: disabled, using sampling or greedy)
3. THE Library SHALL implement length penalty to prevent beam search from favoring short sequences: score = log_prob / (length^α) where α is configurable
4. THE Library SHALL support early stopping per beam: stop expanding a beam when EOS token is generated
5. THE Library SHALL return the top-N complete hypotheses ranked by score
6. THE Library SHALL correctly manage model state for each beam (duplicating state for beam expansion)
7. THE Library SHALL support beam search with configurable no_repeat_ngram_size
8. IF all beams generate EOS, THEN THE Library SHALL stop generation early

### Requirement 85: Model Evaluation Metrics

**User Story:** As a researcher, I want standard evaluation metrics computed during validation for assessing model quality.

#### Acceptance Criteria

1. THE Library SHALL compute perplexity (exp(average_cross_entropy_loss)) on validation data
2. THE Library SHALL compute bits-per-character (BPC) when using character-level tokenization
3. THE Library SHALL compute accuracy: fraction of tokens correctly predicted (argmax == target)
4. THE Library SHALL support computing metrics on configurable evaluation datasets
5. THE Library SHALL log all evaluation metrics at each evaluation interval
6. THE Library SHALL support custom metric functions registered by the user
7. THE Library SHALL report metrics per-batch and averaged over the full evaluation set

### Requirement 86: Scalability Validation

**User Story:** As a researcher, I want empirical validation that the library scales linearly with sequence length and that all complexity claims are correct.

#### Acceptance Criteria

1. THE benchmarks SHALL demonstrate O(n) linear time complexity by measuring forward pass time at lengths: 128, 256, 512, 1024, 2048, 4096, 8192 and showing linear fit R² > 0.95
2. THE benchmarks SHALL demonstrate O(1) inference memory by measuring peak memory at generation lengths: 100, 500, 1000, 5000, 10000 and showing constant memory (variance < 5%)
3. THE benchmarks SHALL measure parallel scan speedup over sequential: report parallel_time / sequential_time ratio
4. THE benchmarks SHALL verify that batch processing scales linearly with batch size (tokens/sec proportional to batch_size)
5. THE benchmarks SHALL measure per-layer computation time to identify bottleneck modules
6. THE benchmarks SHALL produce formatted result tables suitable for inclusion in papers

### Requirement 87: Backward Compatibility Testing

**User Story:** As a library maintainer, I want tests ensuring that new versions can load models saved by older versions.

#### Acceptance Criteria

1. THE test suite SHALL include .usn format test fixtures from the initial version
2. THE test suite SHALL verify that the current version loads format v1 files correctly
3. THE test suite SHALL verify that loaded models produce expected outputs for known inputs
4. THE test suite SHALL verify that config deserialization handles missing fields (added in newer versions) gracefully with defaults
5. WHEN a new format version is released, THE test suite SHALL add fixtures for the previous version to the backward compatibility test set
6. THE Library SHALL maintain a format changelog documenting what changed between versions

### Requirement 88: Edge Case Handling

**User Story:** As a developer, I want the library to handle edge cases gracefully: empty inputs, single-token sequences, maximum-length sequences, batch_size=1, and extreme hyperparameters.

#### Acceptance Criteria

1. WHEN input sequence length is 1, THE Library SHALL produce valid output without errors
2. WHEN batch_size is 1, THE Library SHALL produce valid output without broadcasting errors
3. WHEN generating with max_new_tokens=1, THE Library SHALL return a single generated token
4. WHEN all tokens in a batch hit stop tokens at the first step, THE Library SHALL return empty continuations
5. IF vocab_size is very small (e.g., 2), THEN THE Library SHALL still function correctly
6. IF d_model is very small (e.g., 4), THEN THE Library SHALL still function correctly for testing purposes
7. IF num_layers is 1, THEN THE Library SHALL produce a valid model with a single block
8. THE test suite SHALL include parametrized edge case tests covering all the above scenarios
9. WHEN an empty prompt is provided for generation, THE Library SHALL generate from BOS token or raise a clear error

### Requirement 89: API Consistency and Conventions

**User Story:** As a developer, I want consistent API patterns across all library components for predictable behavior.

#### Acceptance Criteria

1. THE Library SHALL follow PyTorch conventions: .forward() for computation, .to(device) for device transfer, .train()/.eval() for mode switching, .parameters() for parameter iteration
2. THE Library SHALL use consistent argument ordering across similar functions: (model, input, config) pattern
3. THE Library SHALL use keyword-only arguments for optional parameters to prevent positional argument errors
4. THE Library SHALL return dataclasses or named tuples for complex return values (not bare tuples)
5. THE Library SHALL use consistent naming: `num_layers` not `n_layers`, `d_model` not `hidden_size` (defined once in glossary, used everywhere)
6. THE Library SHALL provide both functional API (`usn.generate(model, prompt)`) and object-oriented API (`generator.generate(prompt)`)
7. THE Library SHALL ensure all public APIs are importable from the top-level `usn` package

### Requirement 90: Build and CI/CD Support

**User Story:** As a maintainer, I want the library set up for automated testing, building, and publishing.

#### Acceptance Criteria

1. THE Library SHALL include a GitHub Actions CI configuration running: lint, type check, fast tests, slow tests (optional), build, publish
2. THE Library SHALL include a Makefile or equivalent with targets: test, lint, format, typecheck, build, publish, docs, clean
3. THE Library SHALL pass all CI checks before any release
4. THE Library SHALL include a pre-commit configuration with: ruff (lint + format), mypy (type check), pytest (fast tests)
5. THE Library SHALL support building distributions via: python -m build (producing sdist and wheel)
6. THE Library SHALL include proper MANIFEST.in for source distribution completeness
7. THE Library SHALL be publishable to PyPI and TestPyPI
8. THE Library SHALL include release automation scripts or documentation for versioned releases

### Requirement 91: Security Considerations

**User Story:** As a developer loading models from untrusted sources, I want security measures preventing malicious model files from causing harm.

#### Acceptance Criteria

1. THE .usn format SHALL NOT use pickle for serialization (pickle allows arbitrary code execution)
2. THE .usn format SHALL store tensors as raw numerical data with explicit dtype and shape metadata
3. THE Library SHALL validate all loaded data against expected schemas before using it
4. THE Library SHALL validate tensor shapes and dtypes against the config before loading weights
5. THE Library SHALL set maximum file size limits to prevent memory exhaustion from malicious files
6. THE Library SHALL sanitize all string metadata loaded from .usn files
7. IF loading detects inconsistency between config and weights, THEN THE Library SHALL refuse to load and report the discrepancy

### Requirement 92: Extensibility and Plugin Architecture

**User Story:** As a researcher, I want to extend the library with custom modules, losses, schedulers, and datasets without modifying core code.

#### Acceptance Criteria

1. THE Library SHALL support registering custom activation functions via a registry pattern
2. THE Library SHALL support registering custom schedulers via a registry pattern
3. THE Library SHALL support registering custom loss functions via a registry pattern
4. THE Library SHALL support subclassing USNModel for custom architectures while reusing the training infrastructure
5. THE Library SHALL support custom datasets by implementing the PyTorch Dataset interface
6. THE Library SHALL support custom tokenizers by implementing the tokenizer interface
7. THE Library SHALL use dependency injection for swappable components (optimizer, scheduler, loss, dataset)
8. THE Library SHALL document extension points and provide examples for each

### Requirement 93: USN State Dimension Relationships

**User Story:** As a researcher configuring models, I want clear documentation and validation of how state dimensions relate to model dimensions.

#### Acceptance Criteria

1. THE Library SHALL validate that d_s (semantic state dimension) is a positive integer, typically <= d_model
2. THE Library SHALL validate that k (relational state dimension) is a positive integer, noting that state memory is k² × num_layers
3. THE Library SHALL document the total state memory per sequence: num_layers × (d_s + k²) × sizeof(float)
4. THE Library SHALL provide guidance on balancing d_s vs k: larger d_s for feature-rich memory, larger k for rich relational structure
5. THE Library SHALL warn if k² > d_model (relational readout projection becomes larger than standard projections)
6. THE Library SHALL provide recommended d_s and k values for each preset configuration based on scaling analysis
7. THE Library SHALL validate that total state size is computationally feasible for the target hardware

### Requirement 94: End-to-End Integration Testing

**User Story:** As a developer, I want integration tests that exercise the full workflow from model creation through training to generation, proving all components work together.

#### Acceptance Criteria

1. THE test suite SHALL include an integration test: create tiny model → train 100 steps on synthetic data → verify loss decreased → save model → load model → generate text → verify output is valid tokens
2. THE test suite SHALL include an integration test: create model → export to ONNX → load ONNX model → verify output matches
3. THE test suite SHALL include an integration test: create model → export to SafeTensors → reload → verify output matches
4. THE test suite SHALL include an integration test: run CLI train command → verify checkpoint exists → run CLI generate command → verify output
5. THE test suite SHALL include an integration test: distributed training with 2 simulated workers → verify convergence
6. THE test suite SHALL include an integration test: fine-tune pretrained model → verify improvement on target task
7. THE test suite SHALL include an integration test: streaming generation → verify all tokens are valid and state is maintained
8. THE integration tests SHALL be marked with @pytest.mark.integration for selective execution

### Requirement 95: Softplus and Mathematical Function Implementations

**User Story:** As a developer, I want all mathematical building blocks implemented correctly and documented, especially those critical to stability guarantees.

#### Acceptance Criteria

1. THE Library SHALL use PyTorch's F.softplus for the softplus function: softplus(x) = ln(1 + exp(x))
2. THE Library SHALL handle softplus numerical stability: for large x, softplus(x) ≈ x (avoid exp overflow)
3. THE Library SHALL use torch.sigmoid for all sigmoid activations: σ(x) = 1/(1 + exp(-x))
4. THE Library SHALL verify the exp(-softplus(x)) composition: for all real x, exp(-softplus(x)) ∈ (0, 1)
5. THE Library SHALL use log-space computation: log(λ_1 × λ_2 × ... × λ_t) = Σ log(λ_i) to avoid underflow in decay products
6. THE Library SHALL document all mathematical identities and stability tricks used in the implementation
7. THE Library SHALL include unit tests verifying mathematical properties: exp(-softplus(x)) ∈ (0,1), σ(x) ∈ (0,1), softplus(x) > 0

### Requirement 96: Documentation of Design Decisions

**User Story:** As a contributor, I want documentation explaining WHY each design decision was made, not just what was implemented.

#### Acceptance Criteria

1. THE documentation SHALL explain why RMSNorm is preferred over LayerNorm (computational efficiency, similar performance)
2. THE documentation SHALL explain why pre-norm (Norm before Block) is used over post-norm (better gradient flow in deep models)
3. THE documentation SHALL explain why exp(-softplus(·)) was chosen for decay (smooth, bounded, differentiable, learnable)
4. THE documentation SHALL explain why the relational state uses an outer product (captures second-order feature interactions with O(k²) memory)
5. THE documentation SHALL explain why the confidence gate c_t is needed (prevents noisy state readout from corrupting output)
6. THE documentation SHALL explain why the temporal mixing uses only 1-step lookback (keeps it causal and O(1) memory)
7. THE documentation SHALL explain why the parallel scan works (state transition is affine → composition is affine → associative)
8. THE documentation SHALL explain why chunk-based decomposition is used (GPU parallelism within chunks, sequential only between chunks)
9. THE documentation SHALL explain why there is no attention mechanism (O(n²) memory/compute cost, USN achieves comparable quality with O(n) via persistent state)
10. THE documentation SHALL be included as a "Design Rationale" section in the main documentation

### Requirement 97: Performance Comparison Methodology

**User Story:** As a researcher, I want clear methodology for comparing USN performance against baselines, so that benchmark results are meaningful and reproducible.

#### Acceptance Criteria

1. THE benchmarks SHALL document hardware specifications: GPU model, memory, driver version, CUDA version, PyTorch version
2. THE benchmarks SHALL report warm-up runs (discarded) and measurement runs separately
3. THE benchmarks SHALL report mean, standard deviation, min, and max over multiple runs (at least 10 measurement runs)
4. THE benchmarks SHALL use identical batch sizes, sequence lengths, and dtypes across compared configurations
5. THE benchmarks SHALL measure wall-clock time (not just GPU time) for realistic throughput numbers
6. THE benchmarks SHALL document any compile/JIT warmup required before measurement
7. THE benchmarks SHALL be runnable via a single command: `usn benchmark --all` for full benchmark suite
8. THE benchmarks SHALL produce machine-readable output (JSON) for automated analysis

### Requirement 98: Model Parameter Counting and Memory Estimation

**User Story:** As a researcher planning experiments, I want accurate parameter counting and memory estimation for any configuration.

#### Acceptance Criteria

1. THE Library SHALL compute total parameter count: embedding + N×(block_params) + output_head
2. THE Library SHALL compute per-block parameter count: projection_params + gate_params + state_params + MLP_params + norm_params
3. THE Library SHALL estimate training memory: parameters × sizeof(dtype) + optimizer_states × sizeof(float32) + activations + gradients
4. THE Library SHALL estimate inference memory: parameters × sizeof(dtype) + state × sizeof(dtype)
5. THE Library SHALL account for weight tying in parameter count (shared params counted once)
6. THE Library SHALL account for gradient checkpointing in memory estimation (reduced activation memory)
7. THE Library SHALL provide breakdown by component: "Embeddings: X params, Blocks: Y params, Output: Z params"
8. THE Library SHALL validate computed parameter count against actual model.parameters() count

### Requirement 99: Complete __init__.py Structure

**User Story:** As a developer using the library, I want clean imports from the top-level package without needing to know internal module paths.

#### Acceptance Criteria

1. THE top-level usn/__init__.py SHALL export: USNModel, USNConfig, USNTrainer, USNGenerator, USNTrainingConfig, USNGenerationConfig
2. THE top-level usn/__init__.py SHALL export: create_model, train, generate, save, load, export, summary, benchmark, from_pretrained, set_seed
3. THE top-level usn/__init__.py SHALL export: __version__, __author__
4. EACH subpackage __init__.py SHALL export its public API symbols
5. THE Library SHALL support: `from usn import USNModel, USNConfig` (top-level imports)
6. THE Library SHALL support: `from usn.modules import InputProjection, TemporalMixing` (subpackage imports)
7. THE Library SHALL define __all__ in each __init__.py to control wildcard imports
8. THE Library SHALL NOT have circular import issues with any import pattern

### Requirement 100: Final Deliverable Completeness

**User Story:** As the author, I want absolute confirmation that every single element from the paper and specification is implemented, documented, tested, and validated.

#### Acceptance Criteria

1. THE Library SHALL pass all unit tests with 0 failures
2. THE Library SHALL pass all integration tests with 0 failures
3. THE Library SHALL pass mypy strict type checking with 0 errors
4. THE Library SHALL pass ruff linting with 0 warnings
5. THE Library SHALL have 95%+ code coverage
6. THE Library SHALL have the PAPER_VALIDATION.md checklist with 100% items marked ✅ Implemented
7. THE Library SHALL have complete documentation covering all sections specified in Requirement 38
8. THE Library SHALL have the scalability table with 20+ configurations (Requirement 31)
9. THE Library SHALL have the micro-model training validation passing (Requirement 32)
10. THE Library SHALL have all benchmarks documented with results (Requirement 33)
11. THE Library SHALL be installable via pip install from the built distribution
12. THE Library SHALL have all examples runnable without errors
13. THE Library SHALL have no TODO, FIXME, or HACK comments in released code
14. THE Library SHALL contain no placeholder implementations or stub functions
15. THE Library SHALL be consistent from start to finish: naming, style, patterns, and quality level uniform throughout

### Requirement 101: Triton Fused Kernels - Combined Input Projection (GEMM Fused)

**User Story:** As a developer targeting maximum throughput, I want the three input projection operations (W_u, W_α, W_λ) fused into a single GPU kernel that reads x_t from VRAM only once, reducing kernel launch overhead and memory bandwidth by 3x.

#### Acceptance Criteria

1. THE Library SHALL implement a fused GEMM kernel that concatenates W_u, W_α, W_λ into a single weight matrix W_gate = [W_u; W_α; W_λ] ∈ R^{(d_model + d_model + d_s) × d_model}
2. THE fused kernel SHALL compute [u_t, α̃_t, λ̃_t] = x_t W_gate^T + b_gate in a single matrix multiplication
3. THE fused kernel SHALL read input tensor x_t from VRAM exactly once (not three separate loads)
4. THE fused kernel SHALL produce three output slices: u_t for input projection, α̃_t (pre-sigmoid) for temporal mixing gate, λ̃_t (pre-softplus) for decay gate
5. THE fused kernel SHALL reduce kernel launch count from 3 to 1 for the projection stage
6. THE fused kernel SHALL be implemented using Triton JIT compilation when Triton is available
7. WHEN Triton is not available, THE Library SHALL fall back to a PyTorch-native concatenated matmul implementation that achieves the same logical fusion via torch.mm on the combined weight matrix
8. THE fused kernel SHALL produce numerically identical results to the three separate projections (verified by tests)
9. THE Library SHALL measure and document VRAM bandwidth savings from this fusion (expected ~3x reduction in input reads)
10. THE Library SHALL support this kernel in both training (full sequence) and inference (single step) modes

### Requirement 102: Triton Fused Kernels - Temporal Mix + Exponential Gate (Pointwise Fused)

**User Story:** As a developer, I want a pointwise fusion kernel that computes sigmoid, temporal interpolation, and exponential gating entirely in GPU registers/SRAM without writing intermediate values to VRAM.

#### Acceptance Criteria

1. THE Library SHALL implement a fused pointwise kernel that receives u_t, α̃_t, λ̃_t, and u_{t-1} as inputs
2. THE fused kernel SHALL compute α_t = σ(α̃_t) entirely in registers (no VRAM write for intermediate sigmoid result)
3. THE fused kernel SHALL compute m_t = α_t ⊙ u_t + (1-α_t) ⊙ u_{t-1} entirely in registers
4. THE fused kernel SHALL compute λ_t = exp(-softplus(λ̃_t)) entirely in registers (no VRAM write for softplus intermediate)
5. THE fused kernel SHALL write ONLY the final results m_t and λ_t to VRAM (zero intermediate materializations)
6. THE fused kernel SHALL process all elements in parallel across the batch and feature dimensions
7. WHEN Triton is not available, THE Library SHALL fall back to a PyTorch implementation using torch.compile or manual fusion via a custom autograd function that minimizes intermediate allocations
8. THE fused kernel SHALL be numerically identical to the sequential computation (verified by tests)
9. THE Library SHALL document memory bandwidth savings: eliminates at least 4 intermediate tensor writes/reads (σ output, softplus output, 1-α intermediate, interpolation intermediate)
10. THE fused kernel SHALL be numerically stable: softplus clamped for large inputs (softplus(x) ≈ x for x > 20), exp(-softplus) computed in a stable manner

### Requirement 103: Triton Fused Kernels - State Update Core (The "Heart of USN" Kernel)

**User Story:** As a developer, I want the most performance-critical kernel to fuse the entire state update, readout, and confidence gate within GPU shared memory (SRAM), processing entire chunks without intermediate VRAM writes for state.

#### Acceptance Criteria

1. THE Library SHALL implement a fused state-core kernel that performs the entire intra-chunk state computation in SRAM: state update (semantic + relational), readout, and confidence gating
2. THE fused kernel SHALL load the initial chunk state S_0 = (s_0, R_0) into GPU shared memory (SRAM) at chunk start
3. THE fused kernel SHALL iterate through chunk positions (C steps, e.g., 64 or 128) entirely within SRAM for state variables
4. THE fused kernel SHALL compute at each step within SRAM: s_t = λ_t ⊙ s_{t-1} + g_t ⊙ (B_s m_t) without writing s_t to VRAM between steps
5. THE fused kernel SHALL compute at each step within SRAM: R_t = ρ_t R_{t-1} + (B_r m_t)(C_r m_t)^T without writing R_t to VRAM between steps
6. THE fused kernel SHALL compute readout within the same kernel: z_t = W_s s_t + W_r vec(R_t), o_t = σ(W_c m_t + b_c) ⊙ z_t
7. THE fused kernel SHALL write to VRAM ONLY: the output sequence o_1:C and the final chunk state S_C (for inter-chunk propagation)
8. THE fused kernel SHALL exploit that R_t ∈ R^{k×k} with small k (16-32) fits entirely in SRAM (k²×4 bytes ≈ 1-4 KB)
9. THE fused kernel SHALL support configurable chunk sizes (32, 64, 128) tuned for GPU shared memory capacity
10. WHEN Triton is not available, THE Library SHALL fall back to a PyTorch implementation using chunked processing with torch.compile hints for fusion
11. THE Library SHALL provide a CPU fallback that performs the same chunked sequential computation without kernel fusion
12. THE fused kernel SHALL produce results identical to the unfused reference implementation (verified by tests comparing output tensors)
13. THE Library SHALL measure and report: VRAM bandwidth reduction (expected >10x for state variables within chunk), kernel occupancy, throughput improvement vs unfused
14. THE fused kernel SHALL be differentiable: implement custom backward pass that recomputes forward states from chunk boundaries (gradient checkpointing within the fused kernel)

### Requirement 104: Triton Fused Kernels - Channel MLP Fusion

**User Story:** As a developer, I want the channel mixing MLP (W_1 → activation → W_2) fused to minimize intermediate activation materializations.

#### Acceptance Criteria

1. THE Library SHALL implement a fused MLP kernel combining: linear projection W_1, activation function φ, and linear projection W_2
2. THE fused kernel SHALL avoid materializing the full intermediate d_ff-dimensional activation in VRAM by processing in tiles
3. THE fused kernel SHALL support GELU and SiLU activations within the fused path
4. THE fused kernel SHALL process the MLP in tiles: compute W_1 for a tile of d_ff, apply activation, multiply by corresponding W_2 columns, accumulate result
5. WHEN Triton is not available, THE Library SHALL fall back to torch.compile-optimized sequential MLP or memory-efficient implementation via gradient checkpointing of the activation
6. THE fused kernel SHALL produce numerically identical results to the unfused MLP computation
7. THE Library SHALL measure memory savings: eliminates materializing the full (batch×seq_len×d_ff) intermediate tensor

### Requirement 105: Kernel Fallback Strategy and Acceleration Hierarchy

**User Story:** As a developer running on diverse hardware, I want a hierarchical fallback strategy that always uses the fastest available acceleration, never gets stuck, and gracefully degrades.

#### Acceptance Criteria

1. THE Library SHALL implement a 4-level acceleration hierarchy: Level 1 (fastest): Custom Triton kernels → Level 2: torch.compile with inductor backend → Level 3: Custom autograd functions with minimized allocations → Level 4 (baseline): Standard PyTorch eager execution
2. WHEN Triton is installed and CUDA is available, THE Library SHALL use Level 1 (Triton fused kernels) by default
3. WHEN Triton is NOT available but CUDA is available, THE Library SHALL fall back to Level 2 (torch.compile) automatically
4. WHEN torch.compile fails or is unavailable (older PyTorch), THE Library SHALL fall back to Level 3 (optimized eager)
5. WHEN running on CPU or unsupported hardware, THE Library SHALL use Level 4 (standard PyTorch) with no errors
6. THE Library SHALL detect the available acceleration level at import time and log it: "USN acceleration: Triton fused kernels (Level 1)"
7. THE Library SHALL support manual override: `usn.set_acceleration_level(level)` for debugging or benchmarking purposes
8. THE Library SHALL verify at each level that outputs match the reference (Level 4) implementation within tolerance
9. THE Library SHALL NEVER fail or crash due to unavailable acceleration — graceful degradation is mandatory
10. THE Library SHALL provide `usn.benchmark_acceleration()` comparing throughput across available levels
11. THE Library SHALL support different acceleration levels for training vs inference independently
12. THE Library SHALL document expected speedup at each level relative to baseline for reference configurations

### Requirement 106: Advanced Training Stability - Anti-NaN and Anti-Divergence System

**User Story:** As a training engineer, I want a comprehensive stability system that proactively prevents NaN losses, gradient explosion, loss divergence, and non-learning states, ensuring the model always trains stably regardless of data or hyperparameter choices.

#### Acceptance Criteria

1. THE Library SHALL implement a multi-layer normalization strategy: (a) Pre-block RMSNorm/LayerNorm for activation scale control, (b) Post-state-update normalization of s_t and R_t when magnitudes exceed configurable thresholds, (c) Pre-output normalization before the Output_Head
2. THE Library SHALL implement state magnitude monitoring: at each training step, check ‖s_t‖ and ‖R_t‖_F against configurable maximum thresholds (default: 1000.0)
3. IF state magnitude exceeds the threshold, THEN THE Library SHALL apply emergency state clipping: s_t = s_t × (max_norm / ‖s_t‖) and similarly for R_t
4. THE Library SHALL implement gradient norm monitoring with configurable alert thresholds: warn if grad_norm > 10×typical, error if grad_norm > 100×typical
5. THE Library SHALL implement NaN/Inf detection at multiple checkpoints during forward pass: after state update, after readout, after MLP, after output logits
6. IF NaN is detected during training, THEN THE Library SHALL: (a) log which layer and operation produced NaN, (b) log the training step number, (c) optionally skip the corrupted batch and continue, (d) optionally revert to the last valid checkpoint
7. THE Library SHALL implement loss spike detection: if loss increases by more than 5× the running average, trigger a configurable response (skip batch, reduce learning rate, or log warning)
8. THE Library SHALL implement adaptive gradient clipping as an option: clip gradients based on the ratio of gradient norm to parameter norm (AGC - Adaptive Gradient Clipping from NFNet)
9. THE Library SHALL implement a "stability mode" flag that enables all protective measures simultaneously for safety-critical training runs
10. THE Library SHALL implement learning rate warmup that starts from a very small value (1e-7) and linearly increases to target LR over warmup_steps — this prevents early instability
11. THE Library SHALL implement configurable weight norm monitoring: log and optionally constrain maximum weight magnitudes per layer
12. THE Library SHALL implement residual scale monitoring: ensure residual connections do not cause activation growth through depth (monitor per-block output norms)

### Requirement 107: State Normalization for Long-Sequence Stability

**User Story:** As a researcher training on long sequences, I want the persistent state to remain numerically stable over thousands of timesteps even when decay factors are close to 1, preventing accumulated numerical drift.

#### Acceptance Criteria

1. THE Library SHALL implement optional periodic state normalization: every N steps (configurable), normalize s_t to unit norm or bounded norm
2. THE Library SHALL implement a state norm constraint during training: IF ‖s_t‖ > max_state_norm, THEN scale s_t to have norm = max_state_norm (soft constraint via clamp, hard constraint via projection)
3. THE Library SHALL implement Frobenius norm constraint for relational state: IF ‖R_t‖_F > max_R_norm, THEN scale R_t proportionally
4. THE Library SHALL implement decay factor lower bound: ensure λ_t is not too close to 1 during early training (optional λ_max clamp, e.g., 0.999) to prevent slow state washout
5. THE Library SHALL implement state initialization noise: optionally add small noise to initial state to break symmetry and prevent degenerate solutions
6. THE Library SHALL implement a "state health" diagnostic: `model.check_state_health()` reporting per-layer state norms, decay factor statistics, and flagging potential issues
7. THE Library SHALL implement configurable RMSNorm applied to state vectors optionally at each step (inside the state update loop) — disabled by default but available for stability-critical applications
8. THE Library SHALL document the mathematical analysis of state accumulation bounds: given λ_max and bounded inputs, derive the theoretical maximum state norm and show it is finite
9. THE Library SHALL implement per-layer state statistics logging during training (mean, std, max of ‖s_t‖ and ‖R_t‖_F) at configurable intervals for monitoring
10. THE Library SHALL support "safe mode" training that enables all state normalization measures, suitable for initial experimentation before tuning is done

### Requirement 108: Loss Landscape Smoothing and Training Diagnostics

**User Story:** As a training engineer, I want diagnostic tools and loss smoothing techniques that help identify and prevent common training pathologies specific to state-based models.

#### Acceptance Criteria

1. THE Library SHALL implement exponential moving average (EMA) of model weights as an optional stabilization technique
2. THE Library SHALL implement loss smoothing with configurable window for trend detection (separate from raw loss logging)
3. THE Library SHALL provide training diagnostics detecting: (a) loss plateau (no improvement for N steps), (b) loss oscillation (high variance in loss over window), (c) gradient vanishing (grad_norm < threshold for N steps), (d) gate saturation (gates stuck at 0 or 1)
4. THE Library SHALL detect gate saturation: if mean(g_t) < 0.01 or mean(g_t) > 0.99 for N consecutive steps, log a warning "Write gate saturated — model may not be learning/updating state"
5. THE Library SHALL detect decay saturation: if mean(λ_t) > 0.999 for N steps, log warning "Decay nearly 1 — state washing out slowly, effective memory may be very long"
6. THE Library SHALL detect dead state dimensions: if variance(s_t[i]) < epsilon across a batch for specific dimension i, log warning "State dimension i appears dead"
7. THE Library SHALL provide `trainer.diagnose()` method that runs all diagnostic checks and produces a summary report
8. THE Library SHALL support optional automatic interventions: if loss plateau detected, optionally increase learning rate; if gradient vanishing detected, optionally adjust initialization
9. THE Library SHALL log all diagnostic events with step number, severity (INFO/WARNING/CRITICAL), and recommended action
10. THE Library SHALL implement a training "health score" summarizing overall stability (0-100 scale) based on gradient norms, loss trend, state norms, and gate statistics
