# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-01

### Added

- Initial project structure and packaging
- Core type definitions (UnifiedState, ModelState, BlockOutput, etc.)
- USN modules: InputProjection, TemporalMixing, ExponentialGating, SelectiveWriting, StateUpdate, StateReadout, ChannelMixing
- USNBlock layer with pre-norm residual architecture
- Parallel scan and chunked parallel scan implementations
- USNModel with embedding, blocks, and output head
- Configuration system with presets (tiny through xxl)
- Training system with mixed precision, curriculum, and distributed support
- Inference system with greedy, top-k, top-p, and beam search decoding
- `.usn` single-file serialization format
- Triton fused kernels with PyTorch fallbacks
- CLI interface (train, generate, benchmark, info, export, validate)
- Comprehensive test suite with property-based tests
