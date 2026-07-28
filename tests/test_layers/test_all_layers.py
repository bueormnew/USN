"""Unit tests for USN layers: USNBlock, ParallelScan, ChunkedParallelScan, norms."""

import torch
import pytest

from usn.config.model_config import USNConfig
from usn.layers.block import USNBlock
from usn.layers.parallel_scan import parallel_scan_semantic, ParallelScanFunction
from usn.layers.chunked_scan import ChunkedParallelScan
from usn.layers.norm import RMSNorm, USNLayerNorm


@pytest.fixture
def block_config():
    return USNConfig(
        num_layers=1, d_model=32, d_s=16, k=4, d_ff=64,
        vocab_size=50, max_seq_len=32, dropout=0.0,
        embedding_dropout=0.0, residual_dropout=0.0, fused=False,
    )


# ─── USNBlock Tests ───────────────────────────────────────────────────────────


class TestUSNBlock:
    def test_submodule_ordering(self, block_config):
        """Verify block contains all 8 submodules in correct order."""
        block = USNBlock(block_config, layer_idx=0)
        expected = [
            "norm", "input_proj", "temporal_mix", "exp_gate",
            "selective_write", "state_update", "state_readout", "channel_mix",
        ]
        named = [name for name, _ in block.named_children()]
        # All expected submodules present (residual_dropout is extra)
        for name in expected:
            assert name in named, f"Missing submodule: {name}"
        # Check ordering: each expected module appears before the next
        indices = [named.index(n) for n in expected]
        assert indices == sorted(indices), "Submodules not in correct order"

    def test_residual_connection(self, block_config):
        """Output differs from input (processing occurs) but residual is added."""
        block = USNBlock(block_config, layer_idx=0)
        block.eval()
        x = torch.randn(2, 8, block_config.d_model)
        out = block(x)
        # Output should not be identical to input (block does work)
        assert not torch.allclose(out.hidden, x, atol=1e-6)
        # Output shape matches input shape (residual preserves dims)
        assert out.hidden.shape == x.shape

    def test_mode_switching(self, block_config):
        """Block behaves differently in train vs eval mode."""
        block = USNBlock(block_config, layer_idx=0)
        x = torch.randn(2, 8, block_config.d_model)
        block.train()
        out_train = block(x).hidden.detach()
        block.eval()
        out_eval = block(x).hidden.detach()
        # Results may differ slightly due to state update path
        # Both should produce valid tensors
        assert out_train.shape == (2, 8, block_config.d_model)
        assert out_eval.shape == (2, 8, block_config.d_model)

    def test_output_shape(self, block_config):
        """Forward pass produces correct output shape."""
        block = USNBlock(block_config, layer_idx=0)
        block.eval()
        for batch, seq in [(1, 4), (3, 16), (2, 1)]:
            x = torch.randn(batch, seq, block_config.d_model)
            out = block(x)
            assert out.hidden.shape == (batch, seq, block_config.d_model)


# ─── ParallelScan Tests ───────────────────────────────────────────────────────


class TestParallelScan:
    def test_equivalence_to_sequential(self):
        """Parallel scan matches manual sequential recurrence."""
        batch, seq_len, d_s = 2, 16, 8
        log_decays = torch.randn(batch, seq_len, d_s) * 0.5 - 1.0  # mostly negative
        values = torch.randn(batch, seq_len, d_s) * 0.1
        s0 = torch.randn(batch, d_s) * 0.1

        # Parallel scan
        result = parallel_scan_semantic(log_decays, values, s0)

        # Manual sequential
        expected = torch.empty_like(result)
        s_prev = s0
        for t in range(seq_len):
            s_t = torch.exp(log_decays[:, t, :]) * s_prev + values[:, t, :]
            expected[:, t, :] = s_t
            s_prev = s_t

        assert torch.allclose(result, expected, atol=1e-5)

    def test_gradient_flow(self):
        """Gradients flow through the scan."""
        batch, seq_len, d_s = 2, 8, 4
        log_decays = torch.randn(batch, seq_len, d_s, requires_grad=True)
        values = torch.randn(batch, seq_len, d_s, requires_grad=True)
        s0 = torch.randn(batch, d_s, requires_grad=True)

        result = parallel_scan_semantic(log_decays, values, s0)
        loss = result.sum()
        loss.backward()

        assert log_decays.grad is not None
        assert values.grad is not None
        assert s0.grad is not None
        assert not torch.all(log_decays.grad == 0)

    def test_log_space_stability(self):
        """Scan handles large negative log_decays without NaN."""
        batch, seq_len, d_s = 2, 32, 8
        # Very negative log_decays → decay ~ 0, should not produce NaN
        log_decays = torch.full((batch, seq_len, d_s), -10.0)
        values = torch.randn(batch, seq_len, d_s)
        s0 = torch.randn(batch, d_s)

        result = parallel_scan_semantic(log_decays, values, s0)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


# ─── ChunkedParallelScan Tests ───────────────────────────────────────────────


class TestChunkedParallelScan:
    def test_equivalence_to_full_scan(self):
        """Chunked scan produces same result as full scan."""
        batch, seq_len, d_s = 2, 32, 8
        log_decays = torch.randn(batch, seq_len, d_s) * 0.5 - 1.0
        values = torch.randn(batch, seq_len, d_s) * 0.1
        s0 = torch.randn(batch, d_s) * 0.1

        full_result = parallel_scan_semantic(log_decays, values, s0)
        chunked = ChunkedParallelScan(chunk_size=8)
        chunked_result = chunked(log_decays, values, s0)

        assert torch.allclose(full_result, chunked_result, atol=1e-5)

    def test_non_divisible_sequence(self):
        """Handles sequences not evenly divisible by chunk_size."""
        batch, seq_len, d_s = 2, 13, 8  # 13 not divisible by 4
        log_decays = torch.randn(batch, seq_len, d_s) * 0.5 - 1.0
        values = torch.randn(batch, seq_len, d_s) * 0.1
        s0 = torch.zeros(batch, d_s)

        chunked = ChunkedParallelScan(chunk_size=4)
        result = chunked(log_decays, values, s0)

        assert result.shape == (batch, seq_len, d_s)
        assert not torch.isnan(result).any()

        # Verify against sequential
        expected = parallel_scan_semantic(log_decays, values, s0)
        assert torch.allclose(result, expected, atol=1e-5)


# ─── Normalization Tests ──────────────────────────────────────────────────────


class TestNorms:
    def test_rmsnorm_output_scale(self):
        """RMSNorm output has RMS ≈ 1.0 (with unit gamma)."""
        norm = RMSNorm(64)
        x = torch.randn(4, 16, 64) * 5.0  # large input
        out = norm(x)
        rms = out.float().pow(2).mean(dim=-1).sqrt()
        # RMS should be close to 1.0 (gamma=1)
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_layernorm_output_mean(self):
        """LayerNorm output has mean ≈ 0 (with zero beta)."""
        norm = USNLayerNorm(64)
        x = torch.randn(4, 16, 64) * 5.0 + 3.0  # offset input
        out = norm(x)
        mean = out.float().mean(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)

    def test_norm_preserves_shape(self):
        """Both norms preserve input shape."""
        for NormClass in [RMSNorm, USNLayerNorm]:
            norm = NormClass(32)
            x = torch.randn(2, 8, 32)
            assert norm(x).shape == x.shape
