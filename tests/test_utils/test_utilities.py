"""Tests for utility functions: count_parameters, set_seed, timer, memory_tracker."""

import pytest
import torch
import torch.nn as nn

from usn.utils import count_parameters, set_seed
from usn.utils.timing import MemoryResult, TimerResult, memory_tracker, timer


class TestCountParameters:
    """Test parameter counting utility."""

    def test_counts_all_parameters(self):
        model = nn.Linear(10, 5, bias=True)
        # weight: 10*5=50, bias: 5 => total 55
        assert count_parameters(model) == 55

    def test_trainable_only_excludes_frozen(self):
        model = nn.Linear(10, 5, bias=True)
        model.weight.requires_grad_(False)
        assert count_parameters(model, trainable_only=True) == 5


class TestSetSeed:
    """Test that set_seed produces deterministic results."""

    def test_same_seed_produces_same_tensor(self):
        set_seed(123)
        a = torch.randn(4, 4)
        set_seed(123)
        b = torch.randn(4, 4)
        assert torch.equal(a, b)

    def test_different_seeds_produce_different_tensors(self):
        set_seed(1)
        a = torch.randn(4, 4)
        set_seed(2)
        b = torch.randn(4, 4)
        assert not torch.equal(a, b)

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError):
            set_seed(-1)


class TestTimer:
    """Test the timer context manager."""

    def test_timer_records_positive_elapsed(self):
        with timer("test_block") as t:
            _ = sum(range(1000))
        assert isinstance(t, TimerResult)
        assert t.elapsed_seconds > 0
        assert t.name == "test_block"

    def test_elapsed_ms_property(self):
        with timer("ms_test") as t:
            _ = sum(range(100))
        assert t.elapsed_ms == t.elapsed_seconds * 1000.0


class TestMemoryTracker:
    """Test the memory_tracker context manager (CPU fallback)."""

    def test_cpu_tracker_returns_zeroed_result(self):
        # On CPU, memory tracker returns zeros gracefully
        with memory_tracker(device="cpu") as mem:
            _ = torch.randn(100, 100)
        assert isinstance(mem, MemoryResult)
        assert mem.peak_allocated_bytes == 0
