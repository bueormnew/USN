"""Tests for usn.backends: DeviceDetector and AccelerationManager."""

import pytest
import torch

from usn.backends import AccelerationLevel, AccelerationManager, DeviceDetector


class TestDeviceDetector:
    """Tests for DeviceDetector hardware detection."""

    def test_detect_returns_dict(self):
        info = DeviceDetector.detect()
        assert isinstance(info, dict)

    def test_detect_has_required_keys(self):
        info = DeviceDetector.detect()
        required_keys = {
            "device",
            "cuda_available",
            "mps_available",
            "gpu_name",
            "gpu_memory",
            "compute_capability",
            "cuda_version",
            "device_count",
        }
        assert required_keys.issubset(info.keys())

    def test_detect_device_is_valid_string(self):
        info = DeviceDetector.detect()
        assert info["device"] in ("cuda", "mps", "cpu")

    def test_detect_cuda_available_is_bool(self):
        info = DeviceDetector.detect()
        assert isinstance(info["cuda_available"], bool)

    def test_detect_mps_available_is_bool(self):
        info = DeviceDetector.detect()
        assert isinstance(info["mps_available"], bool)

    def test_detect_device_count_is_int(self):
        info = DeviceDetector.detect()
        assert isinstance(info["device_count"], int)
        assert info["device_count"] >= 0

    def test_best_device_returns_torch_device(self):
        device = DeviceDetector.best_device()
        assert isinstance(device, torch.device)

    def test_best_device_is_valid(self):
        device = DeviceDetector.best_device()
        assert device.type in ("cuda", "mps", "cpu")

    def test_detect_consistency_with_best_device(self):
        """detect()['device'] should match best_device().type."""
        info = DeviceDetector.detect()
        device = DeviceDetector.best_device()
        assert info["device"] == device.type


class TestAccelerationLevel:
    """Tests for AccelerationLevel enum."""

    def test_enum_values(self):
        assert AccelerationLevel.TRITON == 1
        assert AccelerationLevel.COMPILE == 2
        assert AccelerationLevel.AUTOGRAD == 3
        assert AccelerationLevel.EAGER == 4

    def test_enum_ordering(self):
        assert AccelerationLevel.TRITON < AccelerationLevel.COMPILE
        assert AccelerationLevel.COMPILE < AccelerationLevel.AUTOGRAD
        assert AccelerationLevel.AUTOGRAD < AccelerationLevel.EAGER

    def test_enum_from_int(self):
        assert AccelerationLevel(1) == AccelerationLevel.TRITON
        assert AccelerationLevel(4) == AccelerationLevel.EAGER

    def test_enum_names(self):
        assert AccelerationLevel.TRITON.name == "TRITON"
        assert AccelerationLevel.COMPILE.name == "COMPILE"
        assert AccelerationLevel.AUTOGRAD.name == "AUTOGRAD"
        assert AccelerationLevel.EAGER.name == "EAGER"


class TestAccelerationManager:
    """Tests for AccelerationManager."""

    def setup_method(self):
        """Reset manager before each test."""
        AccelerationManager.reset()

    def test_get_level_returns_acceleration_level(self):
        level = AccelerationManager.get_level()
        assert isinstance(level, AccelerationLevel)

    def test_get_level_never_fails(self):
        """get_level() must always succeed — graceful fallback guarantee."""
        level = AccelerationManager.get_level()
        assert level in AccelerationLevel

    def test_set_level_with_enum(self):
        AccelerationManager.set_level(AccelerationLevel.EAGER)
        assert AccelerationManager.get_level() == AccelerationLevel.EAGER

    def test_set_level_with_int(self):
        AccelerationManager.set_level(4)
        assert AccelerationManager.get_level() == AccelerationLevel.EAGER

    def test_set_level_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid acceleration level"):
            AccelerationManager.set_level(99)

    def test_set_level_zero_raises_value_error(self):
        with pytest.raises(ValueError):
            AccelerationManager.set_level(0)

    def test_detect_best_level_returns_valid_level(self):
        level = AccelerationManager.detect_best_level()
        assert isinstance(level, AccelerationLevel)
        assert level in AccelerationLevel

    def test_detect_best_level_updates_current(self):
        AccelerationManager.set_level(AccelerationLevel.TRITON)
        detected = AccelerationManager.detect_best_level()
        assert AccelerationManager.get_level() == detected

    def test_register_and_get_kernel(self):
        def my_kernel(x):
            return x

        AccelerationManager.register_kernel("test_kernel", AccelerationLevel.EAGER, my_kernel)
        AccelerationManager.set_level(AccelerationLevel.EAGER)
        result = AccelerationManager.get_kernel("test_kernel")
        assert result is my_kernel

    def test_get_kernel_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown kernel"):
            AccelerationManager.get_kernel("nonexistent_kernel")

    def test_get_kernel_fallback(self):
        """If current level kernel is missing, fall back to next available."""

        def eager_impl(x):
            return x

        AccelerationManager.register_kernel("fallback_test", AccelerationLevel.EAGER, eager_impl)
        # Set level to TRITON — no TRITON impl registered, should fall back to EAGER
        AccelerationManager.set_level(AccelerationLevel.TRITON)
        result = AccelerationManager.get_kernel("fallback_test")
        assert result is eager_impl

    def test_get_kernel_prefers_current_level(self):
        """get_kernel should prefer an exact match for current level."""

        def compile_impl(x):
            return x * 2

        def eager_impl(x):
            return x * 3

        AccelerationManager.register_kernel("pref_test", AccelerationLevel.COMPILE, compile_impl)
        AccelerationManager.register_kernel("pref_test", AccelerationLevel.EAGER, eager_impl)
        AccelerationManager.set_level(AccelerationLevel.COMPILE)
        result = AccelerationManager.get_kernel("pref_test")
        assert result is compile_impl

    def test_reset_clears_state(self):
        AccelerationManager.set_level(AccelerationLevel.TRITON)
        AccelerationManager.register_kernel("reset_test", AccelerationLevel.EAGER, lambda x: x)
        AccelerationManager.reset()
        # After reset, registry should be empty
        with pytest.raises(KeyError):
            AccelerationManager.get_kernel("reset_test")
