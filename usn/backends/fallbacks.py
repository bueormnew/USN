"""PyTorch fallback implementations for all USN kernels.

Provides Level 2 (torch.compile) wrappers and registers them with
AccelerationManager. When torch.compile is available, these are
JIT-compiled for better performance. When unavailable, they delegate
directly to the Level 4 (eager) implementations.
"""

from __future__ import annotations

import logging

import torch

from usn.backends.acceleration import AccelerationLevel, AccelerationManager
from usn.backends.triton_kernels import (
    eager_channel_mlp,
    eager_fused_projections,
    eager_state_core,
    eager_temporal_gate,
)

logger = logging.getLogger(__name__)

# Check if torch.compile is available
_HAS_COMPILE = hasattr(torch, "compile")


def _try_compile(fn):
    """Attempt to wrap a function with torch.compile.

    Uses 'reduce-overhead' mode which is optimized for repeated calls
    with the same shapes (common in autoregressive generation and
    fixed-length training batches).

    If torch.compile is unavailable or compilation fails for any reason,
    returns the original function unchanged — ensuring graceful fallback
    to eager execution.

    Note: torch.compile is lazy — errors may occur at first invocation
    rather than at wrap time. The wrapper catches these runtime compilation
    failures and falls back to the original function transparently.

    Args:
        fn: The function to compile.

    Returns:
        Compiled function if torch.compile is available and succeeds,
        otherwise the original function unchanged.
    """
    if _HAS_COMPILE:
        try:
            compiled = torch.compile(fn, mode="reduce-overhead")
        except Exception:
            logger.debug(
                "torch.compile wrapping failed for %s, using eager",
                getattr(fn, "__name__", str(fn)),
            )
            return fn

        # Wrap to catch lazy compilation failures at first call
        def _safe_compiled(*args, **kwargs):
            try:
                return compiled(*args, **kwargs)
            except Exception:
                logger.debug(
                    "torch.compile execution failed for %s, falling back to eager",
                    getattr(fn, "__name__", str(fn)),
                )
                # Replace ourselves with the original to avoid repeated failures
                _safe_compiled.__wrapped_fallback__ = True
                return fn(*args, **kwargs)

        # Preserve metadata for introspection
        _safe_compiled.__name__ = getattr(fn, "__name__", "compiled_fn")
        _safe_compiled.__qualname__ = getattr(fn, "__qualname__", "compiled_fn")
        _safe_compiled.__doc__ = fn.__doc__
        _safe_compiled.__wrapped__ = fn
        _safe_compiled.__wrapped_fallback__ = False
        return _safe_compiled

    return fn


# Compiled versions (Level 2)
compiled_projections = _try_compile(eager_fused_projections)
compiled_temporal_gate = _try_compile(eager_temporal_gate)
compiled_state_core = _try_compile(eager_state_core)
compiled_channel_mlp = _try_compile(eager_channel_mlp)

# Register Level 2 (COMPILE) kernels
AccelerationManager.register_kernel("projections", AccelerationLevel.COMPILE, compiled_projections)
AccelerationManager.register_kernel(
    "temporal_gate", AccelerationLevel.COMPILE, compiled_temporal_gate
)
AccelerationManager.register_kernel("state_core", AccelerationLevel.COMPILE, compiled_state_core)
AccelerationManager.register_kernel("channel_mlp", AccelerationLevel.COMPILE, compiled_channel_mlp)

logger.debug(
    "Registered Level 2 (COMPILE) fallback kernels (torch.compile %s)",
    "available" if _HAS_COMPILE else "NOT available — using eager",
)
