"""Utility functions: counting, seeding, timing, profiling, visualization, and diagnostics."""

from usn.utils.counting import count_parameters, estimate_flops, estimate_memory
from usn.utils.diagnostics import (
    ActivationStats,
    GradientStats,
    StateHealthReport,
    activation_stats,
    check_state_health,
    gradient_stats,
)
from usn.utils.profiling import (
    ProfileResult,
    profile_backward,
    profile_forward,
    profile_memory,
)
from usn.utils.seed import set_seed
from usn.utils.timing import MemoryResult, TimerResult, memory_tracker, timer
from usn.utils.visualization import (
    visualize_gates,
    visualize_state,
)

__all__ = [
    # Counting
    "count_parameters",
    "estimate_memory",
    "estimate_flops",
    # Seed
    "set_seed",
    # Timing
    "timer",
    "memory_tracker",
    "TimerResult",
    "MemoryResult",
    # Profiling
    "ProfileResult",
    "profile_forward",
    "profile_backward",
    "profile_memory",
    # Visualization
    "visualize_state",
    "visualize_gates",
    # Diagnostics
    "GradientStats",
    "ActivationStats",
    "StateHealthReport",
    "gradient_stats",
    "activation_stats",
    "check_state_health",
]
