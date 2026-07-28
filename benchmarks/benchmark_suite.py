"""USN Benchmark Suite.

Measures forward pass latency, inference throughput, memory usage,
parallel scan vs sequential comparison, and serialization time
across model configurations.

Usage:
    python benchmarks/benchmark_suite.py [--config tiny|micro|mini|small]
    python benchmarks/benchmark_suite.py --config tiny --output results.json
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import torch

# Add parent to path for development usage
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from usn.config import USNConfig
from usn.layers.parallel_scan import parallel_scan_semantic
from usn.models import USNModel
from usn.serialization import USNReader, USNWriter


def benchmark_forward_latency(model, config, seq_lengths=None):
    """Measure forward pass latency at multiple sequence lengths.

    Runs warmup iterations followed by timed iterations to measure
    average latency and throughput at each sequence length. Used to
    demonstrate O(n) linearity of the architecture.

    Args:
        model: The USNModel in eval mode.
        config: USNConfig for the model.
        seq_lengths: List of sequence lengths to benchmark.

    Returns:
        Dict mapping seq_len -> {avg_ms, tokens_per_sec, std_ms}.
    """
    if seq_lengths is None:
        seq_lengths = [32, 64, 128, 256, 512]

    results = {}
    model.eval()
    device = next(model.parameters()).device

    for seq_len in seq_lengths:
        if seq_len > config.max_seq_len:
            continue
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                model(input_ids)

        # Measure
        times = []
        with torch.no_grad():
            for _ in range(10):
                start = time.perf_counter()
                model(input_ids)
                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)  # ms

        avg_ms = sum(times) / len(times)
        std_ms = (sum((t - avg_ms) ** 2 for t in times) / len(times)) ** 0.5
        results[seq_len] = {
            "avg_ms": round(avg_ms, 3),
            "std_ms": round(std_ms, 3),
            "tokens_per_sec": round(seq_len / (avg_ms / 1000), 1),
        }

    return results


def benchmark_memory(model, config):
    """Measure parameter memory and state memory.

    Args:
        model: The USNModel.
        config: USNConfig for the model.

    Returns:
        Dict with param_memory_mb, state_memory_bytes, total_parameters.
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # State memory per sample: each layer has d_s floats (semantic) + k*k floats (relational)
    state_floats_per_layer = config.d_s + config.k**2
    state_bytes = config.num_layers * state_floats_per_layer * 4  # float32
    return {
        "param_memory_mb": round(param_bytes / (1024 * 1024), 3),
        "state_memory_bytes": state_bytes,
        "state_memory_per_layer_floats": state_floats_per_layer,
        "total_parameters": model.num_parameters,
    }


def benchmark_parallel_vs_sequential(config, seq_lengths=None):
    """Compare parallel scan vs sequential recurrence.

    Both should produce identical results, but parallel scan leverages
    vectorized operations. This measures their relative performance.

    Args:
        config: USNConfig for generating test data.
        seq_lengths: List of sequence lengths to benchmark.

    Returns:
        Dict mapping seq_len -> {parallel_ms, sequential_ms, speedup, max_diff}.
    """
    if seq_lengths is None:
        seq_lengths = [32, 64, 128, 256]

    results = {}
    device = torch.device("cpu")
    batch_size = 2

    for seq_len in seq_lengths:
        if seq_len > config.max_seq_len:
            continue

        # Generate random inputs
        log_decays = torch.randn(batch_size, seq_len, config.d_s, device=device) * 0.1 - 1.0
        values = torch.randn(batch_size, seq_len, config.d_s, device=device) * 0.1
        initial_state = torch.zeros(batch_size, config.d_s, device=device)

        # Warmup
        for _ in range(2):
            parallel_scan_semantic(log_decays, values, initial_state)

        # Benchmark parallel scan
        times_parallel = []
        for _ in range(10):
            start = time.perf_counter()
            result_parallel = parallel_scan_semantic(log_decays, values, initial_state)
            elapsed = time.perf_counter() - start
            times_parallel.append(elapsed * 1000)

        # Benchmark sequential recurrence (manual loop)
        times_sequential = []
        for _ in range(10):
            start = time.perf_counter()
            all_states = torch.empty_like(values)
            s_prev = initial_state
            for t in range(seq_len):
                decay_t = torch.exp(log_decays[:, t, :])
                s_t = decay_t * s_prev + values[:, t, :]
                all_states[:, t, :] = s_t
                s_prev = s_t
            elapsed = time.perf_counter() - start
            times_sequential.append(elapsed * 1000)

        avg_parallel = sum(times_parallel) / len(times_parallel)
        avg_sequential = sum(times_sequential) / len(times_sequential)
        speedup = avg_sequential / avg_parallel if avg_parallel > 0 else 0.0

        # Verify correctness (max absolute difference)
        max_diff = (result_parallel - all_states).abs().max().item()

        results[seq_len] = {
            "parallel_ms": round(avg_parallel, 3),
            "sequential_ms": round(avg_sequential, 3),
            "speedup": round(speedup, 2),
            "max_diff": float(f"{max_diff:.2e}"),
        }

    return results


def benchmark_serialization(model, config):
    """Measure save/load time and file size.

    Args:
        model: The USNModel.
        config: USNConfig for the model.

    Returns:
        Dict with save_ms, load_ms, file_size_mb.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bench_model.usn"
        writer = USNWriter()

        # Benchmark save
        start = time.perf_counter()
        writer.save(str(path), model, config=config)
        save_ms = (time.perf_counter() - start) * 1000

        # Benchmark load
        reader = USNReader()
        start = time.perf_counter()
        reader.load(str(path))
        load_ms = (time.perf_counter() - start) * 1000

        file_size_mb = path.stat().st_size / (1024 * 1024)

    return {
        "save_ms": round(save_ms, 1),
        "load_ms": round(load_ms, 1),
        "file_size_mb": round(file_size_mb, 3),
    }


def check_linearity(latency_results):
    """Analyze whether forward pass scales linearly with sequence length.

    Computes the ratio of time/seq_len for each measurement. If the
    architecture is truly O(n), these ratios should be roughly constant.

    Args:
        latency_results: Dict from benchmark_forward_latency.

    Returns:
        Dict with ratios and a linearity_score (coefficient of variation).
    """
    if len(latency_results) < 2:
        return {"linearity_score": None, "ratios": {}}

    ratios = {}
    for seq_len, metrics in latency_results.items():
        ratios[seq_len] = metrics["avg_ms"] / seq_len

    ratio_values = list(ratios.values())
    mean_ratio = sum(ratio_values) / len(ratio_values)
    variance = sum((r - mean_ratio) ** 2 for r in ratio_values) / len(ratio_values)
    std_ratio = variance**0.5
    cv = std_ratio / mean_ratio if mean_ratio > 0 else float("inf")

    return {
        "ratios_ms_per_token": {k: round(v, 4) for k, v in ratios.items()},
        "mean_ms_per_token": round(mean_ratio, 4),
        "coefficient_of_variation": round(cv, 4),
        "likely_linear": cv < 0.5,  # Generous threshold for small models/CPU
    }


def main():
    parser = argparse.ArgumentParser(description="USN Benchmark Suite")
    parser.add_argument(
        "--config",
        default="tiny",
        choices=["tiny", "micro", "mini", "small"],
        help="Model configuration preset to benchmark (default: tiny)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save JSON results (optional)",
    )
    parser.add_argument(
        "--no-serialization",
        action="store_true",
        help="Skip serialization benchmark",
    )
    args = parser.parse_args()

    print("╔══════════════════════════════════╗")
    print("║     USN Benchmark Suite          ║")
    print("╚══════════════════════════════════╝")
    print(f"  Config: {args.config}")
    print(f"  Device: cpu")
    print()

    # Create model
    config = USNConfig.from_preset(args.config)
    model = USNModel(config)
    model.eval()

    print(f"  Parameters: {model.num_parameters:,}")
    print(f"  State/layer: {model.state_size_per_layer} floats")
    print(f"  Total state: {model.total_state_size} floats")
    print()

    # ─── Forward Latency ───────────────────────────
    print("─── Forward Latency ───")
    if config.max_seq_len >= 256:
        seq_lengths = [32, 64, 128, 256]
    elif config.max_seq_len >= 64:
        seq_lengths = [8, 16, 32, 64]
    else:
        seq_lengths = [8, 16, 32]

    latency = benchmark_forward_latency(model, config, seq_lengths)
    for seq_len, metrics in latency.items():
        print(
            f"  seq_len={seq_len:4d}: {metrics['avg_ms']:7.2f}ms "
            f"(±{metrics['std_ms']:.2f}ms) "
            f"| {metrics['tokens_per_sec']:,.0f} tok/s"
        )
    print()

    # ─── Linearity Analysis ────────────────────────
    print("─── Linearity Analysis (O(n) check) ───")
    linearity = check_linearity(latency)
    for seq_len, ratio in linearity["ratios_ms_per_token"].items():
        print(f"  seq_len={seq_len:4d}: {ratio:.4f} ms/token")
    print(f"  Mean: {linearity['mean_ms_per_token']:.4f} ms/token")
    print(f"  CV:   {linearity['coefficient_of_variation']:.4f}")
    print(f"  Linear: {'✓' if linearity['likely_linear'] else '✗'}")
    print()

    # ─── Memory ────────────────────────────────────
    print("─── Memory ───")
    mem = benchmark_memory(model, config)
    print(f"  Parameters:    {mem['param_memory_mb']:.2f} MB")
    print(f"  State/sample:  {mem['state_memory_bytes']} bytes")
    print(f"  State/layer:   {mem['state_memory_per_layer_floats']} floats")
    print()

    # ─── Parallel vs Sequential ────────────────────
    print("─── Parallel Scan vs Sequential ───")
    scan_lengths = [32, 64, 128] if config.max_seq_len >= 128 else [8, 16, 32]
    scan_results = benchmark_parallel_vs_sequential(config, scan_lengths)
    for seq_len, metrics in scan_results.items():
        print(
            f"  seq_len={seq_len:4d}: "
            f"parallel={metrics['parallel_ms']:6.2f}ms, "
            f"sequential={metrics['sequential_ms']:6.2f}ms, "
            f"speedup={metrics['speedup']:.2f}x, "
            f"max_diff={metrics['max_diff']:.2e}"
        )
    print()

    # ─── Serialization ─────────────────────────────
    ser = None
    if not args.no_serialization:
        print("─── Serialization ───")
        ser = benchmark_serialization(model, config)
        print(f"  Save: {ser['save_ms']:.1f}ms")
        print(f"  Load: {ser['load_ms']:.1f}ms")
        print(f"  File: {ser['file_size_mb']:.3f} MB")
        print()

    print("╔══════════════════════════════════╗")
    print("║          Done                    ║")
    print("╚══════════════════════════════════╝")

    # Output JSON
    if args.output:
        results = {
            "config": args.config,
            "model_info": {
                "num_parameters": model.num_parameters,
                "state_size_per_layer": model.state_size_per_layer,
                "total_state_size": model.total_state_size,
            },
            "latency": {str(k): v for k, v in latency.items()},
            "linearity": linearity,
            "memory": mem,
            "parallel_vs_sequential": {str(k): v for k, v in scan_results.items()},
        }
        if ser is not None:
            results["serialization"] = ser
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
