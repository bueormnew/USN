"""USN Command-Line Interface.

Accessible via the `usn` command after installation.

Commands:
    usn train --config <path>          Train a model
    usn generate --model <path> --prompt <text>  Generate text
    usn benchmark --model <path>       Run benchmarks
    usn info --model <path>            Display model info
    usn export --model <path> --format <fmt> --output <path>  Export model
    usn validate --model <path>        Validate .usn file
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def cli() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="usn",
        description="USN - Unified State Network Architecture Library",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train command
    train_parser = subparsers.add_parser("train", help="Train a USN model")
    train_parser.add_argument("--config", required=True, help="Path to training config YAML")
    train_parser.add_argument("--verbose", action="store_true", default=True)
    train_parser.add_argument("--quiet", action="store_true")

    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("--model", required=True, help="Path to .usn model file")
    gen_parser.add_argument("--prompt", required=True, help="Generation prompt")
    gen_parser.add_argument("--max-tokens", type=int, default=256)
    gen_parser.add_argument("--temperature", type=float, default=1.0)
    gen_parser.add_argument("--top-k", type=int, default=0)
    gen_parser.add_argument("--top-p", type=float, default=1.0)

    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--model", required=True, help="Path to .usn model file")
    bench_parser.add_argument("--all", action="store_true", help="Run full benchmark suite")

    # info command
    info_parser = subparsers.add_parser("info", help="Display model info")
    info_parser.add_argument("--model", required=True, help="Path to .usn model file")

    # export command
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument("--model", required=True, help="Path to .usn model file")
    export_parser.add_argument(
        "--format",
        required=True,
        choices=["onnx", "safetensors", "state_dict", "torchscript"],
        help="Export format",
    )
    export_parser.add_argument("--output", required=True, help="Output file path")

    # validate command
    val_parser = subparsers.add_parser("validate", help="Validate .usn file")
    val_parser.add_argument("--model", required=True, help="Path to .usn model file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Set up logging
    level = logging.WARNING if getattr(args, "quiet", False) else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Dispatch command
    if args.command == "train":
        _cmd_train(args)
    elif args.command == "generate":
        _cmd_generate(args)
    elif args.command == "benchmark":
        _cmd_benchmark(args)
    elif args.command == "info":
        _cmd_info(args)
    elif args.command == "export":
        _cmd_export(args)
    elif args.command == "validate":
        _cmd_validate(args)


def _cmd_train(args: argparse.Namespace) -> None:
    """Execute training command."""
    import yaml

    from usn.config import USNConfig, USNTrainingConfig
    from usn.models import create_model

    with open(args.config) as f:
        config_data = yaml.safe_load(f)

    model_config = USNConfig.from_dict(config_data.get("model", {}))
    train_config = USNTrainingConfig(**config_data.get("training", {}))

    model = create_model(model_config)
    print(model.summary())
    print(f"Training config loaded: {train_config}")
    print("Use the Python API for full training with datasets.")


def _cmd_generate(args: argparse.Namespace) -> None:
    """Execute generation command."""
    from usn.serialization import USNReader

    print(f"Loading model from: {args.model}")
    reader = USNReader()
    try:
        reader.load(args.model)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}, Top-p: {args.top_p}")
    print("Full generation requires a tokenizer. Use the Python API for generation.")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    """Execute benchmark command."""
    print(f"Benchmarking model: {args.model}")
    if args.all:
        print("Running full benchmark suite...")
    print("Use `usn.benchmark()` Python API for full benchmarks.")


def _cmd_info(args: argparse.Namespace) -> None:
    """Execute info command."""
    from usn.serialization import USNReader

    reader = USNReader()
    try:
        data = reader.load(args.model, sections=["config", "metadata"])
        config = data.get("config")
        metadata = data.get("metadata", {})
        if config:
            from usn.models import USNModel

            model = USNModel(config)
            print(model.summary())
        if metadata:
            print("\nMetadata:")
            for k, v in metadata.items():
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error reading model: {e}", file=sys.stderr)
        sys.exit(1)


def _cmd_export(args: argparse.Namespace) -> None:
    """Execute export command."""
    from usn.serialization import USNReader, export_model

    reader = USNReader()
    try:
        data = reader.load(args.model)
        config = data.get("config")
        if config is None:
            print("Error: could not read model config.", file=sys.stderr)
            sys.exit(1)

        from usn.models import USNModel

        model = USNModel(config)
        weights = data.get("weights")
        if weights is not None:
            model.load_state_dict(weights)

        export_model(model, args.format, args.output)
        print(f"Exported to {args.output} ({args.format})")
    except Exception as e:
        print(f"Error exporting model: {e}", file=sys.stderr)
        sys.exit(1)


def _cmd_validate(args: argparse.Namespace) -> None:
    """Execute validate command."""
    from usn.serialization import FormatValidator

    validator = FormatValidator()

    try:
        version = validator.verify_format_version(args.model)
        print(f"Format version: {version}")
    except Exception as e:
        print(f"INVALID: {e}", file=sys.stderr)
        sys.exit(1)

    if validator.verify_checksum(args.model):
        print("Checksum: VALID")
    else:
        print("Checksum: INVALID", file=sys.stderr)
        sys.exit(1)

    print("File is valid.")


if __name__ == "__main__":
    cli()
