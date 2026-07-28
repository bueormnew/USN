#!/usr/bin/env python3
"""USN Quick Start Example.

Demonstrates the core workflow: create a model, inspect it, train on
synthetic data, save/load, and generate text.

Requirements:
    pip install usn
"""

import torch

import usn
from usn import USNConfig, USNGenerationConfig, USNTrainingConfig
from usn.datasets import MathDataset
from usn.tokenizers import CharTokenizer


def main():
    # ──────────────────────────────────────────────
    # 1. Create a model
    # ──────────────────────────────────────────────
    print("=" * 60)
    print("USN Quick Start")
    print("=" * 60)

    # Set seed for reproducibility
    usn.set_seed(42)

    # Create a tiny model for demonstration
    config = USNConfig.tiny()
    model = usn.create_model(config)
    print(f"\nModel created: {usn.count_parameters(model):,} parameters")
    print(usn.summary(model))

    # ──────────────────────────────────────────────
    # 2. Prepare a dataset
    # ──────────────────────────────────────────────
    # MathDataset generates arithmetic problems like "5+3=8"
    train_dataset = MathDataset(
        num_samples=1000,
        max_digits=2,
        operations=["+", "-"],
        split="train",
    )
    val_dataset = MathDataset(
        num_samples=100,
        max_digits=2,
        operations=["+", "-"],
        split="val",
    )
    print(f"\nDataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # ──────────────────────────────────────────────
    # 3. Train the model
    # ──────────────────────────────────────────────
    training_config = USNTrainingConfig(
        learning_rate=1e-3,
        batch_size=32,
        max_steps=200,
        warmup_steps=20,
        mixed_precision="none",  # Use fp32 for CPU compatibility
        log_interval=50,
        eval_interval=100,
        grad_clip=1.0,
    )

    print("\nTraining...")
    trainer = usn.USNTrainer(model, train_dataset, training_config, val_dataset=val_dataset)
    result = trainer.train()
    print(f"Final loss: {result['final_loss']:.4f}")

    # ──────────────────────────────────────────────
    # 4. Save and load the model
    # ──────────────────────────────────────────────
    save_path = "quick_start_model.usn"
    usn.save(model, save_path)
    print(f"\nModel saved to: {save_path}")

    loaded_model = usn.load(save_path)
    print(f"Model loaded: {usn.count_parameters(loaded_model):,} parameters")

    # ──────────────────────────────────────────────
    # 5. Generate text
    # ──────────────────────────────────────────────
    # For generation, we need a tokenizer that matches the model's vocab
    tokenizer = CharTokenizer(vocab_size=config.vocab_size)

    gen_config = USNGenerationConfig(
        temperature=0.8,
        max_new_tokens=20,
    )

    generator = usn.USNGenerator(loaded_model, tokenizer, gen_config)

    prompt = "5+3="
    print(f"\nGenerating from prompt: '{prompt}'")
    output = generator.generate(prompt)
    generated_text = tokenizer.decode(output.token_ids[0].tolist())
    print(f"Generated: {generated_text}")

    # ──────────────────────────────────────────────
    # 6. Export to other formats
    # ──────────────────────────────────────────────
    usn.export(model, "state_dict", "quick_start_weights.pt")
    print("\nExported weights to: quick_start_weights.pt")

    # ──────────────────────────────────────────────
    # 7. Device and acceleration info
    # ──────────────────────────────────────────────
    print(f"\nDevice info: {usn.device_info()}")
    print(f"Acceleration: {usn.benchmark_acceleration()}")

    print("\n" + "=" * 60)
    print("Quick start complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
