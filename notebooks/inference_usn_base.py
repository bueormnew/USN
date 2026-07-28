"""
USN-Base 350M Inference Script
===============================
Load a pretrained USN-Base model and generate text.

Usage:
    python inference_usn_base.py --model path/to/usn_base_350m_final.usn
    python inference_usn_base.py --model model.usn --prompt "Hello world"
"""

import argparse
import sys
import torch
from transformers import GPT2TokenizerFast

# Install USN: pip install usn
import usn
from usn import USNModel
from usn.serialization.reader import USNReader


def load_model(model_path: str, device: str = "auto"):
    """Load a USN model from .usn format."""
    print(f"Loading model from: {model_path}")

    reader = USNReader()
    data = reader.load(model_path)

    config = data["config"]
    weights = data["weights"]
    metadata = data.get("metadata", {})

    print(f"  Config: {config.num_layers} layers, d_model={config.d_model}")
    print(f"  Metadata: {metadata}")

    # Build model
    model = USNModel(config)

    # Load weights
    state_dict = {k: v for k, v in weights.items()
                  if not k.startswith("__buffer__.")}
    model.load_state_dict(state_dict, strict=False)

    # Move to device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"  Parameters: {model.num_parameters:,}")
    print(f"  Device: {device}")
    print(f"  Ready for inference!")
    return model, config, device


def generate(model, tokenizer, prompt, max_tokens=150,
             temperature=0.8, top_k=50, top_p=0.95, device="cuda"):
    """Generate text autoregressively."""
    tokens = tokenizer.encode(prompt)
    generated = list(tokens)

    # Prefill: process prompt token by token
    state = None
    for tok in tokens:
        input_t = torch.tensor([[tok]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, state = model(input_t, initial_state=state)

    # Generate
    for _ in range(max_tokens):
        next_logits = logits[0, -1, :].float()

        # Temperature
        if temperature > 0:
            next_logits = next_logits / temperature

        # Top-k
        if top_k > 0:
            topk_vals, _ = next_logits.topk(top_k)
            next_logits[next_logits < topk_vals[-1]] = float("-inf")

        # Top-p
        if top_p < 1.0:
            sorted_logits, sorted_idx = next_logits.sort(descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = sorted_probs.cumsum(dim=-1)
            mask = cumulative - sorted_probs > top_p
            sorted_logits[mask] = float("-inf")
            next_logits = next_logits.scatter(0, sorted_idx, sorted_logits)

        # Sample
        probs = torch.softmax(next_logits, dim=-1)
        if temperature == 0:
            next_token = next_logits.argmax().item()
        else:
            next_token = torch.multinomial(probs, 1).item()

        if next_token == tokenizer.eos_token_id:
            break

        generated.append(next_token)
        input_t = torch.tensor([[next_token]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, state = model(input_t, initial_state=state)

    return tokenizer.decode(generated)


def interactive_mode(model, tokenizer, device):
    """Interactive text generation."""
    print("\n" + "=" * 60)
    print("USN-Base Interactive Generation")
    print("=" * 60)
    print("Type a prompt and press Enter. Type 'quit' to exit.\n")

    while True:
        prompt = input(">>> ")
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt.strip():
            continue

        output = generate(model, tokenizer, prompt, device=device)
        print(f"\n{output}\n")
        print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="USN-Base Inference")
    parser.add_argument("--model", required=True, help="Path to .usn model file")
    parser.add_argument("--prompt", default=None, help="Single prompt (or interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model, config, device = load_model(args.model, args.device)

    if args.prompt:
        # Single generation
        output = generate(model, tokenizer, args.prompt,
                         max_tokens=args.max_tokens,
                         temperature=args.temperature,
                         top_k=args.top_k,
                         top_p=args.top_p,
                         device=device)
        print(f"\n{output}")
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, device)
