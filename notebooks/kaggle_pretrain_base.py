"""
USN Base (350M) Pretraining on Kaggle (2x T4)
==============================================
Copy this into a Kaggle notebook with 2x T4 GPU accelerator.

Pretrains a USN-Base model on:
1. English general text (wikitext-103 / openwebtext subset)
2. Math reasoning (gsm8k)

Tokenizer: GPT-2 (50257 vocab)
Target: ~2 hours on 2x T4
"""

# ═══════════════════════════════════════════════════════════════
# CELL 1: Installation
# ═══════════════════════════════════════════════════════════════

# !pip install -q usn transformers datasets tokenizers

# ═══════════════════════════════════════════════════════════════
# CELL 2: Imports and GPU Check
# ═══════════════════════════════════════════════════════════════

import os
import time
import math
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.nn.parallel import DataParallel
from transformers import GPT2TokenizerFast
from datasets import load_dataset

import usn
from usn import USNConfig, USNModel
from usn.losses.cross_entropy import USNCrossEntropyLoss
from usn.optim.factory import OptimizerFactory
from usn.optim.schedulers import WarmupCosineScheduler
from usn.serialization.writer import USNWriter

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

print(f"USN version: {usn.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
          f"({torch.cuda.get_device_properties(i).total_mem // 1024**3}GB)")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()

# ═══════════════════════════════════════════════════════════════
# CELL 3: Model Configuration (USN-Base 350M)
# ═══════════════════════════════════════════════════════════════

# GPT-2 tokenizer (50257 vocab)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = tokenizer.vocab_size  # 50257

# USN-Base configuration (~350M params)
# Adjusted for 2x T4 (16GB each) memory constraints
config = USNConfig(
    num_layers=24,
    d_model=1024,
    d_s=768,        # Large semantic state
    k=24,           # Relational state 24x24 = 576
    d_ff=4096,
    vocab_size=VOCAB_SIZE,
    max_seq_len=512,
    norm_type="rmsnorm",
    activation="gelu",
    dropout=0.1,
    embedding_dropout=0.0,
    residual_dropout=0.1,
    tie_weights=True,
    fused=False,
    chunk_size=64,
)

model = USNModel(config)
print(f"\n{'='*60}")
print(f"USN-Base Model Created")
print(f"{'='*60}")
print(model.summary())

# Multi-GPU with DataParallel
if NUM_GPUS > 1:
    model = DataParallel(model)
    print(f"\nUsing DataParallel on {NUM_GPUS} GPUs")
model = model.to(DEVICE)

# Access the underlying model for saving later
base_model = model.module if isinstance(model, DataParallel) else model

# ═══════════════════════════════════════════════════════════════
# CELL 4: Dataset Preparation
# ═══════════════════════════════════════════════════════════════

SEQ_LEN = 512
BATCH_SIZE = 8 * NUM_GPUS  # 8 per GPU, 16 total for 2x T4


class TextDataset(Dataset):
    """Pre-tokenized dataset for causal LM training."""

    def __init__(self, token_chunks: list[list[int]]):
        self.chunks = token_chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        tokens = self.chunks[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)
        return {"input_ids": input_ids, "targets": targets}


def prepare_text_dataset(name, split, max_samples=50000):
    """Load and tokenize a text dataset into fixed-length chunks."""
    print(f"Loading {name} ({split})...")

    if name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        texts = [t for t in ds["text"] if len(t.strip()) > 50]
    elif name == "openwebtext":
        ds = load_dataset("stas/openwebtext-10k", split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > 50]
    elif name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split=split)
        texts = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in ds]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    texts = texts[:max_samples]
    print(f"  Loaded {len(texts)} texts")

    # Tokenize all texts into one long stream
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_token_id)

    print(f"  Total tokens: {len(all_tokens):,}")

    # Chunk into fixed-length sequences
    chunks = []
    for i in range(0, len(all_tokens) - SEQ_LEN, SEQ_LEN):
        chunk = all_tokens[i:i + SEQ_LEN + 1]  # +1 for target shift
        if len(chunk) == SEQ_LEN + 1:
            chunks.append(chunk)

    print(f"  Chunks (seq_len={SEQ_LEN}): {len(chunks):,}")
    return TextDataset(chunks)


# Load datasets
print("\n--- Loading Datasets ---")
wiki_dataset = prepare_text_dataset("wikitext", "train", max_samples=30000)
math_dataset = prepare_text_dataset("gsm8k", "train", max_samples=8000)

# Combine datasets
train_dataset = ConcatDataset([wiki_dataset, math_dataset])
print(f"\nCombined dataset: {len(train_dataset):,} samples")
print(f"Estimated tokens: {len(train_dataset) * SEQ_LEN:,}")

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)
print(f"Batches per epoch: {len(train_loader):,}")

# ═══════════════════════════════════════════════════════════════
# CELL 5: Training Configuration
# ═══════════════════════════════════════════════════════════════

# Calculate steps for ~2 hours on 2x T4
# Estimate: ~0.8s per step with batch_size=16, seq_len=512 on 2xT4
# 2 hours = 7200s / 0.8s = ~9000 steps
MAX_STEPS = 8000
WARMUP_STEPS = 400
LEARNING_RATE = 6e-4
MIN_LR = 6e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
LOG_INTERVAL = 100
SAVE_INTERVAL = 2000

# Optimizer: AdamW with proper weight decay separation
from usn.config import USNTrainingConfig
train_config = USNTrainingConfig(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    grad_clip=GRAD_CLIP,
    batch_size=BATCH_SIZE,
    max_steps=MAX_STEPS,
    warmup_steps=WARMUP_STEPS,
    mixed_precision="bf16",
)

optimizer = OptimizerFactory.create(base_model, train_config)
scheduler = WarmupCosineScheduler(
    max_lr=LEARNING_RATE,
    min_lr=MIN_LR,
    warmup_steps=WARMUP_STEPS,
    total_steps=MAX_STEPS,
)
loss_fn = USNCrossEntropyLoss(label_smoothing=0.0, ignore_index=-100)
scaler = torch.amp.GradScaler("cuda") if train_config.mixed_precision == "fp16" else None

print(f"\nTraining config:")
print(f"  Max steps: {MAX_STEPS:,}")
print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE})")
print(f"  Sequence length: {SEQ_LEN}")
print(f"  Learning rate: {LEARNING_RATE} -> {MIN_LR}")
print(f"  Mixed precision: bf16")
print(f"  Gradient clip: {GRAD_CLIP}")
print(f"  Estimated time: ~2 hours on 2x T4")

# ═══════════════════════════════════════════════════════════════
# CELL 6: Training Loop
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"Starting USN-Base Pretraining")
print(f"{'='*60}\n")

model.train()
global_step = 0
running_loss = 0.0
running_tokens = 0
start_time = time.time()
log_start = time.time()
loss_history = []

# Use bf16 autocast for T4 (supports bf16 via Ampere compat mode)
amp_dtype = torch.bfloat16

data_iter = iter(train_loader)

for step in range(MAX_STEPS):
    # Get batch (cycle through data)
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch = next(data_iter)

    input_ids = batch["input_ids"].to(DEVICE)
    targets = batch["targets"].to(DEVICE)

    # Forward with mixed precision
    with torch.amp.autocast("cuda", dtype=amp_dtype):
        logits, _ = model(input_ids)
        loss = loss_fn(logits, targets)

    # Backward
    loss.backward()

    # Gradient clipping
    if isinstance(model, DataParallel):
        grad_norm = torch.nn.utils.clip_grad_norm_(model.module.parameters(), GRAD_CLIP)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Scheduler
    lr = scheduler.get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # Metrics
    global_step += 1
    batch_tokens = input_ids.numel()
    running_loss += loss.item()
    running_tokens += batch_tokens

    # Logging
    if global_step % LOG_INTERVAL == 0:
        elapsed = time.time() - log_start
        avg_loss = running_loss / LOG_INTERVAL
        tokens_per_sec = running_tokens / elapsed
        ppl = math.exp(min(avg_loss, 20.0))
        total_elapsed = time.time() - start_time
        eta = (MAX_STEPS - global_step) * (total_elapsed / global_step)

        loss_history.append(avg_loss)
        logger.info(
            f"step={global_step:5d}/{MAX_STEPS} | "
            f"loss={avg_loss:.4f} | ppl={ppl:.1f} | "
            f"lr={lr:.2e} | grad={grad_norm:.2f} | "
            f"tok/s={tokens_per_sec:,.0f} | "
            f"elapsed={total_elapsed/60:.1f}min | "
            f"ETA={eta/60:.1f}min"
        )
        running_loss = 0.0
        running_tokens = 0
        log_start = time.time()

    # Checkpoint
    if global_step % SAVE_INTERVAL == 0:
        ckpt_path = f"/kaggle/working/usn_base_step{global_step}.usn"
        writer = USNWriter()
        writer.save(ckpt_path, base_model, config=config,
                    metadata={"step": str(global_step), "loss": str(loss_history[-1])})
        logger.info(f"Checkpoint saved: {ckpt_path}")

total_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"Training Complete!")
print(f"{'='*60}")
print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
print(f"Final loss: {loss_history[-1]:.4f}")
print(f"Final perplexity: {math.exp(loss_history[-1]):.1f}")
print(f"Total steps: {global_step:,}")

# ═══════════════════════════════════════════════════════════════
# CELL 7: Save Final Model
# ═══════════════════════════════════════════════════════════════

SAVE_PATH = "/kaggle/working/usn_base_350m_final.usn"

writer = USNWriter()
writer.save(
    SAVE_PATH,
    base_model,
    config=config,
    metadata={
        "model_name": "USN-Base-350M",
        "training_steps": str(global_step),
        "final_loss": f"{loss_history[-1]:.4f}",
        "final_perplexity": f"{math.exp(loss_history[-1]):.1f}",
        "training_time_hours": f"{total_time/3600:.2f}",
        "datasets": "wikitext-103 + gsm8k",
        "tokenizer": "gpt2",
        "seq_len": str(SEQ_LEN),
        "hardware": "2x NVIDIA T4",
    },
)
print(f"\nModel saved to: {SAVE_PATH}")
print(f"File size: {os.path.getsize(SAVE_PATH) / 1024**2:.1f} MB")

# ═══════════════════════════════════════════════════════════════
# CELL 8: Test Generation
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"Generation Test")
print(f"{'='*60}\n")

base_model.eval()

def generate_text(prompt, max_tokens=100, temperature=0.8, top_k=50):
    """Generate text from a prompt using the trained USN model."""
    tokens = tokenizer.encode(prompt)
    generated = list(tokens)

    # Process prompt token by token to build state
    state = None
    for tok in tokens:
        input_t = torch.tensor([[tok]], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits, state = base_model(input_t, initial_state=state)

    # Generate new tokens
    for _ in range(max_tokens):
        next_logits = logits[0, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            topk_vals, _ = next_logits.topk(top_k)
            threshold = topk_vals[-1]
            next_logits[next_logits < threshold] = float("-inf")

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        if next_token == tokenizer.eos_token_id:
            break

        generated.append(next_token)

        # Next step
        input_t = torch.tensor([[next_token]], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits, state = base_model(input_t, initial_state=state)

    return tokenizer.decode(generated)


# Test prompts
prompts = [
    "The meaning of life is",
    "Artificial intelligence will",
    "Question: What is 25 + 37?\nAnswer:",
    "The solar system contains",
    "In the year 2050,",
]

for prompt in prompts:
    print(f"Prompt: '{prompt}'")
    output = generate_text(prompt, max_tokens=80, temperature=0.8, top_k=50)
    print(f"Output: {output}\n")
    print("-" * 40)

# ═══════════════════════════════════════════════════════════════
# CELL 9: Loss Curve Visualization
# ═══════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
ax1.plot(range(LOG_INTERVAL, len(loss_history) * LOG_INTERVAL + 1, LOG_INTERVAL),
         loss_history, 'b-', linewidth=1.5)
ax1.set_xlabel("Training Steps")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_title("USN-Base 350M Training Loss")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# Perplexity curve
ppl_history = [math.exp(min(l, 10)) for l in loss_history]
ax2.plot(range(LOG_INTERVAL, len(ppl_history) * LOG_INTERVAL + 1, LOG_INTERVAL),
         ppl_history, 'r-', linewidth=1.5)
ax2.set_xlabel("Training Steps")
ax2.set_ylabel("Perplexity")
ax2.set_title("USN-Base 350M Perplexity")
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig("/kaggle/working/usn_base_training_curves.png", dpi=150)
plt.show()
print("Training curves saved to /kaggle/working/usn_base_training_curves.png")

# ═══════════════════════════════════════════════════════════════
# CELL 10: Model Info Summary
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"USN-Base 350M — Final Report")
print(f"{'='*60}")
print(f"  Architecture: USN (Unified State Network)")
print(f"  Parameters: {base_model.num_parameters:,}")
print(f"  Layers: {config.num_layers}")
print(f"  d_model: {config.d_model}")
print(f"  d_s (semantic state): {config.d_s}")
print(f"  k (relational state): {config.k} (R ∈ {config.k}×{config.k})")
print(f"  d_ff: {config.d_ff}")
print(f"  Vocab: {config.vocab_size} (GPT-2 tokenizer)")
print(f"  Sequence length: {config.max_seq_len}")
print(f"  State per layer: {config.d_s + config.k**2} floats")
print(f"  Total state: {config.num_layers * (config.d_s + config.k**2)} floats")
print(f"  Inference memory: O(1) — no KV cache")
print(f"  Training complexity: O(n) — no attention")
print(f"  Training time: {total_time/3600:.2f} hours on 2x T4")
print(f"  Final loss: {loss_history[-1]:.4f}")
print(f"  Final perplexity: {math.exp(loss_history[-1]):.1f}")
print(f"  Datasets: wikitext-103 + gsm8k")
print(f"  Model file: {SAVE_PATH}")
print(f"{'='*60}")
