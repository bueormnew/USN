# %% [markdown]
# # USN-Base Pretraining on Kaggle (2x T4)
#
# Pretrains a USN model (~180M params, optimized for 2x T4 15GB each) on:
# - English text (wikitext-103)
# - Math reasoning (GSM8K)
#
# **Hardware**: 2x NVIDIA T4 (15GB each)
# **Tokenizer**: GPT-2 (50257 vocab)
# **Training time**: ~2 hours
# **Architecture**: USN — no attention, O(n) training, O(1) inference

# %% Installation
# !pip install -q usn transformers datasets

# %% Imports
import os, time, math, logging
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

print(f"USN: {usn.__version__} | PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = max(torch.cuda.device_count(), 1)

# %% Model Configuration
# Optimized for 2x T4 (15GB each): ~180M params fits in memory
# with seq_len=256, batch=4 per GPU, bf16 mixed precision

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = tokenizer.vocab_size  # 50257
SEQ_LEN = 128  # Short sequences for fast sequential loop on GPU
BATCH_SIZE = 16 * NUM_GPUS  # Larger batch compensates shorter seq

# USN model sized for T4 memory (~180M params)
config = USNConfig(
    num_layers=12,       # Reduced for speed
    d_model=768,
    d_s=512,             # Semantic state
    k=16,                # Relational 16x16=256
    d_ff=3072,           # 4x d_model
    vocab_size=VOCAB_SIZE,
    max_seq_len=SEQ_LEN,
    norm_type="rmsnorm",
    activation="gelu",
    dropout=0.1,
    residual_dropout=0.1,
    embedding_dropout=0.0,
    tie_weights=True,
    fused=True,  # Enable fused kernels (falls back gracefully if unavailable)
    chunk_size=64,
)

model = USNModel(config)
print(f"\n{'='*60}")
print(model.summary())
print(f"{'='*60}")

# Multi-GPU
if NUM_GPUS > 1:
    model = DataParallel(model)
    print(f"DataParallel on {NUM_GPUS} GPUs")
model = model.to(DEVICE)
base_model = model.module if isinstance(model, DataParallel) else model

# %% Dataset Preparation

class TokenizedDataset(Dataset):
    """Fixed-length token chunks for causal LM."""
    def __init__(self, chunks):
        self.chunks = chunks
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, idx):
        t = self.chunks[idx]
        return {"input_ids": torch.tensor(t[:-1], dtype=torch.long),
                "targets": torch.tensor(t[1:], dtype=torch.long)}

def load_and_chunk(name, split, max_samples=50000):
    """Load dataset, tokenize, chunk into SEQ_LEN+1 pieces."""
    print(f"Loading {name}...")
    if name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        texts = [t for t in ds["text"] if len(t.strip()) > 100]
    elif name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split=split)
        texts = [f"Q: {x['question']}\nA: {x['answer']}" for x in ds]
    texts = texts[:max_samples]
    
    # Tokenize into one stream
    all_tokens = []
    for t in texts:
        all_tokens.extend(tokenizer.encode(t, add_special_tokens=False))
        all_tokens.append(tokenizer.eos_token_id)
    
    # Chunk
    chunks = [all_tokens[i:i+SEQ_LEN+1] 
              for i in range(0, len(all_tokens)-SEQ_LEN, SEQ_LEN)
              if len(all_tokens[i:i+SEQ_LEN+1]) == SEQ_LEN+1]
    print(f"  {len(texts)} texts -> {len(all_tokens):,} tokens -> {len(chunks):,} chunks")
    return TokenizedDataset(chunks)

wiki_ds = load_and_chunk("wikitext", "train", 50000)
math_ds = load_and_chunk("gsm8k", "train", 8000)
train_dataset = ConcatDataset([wiki_ds, math_ds])
print(f"\nTotal: {len(train_dataset):,} chunks ({len(train_dataset)*SEQ_LEN:,} tokens)")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                          shuffle=True, num_workers=2, 
                          pin_memory=True, drop_last=True)

# %% Training Setup

MAX_STEPS = 6000
WARMUP = 300
LR = 6e-4
MIN_LR = 6e-5
GRAD_CLIP = 1.0
LOG_EVERY = 100
SAVE_EVERY = 2000
GRAD_ACCUM = 2  # Effective batch = BATCH_SIZE * GRAD_ACCUM

from usn.config import USNTrainingConfig
tcfg = USNTrainingConfig(learning_rate=LR, weight_decay=0.1, 
                         grad_clip=GRAD_CLIP, batch_size=BATCH_SIZE,
                         max_steps=MAX_STEPS, warmup_steps=WARMUP)

optimizer = OptimizerFactory.create(base_model, tcfg)
scheduler = WarmupCosineScheduler(max_lr=LR, min_lr=MIN_LR,
                                  warmup_steps=WARMUP, total_steps=MAX_STEPS)
loss_fn = USNCrossEntropyLoss()

print(f"Steps: {MAX_STEPS} | Batch: {BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
print(f"LR: {LR} -> {MIN_LR} (cosine) | Precision: bf16")
print(f"Tokens/step: {BATCH_SIZE * GRAD_ACCUM * SEQ_LEN:,}")
print(f"Total tokens: ~{MAX_STEPS * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN:,}")

# %% Training Loop

from tqdm.auto import tqdm

# Try to enable fused kernels
try:
    from usn.backends import AccelerationManager, AccelerationLevel
    level = AccelerationManager.get_level()
    print(f"Acceleration level: {level.name}")
except Exception as e:
    print(f"Acceleration detection: {e} (using eager)")

print(f"\n{'='*60}")
print(f"USN Pretraining — {base_model.num_parameters:,} params")
print(f"{'='*60}\n")

model.train()
optimizer.zero_grad(set_to_none=True)
step = 0
accum_loss = 0.0
running_loss = 0.0
running_tokens = 0
t0 = time.time()
log_t = time.time()
loss_history = []
data_iter = iter(train_loader)

pbar = tqdm(total=MAX_STEPS, desc="Training", unit="step",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] loss={postfix}")
pbar.set_postfix_str("...")

while step < MAX_STEPS:
    # Get batch
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch = next(data_iter)

    ids = batch["input_ids"].to(DEVICE)
    tgt = batch["targets"].to(DEVICE)

    # Forward (bf16)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(ids)
        loss = loss_fn(logits, tgt) / GRAD_ACCUM

    loss.backward()
    accum_loss += loss.item()
    running_tokens += ids.numel()

    # Accumulate gradients
    if (step + 1) % GRAD_ACCUM == 0 or step == 0:
        # Clip + step
        nn.utils.clip_grad_norm_(base_model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # LR schedule
        actual_step = step // GRAD_ACCUM
        lr = scheduler.get_lr(actual_step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    step += 1
    running_loss += accum_loss
    accum_loss = 0.0

    # Update progress bar
    pbar.update(1)

    # Log
    if step % LOG_EVERY == 0:
        elapsed = time.time() - log_t
        avg = running_loss / LOG_EVERY
        tps = running_tokens / elapsed
        ppl = math.exp(min(avg * GRAD_ACCUM, 20))
        total_t = time.time() - t0
        eta = (MAX_STEPS - step) / step * total_t

        loss_history.append(avg * GRAD_ACCUM)
        pbar.set_postfix_str(f"{avg*GRAD_ACCUM:.3f} ppl={ppl:.0f} lr={lr:.1e} tok/s={tps:,.0f}")
        logger.info(f"step={step:5d}/{MAX_STEPS} loss={avg*GRAD_ACCUM:.3f} "
                    f"ppl={ppl:.0f} lr={lr:.1e} tok/s={tps:,.0f} "
                    f"time={total_t/60:.0f}m ETA={eta/60:.0f}m")
        running_loss = 0.0
        running_tokens = 0
        log_t = time.time()

    # Save
    if step % SAVE_EVERY == 0:
        path = f"/kaggle/working/usn_step{step}.usn"
        USNWriter().save(path, base_model, config=config,
                        metadata={"step": str(step), "loss": f"{loss_history[-1]:.4f}"})
        logger.info(f"Saved: {path}")

total_time = time.time() - t0
pbar.close()
print(f"\n{'='*60}")
print(f"Done! {total_time/60:.0f} min | Final loss: {loss_history[-1]:.3f} | PPL: {math.exp(loss_history[-1]):.0f}")
print(f"{'='*60}")

# %% Save Final Model

FINAL_PATH = "/kaggle/working/usn_pretrained_final.usn"
USNWriter().save(FINAL_PATH, base_model, config=config, metadata={
    "name": "USN-Pretrained",
    "params": str(base_model.num_parameters),
    "steps": str(step),
    "loss": f"{loss_history[-1]:.4f}",
    "ppl": f"{math.exp(loss_history[-1]):.0f}",
    "time_hours": f"{total_time/3600:.2f}",
    "data": "wikitext-103+gsm8k",
    "tokenizer": "gpt2",
    "hardware": f"{NUM_GPUS}x T4",
})
print(f"Final model: {FINAL_PATH} ({os.path.getsize(FINAL_PATH)/1024**2:.0f}MB)")

# %% Generation Test

print(f"\n{'='*60}")
print("GENERATION TEST")
print(f"{'='*60}\n")

base_model.eval()

@torch.no_grad()
def gen(prompt, max_tok=100, temp=0.8, top_k=40):
    toks = tokenizer.encode(prompt)
    state = None
    for t in toks:
        inp = torch.tensor([[t]], device=DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, state = base_model(inp, initial_state=state)
    
    out = list(toks)
    for _ in range(max_tok):
        l = logits[0,-1,:].float() / temp
        if top_k > 0:
            v, _ = l.topk(top_k)
            l[l < v[-1]] = float("-inf")
        p = torch.softmax(l, -1)
        tok = torch.multinomial(p, 1).item()
        if tok == tokenizer.eos_token_id:
            break
        out.append(tok)
        inp = torch.tensor([[tok]], device=DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, state = base_model(inp, initial_state=state)
    return tokenizer.decode(out)

for p in ["The meaning of life is", "Artificial intelligence",
           "Q: What is 15 + 28?\nA:", "The solar system",
           "In machine learning,"]:
    print(f">>> {p}")
    print(gen(p))
    print()

# %% Loss Curve

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(LOG_EVERY, len(loss_history)*LOG_EVERY+1, LOG_EVERY), loss_history)
ax.set(xlabel="Step", ylabel="Loss", title="USN Pretraining Loss")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/kaggle/working/loss_curve.png", dpi=150)
plt.show()

# %% [markdown]
# ## Loading the model for inference (separate script)
# ```python
# import torch
# from transformers import GPT2TokenizerFast
# from usn import USNModel
# from usn.serialization.reader import USNReader
#
# tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# reader = USNReader()
# data = reader.load("usn_pretrained_final.usn")
# model = USNModel(data["config"])
# weights = {k:v for k,v in data["weights"].items() if not k.startswith("__buffer__.")}
# model.load_state_dict(weights, strict=False)
# model.eval().cuda()
#
# # Generate
# state = None
# for t in tokenizer.encode("Hello world"):
#     logits, state = model(torch.tensor([[t]], device="cuda"), initial_state=state)
# # Continue generating from state...
# ```
