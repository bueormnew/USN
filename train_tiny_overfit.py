"""Extreme overfit test: 5 short phrases, prove the model CAN memorize."""
import sys, time
sys.path.insert(0, ".")
import torch
from usn.config import USNConfig, USNTrainingConfig
from usn.models import USNModel
from usn.datasets.usn_dataset import USNDataset
from usn.tokenizers.char_tokenizer import CharTokenizer
from usn.training import USNTrainer

TEXTS = [
    "hola mundo",
    "gato azul",
    "sol grande",
    "rio verde",
    "luna roja",
]

print("=== EXTREME OVERFIT TEST (5 phrases) ===")
tokenizer = CharTokenizer.from_text(" ".join(TEXTS))
print(f"Vocab: {tokenizer.vocab_size} chars")

config = USNConfig(
    num_layers=3, d_model=64, d_s=48, k=4, d_ff=128,
    vocab_size=tokenizer.vocab_size, max_seq_len=12,
    tie_weights=True, fused=False, dropout=0.0,
    embedding_dropout=0.0, residual_dropout=0.0,
)
model = USNModel(config)
print(f"Params: {model.num_parameters:,}")

dataset = USNDataset(TEXTS, tokenizer, max_seq_len=11)
print(f"Samples: {len(dataset)}")

train_config = USNTrainingConfig(
    learning_rate=1e-2, batch_size=5, max_steps=5000,
    warmup_steps=100, mixed_precision="none",
    gradient_accumulation_steps=1, log_interval=500,
    eval_interval=0, checkpoint_interval=0,
    weight_decay=0.0, scheduler_type="cosine", min_lr=1e-5,
)

start = time.time()
trainer = USNTrainer(model, dataset, train_config)
result = trainer.train()
elapsed = time.time() - start
print(f"\nTime: {elapsed:.0f}s")
print(f"Loss: {result['loss_history'][0]:.4f} -> {result['loss_history'][-1]:.6f}")

# Direct autoregressive generation (bypass Generator, use model directly)
print("\n=== DIRECT AUTOREGRESSIVE GENERATION ===")
model.eval()
for text in TEXTS:
    tokens = tokenizer.encode(text)
    prompt_len = 3  # Use first 3 chars as prompt
    prompt_tokens = tokens[:prompt_len]

    # Generate token by token using model forward
    state = None
    generated = list(prompt_tokens)

    # Prefill: process prompt tokens one by one
    for tok in prompt_tokens:
        input_t = torch.tensor([[tok]], dtype=torch.long)
        with torch.no_grad():
            logits, state = model(input_t, initial_state=state)

    # Generate: use argmax of last logit as next token
    for _ in range(len(tokens) - prompt_len):
        next_tok = logits[0, -1].argmax().item()
        generated.append(next_tok)
        input_t = torch.tensor([[next_tok]], dtype=torch.long)
        with torch.no_grad():
            logits, state = model(input_t, initial_state=state)

    gen_text = tokenizer.decode(generated)
    match = gen_text.strip() == text.strip()
    print(f"  {'✓' if match else '✗'} expected='{text}' generated='{gen_text}'")
