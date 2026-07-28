"""Direct inference test: verify token-by-token matches full sequence."""
import sys
sys.path.insert(0, ".")
import torch
from usn.config import USNConfig
from usn.models import USNModel
from usn.tokenizers.char_tokenizer import CharTokenizer

TEXTS = ["hola mundo", "gato azul", "sol grande", "rio verde", "luna roja"]
tokenizer = CharTokenizer.from_text(" ".join(TEXTS))

config = USNConfig(
    num_layers=3, d_model=64, d_s=48, k=4, d_ff=128,
    vocab_size=tokenizer.vocab_size, max_seq_len=12,
    tie_weights=True, fused=False, dropout=0.0,
    embedding_dropout=0.0, residual_dropout=0.0,
)

torch.manual_seed(42)
model = USNModel(config)
model.eval()

# Test: process "hola mundo" as full sequence vs token-by-token
text = "hola mundo"
tokens = tokenizer.encode(text)
print(f"Tokens for '{text}': {tokens}")

# Full sequence forward
input_full = torch.tensor([tokens], dtype=torch.long)
with torch.no_grad():
    logits_full, state_full = model(input_full)

print(f"\nFull sequence logits shape: {logits_full.shape}")
print(f"Full seq argmax at each pos: {logits_full[0].argmax(dim=-1).tolist()}")

# Token-by-token forward (simulating inference)
state = None
logits_step = []
for i, tok in enumerate(tokens):
    input_tok = torch.tensor([[tok]], dtype=torch.long)
    with torch.no_grad():
        logits_t, state = model(input_tok, initial_state=state)
    logits_step.append(logits_t[0, 0])

logits_step_tensor = torch.stack(logits_step)
print(f"\nStep-by-step argmax at each pos: {logits_step_tensor.argmax(dim=-1).tolist()}")

# Compare
diff = (logits_full[0] - logits_step_tensor).abs().max().item()
print(f"\nMax diff between full and step-by-step: {diff:.6f}")

if diff < 0.01:
    print("✓ Full sequence and step-by-step produce SAME results")
else:
    print("✗ MISMATCH between full and step-by-step!")
    # Show where they differ
    for i in range(len(tokens)):
        d = (logits_full[0, i] - logits_step[i]).abs().max().item()
        print(f"  Position {i} ('{tokenizer.decode([tokens[i]])}') diff: {d:.4f}")
