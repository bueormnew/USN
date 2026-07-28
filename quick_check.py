import sys
sys.path.insert(0, ".")
import torch
from usn.config import USNConfig
from usn.models import USNModel

config = USNConfig(num_layers=2, d_model=32, d_s=16, k=4, d_ff=64, vocab_size=20, max_seq_len=12, fused=False)
model = USNModel(config)
model.eval()
x = torch.randint(0, 20, (1, 5))
with torch.no_grad():
    logits, state = model(x)
print(f"Full: logits={logits.shape}")
with torch.no_grad():
    logits2, state2 = model(torch.randint(0, 20, (1, 1)), initial_state=state)
print(f"Step: logits={logits2.shape}")
print("ALL OK")
