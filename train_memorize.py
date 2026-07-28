"""Memorization test: 50 short phrases, heavy overfit to prove architecture works."""
import sys, time
sys.path.insert(0, ".")
import torch
from usn.config import USNConfig, USNTrainingConfig
from usn.models import USNModel
from usn.datasets.usn_dataset import USNDataset
from usn.tokenizers.char_tokenizer import CharTokenizer
from usn.training import USNTrainer
from usn.inference import USNGenerator

TEXTS = [
    "el gato duerme en la casa",
    "el perro corre en el parque",
    "maria come una manzana roja",
    "pedro lee un libro grande",
    "el sol brilla en el cielo",
    "la luna sale por la noche",
    "los pajaros vuelan muy alto",
    "el rio fluye hacia el mar",
    "ana estudia en la escuela",
    "juan trabaja en la oficina",
    "el arbol crece en el jardin",
    "la flor es roja y bonita",
    "carlos cocina una sopa rica",
    "elena pinta un cuadro azul",
    "el viento sopla con fuerza",
    "la lluvia cae desde el cielo",
    "los ninos juegan en la calle",
    "el tren llega a las ocho",
    "sofia canta una cancion bella",
    "miguel nada en la piscina",
    "the cat sleeps on the bed",
    "the dog runs in the park",
    "mary eats a red apple now",
    "peter reads a big book here",
    "the sun shines in the sky",
    "the moon rises every night",
    "the birds fly very high up",
    "the river flows to the sea",
    "anna studies at the school",
    "john works in the office",
    "the tree grows in garden",
    "the flower is red and nice",
    "carlos cooks a tasty soup",
    "elena paints a blue picture",
    "the wind blows with force",
    "the rain falls from above",
    "children play on the street",
    "the train arrives at eight",
    "sofia sings a pretty song",
    "miguel swims in the pool",
    "hoy es un dia muy bonito",
    "tengo hambre quiero comer",
    "mañana vamos a la playa",
    "me gusta mucho el helado",
    "today is a very nice day",
    "i am hungry i want to eat",
    "tomorrow we go to beach",
    "i really like ice cream",
    "la vida es bella y corta",
    "life is beautiful and short",
]

print(f"Corpus: {len(TEXTS)} phrases")
all_text = " ".join(TEXTS)
tokenizer = CharTokenizer.from_text(all_text)
print(f"Tokenizer: {tokenizer.vocab_size} chars")

config = USNConfig(
    num_layers=4, d_model=128, d_s=96, k=8, d_ff=256,
    vocab_size=tokenizer.vocab_size, max_seq_len=32,
    tie_weights=True, fused=False, dropout=0.0,
    embedding_dropout=0.0, residual_dropout=0.0,
)
model = USNModel(config)
print(f"Model: {model.num_parameters:,} params")
print(f"State: {model.total_state_size * 4} bytes total")

dataset = USNDataset(TEXTS, tokenizer, max_seq_len=30)
print(f"Dataset: {len(dataset)} samples")

# Heavy overfit: 2000 steps, batch covers most of dataset
train_config = USNTrainingConfig(
    learning_rate=5e-3, batch_size=25, max_steps=2000,
    warmup_steps=50, mixed_precision="none",
    gradient_accumulation_steps=1, log_interval=200,
    eval_interval=0, checkpoint_interval=0,
    weight_decay=0.0, scheduler_type="cosine", min_lr=1e-4,
)

print(f"\nTraining 2000 steps (~{2000*25//len(dataset)} epochs)...")
start = time.time()
trainer = USNTrainer(model, dataset, train_config)
result = trainer.train()
elapsed = time.time() - start

print(f"\nTime: {elapsed:.0f}s | Loss: {result['loss_history'][0]:.3f} -> {result['loss_history'][-1]:.4f}")
print(f"Loss curve: {[f'{l:.3f}' for l in result['loss_history']]}")

# Generation
print("\n=== GENERATION ===")
model.eval()
gen = USNGenerator(model, tokenizer)
correct = 0
for text in TEXTS[:20]:
    prompt = text[:10]
    expected = text[10:]
    out = gen.generate(prompt, max_new_tokens=len(expected)+2, temperature=0)
    toks = [t for t in out.token_ids[0].tolist() if t != tokenizer.pad_token_id]
    generated = tokenizer.decode(toks)
    match = 0
    for a, b in zip(generated, expected):
        if a == b: match += 1
        else: break
    acc = match / max(len(expected), 1) * 100
    status = "✓" if acc > 50 else "✗"
    if acc > 50: correct += 1
    print(f"  [{status}] '{prompt}' -> '{prompt}{generated[:20]}' (acc={acc:.0f}%)")

print(f"\nResult: {correct}/20 correct (>50%)")
