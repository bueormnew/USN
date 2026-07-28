"""Train a USN model to memorize 100 short text phrases.

Tests whether the architecture can represent and reproduce natural language text.
Uses a character-level tokenizer and a small model (~500K-2M params).
"""

import sys
import time

sys.path.insert(0, ".")

import torch
from torch.utils.data import DataLoader

from usn.config import USNConfig, USNTrainingConfig
from usn.datasets.collate import usn_collate_fn
from usn.datasets.usn_dataset import USNDataset
from usn.inference import USNGenerator
from usn.models import USNModel
from usn.tokenizers.char_tokenizer import CharTokenizer
from usn.training import USNTrainer

# ─── 100 frases cortas en español e inglés ───────────────────────────────────

TEXTS = [
    "El sol brilla con fuerza hoy en la ciudad",
    "La inteligencia artificial cambiara el mundo",
    "Los gatos duermen dieciseis horas al dia",
    "Python es un lenguaje de programacion versatil",
    "El oceano cubre mas del setenta por ciento de la tierra",
    "La musica clasica relaja la mente y el cuerpo",
    "Los arboles producen oxigeno para el planeta",
    "El cafe es la bebida mas popular del mundo",
    "La luna orbita la tierra cada veintiocho dias",
    "Las abejas son esenciales para la polinizacion",
    "El cerebro humano tiene cien mil millones de neuronas",
    "La gravedad mantiene los planetas en orbita",
    "El agua hierve a cien grados centigrados",
    "Los dinosaurios se extinguieron hace millones de anos",
    "La fotosintesis convierte luz solar en energia",
    "El universo tiene aproximadamente catorce mil millones de anos",
    "Los delfines son mamiferos muy inteligentes",
    "La electricidad fluye por cables de cobre",
    "El ADN contiene la informacion genetica de los seres vivos",
    "Las estrellas brillan por fusion nuclear de hidrogeno",
    "The quick brown fox jumps over the lazy dog",
    "Artificial intelligence is transforming every industry",
    "Deep learning models require large amounts of data",
    "Neural networks learn patterns from examples",
    "The human brain processes information in parallel",
    "Quantum computers will solve complex problems faster",
    "Climate change affects every ecosystem on earth",
    "Renewable energy sources include solar and wind",
    "The internet connects billions of people worldwide",
    "Machine learning algorithms improve with more data",
    "Space exploration reveals the mysteries of the cosmos",
    "DNA sequencing has revolutionized modern medicine",
    "Electric vehicles reduce carbon emissions significantly",
    "Blockchain technology enables decentralized systems",
    "Virtual reality creates immersive digital experiences",
    "Robotics automation increases manufacturing efficiency",
    "Natural language processing understands human text",
    "Computer vision enables machines to see and interpret",
    "Cloud computing provides scalable resources on demand",
    "Cybersecurity protects digital systems from threats",
    "La tierra gira alrededor del sol en un ano",
    "Los volcanes expulsan lava y ceniza al exterior",
    "El corazon humano late cien mil veces al dia",
    "Las plantas necesitan agua luz y nutrientes",
    "El hierro es el metal mas utilizado en construccion",
    "Los rios llevan agua dulce hacia el oceano",
    "La velocidad de la luz es trescientos mil kilometros",
    "Los antibioticos combaten las infecciones bacterianas",
    "El telescopio permite observar galaxias lejanas",
    "La matematica es el lenguaje de la ciencia",
    "Los glaciares almacenan el setenta por ciento del agua dulce",
    "El viento es aire en movimiento por diferencias de presion",
    "Las vitaminas son nutrientes esenciales para la salud",
    "El acero es una aleacion de hierro y carbono",
    "Los huesos del cuerpo humano son doscientos seis",
    "La fotografia captura momentos en el tiempo",
    "Los satelites orbitan la tierra a gran velocidad",
    "El petroleo es un recurso natural no renovable",
    "Las neuronas se comunican mediante impulsos electricos",
    "La penicilina fue descubierta por Alexander Fleming",
    "Transformers use attention mechanisms for sequence modeling",
    "State space models achieve linear complexity in training",
    "Recurrent networks maintain hidden state across timesteps",
    "Gradient descent optimizes neural network parameters",
    "Backpropagation computes gradients through the network",
    "Batch normalization stabilizes deep network training",
    "Dropout regularization prevents model overfitting",
    "Learning rate schedules improve convergence speed",
    "Convolutional layers extract local spatial features",
    "Embedding layers map discrete tokens to continuous vectors",
    "Softmax converts logits to probability distributions",
    "Cross entropy loss measures prediction quality",
    "Adam optimizer combines momentum with adaptive learning",
    "Weight decay prevents parameters from growing too large",
    "Residual connections enable training very deep networks",
    "Layer normalization stabilizes activations across features",
    "Attention mechanisms weigh the importance of each position",
    "Positional encoding adds sequence order information",
    "Beam search explores multiple generation hypotheses",
    "Temperature scaling controls generation randomness",
    "Los modelos de lenguaje predicen la siguiente palabra",
    "El entrenamiento requiere muchos datos y computo",
    "La memoria del modelo almacena informacion del contexto",
    "Las compuertas controlan el flujo de informacion",
    "El estado persistente resume el historial completo",
    "La normalizacion estabiliza el entrenamiento profundo",
    "Los optimizadores ajustan los pesos del modelo",
    "La funcion de perdida mide el error de prediccion",
    "El gradiente indica la direccion de mejora",
    "La tasa de aprendizaje controla el tamano del paso",
    "Un modelo grande tiene miles de millones de parametros",
    "La inferencia genera texto token por token",
    "El vocabulario mapea palabras a numeros enteros",
    "La tokenizacion divide el texto en unidades minimas",
    "El contexto largo requiere memoria eficiente",
    "La paralelizacion acelera el entrenamiento en GPU",
    "Las redes profundas aprenden representaciones jerarquicas",
    "El sobreajuste ocurre cuando el modelo memoriza datos",
    "La generalizacion mide el rendimiento en datos nuevos",
    "El preentrenamiento aprende conocimiento general del lenguaje",
]

print("=" * 60)
print("USN Text Memorization Test")
print("=" * 60)
print(f"Corpus: {len(TEXTS)} phrases")
print(f"Total chars: {sum(len(t) for t in TEXTS):,}")
print()

# ─── Build tokenizer from corpus ─────────────────────────────────────────────

all_text = " ".join(TEXTS)
tokenizer = CharTokenizer.from_text(all_text)
print(f"Tokenizer: {tokenizer.vocab_size} chars (character-level)")

# ─── Create model ────────────────────────────────────────────────────────────

config = USNConfig(
    num_layers=3,
    d_model=96,
    d_s=48,
    k=6,
    d_ff=192,
    vocab_size=tokenizer.vocab_size,
    max_seq_len=48,
    tie_weights=True,
    fused=False,
    dropout=0.0,
    embedding_dropout=0.0,
    residual_dropout=0.0,
)
model = USNModel(config)
print(f"Model: {model.num_parameters:,} parameters")
print(f"State per layer: {model.state_size_per_layer} floats ({model.state_size_per_layer * 4} bytes)")
print(f"Total state: {model.total_state_size} floats ({model.total_state_size * 4} bytes)")
print()

# ─── Create dataset ──────────────────────────────────────────────────────────

dataset = USNDataset(TEXTS, tokenizer, max_seq_len=45)
print(f"Dataset: {len(dataset)} samples, max_seq_len=45")
print()

# ─── Train ───────────────────────────────────────────────────────────────────

train_config = USNTrainingConfig(
    learning_rate=5e-3,
    batch_size=50,
    max_steps=1000,
    warmup_steps=10,
    mixed_precision="none",
    gradient_accumulation_steps=1,
    log_interval=100,
    eval_interval=0,
    checkpoint_interval=0,
    weight_decay=0.0,
    scheduler_type="cosine",
    min_lr=1e-4,
)

print("Training (1000 steps)...")
print("-" * 60)
start_time = time.time()

trainer = USNTrainer(model, dataset, train_config)
result = trainer.train()

elapsed = time.time() - start_time
print("-" * 60)
print(f"Training time: {elapsed:.1f}s")
print(f"Final loss: {result['loss_history'][-1]:.4f}")
print(f"Loss reduction: {(1 - result['loss_history'][-1] / result['loss_history'][0]) * 100:.1f}%")
print()

# ─── Test memorization via generation ────────────────────────────────────────

print("=" * 60)
print("GENERATION TEST (memorization check)")
print("=" * 60)
print()

model.eval()
generator = USNGenerator(model, tokenizer)

# Test with beginnings of phrases from the training set
test_prompts = [
    "El sol brilla",
    "The quick brown",
    "Neural networks",
    "La tierra gira",
    "Deep learning",
    "Los gatos duer",
    "Artificial intel",
    "El cerebro human",
    "Gradient descent",
    "La musica clasica",
]

for prompt in test_prompts:
    output = generator.generate(prompt, max_new_tokens=30, temperature=0)
    tokens = output.token_ids[0].tolist()
    # Filter padding
    tokens = [t for t in tokens if t != tokenizer.pad_token_id]
    generated = tokenizer.decode(tokens)
    print(f"  Prompt:    '{prompt}'")
    print(f"  Generated: '{prompt}{generated}'")
    
    # Check if it matches any training phrase
    full = prompt + generated
    matches = [t for t in TEXTS if t.startswith(full[:len(t)])]
    if matches:
        print(f"  Match:     ✓ (matches training data)")
    else:
        # Partial match check
        partial = [t for t in TEXTS if t.startswith(prompt)]
        if partial:
            expected = partial[0][len(prompt):]
            overlap = 0
            for i, (a, b) in enumerate(zip(generated, expected)):
                if a == b:
                    overlap += 1
                else:
                    break
            accuracy = overlap / max(len(expected), 1) * 100
            print(f"  Expected:  '{expected[:30]}'")
            print(f"  Accuracy:  {accuracy:.0f}% ({overlap}/{min(len(expected), 30)} chars)")
        else:
            print(f"  (no matching phrase in corpus)")
    print()

# ─── Loss curve summary ──────────────────────────────────────────────────────

print("=" * 60)
print("LOSS CURVE")
print("=" * 60)
history = result["loss_history"]
for i, loss in enumerate(history):
    step = (i + 1) * train_config.log_interval
    bar_len = int(loss * 10)
    bar = "█" * bar_len
    print(f"  Step {step:4d}: {loss:.4f} {bar}")

print()
print(f"Initial: {history[0]:.4f}")
print(f"Final:   {history[-1]:.4f}")
print(f"Reduction: {(1 - history[-1] / history[0]) * 100:.1f}%")

# Verdict
if history[-1] < 1.0:
    print("\n✓ Model is memorizing the corpus well (loss < 1.0)")
elif history[-1] < 2.0:
    print("\n~ Model is learning but needs more training (1.0 < loss < 2.0)")
else:
    print("\n✗ Model needs more capacity or training steps (loss > 2.0)")
