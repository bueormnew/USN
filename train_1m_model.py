"""Train a ~1M parameter USN model on 600 text samples for memorization.

Larger state, more data, multiple epochs to test generation quality.
"""

import sys
import time
import random

sys.path.insert(0, ".")

import torch
from usn.config import USNConfig, USNTrainingConfig
from usn.models import USNModel
from usn.datasets.usn_dataset import USNDataset
from usn.tokenizers.char_tokenizer import CharTokenizer
from usn.training import USNTrainer
from usn.inference import USNGenerator

# ─── Generate 600 diverse text samples ────────────────────────────────────────

TEMPLATES_ES = [
    "El {adj} {animal} {verbo} en el {lugar}",
    "La {cosa} es {adj2} y {adj3}",
    "Los {plural} necesitan {cosa2} para vivir",
    "{nombre} trabaja como {profesion} en {ciudad}",
    "El color {color} representa {emocion} y {concepto}",
    "Durante el {estacion} las {plural2} {verbo2}",
    "La temperatura hoy es de {num} grados",
    "{nombre2} y {nombre3} estudian {materia} juntos",
    "El mejor {comida} se prepara con {ingrediente}",
    "Un buen {profesion2} siempre {accion} con cuidado",
]

TEMPLATES_EN = [
    "The {adj_en} {animal_en} {verb_en} in the {place_en}",
    "A {thing_en} is {adj2_en} and {adj3_en}",
    "{name_en} works as a {job_en} in {city_en}",
    "The {color_en} color represents {emotion_en}",
    "During {season_en} the {plural_en} {verb2_en}",
    "The temperature today is {num} degrees",
    "{name2_en} and {name3_en} study {subject_en} together",
    "The best {food_en} is made with {ingredient_en}",
    "A good {job2_en} always {action_en} carefully",
    "Every {day_en} we learn something {adj4_en}",
]

VOCAB = {
    "adj": ["grande", "pequeno", "rapido", "lento", "fuerte", "debil", "viejo", "joven", "alto", "bajo"],
    "animal": ["gato", "perro", "pajaro", "pez", "caballo", "leon", "tigre", "oso", "lobo", "aguila"],
    "verbo": ["corre", "duerme", "come", "juega", "descansa", "nada", "vuela", "salta", "camina", "canta"],
    "lugar": ["parque", "bosque", "rio", "monte", "jardin", "campo", "lago", "mar", "desierto", "valle"],
    "cosa": ["ciencia", "musica", "pintura", "lectura", "cocina", "medicina", "historia", "fisica", "quimica", "biologia"],
    "adj2": ["importante", "necesaria", "compleja", "simple", "antigua", "moderna", "util", "bella", "dificil", "facil"],
    "adj3": ["poderosa", "creativa", "educativa", "practica", "natural", "artificial", "profunda", "clara", "directa", "libre"],
    "plural": ["animales", "plantas", "personas", "ninos", "arboles", "flores", "rios", "montanas", "ciudades", "pueblos"],
    "cosa2": ["agua", "comida", "luz", "aire", "calor", "espacio", "tiempo", "energia", "oxigeno", "nutrientes"],
    "nombre": ["Carlos", "Maria", "Pedro", "Ana", "Luis", "Elena", "Jorge", "Sofia", "Miguel", "Laura"],
    "profesion": ["medico", "profesor", "ingeniero", "artista", "musico", "escritor", "cientfico", "cocinero", "piloto", "abogado"],
    "ciudad": ["Madrid", "Barcelona", "Lima", "Mexico", "Buenos Aires", "Bogota", "Santiago", "Quito", "Havana", "Caracas"],
    "color": ["rojo", "azul", "verde", "amarillo", "naranja", "violeta", "blanco", "negro", "gris", "rosa"],
    "emocion": ["alegria", "calma", "fuerza", "pasion", "paz", "amor", "esperanza", "valor", "libertad", "armonia"],
    "concepto": ["poder", "verdad", "belleza", "sabiduria", "justicia", "honor", "respeto", "unidad", "progreso", "futuro"],
    "estacion": ["verano", "invierno", "otono", "primavera"],
    "plural2": ["hojas", "flores", "aves", "nubes", "lluvias", "noches", "estrellas", "mareas", "tormentas", "nieves"],
    "verbo2": ["caen", "florecen", "migran", "brillan", "crecen", "cambian", "aparecen", "danzan", "surgen", "vuelven"],
    "num": ["diez", "veinte", "treinta", "quince", "cinco", "ocho", "doce", "siete", "nueve", "once"],
    "nombre2": ["Daniel", "Carmen", "Pablo", "Isabel", "Andres", "Rosa", "Diego", "Clara", "Oscar", "Julia"],
    "nombre3": ["Fernando", "Lucia", "Ricardo", "Marta", "Alberto", "Teresa", "Sergio", "Alicia", "Raul", "Monica"],
    "materia": ["matematicas", "historia", "biologia", "fisica", "quimica", "literatura", "filosofia", "economia", "arte", "musica"],
    "comida": ["pan", "arroz", "pasta", "pescado", "pollo", "ensalada", "sopa", "postre", "guiso", "caldo"],
    "ingrediente": ["tomate", "cebolla", "ajo", "aceite", "sal", "pimienta", "limon", "mantequilla", "harina", "huevo"],
    "profesion2": ["maestro", "doctor", "chef", "arquitecto", "programador", "detective", "bombero", "enfermero", "veterinario", "farmaceutico"],
    "accion": ["trabaja", "piensa", "planifica", "observa", "escucha", "analiza", "resuelve", "construye", "disena", "investiga"],
    # English
    "adj_en": ["big", "small", "fast", "slow", "strong", "weak", "old", "young", "tall", "short"],
    "animal_en": ["cat", "dog", "bird", "fish", "horse", "lion", "tiger", "bear", "wolf", "eagle"],
    "verb_en": ["runs", "sleeps", "eats", "plays", "rests", "swims", "flies", "jumps", "walks", "sings"],
    "place_en": ["park", "forest", "river", "mountain", "garden", "field", "lake", "sea", "desert", "valley"],
    "thing_en": ["science", "music", "painting", "reading", "cooking", "medicine", "history", "physics", "math", "biology"],
    "adj2_en": ["important", "necessary", "complex", "simple", "ancient", "modern", "useful", "beautiful", "hard", "easy"],
    "adj3_en": ["powerful", "creative", "educational", "practical", "natural", "deep", "clear", "direct", "free", "open"],
    "name_en": ["John", "Mary", "Peter", "Anna", "James", "Elena", "George", "Sarah", "Michael", "Laura"],
    "job_en": ["doctor", "teacher", "engineer", "artist", "musician", "writer", "scientist", "chef", "pilot", "lawyer"],
    "city_en": ["London", "Paris", "Tokyo", "Berlin", "Rome", "Sydney", "Toronto", "Dublin", "Oslo", "Vienna"],
    "color_en": ["red", "blue", "green", "yellow", "orange", "purple", "white", "black", "gray", "pink"],
    "emotion_en": ["joy", "calm", "strength", "passion", "peace", "love", "hope", "courage", "freedom", "harmony"],
    "season_en": ["summer", "winter", "autumn", "spring"],
    "plural_en": ["leaves", "flowers", "birds", "clouds", "rains", "nights", "stars", "waves", "storms", "trees"],
    "verb2_en": ["fall", "bloom", "migrate", "shine", "grow", "change", "appear", "dance", "emerge", "return"],
    "name2_en": ["Daniel", "Carmen", "Paul", "Isabel", "Andrew", "Rose", "David", "Claire", "Oscar", "Julia"],
    "name3_en": ["Fernando", "Lucy", "Richard", "Martha", "Albert", "Teresa", "Steven", "Alice", "Ralph", "Monica"],
    "subject_en": ["math", "history", "biology", "physics", "chemistry", "literature", "philosophy", "economics", "art", "music"],
    "food_en": ["bread", "rice", "pasta", "fish", "chicken", "salad", "soup", "dessert", "stew", "broth"],
    "ingredient_en": ["tomato", "onion", "garlic", "oil", "salt", "pepper", "lemon", "butter", "flour", "egg"],
    "job2_en": ["teacher", "doctor", "chef", "architect", "programmer", "detective", "nurse", "vet", "pharmacist", "designer"],
    "action_en": ["works", "thinks", "plans", "observes", "listens", "analyzes", "solves", "builds", "designs", "studies"],
    "day_en": ["morning", "day", "evening", "night", "week", "month", "year", "moment", "hour", "second"],
    "adj4_en": ["new", "useful", "important", "interesting", "surprising", "valuable", "meaningful", "practical", "wonderful", "essential"],
}


def generate_samples(n=600, seed=42):
    """Generate n diverse text samples from templates."""
    rng = random.Random(seed)
    samples = set()
    
    while len(samples) < n:
        # Pick a template
        if rng.random() < 0.5:
            template = rng.choice(TEMPLATES_ES)
        else:
            template = rng.choice(TEMPLATES_EN)
        
        # Fill in slots
        text = template
        for key, values in VOCAB.items():
            placeholder = "{" + key + "}"
            if placeholder in text:
                text = text.replace(placeholder, rng.choice(values), 1)
        
        if len(text) > 10 and text not in samples:
            samples.add(text)
    
    return list(samples)[:n]


# ─── Generate data ───────────────────────────────────────────────────────────

TEXTS = generate_samples(600)

print("=" * 60)
print("USN 1M Parameter Memorization Test")
print("=" * 60)
print(f"Corpus: {len(TEXTS)} phrases")
print(f"Total chars: {sum(len(t) for t in TEXTS):,}")
print(f"Avg length: {sum(len(t) for t in TEXTS) / len(TEXTS):.0f} chars")
print(f"Examples:")
for t in TEXTS[:5]:
    print(f"  - {t}")
print()

# ─── Build tokenizer ─────────────────────────────────────────────────────────

all_text = " ".join(TEXTS)
tokenizer = CharTokenizer.from_text(all_text)
print(f"Tokenizer: {tokenizer.vocab_size} unique chars")

# ─── Create ~1M param model with larger state ────────────────────────────────

config = USNConfig(
    num_layers=4,
    d_model=160,
    d_s=128,       # Larger semantic state
    k=12,          # Larger relational state (12x12 = 144)
    d_ff=320,
    vocab_size=tokenizer.vocab_size,
    max_seq_len=50,
    tie_weights=True,
    fused=False,
    dropout=0.0,
    embedding_dropout=0.0,
    residual_dropout=0.0,
)
model = USNModel(config)
print(f"Model: {model.num_parameters:,} parameters")
print(f"State per layer: {model.state_size_per_layer} floats = {model.state_size_per_layer * 4} bytes")
print(f"Total state: {model.total_state_size} floats = {model.total_state_size * 4} bytes")
print()

# ─── Create dataset ──────────────────────────────────────────────────────────

dataset = USNDataset(TEXTS, tokenizer, max_seq_len=48)
print(f"Dataset: {len(dataset)} samples, max_seq_len=48")
print()

# ─── Train (multiple epochs via high step count) ─────────────────────────────

# 600 samples / batch_size 50 = 12 steps per epoch
# 3 epochs ~ 36 steps, but we want heavier training
# Let's do ~500 steps = ~40 epochs of overfit
steps = 1500
batch_size = 50

train_config = USNTrainingConfig(
    learning_rate=3e-3,
    batch_size=batch_size,
    max_steps=steps,
    warmup_steps=20,
    mixed_precision="none",
    gradient_accumulation_steps=1,
    log_interval=150,
    eval_interval=0,
    checkpoint_interval=0,
    weight_decay=0.0,
    scheduler_type="cosine",
    min_lr=5e-5,
)

epochs_approx = steps * batch_size / len(dataset)
print(f"Training: {steps} steps, batch={batch_size} (~{epochs_approx:.0f} epochs)")
print("-" * 60)
start_time = time.time()

trainer = USNTrainer(model, dataset, train_config)
result = trainer.train()

elapsed = time.time() - start_time
print("-" * 60)
print(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Final loss: {result['loss_history'][-1]:.4f}")
print(f"Loss reduction: {(1 - result['loss_history'][-1] / result['loss_history'][0]) * 100:.1f}%")
print()

# ─── Generation test ─────────────────────────────────────────────────────────

print("=" * 60)
print("GENERATION TEST")
print("=" * 60)
print()

model.eval()
generator = USNGenerator(model, tokenizer)

# Pick some prompts from actual training data
test_indices = [0, 10, 50, 100, 150, 200, 300, 400, 500, 550]
correct = 0
total = 0

for idx in test_indices:
    if idx >= len(TEXTS):
        continue
    full_text = TEXTS[idx]
    # Use first ~15 chars as prompt
    prompt_len = min(15, len(full_text) // 2)
    prompt = full_text[:prompt_len]
    expected_rest = full_text[prompt_len:]
    
    output = generator.generate(prompt, max_new_tokens=len(expected_rest) + 5, temperature=0)
    tokens = output.token_ids[0].tolist()
    tokens = [t for t in tokens if t != tokenizer.pad_token_id]
    generated = tokenizer.decode(tokens)
    
    # Check character-level accuracy
    overlap = 0
    for i, (a, b) in enumerate(zip(generated, expected_rest)):
        if a == b:
            overlap += 1
        else:
            break
    accuracy = overlap / max(len(expected_rest), 1) * 100
    
    total += 1
    if accuracy > 50:
        correct += 1
    
    status = "✓" if accuracy > 50 else "✗"
    print(f"  [{status}] Prompt: '{prompt}...'")
    print(f"       Generated: '{prompt}{generated[:35]}'")
    print(f"       Expected:  '{full_text[:50]}'")
    print(f"       Accuracy:  {accuracy:.0f}% ({overlap} chars correct)")
    print()

print("=" * 60)
print(f"OVERALL: {correct}/{total} phrases reproduced >50% ({correct/max(total,1)*100:.0f}%)")
print()

# ─── Loss curve ──────────────────────────────────────────────────────────────

print("LOSS CURVE:")
history = result["loss_history"]
for i, loss in enumerate(history):
    step = (i + 1) * train_config.log_interval
    bar_len = min(int(loss * 8), 50)
    bar = "█" * bar_len
    print(f"  Step {step:4d}: {loss:.4f} {bar}")

print()
print(f"Initial: {history[0]:.4f}")
print(f"Final:   {history[-1]:.4f}")

if history[-1] < 0.01:
    print("\n✓✓ PERFECT MEMORIZATION (loss < 0.01)")
elif history[-1] < 0.1:
    print("\n✓ STRONG MEMORIZATION (loss < 0.1)")
elif history[-1] < 0.5:
    print("\n~ PARTIAL MEMORIZATION (loss < 0.5)")
else:
    print("\n✗ NEEDS MORE TRAINING (loss > 0.5)")
