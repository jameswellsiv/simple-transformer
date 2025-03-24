import torch
from simple_transformer.transformer.model import SimpleTransformer
from simple_transformer.tokenizers.character import CharacterTokenizer

# === Load Tokenizer & Vocab ===
with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

tokenizer = CharacterTokenizer(full_text)
vocab_size = tokenizer.vocab_size

# === Model Setup ===
# For these, check the hyperparameters in train.py
embedding_dim = 256 # Must match training
block_size = 64     # Must match training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleTransformer(embedding_dim, vocab_size).to(device)

# === Load Checkpoint ===
checkpoint = torch.load("checkpoint_epoch_10.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# === Sampling Function ===
@torch.no_grad()
def generate(model, tokenizer, start_text, max_new_tokens=100):
    model.eval()
    input_ids = tokenizer.encode(start_text)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # Trim input to block size if necessary
        input_crop = input_tensor[:, -block_size:]

        logits = model(input_crop)  # (1, seq_len, vocab_size)
        next_token_logits = logits[:, -1, :]  # (1, vocab_size)
        probs = torch.softmax(next_token_logits, dim=-1)  # (1, vocab_size)

        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        input_tensor = torch.cat((input_tensor, next_token), dim=1)

    generated_text = tokenizer.decode(input_tensor[0].tolist())
    return generated_text

# === Generate ===
prompt = "To be or not..."
generated = generate(model, tokenizer, start_text=prompt, max_new_tokens=100)
print("Prompt:", prompt)
print("Generated:\n", generated)
