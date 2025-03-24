

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from simple_transformer.transformer.model import SimpleTransformer
from simple_transformer.dataloaders.shakespeare import ShakespeareDataset
from simple_transformer.tokenizers.character import CharacterTokenizer

# === Hyperparameters ===
embedding_dim = 256
block_size = 64
batch_size = 32
num_epochs = 10
learning_rate = 3e-4

# === Load & Split Text ===
with open("/root/workspace/my-transformer/data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

split_ratio = 0.9
split_idx = int(len(text) * split_ratio)
train_text = text[:split_idx]
val_text = text[split_idx:]

# === Tokenizer Setup ===
tokenizer = CharacterTokenizer(train_text)
vocab_size = tokenizer.vocab_size  # assumes your tokenizer has a .vocab attribute

# === Datasets & Dataloaders ===
train_dataset = ShakespeareDataset(train_text, tokenizer, block_size)
val_dataset = ShakespeareDataset(val_text, tokenizer, block_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)

# === Model, Loss, Optimizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleTransformer(embedding_dim, vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# === Training Loop ===
for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # === Validation ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    # === Save checkpoint ===
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
    }
    torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pt")
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    