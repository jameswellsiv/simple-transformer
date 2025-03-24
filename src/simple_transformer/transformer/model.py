import torch
import torch.nn as nn
from simple_transformer.transformer.positional_encoding import get_positional_encoding
from simple_transformer.transformer.transformer_block import TransformerBlock

class SimpleTransformer(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)

        self.transformer_block = TransformerBlock(embedding_dim)

        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor):
        # 'x' should be tokenized at this point
        
        # Embed the tokens
        x_embedding = self.embedding_layer(x)

        # Add positional encodings to embedding
        seq_len = x_embedding.shape[1]
        pos_enc = get_positional_encoding(seq_len, self.embedding_dim, device=x.device)
        x_w_pos = x_embedding + pos_enc

        # Pass through transformer block
        attn_out = self.transformer_block(x_w_pos)

        # Pass through feedforward to get logits
        logits = self.output_layer(attn_out)
        
        return logits

        
if __name__ == "__main__":
    vocab = list("abcdefghijklmnopqrstuvwxyz ")  # 27 characters
    stoi = {ch: i for i, ch in enumerate(vocab)}

    sample_text = "hello world"
    token_ids = [stoi[ch] for ch in sample_text]

    x = torch.tensor(token_ids).unsqueeze(0)  # shape: (1, seq_len)

    model = SimpleTransformer(embedding_dim=32, vocab_size=len(vocab))
    output_logits = model(x)

    print("Input shape:", x.shape)
    print("Logits shape:", output_logits.shape)  # Should be (1, seq_len, vocab_size)