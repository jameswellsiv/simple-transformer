import torch
import torch.nn as nn
from simple_transformer.transformer.attention import SelfAttention, MultiHeadAttention
from simple_transformer.transformer.attention import generate_causal_mask, generate_causal_mask_multi_head

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_heads: int = 2):
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        ################################
        # Self attention layer
        ################################
        # If doing single head, uncomment this and then comment out the MultiHeadAttention
        # self.attn_layer = SelfAttention(embedding_dim, embedding_dim)
        
        # If doing multi head, uncomment this and then comment out the SelfAttention
        self.attn_layer = MultiHeadAttention(embedding_dim, num_heads=num_heads)

        # Fully connected layer
        self.fc1 = nn.Linear(embedding_dim, embedding_dim*2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embedding_dim*2, embedding_dim)

        # Normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor):
        # Pass through attention head(s)
        seq_len = x.shape[1]

        # If doing single head, uncomment this and comment out the generate_causal_mask_multi_head (line below)
        # mask = generate_causal_mask(seq_len, device=x.device)
        
        # If doing multi head, uncomment this and comment out the generate_causal_mask (line above)
        mask = generate_causal_mask_multi_head(self.num_heads, seq_len, device=x.device)
        attn_out, _ = self.attn_layer(x, mask)
   
        # Add & Normalize
        attn_norm = self.norm1(x + attn_out)

        # Pass through feed forward network
        out = self.relu(self.fc1(attn_norm))
        out = self.fc2(out)

        # Add & Normalize
        out_normed = self.norm2(out + attn_norm)

        return out_normed

