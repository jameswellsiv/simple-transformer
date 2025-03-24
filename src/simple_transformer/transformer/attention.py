import math
import torch
import torch.nn as nn

# Setting for reproducibility
torch.manual_seed(42)

def generate_causal_mask(seq_len:int, device:str='cpu'):
    """Causal Mask

    This forces the model to only attend to previous tokens when doing
    next token prediction. If you dont have this then the model will
    attend to future tokens too during training. This is cheating because
    the model will not have access to future tokens during inference.

    Args:
        seq_len (int): Length of input sequence into the model
        device (str, optional): Device for PyTorch to use. Defaults to "cpu"

    Returns:
        torch.Tensor: Causal mask that only lets the lower left triangle of the attention weights to be useds
    """
    # shape: (1, seq_len, seq_len) to broadcast over batch
    return torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0)

def generate_causal_mask_multi_head(num_heads:int, seq_len:int, device:bool = 'cpu'):
    """Causal Mask

    This forces the model to only attend to previous tokens when doing
    next token prediction. If you dont have this then the model will
    attend to future tokens too during training. This is cheating because
    the model will not have access to future tokens during inference.

    Args:
        num_heads (int): Number of attention heads
        seq_len (int): Length of input sequence into the model
        device (str, optional): Device for PyTorch to use. Defaults to "cpu"

    Returns:
        torch.Tensor: Causal mask that only lets the lower left triangle of the attention weights to be useds
    """
    # shape: (1, num_heads, seq_len, seq_len) to broadcast over batch
    return torch.tril(torch.ones((num_heads, seq_len, seq_len), device=device)).unsqueeze(0)

class SelfAttention(nn.Module):
    """ SelfAttention Mechanism as described in the paper.

    The purpose of self-attention is to project the input embeddings
    into three separate spaces.

    Q - Query

    K - Key
    
    V - Value

    Furthermore, each of these spaces serve a different role:
    
    Q asks "What am I looking for?"

    K asks "What do I contain?"
    
    V asks "What information do I pass along?"
    
    
    """
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        
        # Embedding dimension
        self.embed_dim = embed_dim 
        
        # Head dimension, determined by embedding dimension and number of heads
        self.head_dim = head_dim 

        # Define linear transformations for Q, K, and V spaces
        self.q_proj = nn.Linear(embed_dim, head_dim)
        self.k_proj = nn.Linear(embed_dim, head_dim)
        self.v_proj = nn.Linear(embed_dim, head_dim)

        # Softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # Project input into Q, K, V spaces
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # MatMul Q & K_transposed and then scale
        # Note, in K.tranpose, we dont want to effect the batch dimension
        # So we pass -2, -1 as arguments

        # This is gonna give you scaled dot products
        # Which essentially answers how much should token i attend to the rest of the tokens
        # So basically, which other tokens does i correlate to a lot
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        # Which now will give us probabilities
        attn_probas = self.softmax(attn_scores)

        # Mat multiply probabilities with the V vector
        out = torch.matmul(attn_probas, V)

        return out, attn_probas


class MultiHeadAttention(nn.Module):
    """ MultiHead Attention Mechanism

    Does the same thing as SelfAttention, but there is some extra linear algebra
    to get it working across multi heads. Additionally, some funky transposing and reshaping
    is needed too so that the output is in the correct shape.
    
    """
    def __init__(self, embedding_dim:int, num_heads:int=2):
        super().__init__()

        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of attention heads"

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads

        # Define linear transformations for Q, K, and V spaces
        # Now, we project into num_heads*head_dim, so weve effectively
        # increased our Q/K/V space to capture more semantics
        self.q_proj = nn.Linear(embedding_dim, num_heads*self.head_dim)
        self.k_proj = nn.Linear(embedding_dim, num_heads*self.head_dim)
        self.v_proj = nn.Linear(embedding_dim, num_heads*self.head_dim)

        # Softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        Q = self.q_proj(x) # Projected from (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, num_heads*head_dim)
        K = self.k_proj(x) # Projected from (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, num_heads*head_dim)
        V = self.v_proj(x) # Projected from (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, num_heads*head_dim)

        # Reshape linear projects to (batch_size, seq_len, num_heads, head_dim)
        Q = torch.reshape(Q, (Q.shape[0], Q.shape[1], self.num_heads, self.head_dim))
        K = torch.reshape(K, (K.shape[0], K.shape[1], self.num_heads, self.head_dim))
        V = torch.reshape(V, (V.shape[0], V.shape[1], self.num_heads, self.head_dim))

        # Now we need to move num_heads forward 1 dimension so attention and masking works
        # So our shape should be (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Now we can compute our attention scores, just as before with a single head
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embedding_dim)

        # Causal masking, blank out all the attention weights in the upper triangle
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Pass scores through softmax to turn into probabilities
        attn_probas = self.softmax(attn_scores)

        # Mat multply with V
        out = torch.matmul(attn_probas, V)

        # Transpose back and reshape
        out = out.transpose(1, 2)
        out = out.reshape(out.shape[0], out.shape[1], self.embedding_dim)

        return out, attn_probas



if __name__ == "__main__":
    #############################
    # Self contained example
    #############################
    import matplotlib.pyplot as plt
    import seaborn as sns


    #######################################################################
    # Initialize SelfAttention module and run random example through it
    ######################################################################

    # Set parameters
    # batch_size = 1
    # seq_len = 6
    # num_heads = 2
    # embed_dim = 8
    # head_dim = embed_dim // num_heads

    # Random "token embeddings"
    # x = torch.randn(batch_size, seq_len, embed_dim)
    # mask = generate_causal_mask(seq_len, device=x.device)
    
    # attn = SelfAttention(embed_dim, head_dim)
    # output, attn_weights = attn(x, mask)

    # Shapes
    # print(f"Input shape: {x.shape}")
    # print(f"Output shape: {output.shape}")
    # print(f"Attention weights shape: {attn_weights.shape}")  # (1, seq_len, seq_len)


    # Visualize attention matrix
    # plt.figure(figsize=(6, 5))
    # sns.heatmap(attn_weights[0].detach().numpy(), cmap="viridis", annot=True, fmt=".2f")
    # plt.title("Attention Matrix (Single Head)")
    # plt.xlabel("Key Positions")
    # plt.ylabel("Query Positions")
    # plt.show()

    #######################################################################
    # Initialize a MultiHeadAttention module and run a random example through it
    #######################################################################
    batch_size = 1
    seq_len = 6
    num_heads = 2
    embed_dim = 8
    head_dim = embed_dim // num_heads

    x = torch.rand(batch_size, seq_len, embed_dim)
    mask = generate_causal_mask_multi_head(num_heads, seq_len)
    attn = MultiHeadAttention(embed_dim, num_heads=num_heads)
    output, attn_weights = attn(x, mask)

    # Shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")  # (1, seq_len, seq_len)

    plt.figure(figsize=(12, 6))
    plt.suptitle("Attention Matrix (Multi Head)")

    plt.subplot(1, 2, 1)
    sns.heatmap(attn_weights[0][0].detach().numpy(), cmap="viridis", annot=True, fmt=".2f")
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(attn_weights[0][1].detach().numpy(), cmap="viridis", annot=True, fmt=".2f")
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    plt.show()
