
import math
import torch

def get_positional_encoding(seq_len: int, n_dim:int, device:str="cpu") -> torch.Tensor:
    """Creates positional encodings as described in the paper.

    Uses cosine and sine functions for generating positions for even and odd tokens 

    Args:
        seq_len (int): Length of input sequence
        n_dim (int): Number of dimensions in the model
        device (str, optional): Which device to do computation on. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Vector of positional encodings to add to embedding
    """
    # Initialize positional encoding vector
    pe = torch.zeros(seq_len, n_dim, device=device)

    # Get the indexes into a vector
    # Unsqueeze adds a dimension
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)

    # Torch way of doing
    # 1 / 10000^(2i/d_model)
    # Note, d_model is the dimensions of the embedding
    divisor = torch.exp(torch.arange(0, n_dim, 2).float() * (-math.log(10000.0) / n_dim)).to(device)

    # This is super cool slicing
    # 0::2 means start at 0 index and index every 2
    pe[:, 0::2] = torch.sin(position*divisor)
    pe[:, 1::2] = torch.cos(position*divisor)
    
    # Add dimension to the positional encodings so they can be broadcast across all batches
    return pe.unsqueeze(0)


if __name__ == "__main__":
    # Example: sequence length 50, embedding dimension 16
    import matplotlib.pyplot as plt
    seq_len = 50
    dim = 16

    pe = get_positional_encoding(seq_len, dim)
    print(f"Shape of positional encoding: {pe.shape}")  # (1, 50, 16)

    # Let's plot the first 4 dimensions across the sequence
    pe_np = pe[0].cpu().numpy()  # (seq_len, dim)

    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.plot(range(seq_len), pe_np[:, i], label=f"dim {i}")
    plt.legend()
    plt.title("Positional Encoding over Sequence Length")
    plt.xlabel("Position")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()