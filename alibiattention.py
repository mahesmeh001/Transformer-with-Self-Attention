import torch
import torch.nn as nn
import math


class AliBiMultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_dim):
        super().__init__()
        assert n_dim % n_heads == 0, "n_dim must be divisible by n_heads"

        self.n_heads = n_heads
        self.n_dim = n_dim
        self.head_dim = n_dim // n_heads

        # Linear projections
        self.q_proj = nn.Linear(n_dim, n_dim)
        self.k_proj = nn.Linear(n_dim, n_dim)
        self.v_proj = nn.Linear(n_dim, n_dim)
        self.out_proj = nn.Linear(n_dim, n_dim)

        self.scale = math.sqrt(self.head_dim)

        # Create the AliBi slopes as per Press et al.
        self.slopes = torch.Tensor(self.get_slopes(n_heads)).unsqueeze(0).unsqueeze(0)

    def get_slopes(self, n_heads):
        # Compute slopes as per AliBi formula
        slopes = []
        for i in range(1, n_heads + 1):
            slopes.append(1.0 / (2 ** (i / n_heads)))
        return slopes

    def forward(self, x):
        B, T, C = x.shape  # batch size, sequence length, embedding dimension
        H = self.n_heads

        # Project and reshape
        q = self.q_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)  # B, H, T, head_dim
        k = self.k_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)

        # Compute attention scores with AliBi bias
        attn_scores = (q @ k.transpose(-2, -1)) / self.scale  # Shape: (B, H, T, T)

        # AliBi positional bias
        pos_bias = torch.arange(T, device=x.device).view(1, 1, T, 1) - torch.arange(T, device=x.device).view(1, 1, 1, T)
        slopes = self.slopes.to(x.device).view(1, H, 1, 1)  # Reshape slopes to (1, n_heads, 1, 1)
        pos_bias = pos_bias * slopes  # Shape: (1, H, T, T)
        attn_scores += pos_bias

        # Apply softmax over the last dimension (key dimension)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # For visualization: average attention probabilities across heads
        avg_attn_probs = attn_probs.mean(dim=1)

        # Apply attention to values
        out = attn_probs @ v  # B, H, T, head_dim
        out = out.transpose(1, 2).reshape(B, T, C)  # B, T, C
        out = self.out_proj(out)

        return out, avg_attn_probs.detach()

class AliBiTransformerEncoderLayer(nn.Module):
    def __init__(self, n_heads, n_dim, hidden_dim):
        super().__init__()
        self.attention = AliBiMultiHeadAttention(n_heads, n_dim)
        self.ff = FeedForward(n_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(n_dim)
        self.ln2 = nn.LayerNorm(n_dim)

    def forward(self, x):
        # Self-attention block with pre-norm
        attn_out, attn_weights = self.attention(self.ln1(x))
        x = x + attn_out

        # Feedforward block with pre-norm
        x = x + self.ff(self.ln2(x))

        return x, attn_weights

class FeedForward(nn.Module):
    def __init__(self, n_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_dim)
        )

    def forward(self, x):
        return self.net(x)

class AliBiTransformerEncoder(nn.Module):
    def __init__(self, n_layers, n_heads, n_dim, hidden_dim, vocab_size=None):
        super().__init__()
        self.n_dim = n_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, n_dim) if vocab_size else None

        # Transformer layers
        self.layers = nn.ModuleList([
            AliBiTransformerEncoderLayer(n_heads, n_dim, hidden_dim)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(n_dim)

    def forward(self, x):
        # If input is integer tokens and we have an embedding layer, apply embedding
        if self.embedding is not None and x.dtype == torch.long:
            x = self.embedding(x)
        elif x.dtype != torch.float32:
            x = x.float()

        attn_maps = []

        # Process through transformer layers
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_maps.append(attn_weights)

        x = self.ln_f(x)

        # Mean pooling over sequence length
        output = x.mean(dim=1)

        return output, attn_maps
