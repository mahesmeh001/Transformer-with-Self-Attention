import torch
import torch.nn as nn
import math

from CSE256_PA2_FA24.utilities import Utilities

import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
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

    def forward(self, x, mask=None):
        B, T, C = x.shape  # Batch size, sequence length, embedding dimension

        H = self.n_heads

        # Project and reshape query, key, and value
        q = self.q_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)  # Shape: (B, H, T, head_dim)
        k = self.k_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)  # Shape: (B, H, T, head_dim)
        v = self.v_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)  # Shape: (B, H, T, head_dim)

        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / self.scale  # Shape: (B, H, T, T)

        # Apply mask if provided
        if mask is not None:
            # Ensure mask shape matches (B, H, T, T)
            if mask.dim() == 2:  # If mask shape is (T, T), expand it
                mask = mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, T, T)
            elif mask.dim() == 3:  # If mask shape is (B, 1, T, T)
                mask = mask.unsqueeze(1)  # Shape: (B, 1, T, T)

            B, _, T, _ = attn_scores.shape
            H = attn_scores.shape[1]  # Number of attention heads

            mask = mask.expand(B, H, T, T)  # Expand to match (B, H, T, T)

            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        else:
            # For autoregressive (triangular) mask, apply a lower triangular mask
            tril = torch.tril(torch.ones(T, T, device=x.device))  # Shape: (T, T)
            tril = tril.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, T, T)
            tril = tril.expand(B, H, T, T)  # Shape: (B, H, T, T)
            attn_scores = attn_scores.masked_fill(tril == 0, float('-inf'))

        # Apply softmax to attention scores
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # For visualization: average attention probabilities across heads (B, T, T)
        avg_attn_probs = attn_probs.mean(dim=1)


        # Apply the attention weights to the value vectors
        output = attn_probs @ v  # Shape: (B, H, T, head_dim)

        # Concatenate heads and project back
        output = output.transpose(1, 2).contiguous().view(B, T, self.n_dim)  # Shape: (B, T, n_dim)
        output = self.out_proj(output)  # Final projection


        return output, avg_attn_probs.detach()



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



class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_heads, n_dim, hidden_dim):
        super().__init__()
        self.masked_attention = MultiHeadAttention(n_heads, n_dim)
        self.cross_attention = MultiHeadAttention(n_heads, n_dim)
        self.ff = FeedForward(n_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(n_dim)
        self.ln2 = nn.LayerNorm(n_dim)
        self.ln3 = nn.LayerNorm(n_dim)
        self.n_heads = n_heads

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        mask = self.get_mask(batch_size, seq_len)

        # Masked self-attention block with pre-norm
        attn_out, attn_weights = self.masked_attention(self.ln1(x), mask)
        x = x + attn_out

        # Feedforward block with pre-norm
        x = x + self.ff(self.ln3(x))

        return x, attn_weights

    def get_mask(self, batch_size, seq_len):
        """Generate causal mask with the correct shape to match attn_scores."""
        # Create a lower triangular mask of shape [seq_len, seq_len]
        mask = torch.tril(torch.ones(seq_len, seq_len))  # Shape: [seq_len, seq_len]

        # Add batch and head dimensions -> [1, 1, seq_len, seq_len]
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Hardcode the expansion with expected dimensions
        B, H, T = batch_size, self.n_heads, seq_len
        mask = mask.expand(B, H, T, T)  # Ensure it expands to [B, H, seq_len, seq_len]

        return mask

class TransformerDecoder(nn.Module):
    def __init__(self, n_layers, n_heads, n_dim, hidden_dim, vocab_size):
        super().__init__()
        self.n_dim = n_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, n_dim)

        # Position embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, 512, n_dim))  # Fixed size for block_size=512

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(n_heads, n_dim, hidden_dim)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(n_dim)
        self.head = nn.Linear(n_dim, vocab_size)

    def forward(self, x):
        # Apply embedding
        x = self.embedding(x)

        # Add positional embeddings
        x = x + self.pos_embedding[:, :x.size(1), :]

        attn_maps = []

        # Process through transformer layers
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_maps.append(attn_weights)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits, attn_maps

def pretrain_decoder(decoder, train_LM_loader, criterion, optimizer, device, max_iters=500):
    """
    Pretrains the Transformer Decoder on the language modeling task.
    """
    decoder.train()
    for i, (x, y) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        x, y = x.to(device), y.to(device)

        # Shift the input one position to the right to predict the next token
        x_shifted = x[:, :-1]
        y_shifted = x[:, 1:]

        logits, _ = decoder(x_shifted)

        loss = criterion(logits.reshape(-1, logits.size(-1)), y_shifted.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Iteration {i}/{max_iters}, Loss: {loss.item()}")

    return decoder
