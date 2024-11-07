import torch
from torch import nn
import torch.nn.functional as F


class FeedForwardClassifier(nn.Module):
    def __init__(self, n_input, hidden_dim, n_output, encoder, dropout_rate=0.1):
        """
        Simple feedforward classifier with one hidden layer.

        Args:
            n_input (int): Input dimension (should match encoder output dimension)
            hidden_dim (int): Hidden layer dimension
            n_output (int): Number of output classes
            encoder (nn.Module): The transformer encoder to use
            dropout_rate (float): Dropout rate for regularization
        """
        super().__init__()

        # Store the encoder
        self.encoder = encoder

        # First layer: input to hidden
        self.fc1 = nn.Linear(n_input, hidden_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Output layer: hidden to n_output classes
        self.fc2 = nn.Linear(hidden_dim, n_output)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights using Xavier initialization"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        Forward pass of the classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length)
                            This can be raw token indices

        Returns:
            torch.Tensor: Logits for each class, shape (batch_size, n_output)
        """
        # First get embeddings from encoder
        embeddings, _ = self.encoder(x)  # This handles both type conversion and dimensionality

        # First layer with ReLU activation
        x = F.relu(self.fc1(embeddings))

        # Apply dropout
        x = self.dropout(x)

        # Output layer (logits)
        x = self.fc2(x)

        return x

    def predict(self, x):
        """
        Make predictions for input x.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Predicted class indices
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)