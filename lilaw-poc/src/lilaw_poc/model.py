"""5-layer MLP matching MLMD's feed-forward neural network architecture."""

import torch
from torch import nn


class MLMDNet(nn.Module):
    """5-layer fully connected network with ReLU activations and sigmoid output.

    Architecture mirrors Singh et al. (2022): 5 linear layers, ReLU on first 4,
    sigmoid on the final output for binary classification.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns probabilities in (0, 1)."""
        return self.net(x)
