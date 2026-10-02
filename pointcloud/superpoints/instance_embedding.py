"""Small instance-boundary embedding for the EZ-SP-style partition pilot."""
from __future__ import annotations

import torch
from torch import nn


class InstanceBoundaryEmbedding(nn.Module):
    """Project cached LitePT features to a 16-D local instance space.

    This is not the sparse CNN from the original EZ-SP publication. It is a
    controlled, inexpensive adaptation of its contrastive-boundary objective
    to the existing frozen backbone and available tree-instance IDs.
    """

    def __init__(self, input_dim: int = 72, hidden_dim: int = 64,
                 output_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.network(feature)


def edge_affinity(embedding: torch.Tensor, edge: torch.Tensor,
                  temperature: float = 1.0) -> torch.Tensor:
    """Original EZ-SP form: exp(-Euclidean distance / temperature)."""
    if temperature <= 0:
        raise ValueError("Positive temperature required")
    distance = (embedding[edge[:, 0]] - embedding[edge[:, 1]]).norm(dim=1)
    return torch.exp(-distance / temperature)
