"""Small symmetric edge predictor and bounded, label-free partitioning."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class EdgeAffinityHead(nn.Module):
    def __init__(self, feature_dim: int = 72, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * feature_dim + 5, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, feature: torch.Tensor, coord: torch.Tensor,
                edge: torch.Tensor) -> torch.Tensor:
        left, right = edge[:, 0].long(), edge[:, 1].long()
        first, second = feature[left], feature[right]
        delta = (coord[left] - coord[right]).abs()
        geometry = torch.cat((delta / 2., delta.norm(dim=1, keepdim=True) / 2.,
                              (coord[left, 2:3] + coord[right, 2:3]) / 60.), dim=1)
        symmetric = torch.cat(((first - second).abs(), first * second, geometry), dim=1)
        return self.network(symmetric).squeeze(1)


def bounded_partition(coord: np.ndarray, edges: np.ndarray,
                      probability: np.ndarray, *, threshold: float,
                      max_extent_m: float = 1., max_points: int = 64) -> np.ndarray:
    """Merge high-affinity neighbors while capping cluster size and 3-D extent.

    No reference labels are used. The bounds prevent one false-positive bridge
    from swallowing a large group; instance grouping is a separate operation.
    """
    xyz = np.asarray(coord, np.float32)
    edge = np.asarray(edges, np.int32)
    prob = np.asarray(probability, np.float32)
    if not 0 <= threshold <= 1 or max_extent_m <= 0 or max_points < 1:
        raise ValueError("Invalid partition constraint")
    if edge.shape != (len(prob), 2) or not np.isfinite(prob).all():
        raise ValueError("Edge and probability arrays must align")
    n = len(xyz)
    parent = np.arange(n, dtype=np.int32)
    size = np.ones(n, np.int32)
    lower, upper = xyz.copy(), xyz.copy()

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for item in np.argsort(-prob, kind="stable"):
        if prob[item] < threshold:
            break
        a, b = root(int(edge[item, 0])), root(int(edge[item, 1]))
        if a == b or size[a] + size[b] > max_points:
            continue
        merged_lower = np.minimum(lower[a], lower[b])
        merged_upper = np.maximum(upper[a], upper[b])
        if np.linalg.norm(merged_upper - merged_lower) > max_extent_m:
            continue
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
        lower[a], upper[a] = merged_lower, merged_upper
    roots = np.fromiter((root(i) for i in range(n)), dtype=np.int32, count=n)
    _, inverse = np.unique(roots, return_inverse=True)
    return inverse.astype(np.int32)
