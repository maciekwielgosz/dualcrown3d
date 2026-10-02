"""Trainable 192-D mask-query refinement with zero-impact warm initialization."""
import torch
from torch import nn
from torch.nn import functional as F

from pointcloud.dual_head import MaskAttentionLayer


class WideMaskRefiner(nn.Module):
    """Refine the inherited 128-D decoder with wider point/query attention."""

    def __init__(self, source_dim=128, width=192, layers=2, memory_tokens=512):
        super().__init__()
        self.memory_tokens = memory_tokens
        self.point = nn.Sequential(nn.Linear(source_dim, width), nn.LayerNorm(width), nn.GELU())
        self.query = nn.Sequential(nn.Linear(source_dim, width), nn.LayerNorm(width), nn.GELU())
        self.layers = nn.ModuleList(MaskAttentionLayer(width) for _ in range(layers))
        self.point_out = nn.Linear(width, width)
        self.query_out = nn.Linear(width, width)
        self.mask_gain = nn.Parameter(torch.zeros(()))
        self.score = nn.Linear(width, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, result):
        if not len(result['query_features']) or not len(result['point_features']):
            return result
        points = self.point(result['point_features'])
        queries = self.query(result['query_features'])
        memory_index = torch.linspace(0, len(points) - 1,
                                      min(len(points), self.memory_tokens),
                                      device=points.device).long()
        memory = points[memory_index]
        blocked = result['mask_logits'][:, memory_index].detach() < 0
        blocked[blocked.all(1)] = False
        for layer in self.layers:
            queries = layer(queries, memory, blocked)
        delta = 8. * (F.normalize(self.query_out(queries), dim=1) @
                       F.normalize(self.point_out(points), dim=1).T)
        result['mask_logits'] = result['mask_logits'] + self.mask_gain * delta
        result['object_logits'] = result['object_logits'] + self.score(queries).squeeze(1)
        return result
