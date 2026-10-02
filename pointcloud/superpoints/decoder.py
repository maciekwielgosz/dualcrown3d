"""Revisable superpoint context for point-resolution instance masks.

Groups only aggregate context. The original point features, geometry and mask
logits remain independent, so two points in one group may get different IDs.
This is an experimental decoder, not a replacement for production checkpoints.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pointcloud.decoder_v4 import HybridTreeMaskDecoder
from pointcloud.superpoints.partition import neighbor_pairs


def group_geometry(coord, groups):
    groups = np.unique(groups, return_inverse=True)[1].astype(np.int64)
    count = np.bincount(groups)
    centers = np.column_stack([
        np.bincount(groups, weights=coord[:, axis]) / count for axis in range(3)
    ]).astype(np.float32)
    edges = neighbor_pairs(centers, radius_m=3., k=16).astype(np.int64)
    return groups, centers, edges


def pool_mean(values, group, count):
    pooled = values.new_zeros((len(count), values.shape[1]))
    pooled.index_add_(0, group, values)
    return pooled / count[:, None].clamp_min(1)


class GraphContext(nn.Module):
    def __init__(self, feature_dim=72, width=96, layers=3):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(feature_dim + 7, width), nn.LayerNorm(width), nn.GELU())
        self.messages = nn.ModuleList([
            nn.Sequential(nn.Linear(width + 3, width), nn.GELU(), nn.Linear(width, width))
            for _ in range(layers)
        ])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, width), nn.GELU())
            for _ in range(layers)
        ])
        # Point-specific residual is retained after graph broadcast.
        self.point_update = nn.Sequential(nn.Linear(feature_dim + width + 3, width),
                                          nn.GELU(), nn.Linear(width, feature_dim))
        nn.init.zeros_(self.point_update[-1].weight)
        nn.init.zeros_(self.point_update[-1].bias)

    def forward(self, features, coord, groups, centers, edges):
        count = torch.bincount(groups, minlength=len(centers)).to(features.dtype)
        relative = coord - centers[groups]
        spread = pool_mean(relative.square(), groups, count).clamp_min(0).sqrt()
        geometry = torch.cat((centers / centers.new_tensor([10., 10., 30.]),
                              spread / 3., torch.log1p(count)[:, None] / 5.), 1)
        hidden = self.input(torch.cat((pool_mean(features, groups, count), geometry), 1))
        source = torch.cat((edges[:, 0], edges[:, 1]))
        target = torch.cat((edges[:, 1], edges[:, 0]))
        degree = torch.bincount(target, minlength=len(centers)).to(features.dtype).clamp_min(1)
        delta = (centers[source] - centers[target]) / 3.
        for message, update in zip(self.messages, self.updates):
            aggregate = hidden.new_zeros(hidden.shape)
            aggregate.index_add_(0, target, message(torch.cat((hidden[source], delta), 1)))
            hidden = hidden + update(torch.cat((hidden, aggregate / degree[:, None]), 1))
        correction = self.point_update(torch.cat((features, hidden[groups], relative), 1))
        return features + correction


class SuperpointMaskDecoder(HybridTreeMaskDecoder):
    def __init__(self, *, use_graph=True, hidden_dim=128, queries=96, layers=3,
                 memory_tokens=1024, graph_width=96, graph_layers=3, wide_dim=0, wide_layers=2):
        super().__init__(hidden_dim, queries, layers, memory_tokens)
        self.use_graph = use_graph
        self.all_canopy_candidates = True
        self.teacher_probability = 0.
        if use_graph:
            self.graph = GraphContext(width=graph_width, layers=graph_layers)
        if wide_dim:
            from pointcloud.superpoints.wide_decoder import WideMaskRefiner
            self.wide = WideMaskRefiner(source_dim=hidden_dim, width=wide_dim, layers=wide_layers)

    def forward(self, features, data, logits, offset):
        if self.use_graph:
            features = self.graph(features, data['coord'], data['groups'],
                                  data['centers'], data['group_edges'])
        result = super().forward(features, data, logits, offset)
        if hasattr(self, 'wide'):
            result = self.wide(result)
        return result


def assign_masks(mask_probability, object_probability, *, object_threshold=.25,
                 mask_threshold=.5, minimum_points=8, duplicate_iou=.6):
    """Pairwise duplicate suppression followed by revisable point competition.

    Never reject a query merely because its support overlaps the union of other
    crowns. After removing undersized instances, their points compete again.
    Ground-truth labels and semantic foreground gates are not inputs.
    """
    p = np.asarray(mask_probability, np.float32)
    objects = np.asarray(object_probability, np.float32)
    n = p.shape[1]
    support = p >= mask_threshold
    counts = support.sum(1)
    ranks = objects * (p * support).sum(1) / np.maximum(counts, 1)
    candidates = np.flatnonzero((objects >= object_threshold) & (counts >= minimum_points))
    accepted = []
    for q in candidates[np.argsort(-ranks[candidates], kind='stable')]:
        if accepted:
            overlap = (support[accepted] & support[q]).sum(1)
            if np.any(overlap / np.maximum(counts[accepted] + counts[q] - overlap, 1) >= duplicate_iou):
                continue
        accepted.append(int(q))
    while accepted:
        offers = np.where(support[accepted], p[accepted] * objects[accepted, None], -1.)
        winner = offers.argmax(0)
        confidence = offers[winner, np.arange(n)]
        counts = np.bincount(winner[confidence >= 0], minlength=len(accepted))
        keep = counts >= minimum_points
        if keep.all():
            labels = np.where(confidence >= 0, winner + 1, 0).astype(np.int32)
            return labels, np.maximum(confidence, 0), np.asarray(accepted, np.int64)
        accepted = [q for q, retain in zip(accepted, keep) if retain]
    return np.zeros(n, np.int32), np.zeros(n, np.float32), np.empty(0, np.int64)
