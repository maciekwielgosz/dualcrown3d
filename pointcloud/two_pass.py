"""Second pass: residual refinement conditioned on a frozen first-pass result.

Stage 1 is the retained DualCrown3D (Model20). Stage 2 sees the same window
together with label-free descriptors of the stage-1 assignment and decides, per
learned query, whether its mask is a tree stage 1 does not have (``new``), a
tree stage 1 already represents (``existing``: fragment, duplicate or boundary
correction) or ``background``. Ground truth only builds training targets; it is
never an inference input. Seeds come from a learned per-point ``need`` head so
absorbed small trees inside large stage-1 instances can still start a query.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from pointcloud.dual_head import MaskAttentionLayer, TreeMaskDecoder, pooled_context

SOURCE_CODES = 5          # stage-1 assignment provenance codes 1..5, 0 unassigned
DECISION_NAMES = ('background', 'new', 'existing')
CONDITION_NAMES = (
    'assigned', 'log_instance_voxels', 'instance_height', 'relative_height',
    'distance_to_top', 'distance_to_vote_center', 'vote_disagreement', 'vote_length',
    'confidence', 'tree_probability', 'point_probability',
    'source_1', 'source_2', 'source_3', 'source_4', 'source_5',
    'local_peak', 'instance_top_cell', 'log_candidate_count', 'foreign_support',
    'cover_depth', 'gap_above', 'sub_canopy_peak')
CONDITION_DIM = len(CONDITION_NAMES)
LEGACY_CONDITION_DIM = 20   # v1 checkpoints were trained without the vertical-structure features


def match_instances(truth, labels, threshold=.5):
    """Cardinality-first IoU matching on known points; returns per-GT matches.

    ``matched_label[g]`` is the predicted ID matched to GT ID ``gt_ids[g]`` or 0.
    """
    truth = np.asarray(truth)
    labels = np.asarray(labels)
    known = truth >= 0
    gt_ids = np.unique(truth[truth > 0])
    if not len(gt_ids):
        return gt_ids, np.zeros(0, np.int64), np.zeros(0, np.float64)
    t, l = truth[known], labels[known]
    pr_ids = np.unique(l[l > 0])
    if not len(pr_ids):
        return gt_ids, np.zeros(len(gt_ids), np.int64), np.zeros(len(gt_ids))
    gi = np.searchsorted(gt_ids, t)
    pi = np.searchsorted(pr_ids, l)
    both = (t > 0) & (l > 0)
    table = np.bincount(gi[both] * len(pr_ids) + pi[both],
                        minlength=len(gt_ids) * len(pr_ids)).reshape(len(gt_ids), len(pr_ids))
    gt_count = np.bincount(gi[t > 0], minlength=len(gt_ids))
    pr_count = np.bincount(pi[l > 0], minlength=len(pr_ids))
    iou = table / np.maximum(gt_count[:, None] + pr_count[None] - table, 1)
    valid = iou >= threshold
    score = valid * (min(iou.shape) + 1. + iou)
    rows, cols = linear_sum_assignment(-score)
    good = iou[rows, cols] >= threshold
    matched = np.zeros(len(gt_ids), np.int64)
    matched[rows[good]] = pr_ids[cols[good]]
    return gt_ids, matched, iou.max(1)


def per_point_matched_label(truth, labels, threshold=.5):
    """Predicted ID that represents each point's GT tree (0 = missed / no tree)."""
    truth = np.asarray(truth)
    gt_ids, matched, _ = match_instances(truth, labels, threshold)
    result = np.zeros(len(truth), np.int64)
    if len(gt_ids):
        positive = truth > 0
        result[positive] = matched[np.searchsorted(gt_ids, truth[positive])]
    return result


def local_height_peaks(xyz, cell=.5, radius=1.5, minimum_height=1.):
    """Points that are the highest return of a local CHM maximum cell."""
    xyz = np.asarray(xyz, np.float32)
    peak = np.zeros(len(xyz), bool)
    if not len(xyz):
        return peak
    grid = np.floor((xyz[:, :2] - xyz[:, :2].min(0)) / cell).astype(np.int64)
    shape = grid.max(0) + 1
    key = grid[:, 0] * shape[1] + grid[:, 1]
    top = np.full(shape[0] * shape[1], -np.inf, np.float32)
    np.maximum.at(top, key, xyz[:, 2])
    raster = top.reshape(shape[0], shape[1])
    window = 2 * int(np.ceil(radius / cell)) + 1
    local = maximum_filter(raster, size=window, mode='constant', cval=-np.inf)
    peak_cell = (raster == local) & (raster >= minimum_height)
    peak = peak_cell.reshape(-1)[key] & (xyz[:, 2] >= top[key] - 1e-4)
    return peak


def column_structure(xyz, cell=.5, radius=1.5, minimum_gap=2.):
    """Label-free vertical structure for understory trees.

    Returns depth below the local canopy top (m), free space above the voxel in
    its own XY column (m, capped at 10; 10 for column tops) and a sub-canopy top
    flag: the ``minimum_gap`` slab above the voxel is empty across its 3x3 cell
    neighbourhood although something taller stands within ``radius``.
    """
    xyz = np.asarray(xyz, np.float32)
    n = len(xyz)
    if not n:
        return np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, bool)
    z = xyz[:, 2].astype(np.float64)
    grid = np.floor((xyz[:, :2] - xyz[:, :2].min(0)) / cell).astype(np.int64)
    shape = grid.max(0) + 1
    key = grid[:, 0] * shape[1] + grid[:, 1]
    top = np.full(shape[0] * shape[1], -np.inf, np.float32)
    np.maximum.at(top, key, xyz[:, 2])
    window = 2 * int(np.ceil(radius / cell)) + 1
    canopy = maximum_filter(top.reshape(shape[0], shape[1]), size=window, mode='constant', cval=-np.inf)
    cover = canopy.reshape(-1)[key] - xyz[:, 2]
    order = np.lexsort((z, key))
    sorted_key, sorted_z = key[order], z[order]
    gap_sorted = np.full(n, np.inf, np.float32)
    same = sorted_key[1:] == sorted_key[:-1]
    gap_sorted[:-1][same] = (sorted_z[1:] - sorted_z[:-1])[same]
    gap = np.empty(n, np.float32)
    gap[order] = gap_sorted
    # Count voxels in the slab (z + 0.25, z + minimum_gap] of the 3x3 cell block.
    span = 1e4
    combined = sorted_key.astype(np.float64) * span + sorted_z
    blocked = np.zeros(n, np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            gx, gy = grid[:, 0] + dx, grid[:, 1] + dy
            inside = (gx >= 0) & (gx < shape[0]) & (gy >= 0) & (gy < shape[1])
            base = (gx * shape[1] + gy).astype(np.float64) * span + z
            low = np.searchsorted(combined, base + .25, side='right')
            high = np.searchsorted(combined, base + minimum_gap, side='right')
            blocked += np.where(inside, high - low, 0)
    peak = (blocked == 0) & (cover >= minimum_gap) & (xyz[:, 2] >= 1.5)
    return cover.astype(np.float32), np.where(np.isinf(gap), 10., np.minimum(gap, 10.)).astype(np.float32), peak


def candidate_competition(raw, labels, minimum_object_score=.1, minimum_mask=.2):
    """Per point: how many raw stage-1 masks cover it and the strongest mask whose
    majority lies in another stage-1 instance. Label-free conflict descriptors."""
    labels = np.asarray(labels, np.int64)
    count = np.zeros(len(labels), np.float32)
    foreign = np.zeros(len(labels), np.float32)
    offsets = raw['candidate_offset']
    scores = raw['object_score']
    index = raw['point_index']
    probability = raw['point_score'].astype(np.float32)
    for i in range(len(scores)):
        if scores[i] < minimum_object_score:
            continue
        a, b = offsets[i:i + 2]
        keep = probability[a:b] >= minimum_mask
        members = index[a:b][keep]
        if not len(members):
            continue
        prob = probability[a:b][keep]
        owners = labels[members]
        majority = np.bincount(owners).argmax()
        count[members] += 1.
        other = owners != majority
        np.maximum.at(foreign, members[other], prob[other])
    return count, foreign


def stage1_conditioning(xyz, vote_xy, labels, confidence, source, tree_probability,
                        point_probability, candidate_count, foreign_support, cell=.5):
    """Label-free per-point descriptors of a stage-1 result on any point subset.

    Distances are translation and rotation invariant, so the same function serves
    rotated training crops and plot-frame inference windows.
    """
    xyz = np.asarray(xyz, np.float32)
    n = len(xyz)
    labels = np.asarray(labels, np.int64)
    result = np.zeros((n, CONDITION_DIM), np.float32)
    if not n:
        return result
    assigned = labels > 0
    result[:, 0] = assigned
    if assigned.any():
        ids, inverse = np.unique(labels[assigned], return_inverse=True)
        count = np.bincount(inverse).astype(np.float32)
        z = xyz[assigned, 2]
        top_height = np.full(len(ids), -np.inf, np.float32)
        np.maximum.at(top_height, inverse, z)
        # Instance top cell: highest point of each instance.
        order = np.lexsort((-z, inverse))
        first = np.ones(len(order), bool)
        first[1:] = inverse[order[1:]] != inverse[order[:-1]]
        top_index = np.flatnonzero(assigned)[order[first]]
        top_xy = xyz[top_index, :2]
        votes = xyz[:, :2] + np.asarray(vote_xy, np.float32)
        center = np.zeros((len(ids), 2), np.float32)
        for k, members in enumerate(np.split(np.flatnonzero(assigned)[np.argsort(inverse, kind='stable')],
                                            np.cumsum(count.astype(np.int64))[:-1])):
            center[k] = np.median(votes[members], axis=0)
        result[assigned, 1] = np.log1p(count[inverse]) / 8.
        result[assigned, 2] = top_height[inverse] / 30.
        result[assigned, 3] = np.clip(z / np.maximum(top_height[inverse], .25), 0., 1.5)
        result[assigned, 4] = np.linalg.norm(xyz[assigned, :2] - top_xy[inverse], axis=1) / 10.
        result[assigned, 5] = np.linalg.norm(xyz[assigned, :2] - center[inverse], axis=1) / 10.
        result[assigned, 6] = np.linalg.norm(votes[assigned] - center[inverse], axis=1) / 10.
        top_cell = (np.linalg.norm(xyz[assigned, :2] - top_xy[inverse], axis=1) <= cell) & \
                   (z >= top_height[inverse] - 1.)
        result[assigned, 17] = top_cell
    result[:, 7] = np.linalg.norm(np.asarray(vote_xy, np.float32), axis=1) / 10.
    result[:, 8] = np.asarray(confidence, np.float32)
    result[:, 9] = np.asarray(tree_probability, np.float32)
    result[:, 10] = np.asarray(point_probability, np.float32)
    code = np.clip(np.asarray(source, np.int64), 0, SOURCE_CODES)
    for s in range(1, SOURCE_CODES + 1):
        result[:, 10 + s] = code == s
    result[:, 16] = local_height_peaks(xyz, cell=cell)
    result[:, 18] = np.log1p(np.asarray(candidate_count, np.float32)) / 3.
    result[:, 19] = np.asarray(foreign_support, np.float32)
    cover, gap, peak = column_structure(xyz, cell=cell)
    result[:, 20] = cover / 30.
    result[:, 21] = gap / 10.
    result[:, 22] = peak
    return result


def unique_voxel_index(coord, voxel_size):
    grid = np.floor((coord - coord.min(0)) / voxel_size).astype(np.int64)
    _, index = np.unique(grid, axis=0, return_index=True)
    return np.sort(index), grid


def crop_with_extras(arrays, extras, vectors, rng, crop_size_m=20., max_points=16000,
                     augment=True, preserve_height=True, anchor_index=None,
                     density_keep_fractions=(.5, .75, 1.)):
    """``prepare_crop`` semantics with arbitrary per-point scalars and 2-D vectors.

    Scalars are carried unchanged; vectors rotate and scale with the coordinates.
    """
    coord = arrays['coord']
    tree_id = arrays['tree_id']
    if anchor_index is None:
        positive = np.flatnonzero(tree_id > 0)
        anchor_index = int(rng.choice(positive if len(positive) else len(coord)))
    anchor = coord[anchor_index, :2]
    half = crop_size_m / 2.
    inside = ((coord[:, 0] >= anchor[0] - half) & (coord[:, 0] < anchor[0] + half)
              & (coord[:, 1] >= anchor[1] - half) & (coord[:, 1] < anchor[1] + half))
    selected = np.flatnonzero(inside)
    if len(selected) > max_points:
        selected = np.sort(rng.choice(selected, size=max_points, replace=False))
    semantic = arrays.get('semantic_target', np.where(tree_id < 0, -1, (tree_id > 0).astype(np.int64)))
    columns = dict(coord=coord[selected].astype(np.float32).copy(),
                   intensity=arrays['intensity'][selected].astype(np.float32).copy(),
                   tree_id=tree_id[selected].copy(),
                   semantic_target=semantic[selected].astype(np.int64).copy(),
                   instance_offset=arrays['instance_offset'][selected].astype(np.float32).copy())
    for name, value in extras.items():
        columns[name] = np.asarray(value)[selected].copy()
    for name, value in vectors.items():
        columns[name] = np.asarray(value, np.float32)[selected].copy()

    def take(keep):
        for name in list(columns):
            columns[name] = columns[name][keep]

    if augment and density_keep_fractions:
        fraction = float(rng.choice(density_keep_fractions))
        if fraction < 1. and len(columns['coord']) > 256:
            keep = min(len(columns['coord']), max(256, int(round(len(columns['coord']) * fraction))))
            take(np.sort(rng.choice(len(columns['coord']), size=keep, replace=False)))
    scale = 1.
    center = columns['coord'].mean(0, keepdims=True)
    if augment:
        angle = float(rng.uniform(-np.pi, np.pi))
        scale = float(rng.uniform(.9, 1.1))
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.asarray([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], np.float32)
        columns['coord'] = (columns['coord'] - center) @ rotation.T * scale + center
        columns['instance_offset'] = columns['instance_offset'] @ rotation.T * scale
        for name in vectors:
            columns[name] = columns[name] @ rotation[:2, :2].T * scale
        if rng.random() < .3:
            take(rng.random(len(columns['coord'])) >= rng.uniform(0., .15))
        columns['intensity'] = np.clip(columns['intensity'] * rng.uniform(.9, 1.1)
                                       + rng.normal(0., .01, len(columns['intensity'])), 0., 1.).astype(np.float32)
    voxel_size = float(arrays['voxel_size'])
    retained, _ = unique_voxel_index(columns['coord'], voxel_size)
    take(retained)
    columns['coord'][:, :2] -= columns['coord'][:, :2].mean(0)
    if not preserve_height:
        columns['coord'][:, 2] -= columns['coord'][:, 2].min()
    grid = np.floor((columns['coord'] - columns['coord'].min(0)) / voxel_size).astype(np.int32)
    _, final = np.unique(grid, axis=0, return_index=True)
    final.sort()
    take(final)
    grid = grid[final]
    result = {'coord': torch.from_numpy(columns['coord']), 'grid_coord': torch.from_numpy(grid),
              'feat': torch.from_numpy(np.column_stack((columns['coord'], columns['intensity'])).astype(np.float32)),
              'offset': torch.tensor([len(grid)], dtype=torch.long)}
    for name, value in columns.items():
        if name not in ('coord', 'intensity'):
            result[name] = torch.from_numpy(np.ascontiguousarray(value))
    return result


def corrupt_stage1(xyz, labels, confidence, source, rng, absorb_probability=.3,
                   drop_probability=.15, max_voxels=400):
    """Simulate stage-1 failure modes on a crop: absorb a small instance into its
    nearest neighbour, or drop one entirely. Returns copies."""
    labels = np.asarray(labels, np.int64).copy()
    confidence = np.asarray(confidence, np.float32).copy()
    source = np.asarray(source, np.uint8).copy()
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    small = ids[counts <= max_voxels]
    if len(ids) >= 2 and len(small) and rng.random() < absorb_probability:
        weight = 1. / counts[np.isin(ids, small)]
        victim = int(rng.choice(small, p=weight / weight.sum()))
        centers = np.stack([xyz[labels == i, :2].mean(0) for i in ids])
        own = int(np.flatnonzero(ids == victim)[0])
        distance = np.linalg.norm(centers - centers[own], axis=1)
        distance[own] = np.inf
        host = int(ids[distance.argmin()])
        members = labels == victim
        labels[members] = host
        source[members] = 2
        ids, counts = np.unique(labels[labels > 0], return_counts=True)
        small = ids[counts <= max_voxels]
    if len(small) and rng.random() < drop_probability:
        victim = int(rng.choice(small))
        members = labels == victim
        labels[members], confidence[members], source[members] = 0, 0., 0
    return labels, confidence, source


def need_target(tree_id, labels, matched_label):
    """1 where a known tree point is missed or carries the wrong stage-1 ID."""
    tree_id = np.asarray(tree_id)
    target = np.full(len(tree_id), -1., np.float32)
    known = tree_id >= 0
    target[known] = 0.
    positive = tree_id > 0
    wrong = (np.asarray(matched_label) == 0) | (np.asarray(labels) != np.asarray(matched_label))
    target[positive & wrong] = 1.
    return target


class RefinementDecoder(nn.Module):
    """Conditioned query decoder for the second pass."""
    teacher_probability = 0.

    def __init__(self, backbone_dim=72, dense_dim=128, condition_dim=CONDITION_DIM, hidden_dim=128,
                 queries=64, layers=3, memory_tokens=1024, scales=(.75, 2., 6.), coverage_fraction=.25,
                 memory_cell=.75, context_priors=True, center_shift=False, warm_start=False):
        super().__init__()
        # v3 warm start: per-point features are Model20's dense hidden state plus a
        # zero-initialised conditioned adapter, and the mask decoder starts as a copy
        # of Model20's, so masks begin at stage-1 quality.
        self.warm_start = bool(warm_start)
        if self.warm_start and hidden_dim != dense_dim:
            raise ValueError('warm_start requires hidden_dim == dense_dim')
        # v3: each query may move the centre of its spatial prior away from the seed.
        self.center_shift = bool(center_shift)
        # v2: same-instance prior, vertical lid and mask/stage-1 overlap statistics.
        self.context_priors = bool(context_priors)
        self.condition_dim = condition_dim
        self.queries, self.memory_tokens, self.scales = int(queries), int(memory_tokens), tuple(scales)
        self.coverage = max(1, int(round(queries * coverage_fraction)))
        self.memory_cell = memory_cell
        self.input = nn.Sequential(nn.Linear(backbone_dim + dense_dim + condition_dim + 6, hidden_dim),
                                   nn.LayerNorm(hidden_dim), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(hidden_dim * (1 + len(self.scales)), hidden_dim),
                                  nn.LayerNorm(hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2 * hidden_dim),
                          nn.GELU(), nn.Linear(2 * hidden_dim, hidden_dim)) for _ in range(2)])
        self.need = nn.Linear(hidden_dim, 1)
        self.attention_memory = nn.Linear(hidden_dim, hidden_dim)
        self.mask_memory = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                         nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.position = nn.Linear(3, hidden_dim)
        self.layers = nn.ModuleList([MaskAttentionLayer(hidden_dim) for _ in range(layers)])
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.mask_query = nn.Linear(hidden_dim, hidden_dim)
        self.radius = nn.Linear(hidden_dim, 1)
        stats = 7 if self.context_priors else 0
        self.decision = nn.Sequential(nn.Linear(hidden_dim + stats, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.quality = nn.Sequential(nn.Linear(hidden_dim + stats, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.radius.weight)
        nn.init.zeros_(self.radius.bias)
        if self.warm_start:
            self.adapter = nn.Linear(hidden_dim, hidden_dim)
            nn.init.zeros_(self.adapter.weight)
            nn.init.zeros_(self.adapter.bias)
        if self.center_shift:
            self.shift = nn.Linear(hidden_dim, 2)
            nn.init.zeros_(self.shift.weight)
            nn.init.zeros_(self.shift.bias)
        if self.context_priors:
            self.same = nn.Linear(hidden_dim, 1)
            self.lid = nn.Linear(hidden_dim, 1)
            for layer in (self.same, self.lid):
                nn.init.zeros_(layer.weight)
            nn.init.zeros_(self.same.bias)
            nn.init.constant_(self.lid.bias, 2.)     # ~14 m: effectively open at the start
        with torch.no_grad():
            self.decision[-1].bias.copy_(torch.tensor([1., -1., 0.]))
            self.need.bias.fill_(-2.)

    COPIED = ('attention_memory', 'mask_memory', 'position', 'layers', 'query_norm', 'mask_query', 'radius')

    def load_model20_decoder(self, point_decoder):
        """Copy Model20's trained mask-decoder weights (shapes must match)."""
        for name in self.COPIED:
            getattr(self, name).load_state_dict(getattr(point_decoder, name).state_dict(), strict=True)

    def _memory(self, xyz, pool):
        grid = torch.floor((xyz[pool] - xyz[pool].amin(0)) / self.memory_cell).long()
        extent = grid.amax(0) + 1
        key = grid[:, 0] + extent[0] * (grid[:, 1] + extent[1] * grid[:, 2])
        order = torch.argsort(key, stable=True)
        first = torch.ones(len(order), dtype=torch.bool, device=xyz.device)
        first[1:] = key[order[1:]] != key[order[:-1]]
        representatives = pool[order[first]]
        if len(representatives) > self.memory_tokens:
            representatives = representatives[torch.linspace(0, len(representatives) - 1, self.memory_tokens,
                                                             device=xyz.device).long()]
        return representatives

    def _seeds(self, xyz, pool, memory_index, need_logits, teacher_need):
        anisotropic = xyz.new_tensor([1., 1., .5])
        score = need_logits[pool].detach()
        if teacher_need is not None and self.training and torch.rand((), device=xyz.device) < self.teacher_probability:
            score = teacher_need[pool].clamp_min(0.) * 10. + .001 * score
        wanted = self.queries - self.coverage
        # Every confident need voxel competes spatially, so one large missed
        # tree cannot monopolise the seeds of several small ones.
        confident = int((score > 0).sum())
        top = pool[score.topk(min(len(pool), max(4 * wanted, min(confident, 2048)))).indices]
        need_seeds = top[TreeMaskDecoder.fps(xyz[top] * anisotropic, wanted)]
        coverage = memory_index[TreeMaskDecoder.fps(xyz[memory_index] * anisotropic, self.coverage)]
        combined = torch.cat((need_seeds, coverage))
        unique, inverse = torch.unique(combined, return_inverse=True)
        position = torch.full((len(unique),), len(combined), dtype=torch.long, device=xyz.device)
        position.scatter_reduce_(0, inverse, torch.arange(len(combined), device=xyz.device), reduce='amin')
        return unique[torch.argsort(position)[:self.queries]]

    def forward(self, backbone_features, dense_hidden, data, condition, semantic_logits, offset_m,
                teacher_need=None, stage1_labels=None):
        xyz = data['coord']
        condition = condition[:, :self.condition_dim]
        if self.context_priors and stage1_labels is None:
            raise ValueError('context_priors requires the stage-1 labels of the window')
        geometry = torch.cat((data['feat'] / data['feat'].new_tensor([10., 10., 30., 1.]),
                              semantic_logits.softmax(1)[:, 1:2],
                              torch.linalg.vector_norm(offset_m[:, :2], dim=1, keepdim=True) / 10.), 1)
        hidden = self.input(torch.cat((backbone_features, dense_hidden, condition, geometry), 1))
        context = [pooled_context(hidden, xyz, s) for s in self.scales]
        hidden = self.fuse(torch.cat([hidden, *context], 1))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        need_logits = self.need(hidden).squeeze(1)
        if self.warm_start:
            hidden = dense_hidden + self.adapter(hidden)
        pool = torch.nonzero(xyz[:, 2] >= .5, as_tuple=False).flatten()
        if len(pool) < 8:
            pool = torch.arange(len(xyz), device=xyz.device)
        memory_index = self._memory(xyz, pool)
        seed_index = self._seeds(xyz, pool, memory_index, need_logits, teacher_need)
        position = self.position(xyz / xyz.new_tensor([10., 10., 30.]))
        memory = self.attention_memory(hidden[memory_index]) + position[memory_index]
        mask_features = F.normalize(self.mask_memory(hidden) + position, dim=1)
        query = hidden[seed_index] + position[seed_index]
        seed_center = xyz[seed_index, :2]
        if self.warm_start:      # Model20 centres its spatial prior on the seed's vote
            seed_center = seed_center + offset_m[seed_index, :2]
        distance2 = (xyz[:, :2][None] - seed_center[:, None]).square().sum(2)
        if self.context_priors:
            seed_label = stage1_labels[seed_index]
            same = ((stage1_labels[None] == seed_label[:, None]) & (seed_label[:, None] > 0)).to(hidden.dtype)
            unassigned = (stage1_labels == 0).to(hidden.dtype)
            above = xyz[:, 2][None] - xyz[seed_index, 2][:, None]

        def decode(q):
            normalized = self.query_norm(q)
            masks = 8. * (F.normalize(self.mask_query(normalized), dim=1) @ mask_features.T)
            radius = 1. + 5. * self.radius(normalized).sigmoid()
            planar = distance2
            if self.center_shift:
                center = seed_center + 3. * torch.tanh(self.shift(normalized))
                planar = (xyz[:, :2][None] - center[:, None]).square().sum(2)
            masks = masks + (2.5 - .5 * planar / radius.square()).clamp(min=-30.)
            head = normalized
            if self.context_priors:
                lid = .5 + 15. * self.lid(normalized).sigmoid()
                masks = masks + self.same(normalized) * same - F.relu(above - lid).square().clamp(max=30.)
                with torch.no_grad():
                    probability = masks.sigmoid()
                    size = probability.sum(1).clamp_min(1e-3)
                    inside = (probability * same).sum(1)
                    statistics = torch.stack((inside / size, inside / same.sum(1).clamp_min(1.),
                                              (probability * unassigned[None]).sum(1) / size,
                                              torch.log1p(size) / 8., xyz[seed_index, 2] / 30.,
                                              lid.squeeze(1) / 15., radius.squeeze(1) / 6.), 1)
                head = torch.cat((normalized, statistics), 1)
            return dict(mask_logits=masks, decision_logits=self.decision(head),
                        quality_logits=self.quality(head).squeeze(1))

        output = decode(query)
        auxiliary = []
        for layer in self.layers:
            blocked = output['mask_logits'][:, memory_index].detach() < 0
            blocked[blocked.all(1)] = False
            query = layer(query, memory, blocked)
            output = decode(query)
            auxiliary.append(output)
        return {**output, 'need_logits': need_logits, 'seed_index': seed_index,
                'memory_index': memory_index, 'aux_outputs': auxiliary[:-1] if self.training else []}


def refinement_losses(output, batch, missed_weight=2., decision_weights=(1., 2., 1.)):
    """Hungarian set loss with stage-1-aware decisions and a per-point need loss."""
    ids = batch['tree_id']
    valid = ids >= 0
    device = ids.device
    zero = output['need_logits'].sum() * 0.
    need = batch['need_target']
    need_valid = need >= 0
    losses = {}
    if need_valid.any():
        positive = (need[need_valid] > .5).sum().float()
        negative = need_valid.sum().float() - positive
        pos_weight = (negative / positive.clamp_min(1.)).clamp(1., 20.)
        losses['need'] = F.binary_cross_entropy_with_logits(output['need_logits'][need_valid], need[need_valid],
                                                            pos_weight=pos_weight)
    else:
        losses['need'] = zero
    seed = output['seed_index']
    gt_ids, counts = torch.unique(ids[ids > 0], return_counts=True)
    gt_ids = gt_ids[counts >= 4]
    queries = len(seed)
    if not queries or not valid.any():
        losses.update(mask=zero, decision=zero, quality=zero, mask_iou=zero, matched=zero,
                      matched_missed=zero, loss=losses['need'])
        return losses
    matched_label = batch['gt_matched_label']
    gt = (gt_ids[:, None] == ids[None, valid]).float()
    found = torch.zeros(len(gt_ids), dtype=torch.bool, device=device)
    for k, identifier in enumerate(gt_ids):
        found[k] = matched_label[ids == identifier][0] > 0
    logits = output['mask_logits'][:, valid].float()
    with torch.no_grad():
        probability = logits.sigmoid()
        cost = 1. - (2. * (probability @ gt.T) + 1.) / (probability.sum(1)[:, None] + gt.sum(1)[None] + 1.)
        # Missed trees are the purpose of the pass: make them cheaper to claim.
        cost = cost - .2 * (~found).float()[None]
        qi, gi = linear_sum_assignment(cost.cpu().numpy()) if len(gt_ids) else (np.zeros(0, int), np.zeros(0, int))
        qi = torch.as_tensor(qi, device=device, dtype=torch.long)
        gi = torch.as_tensor(gi, device=device, dtype=torch.long)
    targets = torch.zeros_like(logits)
    targets[qi] = gt[gi]
    decision_target = torch.zeros(queries, dtype=torch.long, device=device)
    decision_target[qi] = torch.where(found[gi], 2, 1)
    supervised = ids[seed] >= 0
    supervised[qi] = True
    weight = torch.ones(len(qi), device=device)
    weight[~found[gi]] = missed_weight
    class_weight = torch.tensor(decision_weights, device=device)

    def layer_loss(prediction):
        logit = prediction['mask_logits'][:, valid].float()
        prob = logit.sigmoid()
        if len(qi):
            p, t = prob[qi], targets[qi]
            dice = 1. - (2. * (p * t).sum(1) + 1.) / (p.sum(1) + t.sum(1) + 1.)
            bce = F.binary_cross_entropy_with_logits(logit[qi], t, reduction='none')
            pt = p * t + (1 - p) * (1 - t)
            focal = ((.25 * t + .75 * (1 - t)) * (1 - pt).square() * bce).mean(1)
            mask = ((2 * dice + 2 * focal) * weight).sum() / weight.sum()
        else:
            mask = zero
        with torch.no_grad():
            quality = torch.zeros(queries, device=device)
            if len(qi):
                inter = (prob[qi] * targets[qi]).sum(1)
                quality[qi] = inter / (prob[qi].sum(1) + targets[qi].sum(1) - inter).clamp_min(1e-6)
        decision = F.cross_entropy(prediction['decision_logits'][supervised], decision_target[supervised],
                                   weight=class_weight) if supervised.any() else zero
        regression = ((prediction['quality_logits'].sigmoid() - quality).square()[supervised].mean()
                      if supervised.any() else zero)
        return mask, decision, regression, (quality[qi].mean() if len(qi) else zero)

    mask, decision, regression, iou = layer_loss(output)
    for auxiliary in output.get('aux_outputs', []):
        m, d, r, _ = layer_loss(auxiliary)
        mask, decision, regression = mask + .5 * m, decision + .5 * d, regression + .5 * r
    losses.update(mask=mask, decision=decision, quality=regression, mask_iou=iou,
                  matched=torch.tensor(float(len(qi)), device=device),
                  matched_missed=(~found[gi]).float().sum() if len(gi) else zero)
    losses['loss'] = losses['need'] + mask + decision + .5 * regression
    return losses


def seed_refinement_losses(output, batch, missed_weight=2., decision_weights=(1., 2., 1.)):
    """v2 targets: every query segments the ground-truth tree that contains its seed.

    Decisions follow the stage-1 status of that tree (missed -> new, found ->
    existing); seeds on known non-tree voxels are background, seeds on unknown
    voxels are not supervised. Several queries may share a tree; duplicates are
    removed by reconciliation, not by the loss.
    """
    ids = batch['tree_id']
    valid = ids >= 0
    device = ids.device
    zero = output['need_logits'].sum() * 0.
    need = batch['need_target']
    need_valid = need >= 0
    losses = {}
    if need_valid.any():
        positive = (need[need_valid] > .5).sum().float()
        negative = need_valid.sum().float() - positive
        losses['need'] = F.binary_cross_entropy_with_logits(
            output['need_logits'][need_valid], need[need_valid],
            pos_weight=(negative / positive.clamp_min(1.)).clamp(1., 20.))
    else:
        losses['need'] = zero
    seed = output['seed_index']
    seed_ids = ids[seed]
    tree = seed_ids > 0
    supervised = seed_ids >= 0
    found = batch['gt_matched_label'][seed] > 0
    decision_target = torch.where(tree, torch.where(found, 2, 1), 0)
    targets = (seed_ids[:, None] == ids[None, valid]).float()
    weight = torch.where(found, 1., missed_weight)[tree]
    class_weight = torch.tensor(decision_weights, device=device)

    def layer_loss(prediction):
        logit = prediction['mask_logits'][:, valid].float()
        prob = logit.sigmoid()
        with torch.no_grad():
            inter = (prob * targets).sum(1)
            iou = torch.where(tree, inter / (prob.sum(1) + targets.sum(1) - inter).clamp_min(1e-6), 0.)
        if tree.any():
            p, t, l = prob[tree], targets[tree], logit[tree]
            dice = 1. - (2. * (p * t).sum(1) + 1.) / (p.sum(1) + t.sum(1) + 1.)
            bce = F.binary_cross_entropy_with_logits(l, t, reduction='none')
            pt = p * t + (1 - p) * (1 - t)
            focal = ((.25 * t + .75 * (1 - t)) * (1 - pt).square() * bce).mean(1)
            mask = ((2 * dice + 2 * focal) * weight).sum() / weight.sum()
        else:
            mask = zero
        if supervised.any():
            decision = F.cross_entropy(prediction['decision_logits'][supervised], decision_target[supervised],
                                       weight=class_weight)
            regression = (prediction['quality_logits'].sigmoid() - iou).square()[supervised].mean()
        else:
            decision = regression = zero
        return mask, decision, regression, iou

    mask, decision, regression, iou = layer_loss(output)
    for auxiliary in output.get('aux_outputs', []):
        m, d, r, _ = layer_loss(auxiliary)
        mask, decision, regression = mask + .5 * m, decision + .5 * d, regression + .5 * r
    missed = tree & ~found
    present = tree & found
    with torch.no_grad():
        predicted = output['decision_logits'].argmax(1)
        correct = (predicted == decision_target)[supervised].float().mean() if supervised.any() else zero
    losses.update(mask=mask, decision=decision, quality=regression,
                  mask_iou=iou[tree].mean() if tree.any() else zero,
                  mask_iou_missed=iou[missed].mean() if missed.any() else zero,
                  mask_iou_found=iou[present].mean() if present.any() else zero,
                  decision_accuracy=correct, matched=tree.float().sum(), matched_missed=missed.float().sum())
    losses['loss'] = losses['need'] + mask + decision + .5 * regression
    return losses


def build_refiner(payload):
    """Rebuild a refiner from a checkpoint, including v1 runs without context priors."""
    arguments = dict(payload['refiner_args'])
    if 'context_priors' not in arguments:
        arguments.update(context_priors=False, condition_dim=LEGACY_CONDITION_DIM)
    refiner = RefinementDecoder(**arguments)
    refiner.load_state_dict(payload['refiner'])
    return refiner
