"""Stem-seeded mask decoder: one query per detected stem foot, one mask per query, no NMS.

Model20's ``TreeMaskDecoder`` seeds its 96 queries by farthest-point sampling over the crop and
leaves "how many trees and where" to the masks and the consensus merge. Here the stems come from
the caller (``data['stems']``: local x, y of the foot and the tree height; ``data['stem_score']``
optional), the decoder builds one query per stem from the foot position, the tree height and the
backbone features pooled in a column around the foot, decodes one mask per query with a locality
prior centred on the foot (plus a learned shift of at most ``SHIFT_M``), and the assignment is a
per-point argmax over the stem masks (``assign_points``). Weights of ``TreeMaskDecoder`` are
reused; only the ``stem_*`` modules are new.

Supervision (``stem_mask_losses``): the target of stem ``s`` is the mask of the tree
``stem_tree_id[s]`` (> 0); stems with id 0 are false feet (empty mask, score 0); id -1 is ignored.
A per-point cross-entropy over (stems + background) trains the argmax assignment directly.
Training feet are corrupted (``corrupt_stems``) so the decoder does not trust them blindly.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pointcloud.dual_head import DensePointDecoder, TreeMaskDecoder, discriminative_loss

SHIFT_M = 1.0            # maximum learned xy shift of a mask centre away from its stem foot
COLUMN_SIGMA_M = 1.5     # horizontal kernel of the feature column pooled per stem
COLUMN_CUTOFF_M = 4.0
POSITION_SCALE = (10., 10., 30.)
MIN_TARGET_POINTS = 4


class StemSeededDecoder(TreeMaskDecoder):
    """``decoder_policy = 'stem_seeded'``: queries from stem feet instead of FPS seeds."""

    objectness = False  # v2 checkpoints set this (``train_stem_seeded.py --objectness``)

    def __init__(self, hidden_dim=128, queries=96, layers=3, memory_tokens=1024):
        super().__init__(hidden_dim=hidden_dim, queries=queries, layers=layers, memory_tokens=memory_tokens)
        self.stem_token = nn.Parameter(torch.zeros(hidden_dim))
        self.stem_height = nn.Linear(2, hidden_dim)      # (h / 30, h known)
        self.stem_score_embed = nn.Linear(1, hidden_dim)
        self.stem_column = nn.Linear(hidden_dim, hidden_dim)
        self.stem_shift = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.stem_shift.weight)
        nn.init.zeros_(self.stem_shift.bias)
        nn.init.zeros_(self.stem_score_embed.weight)
        nn.init.zeros_(self.stem_score_embed.bias)

    @staticmethod
    def _memory(xyz, indices, memory_tokens):
        """Spatially deduplicated (0.75 m) candidate points, at most ``memory_tokens``."""
        grid = torch.floor((xyz[indices] - xyz[indices].amin(0)) / .75).long()
        extent = grid.amax(0) + 1
        key = grid[:, 0] + extent[0] * (grid[:, 1] + extent[1] * grid[:, 2])
        order = torch.argsort(key, stable=True)
        first = torch.ones(len(order), dtype=torch.bool, device=xyz.device)
        first[1:] = key[order[1:]] != key[order[:-1]]
        representatives = indices[order[first]]
        if len(representatives) > memory_tokens:
            representatives = representatives[torch.linspace(0, len(representatives) - 1, memory_tokens,
                                                             device=xyz.device).long()]
        return representatives

    def forward(self, features, data, logits, offset):
        dense = DensePointDecoder.forward(self, features, data, logits, offset)
        hidden, xyz = dense.pop('features'), data['coord']
        embedding = self.embedding(hidden)
        probability = dense['semantic_logits'].softmax(1)[:, 1]
        stems = data.get('stems')
        result = {**dense, 'embedding': embedding, 'mask_supervision': 'stem_seeded', 'objectness': self.objectness}
        empty = {**result, 'mask_logits': hidden.new_empty((0, len(xyz))), 'object_logits': hidden.new_empty(0),
                 'seed_index': torch.zeros(0, dtype=torch.long, device=xyz.device), 'aux_outputs': [],
                 'stem_center_xy': hidden.new_empty((0, 2)), 'stem_tree_id': torch.zeros(0, dtype=torch.long, device=xyz.device)}
        if stems is None or not len(stems):
            return empty
        stems = stems.to(hidden.dtype)
        candidates = torch.nonzero(xyz[:, 2] >= .5, as_tuple=False).flatten()
        if not len(candidates):
            candidates = torch.arange(len(xyz), device=xyz.device)
        memory_index = self._memory(xyz, candidates, self.memory_tokens)
        position = self.position(xyz / xyz.new_tensor(POSITION_SCALE))
        memory = self.attention_memory(hidden[memory_index]) + position[memory_index]
        mask_features = F.normalize(self.mask_memory(hidden) + position, dim=1)

        # One query per stem: foot position, tree height, column of features above the foot.
        foot = torch.cat((stems[:, :2], torch.zeros_like(stems[:, :1])), 1)
        height = stems[:, 2:3]
        known = (height > 0).to(hidden.dtype)
        distance2 = (xyz[memory_index, :2][None] - stems[:, None, :2]).square().sum(2)
        weight = torch.exp(-distance2 / (2 * COLUMN_SIGMA_M ** 2)) * (distance2 <= COLUMN_CUTOFF_M ** 2)
        column = (weight @ memory) / weight.sum(1, keepdim=True).clamp_min(1e-6)
        score = data.get('stem_score')
        score = score.to(hidden.dtype)[:, None] if score is not None else torch.ones_like(height)
        query = (self.stem_token + self.position(foot / foot.new_tensor(POSITION_SCALE))
                 + self.stem_height(torch.cat((height / 30. * known, known), 1))
                 + self.stem_score_embed(score) + self.stem_column(column))
        point_distance2_base = xyz[:, :2]
        semantic_term = .25 * torch.logit(probability.clamp(.02, .98))[None]

        def decode(q):
            normalized = self.query_norm(q)
            masks = 8. * (F.normalize(self.mask_query(normalized), dim=1) @ mask_features.T)
            radius = 1. + 5. * self.radius(normalized).sigmoid()
            center = stems[:, :2] + SHIFT_M * torch.tanh(self.stem_shift(normalized))
            d2 = (point_distance2_base[None] - center[:, None]).square().sum(2)
            masks = masks + (2.5 - .5 * d2 / radius.square()).clamp(min=-30.) + semantic_term
            return dict(mask_logits=masks, object_logits=self.score(normalized).squeeze(1), stem_center_xy=center)

        output = decode(query)
        auxiliary = []
        for layer in self.layers:
            blocked = output['mask_logits'][:, memory_index].detach() < 0
            blocked[blocked.all(1)] = False
            query = layer(query, memory, blocked)
            output = decode(query)
            auxiliary.append(output)
        nearest = memory_index[distance2.argmin(1)] if len(memory_index) else torch.zeros(len(stems), dtype=torch.long, device=xyz.device)
        return {**result, **output, 'seed_index': nearest, 'query_features': self.query_norm(query),
                'stem_tree_id': data.get('stem_tree_id', torch.full((len(stems),), -1, dtype=torch.long, device=xyz.device)),
                'aux_outputs': auxiliary[:-1] if self.training else []}


def stem_targets(ids, stem_tree_id, minimum_points=MIN_TARGET_POINTS):
    """Per-stem target masks over all points and the supervision class of every stem.

    Returns ``target`` (S, N) float, ``real`` (S,) bool (tree with enough points),
    ``false`` (S,) bool (id 0) and ``owner`` (N,) long: the stem index of each point's tree, -1 if none.
    """
    device = ids.device
    S, N = len(stem_tree_id), len(ids)
    target = torch.zeros((S, N), dtype=torch.float32, device=device)
    real = torch.zeros(S, dtype=torch.bool, device=device)
    owner = torch.full((N,), -1, dtype=torch.long, device=device)
    positive = stem_tree_id > 0
    if positive.any():
        target[positive] = (stem_tree_id[positive, None] == ids[None]).float()
        real = positive & (target.sum(1) >= minimum_points)
        # The first stem of a tree owns its points (duplicates from corruption share the mask).
        for s in torch.nonzero(real, as_tuple=False).flatten().tolist()[::-1]:
            owner[ids == stem_tree_id[s]] = s
    false = stem_tree_id == 0
    return target, real, false, owner


def stem_mask_losses(output, ids, assignment_weight=1., objectness=None):
    zero = output['embedding'].sum() * 0.
    if objectness is None:
        objectness = bool(output.get('objectness', False))
    stem_tree_id = output.get('stem_tree_id')
    if stem_tree_id is None or not len(stem_tree_id) or not len(output['mask_logits']):
        return dict(mask_loss=zero, embedding_loss=discriminative_loss(output['embedding'], ids), mask_iou=zero,
                    assignment_loss=zero)
    valid = ids >= 0
    target, real, false, owner = stem_targets(ids, stem_tree_id)
    supervised = real | false
    target_valid = target[:, valid]
    assign_target = torch.where(ids > 0, owner, torch.full_like(owner, len(stem_tree_id)))  # background = S
    assign_mask = valid & ((ids == 0) | (owner >= 0))

    def layer_loss(prediction):
        logit = prediction['mask_logits'][:, valid].float()
        prob = logit.sigmoid()
        bce = F.binary_cross_entropy_with_logits(logit, target_valid, reduction='none')
        pt = prob * target_valid + (1 - prob) * (1 - target_valid)
        focal = ((.25 * target_valid + .75 * (1 - target_valid)) * (1 - pt).square() * bce).mean(1)
        dice = 1. - (2 * (prob * target_valid).sum(1) + 1.) / (prob.sum(1) + target_valid.sum(1) + 1.)
        with torch.no_grad():
            inter = (prob * target_valid).sum(1)
            quality = inter / (prob.sum(1) + target_valid.sum(1) - inter).clamp_min(1e-6)
            quality[false] = 0.
        mask = (2 * dice[real] + 2 * focal[real]).mean() if real.any() else zero
        if false.any():
            mask = mask + 2 * focal[false].mean()
        if supervised.any():
            if objectness:  # v2: is this a real stem? (calibrated, so a 0.5 threshold drops false feet)
                logit = prediction['object_logits'][supervised].float()
                target_obj = real[supervised].float()
                weight = torch.where(target_obj > 0, 1., float(min(max(real.sum().item(), 1) / max(false.sum().item(), 1), 4.)))
                mask = mask + F.binary_cross_entropy_with_logits(logit, target_obj, weight=weight)
            else:  # v1: score regressed to the mask IoU (0 for false feet)
                score = (prediction['object_logits'].sigmoid() - quality).square()
                mask = mask + .5 * score[supervised].mean()
        if assign_mask.any():
            logits = prediction['mask_logits'][:, assign_mask].float()
            full = torch.cat((logits, torch.zeros((1, logits.shape[1]), device=logits.device)), 0)
            assignment = F.cross_entropy(full.T, assign_target[assign_mask])
        else:
            assignment = zero
        return mask, assignment, quality[real].mean() if real.any() else zero

    mask, assignment, iou = layer_loss(output)
    total = mask + assignment_weight * assignment
    for aux in output.get('aux_outputs', []):
        m, a, _ = layer_loss(aux)
        total = total + .5 * (m + assignment_weight * a)
    return dict(mask_loss=total, embedding_loss=discriminative_loss(output['embedding'], ids), mask_iou=iou,
                assignment_loss=assignment)


def assign_points(mask_logits, object_logits=None, mask_threshold=.5, object_threshold=0., minimum_points=4):
    """Per-point argmax over the stem masks; stems below ``object_threshold`` or with fewer than
    ``minimum_points`` points get no instance. Returns labels (N,) with stem index + 1, 0 = none."""
    if not len(mask_logits):
        return np.zeros(mask_logits.shape[1] if mask_logits.ndim == 2 else 0, np.int64)
    probability = torch.as_tensor(mask_logits).float().sigmoid()
    if object_logits is not None and object_threshold > 0:
        keep = torch.as_tensor(object_logits).float().sigmoid() >= object_threshold
        probability = probability * keep[:, None]
    best_probability, best = probability.max(0)
    labels = torch.where(best_probability >= mask_threshold, best + 1, torch.zeros_like(best))
    counts = torch.bincount(labels, minlength=len(probability) + 1)
    small = counts[labels] < minimum_points
    labels[small & (labels > 0)] = 0
    return labels.cpu().numpy().astype(np.int64)


def corrupt_stems(stems, tree_id, rng, xy_canopy=None, drop=(0., .25), jitter=(.3, .6, 1.), height_scale=(.85, 1.15),
                  unknown_height=.2, false_rate=.15, duplicate=.05, fragment=0., fragment_distance=(1.5, 4.)):
    """Training noise on true feet so the decoder learns to doubt them.

    ``stems`` (S, 3) local x, y, h; ``tree_id`` (S,) > 0. Returns corrupted stems, their tree ids
    (0 = false foot) and an input score per stem. ``xy_canopy`` (M, 2): canopy point positions used
    to place false feet; without it false feet go next to true ones.
    """
    stems = np.asarray(stems, np.float64).copy()
    tree_id = np.asarray(tree_id, np.int64).copy()
    S = len(stems)
    keep = rng.random(S) >= rng.uniform(*drop)
    stems, tree_id = stems[keep], tree_id[keep]
    sigma = float(rng.choice(jitter))
    stems[:, :2] += rng.normal(0., sigma, (len(stems), 2))
    stems[:, 2] *= rng.uniform(height_scale[0], height_scale[1], len(stems))
    stems[rng.random(len(stems)) < unknown_height, 2] = 0.
    score = rng.uniform(.6, 1., len(stems))
    extra, extra_h = [], []
    n_false = rng.poisson(false_rate * len(stems) + .5) if false_rate > 0 else 0
    for _ in range(n_false):
        if len(stems) >= 2 and rng.random() < .5:
            i, j = rng.choice(len(stems), 2, replace=False)
            xy = (stems[i, :2] + stems[j, :2]) / 2 + rng.normal(0., .5, 2)
            h = max(stems[i, 2], stems[j, 2]) * rng.uniform(.5, 1.)
        elif xy_canopy is not None and len(xy_canopy):
            xy = xy_canopy[rng.integers(len(xy_canopy))] + rng.normal(0., .5, 2)
            h = float(rng.uniform(3., 30.))
        elif len(stems):
            xy = stems[rng.integers(len(stems)), :2] + rng.normal(0., 2., 2)
            h = float(rng.uniform(3., 30.))
        else:
            break
        extra.append(xy); extra_h.append(h)
    dup = rng.random(len(stems)) < duplicate
    for i in np.flatnonzero(dup):
        extra.append(stems[i, :2] + rng.normal(0., 1., 2)); extra_h.append(stems[i, 2])
    # A crown cut in two (Model20 splits 39 % of Weiser trees): a second foot under the fragment,
    # 1.5-4 m from the stem, slightly lower; it owns nothing.
    frag = rng.random(len(stems)) < fragment
    for i in np.flatnonzero(frag):
        angle, distance = rng.uniform(0, 2 * np.pi), rng.uniform(*fragment_distance)
        extra.append(stems[i, :2] + distance * np.array([np.cos(angle), np.sin(angle)]))
        extra_h.append(stems[i, 2] * rng.uniform(.6, 1.))
    if extra:
        added = np.column_stack((np.asarray(extra), np.asarray(extra_h)))
        stems = np.concatenate((stems, added))
        tree_id = np.concatenate((tree_id, np.zeros(len(added), np.int64)))
        score = np.concatenate((score, rng.uniform(.2, .9, len(added))))
    return stems, tree_id, score.astype(np.float32)


def stems_to_local(stems_world, local_to_world):
    """World xy of feet -> the crop's local frame (inverse of ``prepare_crop``'s transform)."""
    A = np.asarray(local_to_world, np.float64)[:2, :2]
    t = np.asarray(local_to_world, np.float64)[:2, 2]
    inverse = np.linalg.inv(A)
    local = (np.asarray(stems_world, np.float64)[:, :2] - t) @ inverse.T
    return np.column_stack((local, np.asarray(stems_world, np.float64)[:, 2]))


def stems_for_crop(stems_world, stem_tree_id, local_to_world, coord, margin=2.):
    """Feet of a crop: inside the crop's bounding box plus ``margin``; trees with too few points
    in the crop are marked -1 (not supervised)."""
    if not len(stems_world):
        return np.zeros((0, 3)), np.zeros(0, np.int64)
    local = stems_to_local(stems_world, local_to_world)
    low, high = coord[:, :2].min(0) - margin, coord[:, :2].max(0) + margin
    inside = ((local[:, :2] >= low) & (local[:, :2] <= high)).all(1)
    local, ids = local[inside], np.asarray(stem_tree_id, np.int64)[inside].copy()
    return local, ids


@torch.no_grad()
def predict_stems(model, arrays, stems_world, stem_score=None, max_points=40000, size=20., stride=6.,
                  mask_threshold=.5, object_threshold=0., device=None):
    """Whole plot in ``size`` windows every ``stride`` metres, with Model20's point ownership: every
    point belongs to exactly one window (its central ``stride`` cell) and is assigned there by the
    argmax over all feet inside that window, so the candidate stems of a point are compared in one
    forward pass and every foot within about (size - stride) / 2 metres of it competes. Windows with
    more than ``max_points`` points are covered in chunks (owned chunk plus random context). Feet in
    the frame of ``arrays['coord'] + arrays['source_origin']``. Returns labels (N,) = stem index + 1
    (0 = none), per-stem object scores (from the window where the foot is most central) and the
    winning probability of every point."""
    from scipy.spatial import cKDTree
    from scripts.evaluate_pointcloud_litept import model_input, ownership_intervals, starts_for_axis
    xyz = arrays['coord']
    stems = np.asarray(stems_world, np.float64).reshape(-1, 3).copy()
    stems[:, :2] -= arrays['source_origin'][:2]
    score_in = np.ones(len(stems), np.float32) if stem_score is None else np.asarray(stem_score, np.float32)
    device = device or next(model.parameters()).device
    best_probability = np.zeros(len(xyz), np.float32)
    best_stem = np.full(len(xyz), -1, np.int64)
    object_score = np.full(len(stems), np.nan, np.float32)
    centrality = np.full(len(stems), np.inf)
    if not len(stems) or not len(xyz):
        return np.zeros(len(xyz), np.int64), object_score, best_probability
    x0, x1 = float(xyz[:, 0].min()), float(xyz[:, 0].max())
    y0, y1 = float(xyz[:, 1].min()), float(xyz[:, 1].max())
    xs = starts_for_axis(x0, x1, size, size - stride)
    ys = starts_for_axis(y0, y1, size, size - stride)
    xo = ownership_intervals(xs, size, x0, float(np.nextafter(np.float32(x1), np.float32(np.inf))))
    yo = ownership_intervals(ys, size, y0, float(np.nextafter(np.float32(y1), np.float32(np.inf))))
    index = cKDTree(xyz[:, :2])
    visited = np.zeros(len(xyz), np.uint8)
    rng = np.random.default_rng(20261014)
    model.eval()
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            window = np.asarray(sorted(index.query_ball_point([x + size / 2, y + size / 2], size / 2 + 1e-4, p=np.inf)), np.int64)
            if not len(window):
                continue
            a, b = xo[xi]; c, d = yo[yi]
            owner = window[(xyz[window, 0] >= a) & (xyz[window, 0] < b) & (xyz[window, 1] >= c) & (xyz[window, 1] < d)]
            if not len(owner):
                continue
            visited[owner] += 1
            inside = np.flatnonzero((stems[:, 0] >= x) & (stems[:, 0] < x + size) & (stems[:, 1] >= y) & (stems[:, 1] < y + size))
            if not len(inside):
                continue
            for owned in np.array_split(owner, max(1, int(np.ceil(len(owner) / max_points)))):
                extra = np.setdiff1d(window, owned, assume_unique=True)
                capacity = max_points - len(owned)
                if len(extra) > capacity:
                    extra = rng.choice(extra, capacity, replace=False)
                chosen = np.sort(np.concatenate((owned, extra)))
                own_local = np.flatnonzero(np.isin(chosen, owned))
                batch = model_input(xyz[chosen], arrays['grid_coord'][chosen], arrays['intensity'][chosen], device,
                                    preserve_height=True)
                shift = xyz[chosen, :2].mean(0)
                local = stems[inside].copy(); local[:, :2] -= shift
                batch['stems'] = torch.from_numpy(local.astype(np.float32)).to(device)
                batch['stem_score'] = torch.from_numpy(score_in[inside]).to(device)
                result = model(batch)['instance_masks']
                probability = result['mask_logits'][:, own_local].sigmoid().cpu().numpy()
                scores = result['object_logits'].sigmoid().cpu().numpy()
                distance = np.abs(stems[inside, :2] - [x + size / 2, y + size / 2]).max(1)
                central = distance < centrality[inside]
                object_score[inside[central]] = scores[central]
                centrality[inside[central]] = distance[central]
                if object_threshold > 0:
                    probability = probability * (scores[:, None] >= object_threshold)
                top = probability.argmax(0)
                points = chosen[own_local]
                best_probability[points] = probability[top, np.arange(len(own_local))]
                best_stem[points] = inside[top]
    if not np.all(visited == 1):
        raise AssertionError(f'Ownership failed: missing={int((visited == 0).sum())}, repeated={int((visited > 1).sum())}')
    labels = np.where(best_probability >= mask_threshold, best_stem + 1, 0).astype(np.int64)
    counts = np.bincount(labels, minlength=len(stems) + 1)
    labels[(counts[labels] < MIN_TARGET_POINTS) & (labels > 0)] = 0
    return labels, object_score, best_probability
