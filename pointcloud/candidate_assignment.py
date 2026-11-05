"""Stage 3: learned assignment of voxels to candidate tree centres.

Why: with the real v7 detections above score 0.2 (86 % of the validation trees
have a candidate), giving every annotated voxel to the candidate of its own tree
would reach 0.93 point SB-PQ on dense validation, 0.91 at the reference-scene
density and 0.97 on HELIOS. Nearest-centre assignment of the same candidates
reaches 0.51 / 0.48 / 0.62. The votes of boundary voxels are imprecise and
unmatched candidates steal voxels; both are decisions a model can learn.

Formulation: every tree voxel sees its ``K`` nearest candidates in scaled vote
space as ordered slots. Per slot it gets relative features (vote-to-centre
offset, distance in units of the candidate's reach, the candidate's score and
size, the voxel's position relative to the centre). A classifier chooses a slot
or "none". A candidate nobody chooses disappears, so this also rejects
duplicates and false centres.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from pointcloud.vote_field import ASSIGN, SCALE

K = 6
VOXEL_FEATURES = ('height', 'tree_probability', 'vote_spread', 'vote_travel', 'vote_rise')
SLOT_FEATURES = ('vote_distance', 'vote_distance_over_extent', 'vote_dxy', 'vote_dz', 'point_dxy', 'point_dz',
                 'point_height_ratio', 'score', 'extent', 'centre_height', 'is_stage1_centre', 'valid')
PAIR_EXTRA = ('slot', 'best_other_distance_over_extent', 'best_other_score', 'margin_to_best_other')


def match_candidates(candidates, true_ids, true_centres, xy=1.5, z=4.):
    """Tree id per candidate (0 = none): strongest candidate within ``xy``/``z`` of each
    tree's median vote, one candidate per tree."""
    centres, scores = candidates['centres'], candidates['scores']
    owner = np.zeros(len(centres), np.int64)
    truth = np.asarray(true_centres, np.float32).reshape(-1, 3)
    if not len(centres) or not len(truth):
        return owner
    tree = cKDTree(truth[:, :2])
    used = set()
    for i in np.argsort(-scores):
        best, gap = -1, np.inf
        for j in tree.query_ball_point(centres[i, :2], xy):
            distance = np.linalg.norm(truth[j, :2] - centres[i, :2])
            if j not in used and abs(truth[j, 2] - centres[i, 2]) <= z and distance < gap:
                best, gap = j, distance
        if best >= 0:
            used.add(best)
            owner[i] = int(true_ids[best])
    return owner


def slot_inputs(arrays, votes, probability, spread, candidates, k=K, reach=6.):
    """Voxel and slot features for every tree voxel.

    Returns ``index`` [M] (rows of ``arrays``), ``slots`` [M, k] (candidate row or -1),
    ``voxel`` [M, len(VOXEL_FEATURES)] and ``slot`` [M, k, len(SLOT_FEATURES)].
    """
    xyz = arrays['coord']
    index = np.flatnonzero((probability >= ASSIGN['probability']) & (xyz[:, 2] >= .5) & np.isfinite(votes).all(1))
    centres, scores, extents = candidates['centres'], candidates['scores'], candidates['extents']
    M = len(index)
    slots = np.full((M, k), -1, np.int64)
    voxel = np.zeros((M, len(VOXEL_FEATURES)), np.float32)
    slot = np.zeros((M, k, len(SLOT_FEATURES)), np.float32)
    if not M:
        return index, slots, voxel, slot
    point, vote = xyz[index], votes[index]
    delta = vote - point
    voxel[:, 0] = point[:, 2] / 30.
    voxel[:, 1] = probability[index]
    voxel[:, 2] = spread[index]
    voxel[:, 3] = np.linalg.norm(delta[:, :2], axis=1) / 5.
    voxel[:, 4] = delta[:, 2] / 10.
    if not len(centres):
        return index, slots, voxel, slot
    kk = min(k, len(centres))
    distance, nearest = cKDTree(centres * SCALE).query(vote * SCALE, k=kk, distance_upper_bound=reach)
    distance, nearest = distance.reshape(M, kk), nearest.reshape(M, kk)
    valid = np.isfinite(distance)
    nearest = np.where(valid, nearest, 0)
    slots[:, :kk] = np.where(valid, nearest, -1)
    c = centres[nearest]                                    # [M, kk, 3]
    to_vote = vote[:, None, :] - c
    to_point = point[:, None, :] - c
    reachable = np.where(valid, distance, reach)
    block = slot[:, :kk]
    block[..., 0] = reachable / reach
    block[..., 1] = np.minimum(reachable / extents[nearest], 6.) / 6.
    block[..., 2] = np.linalg.norm(to_vote[..., :2], axis=2) / reach
    block[..., 3] = to_vote[..., 2] / 10.
    block[..., 4] = np.linalg.norm(to_point[..., :2], axis=2) / reach
    block[..., 5] = to_point[..., 2] / 10.
    block[..., 6] = point[:, None, 2] / np.maximum(c[..., 2], 1.) / 3.
    block[..., 7] = scores[nearest]
    block[..., 8] = extents[nearest] / 4.
    block[..., 9] = c[..., 2] / 30.
    block[..., 10] = candidates['is_anchor'][nearest]
    block[..., 11] = 1.
    block[~valid] = 0.
    return index, slots, voxel, slot


def slot_targets(tree_id, slots, owner):
    """Class per voxel: the slot holding the candidate of its own tree, ``k`` for
    "none", and -1 (ignored) for unannotated voxels."""
    k = slots.shape[1]
    truth = np.asarray(tree_id, np.int64)
    candidate_tree = np.where(slots >= 0, owner[np.maximum(slots, 0)], 0)
    hit = (candidate_tree == truth[:, None]) & (truth[:, None] > 0)
    target = np.where(hit.any(1), hit.argmax(1), k).astype(np.int64)
    target[truth < 0] = -1
    return target


def pair_table(voxel, slot):
    """Flat (voxel, slot) rows for a tabular classifier: voxel + slot + competition features."""
    M, k, F = slot.shape
    ratio = np.where(slot[..., 11] > 0, slot[..., 1], 1.)                # distance over extent, 1 = unreachable
    order = np.sort(ratio, axis=1)
    best_other = np.where(ratio <= order[:, :1], order[:, 1:2] if k > 1 else 1., order[:, :1])
    other_score = np.zeros((M, k), np.float32)
    first = ratio.argmin(1)
    masked = ratio.copy()
    masked[np.arange(M), first] = np.inf
    second = masked.argmin(1)
    rows = np.arange(M)
    other_score[:] = slot[rows, first, 7][:, None]
    other_score[rows, first] = slot[rows, second, 7]
    extra = np.stack((np.broadcast_to(np.arange(k, dtype=np.float32) / k, (M, k)), best_other, other_score,
                      best_other - ratio), axis=2)
    table = np.concatenate((np.broadcast_to(voxel[:, None, :], (M, k, voxel.shape[1])), slot, extra), axis=2)
    return table.reshape(M * k, -1), slot[..., 11].reshape(-1) > 0


def labels_from_choice(arrays, index, slots, choice, config=None):
    """Instance labels from a per-voxel slot choice (``k`` or -1 = none), with Model20's size filters.
    Returns labels and, per label (index 0 unused), the candidate row."""
    from shapely.geometry import MultiPoint
    cfg = {**ASSIGN, **(config or {})}
    xyz = arrays['coord']
    labels = np.zeros(len(xyz), np.uint32)
    origin = [-1]
    k = slots.shape[1]
    chosen = (choice >= 0) & (choice < k)
    candidate = np.where(chosen, slots[np.arange(len(index)), np.clip(choice, 0, k - 1)], -1)
    keep = candidate >= 0
    temporary = np.zeros(len(xyz), np.int64)
    temporary[index[keep]] = candidate[keep] + 1
    members_index = np.flatnonzero(temporary)
    members_index = members_index[np.argsort(temporary[members_index], kind='stable')]
    identifier = 0
    voxel_size = float(arrays['voxel_size'])
    for members in (np.split(members_index, np.flatnonzero(np.diff(temporary[members_index])) + 1) if len(members_index) else []):
        if len(members) < cfg['min_voxels'] or xyz[members, 2].max() < cfg['min_height_m']:
            continue
        if MultiPoint(xyz[members, :2]).convex_hull.buffer(voxel_size / 2.).area < cfg['min_area_m2']:
            continue
        identifier += 1
        labels[members] = identifier
        origin.append(int(temporary[members[0]]) - 1)
    return labels, np.asarray(origin)
