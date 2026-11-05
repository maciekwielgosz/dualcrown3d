"""Rotation-ensembled centre votes of the frozen stage-1 model.

Model20's votes carry about 0.35 m of error that changes with the orientation of
the window. Averaging the votes over the lattice symmetries of the voxel grid
(90-degree rotations, optionally mirrored) removes that part without retraining:
on the 14 validation plots the true-centre oracle rises from 0.671 / 0.602 to
0.705 / 0.627 point / crown SB-PQ with four rotations. Symmetry 0 is the
identity, so the first member reproduces the ordinary single-pass votes exactly.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import cKDTree

# D4 as integer matrices acting on (x, y); the first four are the rotations.
SYMMETRIES = tuple(np.asarray(m, np.int64) for m in (
    [[1, 0], [0, 1]], [[0, -1], [1, 0]], [[-1, 0], [0, -1]], [[0, 1], [-1, 0]],
    [[-1, 0], [0, 1]], [[1, 0], [0, -1]], [[0, 1], [1, 0]], [[0, -1], [-1, 0]]))


def transform_window(coord, grid_coord, symmetry):
    """Apply a lattice symmetry about the window's mean; returns new coord and grid_coord."""
    coord = coord.copy()
    middle = coord[:, :2].mean(0)
    coord[:, :2] = (coord[:, :2] - middle) @ symmetry.T.astype(np.float32)
    grid = grid_coord.copy()
    grid[:, :2] = grid[:, :2] @ symmetry.T.astype(grid.dtype)
    return coord, grid


def restore_offsets(offsets, symmetry):
    """Map offsets predicted in the transformed frame back (the matrices are orthogonal)."""
    offsets = offsets.copy()
    offsets[:, :2] = offsets[:, :2] @ symmetry.astype(np.float32)
    return offsets


@torch.no_grad()
def ensemble_votes(model, arrays, symmetries=4, max_points=40000, size=20., overlap=8., embedding=False):
    """Votes and tree probability averaged over ``symmetries`` lattice symmetries.

    Uses the window sweep and centre ownership of ``scripts.predict_dual_head.predict``.
    Returns ``shifted_center`` [N, 3], ``tree_probability`` [N] and ``vote_spread`` [N]
    (mean horizontal distance of the members from their average, metres). With
    ``embedding`` also returns the stage-1 instance embedding [N, 5] of the
    unrotated pass, a head Model20 trains with a discriminative loss but never
    uses for its final labels.
    """
    from scripts.evaluate_pointcloud_litept import model_input, ownership_intervals, starts_for_axis
    xyz = arrays['coord']
    xs = starts_for_axis(float(xyz[:, 0].min()), float(xyz[:, 0].max()), size, overlap)
    ys = starts_for_axis(float(xyz[:, 1].min()), float(xyz[:, 1].max()), size, overlap)
    maximum_x, maximum_y = float(xyz[:, 0].max()), float(xyz[:, 1].max())
    xo = ownership_intervals(xs, size, float(xyz[:, 0].min()), float(np.nextafter(xyz[:, 0].max(), np.float32(np.inf))))
    yo = ownership_intervals(ys, size, float(xyz[:, 1].min()), float(np.nextafter(xyz[:, 1].max(), np.float32(np.inf))))
    index = cKDTree(xyz[:, :2])
    rng = np.random.default_rng(20260925)
    device = next(model.parameters()).device
    total = np.zeros((len(xyz), 3), np.float64)
    squares = np.zeros(len(xyz), np.float64)
    probability = np.zeros(len(xyz), np.float64)
    visited = np.zeros(len(xyz), np.uint8)
    codes = np.zeros((len(xyz), 5), np.float32) if embedding else None
    model.eval()
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            context = np.asarray(sorted(index.query_ball_point([x + size / 2, y + size / 2], size / 2 + 1e-4, p=np.inf)), dtype=np.int64)
            if not len(context):
                continue
            x_end = maximum_x if xi == len(xs) - 1 else x + size
            y_end = maximum_y if yi == len(ys) - 1 else y + size
            context = context[(xyz[context, 0] >= x) & (xyz[context, 0] <= x_end)
                              & (xyz[context, 1] >= y) & (xyz[context, 1] <= y_end)]
            a, b = xo[xi]
            c, d = yo[yi]
            owner = context[(xyz[context, 0] >= a) & (xyz[context, 0] < b) & (xyz[context, 1] >= c) & (xyz[context, 1] < d)]
            if not len(owner):
                continue
            for owned in np.array_split(owner, max(1, math.ceil(len(owner) / max_points))):
                extra = np.setdiff1d(context, owned, assume_unique=True)
                capacity = max_points - len(owned)
                if len(extra) > capacity:
                    extra = rng.choice(extra, capacity, replace=False)
                chosen = np.sort(np.concatenate((owned, extra)))
                local = np.flatnonzero(np.isin(chosen, owned))
                members = []
                for symmetry in SYMMETRIES[:symmetries]:
                    coord, grid = transform_window(xyz[chosen], arrays['grid_coord'][chosen], symmetry)
                    result = model(model_input(coord, grid, arrays['intensity'][chosen], device, preserve_height=True))
                    members.append(restore_offsets(result['offset_m'][local].cpu().numpy(), symmetry))
                    probability[owned] += result['semantic_logits'].softmax(1)[local, 1].cpu().numpy()
                    if embedding and symmetry is SYMMETRIES[0]:
                        codes[owned] = result['instance_masks']['embedding'][local].float().cpu().numpy()
                members = np.stack(members)
                mean = members.mean(0)
                total[owned] = mean
                squares[owned] = np.linalg.norm(members[..., :2] - mean[None, :, :2], axis=2).mean(0)
                visited[owned] += 1
    if not np.all(visited == 1):
        raise AssertionError(f'Ownership failed: missing={int((visited == 0).sum())}, repeated={int((visited > 1).sum())}')
    result = dict(shifted_center=(xyz + total).astype(np.float32),
                  tree_probability=(probability / symmetries).astype(np.float32),
                  vote_spread=squares.astype(np.float32))
    if embedding:
        result['embedding'] = codes
    return result
