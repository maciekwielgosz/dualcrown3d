"""Label-free geometric partitions and label-only diagnostic oracles."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from pointcloud.instance_output import point_instance_metrics


def geometric_partition(coord: np.ndarray, cell_m: float) -> np.ndarray:
    """Assign compact 3-D cell IDs without looking at instance labels."""
    if cell_m <= 0 or len(coord) == 0:
        raise ValueError("Expected non-empty coordinates and positive cell size")
    xyz = np.asarray(coord, np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Expected finite Nx3 coordinates")
    grid = np.floor((xyz - xyz.min(0)) / cell_m).astype(np.int64)
    _, inverse = np.unique(grid, axis=0, return_inverse=True)
    return inverse.astype(np.int32)


def neighbor_pairs(coord: np.ndarray, radius_m: float = 1.0,
                   k: int = 8) -> np.ndarray:
    """Undirected local point graph, rebuilt for each input cloud."""
    if radius_m <= 0 or k < 2:
        raise ValueError("Invalid neighborhood")
    n = len(coord)
    if n < 2:
        return np.empty((0, 2), np.int32)
    _, index = cKDTree(coord).query(coord, k=min(k, n),
                                    distance_upper_bound=radius_m, workers=-1)
    index = np.atleast_2d(index)
    if index.shape[0] != n:
        index = index.T
    a = np.broadcast_to(np.arange(n)[:, None], index.shape)
    valid = (index < n) & (index != a)
    edge = np.column_stack((a[valid], index[valid])).astype(np.int32)
    edge.sort(axis=1)
    return np.unique(edge, axis=0)


def _majority_labels(groups: np.ndarray, truth: np.ndarray) -> np.ndarray:
    count = int(groups.max()) + 1
    known = truth >= 0
    # Unknown points cannot vote for either a tree or known background.
    pair, pair_count = np.unique(np.column_stack((groups[known], truth[known])),
                                 axis=0, return_counts=True)
    majority = np.zeros(count, np.int64)
    if len(pair):
        order = np.lexsort((pair[:, 1], -pair_count, pair[:, 0]))
        chosen = pair[order]
        _, first = np.unique(chosen[:, 0], return_index=True)
        majority[chosen[first, 0]] = chosen[first, 1]
    return majority


def _graph_components(groups: np.ndarray, majority: np.ndarray,
                      truth: np.ndarray, edges: np.ndarray) -> tuple[int, int]:
    n = len(majority)
    parent = np.arange(n, dtype=np.int32)

    def root(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    ge = groups[edges] if len(edges) else np.empty((0, 2), np.int32)
    for a, b in np.unique(np.sort(ge, axis=1), axis=0):
        a, b = int(a), int(b)
        if a != b and majority[a] > 0 and majority[a] == majority[b]:
            parent[root(a)] = root(b)
    tree_count = 0
    disconnected = 0
    for tree in np.unique(truth[truth > 0]):
        parts = np.flatnonzero(majority == tree)
        if len(parts):
            tree_count += 1
            disconnected += len({root(int(g)) for g in parts}) > 1
    return tree_count, disconnected


def partition_diagnostics(coord: np.ndarray, truth: np.ndarray,
                          groups: np.ndarray, edges: np.ndarray) -> dict:
    """GT-assisted *diagnostic*, never an inference prediction.

    Majority-vote labels form an optimistic partition-specific prediction; it
    can still split/lose true trees because mixed groups are irreversible.
    """
    truth = np.asarray(truth, np.int64)
    groups = np.asarray(groups, np.int32)
    if len(coord) != len(truth) or len(truth) != len(groups):
        raise ValueError("Coordinates, labels and partitions must align")
    n_groups = int(groups.max()) + 1
    majority = _majority_labels(groups, truth)
    prediction = majority[groups]
    result = point_instance_metrics(truth, prediction)
    denom = 2 * result["tp"] + result["fp"] + result["fn"]
    result["f1"] = 2 * result["tp"] / denom if denom else 0.
    result["pq"] = 2 * result["iou_sum"] / denom if denom else 0.
    counts = np.bincount(groups, minlength=n_groups)
    known = truth >= 0
    correct = (prediction == truth) & known
    valid_groups = np.unique(groups[known])
    label_pair = np.unique(np.column_stack((groups[known], truth[known])), axis=0)
    per_group_labels = np.bincount(label_pair[:, 0], minlength=n_groups)
    mixed = per_group_labels > 1
    cross_tree = ((truth[edges[:, 0]] > 0) & (truth[edges[:, 1]] > 0) &
                  (truth[edges[:, 0]] != truth[edges[:, 1]])) if len(edges) else np.zeros(0, bool)
    boundary = edges[cross_tree]
    boundary_recall = (float(np.mean(groups[boundary[:, 0]] != groups[boundary[:, 1]]))
                       if len(boundary) else None)
    graph_trees, disconnected = _graph_components(groups, majority, truth, edges)
    return dict(points=len(truth), superpoints=n_groups,
                compression=len(truth) / n_groups,
                median_superpoint_points=float(np.median(counts)),
                p95_superpoint_points=float(np.quantile(counts, .95)),
                mixed_group_fraction=float(np.mean(mixed[valid_groups])) if len(valid_groups) else 0.,
                known_point_purity=float(np.mean(correct[known])) if known.any() else 0.,
                boundary_pairs=len(boundary), boundary_recall=boundary_recall,
                graph_trees=graph_trees, graph_disconnected_trees=disconnected,
                gt_trees=int(len(np.unique(truth[truth > 0]))),
                oracle=result)


def centroid_graph_connectivity(coord: np.ndarray, truth: np.ndarray,
                                groups: np.ndarray, *, radius_m: float,
                                k: int = 32) -> dict:
    """Assess a bounded candidate graph without granting it GT edges.

    GT IDs are used only *after* construction to count trees whose majority-
    labeled superpoints remain disconnected in the proposed local graph.
    """
    groups = np.asarray(groups, np.int32)
    count = int(groups.max()) + 1
    sizes = np.bincount(groups, minlength=count)
    centers = np.column_stack([
        np.bincount(groups, weights=coord[:, axis], minlength=count) / sizes
        for axis in range(3)
    ])
    edges = neighbor_pairs(centers, radius_m=radius_m, k=k)
    majority = _majority_labels(groups, np.asarray(truth, np.int64))
    tree_count, disconnected = _graph_components(
        np.arange(count, dtype=np.int32), majority, np.asarray(truth, np.int64), edges)
    return dict(superpoint_edges=len(edges), graph_trees=tree_count,
                graph_disconnected_trees=disconnected,
                disconnected_fraction=disconnected / max(tree_count, 1))
