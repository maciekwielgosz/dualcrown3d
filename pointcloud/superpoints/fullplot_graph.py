"""Reconcile overlapping-window mask proposals before assigning instance IDs."""
from __future__ import annotations

import numpy as np
from scipy import sparse


FEATURES = ('log_points', 'log_area_cells', 'max_height', 'height_std',
            'mean_mask', 'std_mask', 'max_object', 'mean_object',
            'log_votes', 'xy_compactness', 'xy_aspect', 'top_offset',
            'point_density', 'member_agreement')
DEFAULT_GRAPH = dict(mask_threshold=.5, min_points=8, min_height_m=1.5,
                     link_min_overlap=.55, link_min_iou=.20,
                     link_max_centroid_distance_m=4., link_max_top_difference_m=3.)


def _features(xyz, indices, scores, object_scores, member_sizes):
    points = xyz[indices]
    xy = points[:, :2]
    bins = np.floor((xy - xy.min(0)) / .5).astype(np.int32)
    cells = np.unique(bins, axis=0)
    box = (bins.max(0) - bins.min(0) + 1).astype(np.float32)
    centered = xy - xy.mean(0)
    covariance = centered.T @ centered / max(len(xy), 1)
    eigen = np.linalg.eigvalsh(covariance).clip(min=1e-6)
    top_xy = xy[int(np.argmax(points[:, 2]))]
    radius = np.sqrt(float(eigen.sum())).clip(min=.25)
    vector = np.asarray([
        np.log1p(len(indices)), np.log1p(len(cells)), points[:, 2].max(),
        points[:, 2].std(), scores.mean(), scores.std(),
        np.max(object_scores), np.mean(object_scores), np.log1p(len(object_scores)),
        len(cells) / max(float(np.prod(box)), 1.),
        np.sqrt(float(eigen[0] / eigen[1])),
        np.linalg.norm(top_xy - xy.mean(0)) / radius,
        len(indices) / max(len(cells), 1),
        sum(member_sizes) / max(len(indices), 1),
    ], np.float32)
    return vector


def reconcile(arrays, raw, config=DEFAULT_GRAPH):
    """Return graph-fused whole-mask proposals; no residual mask becomes an ID."""
    xyz = arrays['coord']
    offsets = raw['candidate_offset']
    all_idx = raw['point_index']
    all_prob = raw['point_score']
    objects = raw['object_score']
    nodes = []
    for i, obj in enumerate(objects):
        if obj < .1:
            continue
        a, b = int(offsets[i]), int(offsets[i + 1])
        keep = all_prob[a:b] >= config['mask_threshold']
        idx = all_idx[a:b][keep].astype(np.int32)
        if len(idx) < config['min_points'] or xyz[idx, 2].max() < config['min_height_m']:
            continue
        prob = all_prob[a:b][keep].astype(np.float32)
        center = np.average(xyz[idx, :2], axis=0, weights=prob)
        nodes.append((idx, prob, float(obj), center, float(xyz[idx, 2].max())))
    if not nodes:
        return dict(offset=np.asarray([0], np.int64), point_index=np.empty(0, np.int32),
                    point_score=np.empty(0, np.float32), feature=np.empty((0, len(FEATURES)), np.float32),
                    source_count=np.empty(0, np.int32), graph_edges=0)
    row = np.repeat(np.arange(len(nodes), dtype=np.int32), [len(n[0]) for n in nodes])
    col = np.concatenate([n[0] for n in nodes])
    membership = sparse.csr_matrix((np.ones(len(col), np.int32), (row, col)),
                                   shape=(len(nodes), len(xyz)), dtype=np.int32)
    overlap = (membership @ membership.T).tocoo()
    sizes = np.asarray([len(n[0]) for n in nodes], np.int32)
    centers = np.asarray([n[3] for n in nodes], np.float32)
    heights = np.asarray([n[4] for n in nodes], np.float32)
    parent = np.arange(len(nodes), dtype=np.int32)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = 0
    for i, j, n in zip(overlap.row, overlap.col, overlap.data):
        if i >= j:
            continue
        if (n / min(sizes[i], sizes[j]) < config['link_min_overlap'] or
            n / (sizes[i] + sizes[j] - n) < config['link_min_iou'] or
            min(sizes[i], sizes[j]) / max(sizes[i], sizes[j]) <
                config.get('link_min_size_ratio', 0.) or
            np.linalg.norm(centers[i] - centers[j]) > config['link_max_centroid_distance_m'] or
            abs(heights[i] - heights[j]) > config['link_max_top_difference_m']):
            continue
        a, b = root(int(i)), root(int(j))
        if a != b:
            parent[b] = a
        edges += 1
    groups = {}
    for i in range(len(nodes)):
        groups.setdefault(root(i), []).append(i)
    offsets_out = [0]
    indices_out, scores_out, feature_out, votes_out = [], [], [], []
    for members in groups.values():
        idx = np.concatenate([nodes[i][0] for i in members])
        prob = np.concatenate([nodes[i][1] for i in members])
        order = np.argsort(idx, kind='stable')
        idx, prob = idx[order], prob[order]
        first = np.r_[0, np.flatnonzero(np.diff(idx)) + 1]
        unique = idx[first]
        best = np.maximum.reduceat(prob, first)
        object_scores = [nodes[i][2] for i in members]
        feature_out.append(_features(xyz, unique, best, object_scores,
                                     [len(nodes[i][0]) for i in members]))
        indices_out.append(unique)
        scores_out.append(best)
        votes_out.append(len(members))
        offsets_out.append(offsets_out[-1] + len(unique))
    return dict(offset=np.asarray(offsets_out, np.int64),
                point_index=np.concatenate(indices_out).astype(np.int32),
                point_score=np.concatenate(scores_out).astype(np.float32),
                feature=np.stack(feature_out), source_count=np.asarray(votes_out, np.int32),
                graph_edges=edges)


def target_labels(graph, truth):
    """Label proposals from training annotations; unknown regions are ignored."""
    reference = np.asarray(truth)
    gt_ids, gt_sizes = np.unique(reference[reference > 0], return_counts=True)
    gt_count = dict(zip(gt_ids.tolist(), gt_sizes.tolist()))
    targets = np.full(len(graph['feature']), -1, np.int8)
    best_iou = np.zeros(len(targets), np.float32)
    best_gt = np.zeros(len(targets), np.int32)
    for i in range(len(targets)):
        a, b = graph['offset'][i:i+2]
        ids = graph['point_index'][a:b]
        known = reference[ids] >= 0
        if known.mean() < .5:
            continue
        observed, count = np.unique(reference[ids][known], return_counts=True)
        positive = observed > 0
        observed, count = observed[positive], count[positive]
        if len(observed):
            sizes = np.asarray([gt_count[int(identifier)] for identifier in observed])
            iou = count / (known.sum() + sizes - count)
            winner = int(np.argmax(iou))
            best_iou[i] = float(iou[winner])
            best_gt[i] = int(observed[winner])
        targets[i] = int(best_iou[i] >= .5)
    return targets, best_iou, best_gt
