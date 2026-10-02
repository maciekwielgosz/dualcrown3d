"""Plot-wide reconciliation of second-pass proposals with frozen stage-1 IDs.

A ``new`` proposal may reclaim points from a stage-1 instance (an absorbed small
tree) but never that instance's treetop and never a majority of its points, so
large crowns cannot be destroyed or duplicated. Rejected proposals never leave a
residual ID behind. Source codes continue the stage-1 convention:
6 second-pass new tree, 7 second-pass completion, 8 second-pass merge.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint

from pointcloud.two_pass import match_instances

SOURCE_NEW, SOURCE_COMPLETION, SOURCE_MERGE = 6, 7, 8

DEFAULT_RECONCILE = dict(
    new_probability=.5, min_quality=.3, mask_threshold=.5, minimum_voxels=12,
    minimum_height_m=2., minimum_area_m2=.75, max_take_fraction=.5, protect_top=True,
    dedupe_overlap=.3, allow_merge=False, merge_coverage=.7, apply_completion=False,
    completion_probability=.5, completion_dominance=.8, completion_distance_m=1.5)


def _groups(labels):
    indices = np.flatnonzero(labels)
    if not len(indices):
        return {}
    indices = indices[np.argsort(labels[indices], kind='stable')]
    pieces = np.split(indices, np.flatnonzero(np.diff(labels[indices])) + 1)
    return {int(labels[p[0]]): p for p in pieces}


def reconcile(arrays, stage1_labels, stage1_confidence, stage1_source, proposals, config=None, trace=None):
    """Return labels, confidence, source and a record per accepted operation.

    ``trace`` (optional dict) receives the outcome of every proposal index."""
    note = (lambda k, reason: trace.__setitem__(int(k), reason)) if trace is not None else (lambda k, reason: None)
    cfg = {**DEFAULT_RECONCILE, **(config or {})}
    if not 0 < cfg['max_take_fraction'] <= 1 or not 0 <= cfg['dedupe_overlap'] < 1:
        raise ValueError('Invalid reconciliation fractions')
    xyz = arrays['coord']
    n = len(xyz)
    labels = np.asarray(stage1_labels, np.int64).copy()
    confidence = np.asarray(stage1_confidence, np.float32).copy()
    source = np.asarray(stage1_source, np.uint8).copy()
    if len(labels) != n or len(confidence) != n or len(source) != n:
        raise ValueError('Stage-1 arrays must match the point cloud')
    offsets = proposals['candidate_offset']
    point_index = proposals['point_index']
    point_score = proposals['point_score'].astype(np.float32)
    decision = np.asarray(proposals['decision'], np.float32)
    quality = np.asarray(proposals['quality'], np.float32)
    count = len(quality)
    capacity = int(labels.max(initial=0)) + 1 + count
    size = np.zeros(capacity, np.int64)
    present = np.bincount(labels, minlength=1)
    size[:len(present)] = present
    size[0] = 0
    top_flag = np.zeros(n, bool)
    for members in _groups(labels).values():
        top_flag[members[np.argmax(xyz[members, 2])]] = True
    claimed = np.zeros(n, bool)
    next_id = int(labels.max(initial=0))
    voxel = float(arrays['voxel_size'])
    records = []
    p_new, p_existing = decision[:, 1], decision[:, 2]
    eligible = np.flatnonzero((p_new >= cfg['new_probability']) & (quality >= cfg['min_quality']))
    for k in eligible[np.argsort(-(p_new[eligible] * quality[eligible]), kind='stable')]:
        a, b = offsets[k:k + 2]
        keep = point_score[a:b] >= cfg['mask_threshold']
        idx, prob = point_index[a:b][keep], point_score[a:b][keep]
        if len(idx) < cfg['minimum_voxels']:
            note(k, 'too_few_voxels')
            continue
        if claimed[idx].mean() > cfg['dedupe_overlap']:
            note(k, 'duplicate_of_second_pass')
            continue
        owners = labels[idx]
        ids, counts = np.unique(owners[owners > 0], return_counts=True)
        cover = counts / np.maximum(size[ids], 1)
        heavy = cover >= cfg['max_take_fraction']
        if heavy.any():
            full = cover >= cfg['merge_coverage']
            partial = (~full) & (cover > .1)
            if (cfg['allow_merge'] and full.sum() >= 2 and not partial.any()
                    and counts[full].sum() / len(idx) >= .7):
                target = int(ids[full].min())
                for identifier in ids[full]:
                    if identifier != target:
                        members = labels == identifier
                        labels[members] = target
                        source[members] = SOURCE_MERGE
                        size[target] += size[identifier]
                        size[identifier] = 0
                free = idx[owners == 0]
                labels[free], source[free] = target, SOURCE_MERGE
                confidence[free] = prob[owners == 0] * p_new[k]
                size[target] += len(free)
                records.append(dict(tree_id=target, op='merge', merged=[int(i) for i in ids[full]],
                                    points=int(size[target]), proposal=int(k), score=float(p_new[k] * quality[k])))
                note(k, 'merged')
            else:
                note(k, 'takes_majority_of_instance')
            continue
        if cfg['protect_top'] and top_flag[idx].any():
            note(k, 'contains_existing_top')
            continue
        support = xyz[idx]
        if support[:, 2].max() < cfg['minimum_height_m']:
            note(k, 'too_low')
            continue
        if MultiPoint(support[:, :2]).convex_hull.buffer(voxel / 2.).area < cfg['minimum_area_m2']:
            note(k, 'too_small_area')
            continue
        note(k, 'accepted')
        next_id += 1
        labels[idx], source[idx], claimed[idx] = next_id, SOURCE_NEW, True
        confidence[idx] = prob * p_new[k]
        size[ids] -= counts
        size[next_id] = len(idx)
        records.append(dict(tree_id=next_id, op='new', points=int(len(idx)), taken=int(counts.sum()),
                            taken_from=[int(i) for i in ids], proposal=int(k),
                            score=float(p_new[k] * quality[k])))
    if cfg['apply_completion']:
        trees = {}
        eligible = np.flatnonzero((p_existing >= cfg['completion_probability']) & (quality >= cfg['min_quality']))
        for k in eligible[np.argsort(-(p_existing[eligible] * quality[eligible]), kind='stable')]:
            a, b = offsets[k:k + 2]
            keep = point_score[a:b] >= cfg['mask_threshold']
            idx, prob = point_index[a:b][keep], point_score[a:b][keep]
            owners = labels[idx]
            ids, counts = np.unique(owners[owners > 0], return_counts=True)
            if not len(ids) or counts.max() / counts.sum() < cfg['completion_dominance']:
                continue
            host = int(ids[counts.argmax()])
            free = owners == 0
            if not free.any():
                continue
            if host not in trees:
                trees[host] = cKDTree(xyz[labels == host])
            distance, _ = trees[host].query(xyz[idx[free]], distance_upper_bound=cfg['completion_distance_m'])
            accepted = idx[free][np.isfinite(distance)]
            if not len(accepted):
                continue
            labels[accepted], source[accepted] = host, SOURCE_COMPLETION
            confidence[accepted] = prob[free][np.isfinite(distance)] * p_existing[k]
            records.append(dict(tree_id=host, op='completion', points=int(len(accepted)), proposal=int(k),
                                score=float(p_existing[k] * quality[k])))
    if (labels[np.asarray(stage1_labels) > 0] == 0).any():
        raise AssertionError('Reconciliation left stage-1 points unassigned')
    return labels.astype(np.uint32), confidence, source, records


def segmentation_diagnostics(truth, labels, large_voxels=500, small_voxels=100, fraction=.2):
    """Split/merge counts and size-stratified recall at point IoU >= 0.5."""
    truth = np.asarray(truth)
    labels = np.asarray(labels).astype(np.int64)
    gt_ids, matched, _ = match_instances(truth, labels)
    known = truth >= 0
    result = dict(gt_total=int(len(gt_ids)), oversplit_gt=0, undersegmented_pred=0,
                  large_gt=0, large_tp=0, small_gt=0, small_tp=0)
    if not len(gt_ids):
        return result
    gi = np.searchsorted(gt_ids, truth[truth > 0])
    gt_count = np.bincount(gi, minlength=len(gt_ids))
    labelled = labels[truth > 0]
    pr_ids = np.unique(labels[labels > 0])
    if len(pr_ids):
        pi = np.searchsorted(pr_ids, labelled)
        both = labelled > 0
        table = np.bincount(gi[both] * len(pr_ids) + pi[both],
                            minlength=len(gt_ids) * len(pr_ids)).reshape(len(gt_ids), len(pr_ids))
        share = table / np.maximum(gt_count[:, None], 1)
        result['oversplit_gt'] = int(((share >= fraction).sum(1) >= 2).sum())
        result['undersegmented_pred'] = int(((share >= .5).sum(0) >= 2).sum())
    large = gt_count >= large_voxels
    small = gt_count <= small_voxels
    result.update(large_gt=int(large.sum()), large_tp=int((matched[large] > 0).sum()),
                  small_gt=int(small.sum()), small_tp=int((matched[small] > 0).sum()))
    return result
