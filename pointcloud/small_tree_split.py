"""Label-free features and conservative internal crown-split candidates."""
from __future__ import annotations

import numpy as np


def internal_split_features(arrays, anchor_labels, proposal_labels, proposal_confidence,
                            anchor_records, polygonizer):
    """Measure whether a compact proposal is a plausible tree inside an anchor."""
    xyz = arrays['coord']
    world_xy = xyz[:, :2].astype(np.float64) + arrays['source_origin'][:2]
    proposals = polygonizer({**arrays, 'world_xy': world_xy}, proposal_labels,
                            proposal_confidence)
    anchor_by_id = {int(r['tree_id']): r for r in anchor_records}
    anchor_sizes = np.bincount(anchor_labels.astype(np.int64))
    anchor_tops = {}
    for identifier in np.unique(anchor_labels[anchor_labels > 0]):
        members = np.flatnonzero(anchor_labels == identifier)
        anchor_tops[int(identifier)] = xyz[members[np.argmax(xyz[members, 2])]]
    features = []
    for record in proposals:
        old_id = int(record['tree_id'])
        indices = np.flatnonzero(proposal_labels == old_id)
        existing = anchor_labels[indices]
        assigned = existing[existing > 0]
        if not len(assigned):
            continue
        ids, counts = np.unique(assigned, return_counts=True)
        winner = int(np.argmax(counts))
        anchor_id = int(ids[winner])
        anchor = anchor_by_id.get(anchor_id)
        if anchor is None:
            continue
        top = anchor_tops[anchor_id]
        features.append(dict(record=record, old_id=old_id, anchor_id=anchor_id,
                             indices=indices,
                             dominance=float(counts[winner] / len(assigned)),
                             anchor_fraction=float(counts[winner] / anchor_sizes[anchor_id]),
                             anchor_area_ratio=float(anchor['geometry'].area /
                                                     max(record['geometry'].area, 1e-9)),
                             height_gap=float(top[2] - record['height']),
                             top_distance=float(np.linalg.norm(top[:2] -
                                         xyz[indices[np.argmax(xyz[indices, 2])], :2]))))
    return features


def guarded_internal_splits(anchor_labels, anchor_confidence, proposal_labels,
                            proposal_confidence, anchor_records, features, config):
    """Peel at most one compact lower-height proposal from each anchor.

    Other anchors are never touched. Anchor polygons remain frozen so their
    proven outlines cannot be fragmented by a speculative small-tree split.
    """
    labels = np.asarray(anchor_labels, np.uint32).copy()
    confidence = np.asarray(anchor_confidence, np.float32).copy()
    next_id = max(int(labels.max()), max((int(r['tree_id']) for r in anchor_records), default=0))
    accepted = []
    used_anchors = set()
    ranked = sorted(features, key=lambda x: (-x['record']['confidence'],
                    -x['height_gap'], x['old_id']))
    for item in ranked:
        r = item['record']
        anchor_id = item['anchor_id']
        if anchor_id in used_anchors:
            continue
        if (r['geometry'].area < config['min_area_m2'] or
            r['geometry'].area > config['max_area_m2'] or
            r['points'] < config['min_voxels'] or
            r['height'] < config['min_height_m'] or
            r['confidence'] < config['min_confidence'] or
            item['dominance'] < config['min_anchor_dominance'] or
            item['anchor_fraction'] > config['max_anchor_fraction'] or
            item['anchor_area_ratio'] < config['min_anchor_area_ratio'] or
            item['height_gap'] < config['min_height_gap_m'] or
            item['top_distance'] < config['min_top_distance_m']):
            continue
        points = item['indices']
        points = points[(labels[points] == anchor_id) | (labels[points] == 0)]
        if len(points) < config['min_voxels']:
            continue
        remaining = np.count_nonzero(labels == anchor_id) - np.count_nonzero(labels[points] == anchor_id)
        if remaining < config['min_anchor_remaining_voxels']:
            continue
        next_id += 1
        labels[points] = next_id
        confidence[points] = proposal_confidence[points]
        accepted.append({**r, 'tree_id': next_id,
                         'source_proposal_id': item['old_id'],
                         'source_anchor_id': anchor_id})
        used_anchors.add(anchor_id)
    return labels, confidence, [*anchor_records, *accepted], accepted
