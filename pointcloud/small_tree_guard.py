"""Conservative small-crown additions without revising trusted instance IDs.

The retained dual-head result is an immutable anchor.  A second decoder may
only label previously unassigned voxels where its residual support is compact,
mostly novel, and spatially distinct from anchor crowns.  This is deliberately
not a majority vote: a large tree can never be split by the second decoder.
"""
from __future__ import annotations

import numpy as np
import shapely


def candidate_features(arrays, anchor_labels, proposal_labels, proposal_confidence,
                       anchor_records, polygonizer):
    """Build each residual footprint once, independently of calibration knobs."""
    anchor_labels = np.asarray(anchor_labels)
    proposal_labels = np.asarray(proposal_labels)
    if anchor_labels.shape != proposal_labels.shape or anchor_labels.ndim != 1:
        raise ValueError('Point label arrays must have equal one-dimensional shape')
    if len(proposal_confidence) != len(anchor_labels):
        raise ValueError('Proposal confidence has another point count')
    residual = np.where(anchor_labels == 0, proposal_labels, 0).astype(np.uint32)
    xyz = arrays['coord']
    world_xy = xyz[:, :2].astype(np.float64) + arrays['source_origin'][:2]
    records = polygonizer({**arrays, 'world_xy': world_xy}, residual, proposal_confidence)
    total = np.bincount(proposal_labels.astype(np.int64))
    base = [r['geometry'] for r in anchor_records]
    index = shapely.STRtree(base) if base else None
    features = []
    for record in records:
        old_id = int(record['tree_id'])
        if not record['points'] or old_id >= len(total):
            continue
        geometry = record['geometry']
        nearby = index.query(geometry, predicate='intersects') if index is not None else []
        covered = (shapely.union_all([base[int(i)] for i in nearby]).intersection(geometry).area
                   if len(nearby) else 0.)
        features.append(dict(record=record, old_id=old_id,
                             novelty=float(record['points'] / total[old_id]),
                             base_cover=float(covered / max(geometry.area, 1e-9)),
                             area=float(geometry.area)))
    return residual, features


def guarded_additions(anchor_labels, anchor_confidence, proposal_labels,
                      proposal_confidence, anchor_records, residual, features, config):
    """Return unchanged anchor IDs plus accepted small, independent proposals."""
    labels = np.asarray(anchor_labels, dtype=np.uint32).copy()
    confidence = np.asarray(anchor_confidence, dtype=np.float32).copy()
    next_id = max(int(labels.max()), max((int(r['tree_id']) for r in anchor_records), default=0))
    accepted = []
    for item in sorted(features, key=lambda v: (-v['record']['confidence'], v['old_id'])):
        record = item['record']
        geometry = record['geometry']
        if (item['novelty'] < config['min_novelty'] or
            item['base_cover'] > config['max_base_cover'] or
            item['area'] > config['max_area_m2'] or
            item['area'] < config['min_area_m2'] or
            record['points'] < config['min_voxels'] or
            record['height'] < config['min_height_m'] or
            record['confidence'] < config['min_confidence']):
            continue
        if any(geometry.intersection(old['geometry']).area / max(geometry.area, 1e-9)
               > config['max_added_cover'] for old in accepted):
            continue
        indices = np.flatnonzero(residual == item['old_id'])
        if not len(indices) or np.any(labels[indices] != 0):
            raise AssertionError('Protected anchor or duplicate proposal was changed')
        next_id += 1
        labels[indices] = next_id
        confidence[indices] = proposal_confidence[indices]
        accepted.append({**record, 'tree_id': next_id,
                         'source_proposal_id': item['old_id']})
    if not np.array_equal(labels[anchor_labels > 0], anchor_labels[anchor_labels > 0]):
        raise AssertionError('Protected anchor IDs changed')
    return labels, confidence, [*anchor_records, *accepted], accepted
