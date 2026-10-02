"""Validation-tuned, GT-free consensus of mask instances and centre-vote instances.

The selected anchor branch is immutable. Recovery operates on fixed point
sets, never on labels grown in an earlier iteration. Source codes: 0 unassigned,
1 mask anchor, 2 cross-head completion, 3 vote instance, 4 local growth,
5 new mask instance. The anchor direction is selected on validation only.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Polygon


def groups(labels):
    indices = np.flatnonzero(labels)
    indices = indices[np.argsort(labels[indices], kind='stable')]
    return np.split(indices, np.flatnonzero(np.diff(labels[indices])) + 1) if len(indices) else []


def vote_instances(arrays, raw, config):
    """Bounded-memory centre voting, same 100 m core/8 m context in val and export.

    Peaks belong to one core by their XY centre. Points can cross cores, but
    competing peaks are resolved by vote distance (not by processing order).
    Ground truth, imagery, and polygon labels are not inputs.
    """
    xyz = arrays['coord']
    labels = np.zeros(len(xyz), np.uint32)
    if not len(xyz):
        return labels
    selected = np.flatnonzero((raw['tree_probability'] >= config['probability']) & (xyz[:, 2] >= .5))
    if not len(selected):
        return labels
    votes = raw['shifted_center'][selected, :2].astype(np.float64)
    valid = np.isfinite(votes).all(1)
    selected, votes = selected[valid], votes[valid]
    if not len(selected):
        return labels
    core, halo, resolution = 100., 8., .25
    # World-aligned cells do not depend on crop or tile origin.
    origin = arrays['source_origin'][:2]
    lower = np.floor((xyz[:, :2].min(0) + origin) / core) * core - origin
    upper = xyz[:, :2].max(0)
    index = cKDTree(xyz[selected, :2])
    peak_list = []
    for x in np.arange(lower[0], upper[0] + .001, core):
        for y in np.arange(lower[1], upper[1] + .001, core):
            local = np.asarray(index.query_ball_point([x+core/2, y+core/2], core/2+halo, p=np.inf), dtype=np.int64)
            if not len(local):
                continue
            start = np.array([x-halo, y-halo])
            side = int(round((core+2*halo)/resolution))
            grid = np.floor((votes[local]-start)/resolution).astype(np.int64)
            inside = ((grid >= 0) & (grid < side)).all(1)
            grid = grid[inside]
            density = np.zeros((side, side), np.float32)
            np.add.at(density, (grid[:, 0], grid[:, 1]), 1.)
            density = gaussian_filter(density, config['vote_smoothing']/resolution)
            if not density.max():
                continue
            window = 2*int(np.ceil(config['peak_separation']/resolution))+1
            maxima = maximum_filter(density, size=window, mode='constant')
            peaks = np.argwhere((density == maxima) & (density >= config['peak_threshold_fraction']*density.max()))
            centers = start + (peaks+.5)*resolution
            own = (centers[:, 0] >= x) & (centers[:, 0] < x+core) & (centers[:, 1] >= y) & (centers[:, 1] < y+core)
            peak_list.extend(centers[own])
    if not peak_list:
        return labels
    distances, nearest = cKDTree(np.asarray(peak_list)).query(votes, distance_upper_bound=config['assignment_radius'])
    finite = np.isfinite(distances)
    temporary = np.zeros(len(xyz), np.uint32)
    temporary[selected[finite]] = nearest[finite]+1
    identifier = 0
    for members in groups(temporary):
        if len(members) < config['min_voxels'] or xyz[members, 2].max() < config.get('min_height_m', 2.):
            continue
        geometry = MultiPoint(xyz[members, :2]).convex_hull.buffer(float(arrays['voxel_size'])/2.)
        if geometry.area < config.get('min_area_m2', .75):
            continue
        identifier += 1
        labels[members] = identifier
    return labels


def instance_records(arrays, labels, confidence):
    records = []
    for members in groups(labels):
        xyz = arrays['coord'][members]
        xy = xyz[:, :2].astype(np.float64) + arrays['source_origin'][:2]
        geometry = MultiPoint(xy).convex_hull.buffer(float(arrays['voxel_size'])/2.)
        top = int(xyz[:, 2].argmax())
        records.append(dict(tree_id=int(labels[members[0]]), geometry=Polygon(geometry.exterior),
                            confidence=float(confidence[members].mean()), points=len(members),
                            height=float(xyz[top, 2]), top_x=float(xy[top, 0]), top_y=float(xy[top, 1])))
    return records


def fuse_heads(arrays, raw, mask_labels, mask_confidence, config, auxiliary_labels=None):
    """Return labels, confidence, filled crowns and per-point assignment provenance."""
    for name, default in (('dual_max_distance_m', 2.), ('dual_vote_distance_m', 2.),
                          ('dual_new_separation_m', 1.), ('dual_growth_vote_distance_m', 1.)):
        if not np.isfinite(config.get(name, default)) or config.get(name, default) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    for name, default in (('dual_growth_distance_m', 0.), ('dual_growth_margin_m', .25)):
        if not np.isfinite(config.get(name, default)) or config.get(name, default) < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not (.5 < config.get('dual_dominance', .8) <= 1
            and 0 <= config.get('dual_new_max_overlap', .05) < .5
            and 0 <= config.get('dual_min_probability', .5) <= 1
            and 0 <= config.get('dual_complement_min_probability', .5) <= 1):
        raise ValueError('Invalid consensus probability/overlap thresholds')
    auxiliary = (vote_instances(arrays, raw, config['vote_cluster_config'])
                 if auxiliary_labels is None else auxiliary_labels)
    anchor = config.get('dual_anchor', 'mask')
    if anchor not in ('mask', 'vote'):
        raise ValueError('dual_anchor must be mask or vote')
    if anchor == 'vote':
        complement_probability = (raw['point_probability'] if config.get('dual_complement_own_semantic', False)
                                  else raw['tree_probability'])
        return _fuse(arrays, raw, auxiliary, np.where(auxiliary > 0, raw['tree_probability'], 0.),
                     config, mask_labels, anchor_source=3, new_source=5, mask_support=mask_labels>0,
                     complement_probability=complement_probability,
                     complement_confidence=mask_confidence if config.get('dual_complement_own_semantic', False) else None)
    return _fuse(arrays, raw, mask_labels, mask_confidence, config, auxiliary, mask_support=mask_labels>0)


def _fuse(arrays, raw, anchor_labels, anchor_confidence, config, auxiliary, anchor_source=1, new_source=3,
          mask_support=None, complement_probability=None, complement_confidence=None):
    cfg = config
    xyz, votes = arrays['coord'], raw['shifted_center'][:, :2]
    semantic = raw['tree_probability']
    labels, confidence = anchor_labels.copy(), anchor_confidence.copy()
    source = np.where(labels > 0, anchor_source, 0).astype(np.uint8)
    if len(auxiliary) != len(labels):
        raise ValueError('Auxiliary labels must match the input point cloud')
    members_by_id = {int(labels[m[0]]): m for m in groups(labels)}
    centers = {identifier: np.median(votes[m], axis=0) for identifier, m in members_by_id.items()}
    trees = {}
    center_index = cKDTree(np.asarray(list(centers.values()))) if centers else None
    next_id = int(labels.max(initial=0))
    min_probability = cfg.get('dual_min_probability', .5)
    proposal_probability = semantic if complement_probability is None else complement_probability
    proposal_confidence = semantic if complement_confidence is None else complement_confidence
    for members in groups(auxiliary):
        # ALWAYS match to the selected branch's original support, not earlier
        # recovered points. Neither direction assumes matching integer IDs.
        claimed = anchor_labels[members]
        occupied = claimed > 0
        remaining = members[(~occupied) & (proposal_probability[members] >= cfg.get('dual_complement_min_probability', min_probability))]
        if not len(remaining):
            continue
        if occupied.any():
            identifiers, counts = np.unique(claimed[occupied], return_counts=True)
            best = int(counts.argmax())
            identifier = int(identifiers[best])
            match = (counts[best] >= cfg.get('dual_min_intersection', 4)
                     and counts[best]/counts.sum() >= cfg.get('dual_dominance', .8))
            if match:
                delta = votes[remaining] - centers[identifier]
                consistent = np.linalg.norm(delta, axis=1) <= cfg.get('dual_vote_distance_m', 2.)
                remaining = remaining[consistent]
                if identifier not in trees:
                    trees[identifier] = cKDTree(xyz[members_by_id[identifier]])
                bound = np.nextafter(float(cfg.get('dual_max_distance_m', 2.)), np.inf)
                distances, _ = trees[identifier].query(xyz[remaining], distance_upper_bound=bound)
                accepted = remaining[np.isfinite(distances)]
                labels[accepted], confidence[accepted], source[accepted] = identifier, proposal_confidence[accepted], 2
                continue
        # Add unsupported instances from the complementary branch, not
        # residual fragments of a candidate bridging several existing trees.
        if (not cfg.get('dual_add_instances', True)
                or occupied.mean() > cfg.get('dual_new_max_overlap', .05)
                or len(remaining) < cfg.get('minimum_voxels', 12)):
            continue
        center = np.median(votes[remaining], axis=0)
        if (cfg.get('dual_new_use_vote_guard', True) and center_index is not None
                and center_index.query(center)[0] < cfg.get('dual_new_separation_m', 1.)):
            continue
        support = xyz[remaining]
        if (support[:, 2].max() < cfg.get('minimum_height_m', 2.) or
                MultiPoint(support[:, :2]).convex_hull.buffer(float(arrays['voxel_size'])/2.).area
                < cfg.get('minimum_area_m2', .75)):
            continue
        next_id += 1
        labels[remaining], confidence[remaining], source[remaining] = next_id, proposal_confidence[remaining], new_source

    # Optional one-pass local recovery. Seed support is frozen before this pass.
    # Two competing vote centres must have an explicit margin; background and
    # spatially disconnected points cannot acquire labels by nearest ID alone.
    radius = cfg.get('dual_growth_distance_m', 0.)
    if radius > 0:
        frozen = labels.copy()
        grouped = groups(frozen)
        if grouped:
            ids = np.asarray([frozen[m[0]] for m in grouped])
            centers = np.asarray([np.median(votes[m], axis=0) for m in grouped])
            eligible = (frozen == 0) & (semantic >= min_probability) & (xyz[:, 2] >= .5)
            if cfg.get('dual_growth_support_only', False):
                eligible &= mask_support
            pending = np.flatnonzero(eligible)
            distances, nearest = cKDTree(centers).query(votes[pending], k=2)
            safe = ((distances[:, 0] <= cfg.get('dual_growth_vote_distance_m', 1.))
                    & (distances[:, 1]-distances[:, 0] >= cfg.get('dual_growth_margin_m', .25)))
            pending, nearest = pending[safe], nearest[safe, 0]
            order = np.argsort(nearest, kind='stable')
            for chunk in np.split(order, np.flatnonzero(np.diff(nearest[order]))+1) if len(order) else []:
                target = int(nearest[chunk[0]])
                candidates = pending[chunk]
                spatial, _ = cKDTree(xyz[grouped[target]]).query(xyz[candidates], distance_upper_bound=radius)
                accepted = candidates[np.isfinite(spatial)]
                labels[accepted], confidence[accepted], source[accepted] = ids[target], semantic[accepted], 4
    return labels, confidence, instance_records(arrays, labels, confidence), source
