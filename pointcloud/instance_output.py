"""Scene-wide mask deduplication and integer instance IDs for both GIS and LAS."""
import numpy as np
from shapely import intersects_xy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Polygon


def _mask_candidates(raw, config):
    candidates = []
    offsets = raw['candidate_offset']
    for i, score in enumerate(raw['object_score']):
        if config.get('owner_only', True) and 'candidate_owner' in raw and not raw['candidate_owner'][i]:
            continue
        if score < config['object_threshold']:
            continue
        a, b = offsets[i:i+2]
        scores = raw['point_score'][a:b].astype(np.float32)
        keep = scores >= config['mask_threshold']
        indices = raw['point_index'][a:b][keep]
        scores = scores[keep]
        if len(indices) < config['minimum_voxels']:
            continue
        ranking = float(score * scores.mean())
        candidates.append((ranking, indices, scores, float(score)))
    return sorted(candidates, key=lambda x: -x[0])


def _legacy_merge(arrays, candidates, config):
    count = len(arrays['coord'])
    labels = np.zeros(count, np.uint32)
    confidence = np.zeros(count, np.float32)
    instances = []
    for ranking, indices, scores, object_score in candidates:
        claimed = labels[indices] > 0
        # SATv2-style asymmetric support overlap against the accepted union.
        if claimed.mean() > config['merge_overlap']:
            continue
        ids = indices[~claimed]
        scores = scores[~claimed]
        if len(ids) < config['minimum_voxels']:
            continue
        xyz = arrays['coord'][ids]
        if xyz[:, 2].max() < 2.:
            continue
        xy = xyz[:, :2].astype(np.float64) + arrays['source_origin'][:2]
        geometry = MultiPoint(xy).convex_hull.buffer(float(arrays['voxel_size']) / 2.)
        if geometry.geom_type != 'Polygon' or geometry.is_empty or geometry.area < .75:
            continue
        geometry = Polygon(geometry.exterior)
        identifier = len(instances) + 1
        labels[ids] = identifier
        confidence[ids] = scores * object_score
        top = xyz[:, 2].argmax()
        instances.append(dict(tree_id=identifier, geometry=geometry, confidence=ranking,
                              points=len(ids), height=float(xyz[top, 2]),
                              top_x=float(xy[top, 0]), top_y=float(xy[top, 1])))
    return labels, confidence, instances


def _instance_members(labels):
    """Group labelled indices once, without scanning the entire cloud per tree."""
    indices = np.flatnonzero(labels)
    if not len(indices):
        return []
    indices = indices[np.argsort(labels[indices], kind='stable')]
    boundaries = np.flatnonzero(np.diff(labels[indices])) + 1
    return np.split(indices, boundaries)


def merge_masks(arrays, raw, config):
    """Deduplicate masks and optionally fuse complementary point support.

    Missing ``merge_strategy`` means the historical suppression algorithm, so
    archived selections remain reproducible. ``support_fusion_v2`` freezes its
    accepted masks as anchors, then matches proposals against those ORIGINAL
    point sets. Only previously unassigned, nearby points can be recovered.
    Anchor labels never change; recovered points never create transitive links.
    """
    strategy = config.get('merge_strategy', 'legacy')
    if strategy not in ('legacy', 'support_fusion_v2'):
        raise ValueError(f'Unknown merge strategy: {strategy}')
    candidates = _mask_candidates(raw, config)
    labels, confidence, instances = _legacy_merge(arrays, candidates, config)
    if strategy == 'legacy' or not instances:
        return labels, confidence, instances

    min_iou = float(config.get('fusion_min_iou', .2))
    dominance = float(config.get('fusion_dominance', .8))
    distance = float(config.get('fusion_max_distance_m', 1.5))
    probability = float(config.get('fusion_min_probability', config['mask_threshold']))
    vote_distance = config.get('fusion_max_vote_distance_m')
    if vote_distance is not None:
        vote_distance = float(vote_distance)
        if vote_distance <= 0 or 'shifted_center' not in raw:
            raise ValueError('Vote-guided fusion requires shifted_center and a positive vote distance')
    if not (0 < min_iou <= 1 and .5 < dominance <= 1 and distance > 0
            and config['mask_threshold'] <= probability <= 1):
        raise ValueError('Invalid support fusion thresholds')

    anchors = labels.copy()
    counts = np.bincount(anchors, minlength=len(instances) + 1)
    members = _instance_members(anchors)
    trees = {}
    hulls = {}
    vote_centers = {}
    # Confidence weighted by mask-to-anchor affinity resolves competing offers
    # for unassigned points. This score is separate from exported confidence.
    best_offer = np.zeros(len(labels), np.float32)
    for ranking, indices, scores, object_score in candidates:
        claimed = anchors[indices]
        identifiers, overlap = np.unique(claimed[claimed > 0], return_counts=True)
        if not len(identifiers):
            continue
        winner = int(overlap.argmax())
        identifier, intersection = int(identifiers[winner]), int(overlap[winner])
        if intersection / overlap.sum() < dominance:
            continue  # Ambiguous proposals must not bridge neighbouring trees.
        affinity = intersection / (len(indices) + counts[identifier] - intersection)
        if affinity < min_iou:
            continue
        eligible = (claimed == 0) & (scores >= probability)
        proposed, probabilities = indices[eligible], scores[eligible]
        offers = (probabilities * object_score * affinity).astype(np.float32)
        better = offers > best_offer[proposed]
        proposed, probabilities, offers = proposed[better], probabilities[better], offers[better]
        if not len(proposed):
            continue
        if vote_distance is not None:
            if identifier not in vote_centers:
                vote_centers[identifier] = np.median(raw['shifted_center'][members[identifier - 1], :2], axis=0)
            delta = raw['shifted_center'][proposed, :2] - vote_centers[identifier]
            consistent = np.einsum('ij,ij->i', delta, delta) <= vote_distance**2
            proposed, probabilities, offers = proposed[consistent], probabilities[consistent], offers[consistent]
            if not len(proposed):
                continue
        if config.get('fusion_inside_hull', False):
            if identifier not in hulls:
                hulls[identifier] = MultiPoint(arrays['coord'][members[identifier - 1], :2]).convex_hull
            xy = arrays['coord'][proposed, :2]
            inside = intersects_xy(hulls[identifier], xy[:, 0], xy[:, 1])
            proposed, probabilities, offers = proposed[inside], probabilities[inside], offers[inside]
            if not len(proposed):
                continue
        if identifier not in trees:
            trees[identifier] = cKDTree(arrays['coord'][members[identifier - 1]])
        distances, _ = trees[identifier].query(arrays['coord'][proposed], distance_upper_bound=distance)
        near = np.isfinite(distances)
        proposed, probabilities, offers = proposed[near], probabilities[near], offers[near]
        labels[proposed] = identifier
        confidence[proposed] = probabilities * object_score
        best_offer[proposed] = offers

    # Rebuild polygons AND treetops from the final support, keeping IDs aligned
    # with the point cloud. These filled hulls do not invent point labels.
    for item, indices in zip(instances, _instance_members(labels)):
        xyz = arrays['coord'][indices]
        xy = xyz[:, :2].astype(np.float64) + arrays['source_origin'][:2]
        geometry = MultiPoint(xy).convex_hull.buffer(float(arrays['voxel_size']) / 2.)
        top = xyz[:, 2].argmax()
        item.update(geometry=Polygon(geometry.exterior), points=len(indices),
                    height=float(xyz[top, 2]), top_x=float(xy[top, 0]), top_y=float(xy[top, 1]))
    return labels, confidence, instances


def point_instance_metrics(truth, prediction, threshold=.5):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    valid = truth >= 0
    truth, prediction = truth[valid], prediction[valid]
    gt_ids, gt_inverse, gt_count = np.unique(truth, return_inverse=True, return_counts=True)
    pr_ids, pr_inverse, pr_count = np.unique(prediction, return_inverse=True, return_counts=True)
    table = np.bincount(gt_inverse * len(pr_ids) + pr_inverse,
                        minlength=len(gt_ids)*len(pr_ids)).reshape(len(gt_ids), len(pr_ids))
    gt_keep, pr_keep = gt_ids > 0, pr_ids > 0
    intersection = table[gt_keep][:, pr_keep]
    union = gt_count[gt_keep, None] + pr_count[None, pr_keep] - intersection
    iou = intersection / np.maximum(union, 1)
    matched_gt, matched_pr = linear_sum_assignment(-((iou >= threshold) * (min(iou.shape, default=0) + 1 + iou)))
    matched = iou[matched_gt, matched_pr]
    matched = matched[matched >= threshold]
    tp, fp, fn = len(matched), int(pr_keep.sum()) - len(matched), int(gt_keep.sum()) - len(matched)
    best = iou.max(1) if iou.shape[1] else np.zeros(iou.shape[0])
    return dict(tp=tp, fp=fp, fn=fn, iou_sum=float(matched.sum()),
                mucov=float(best.mean()) if len(best) else 0.,
                mwcov=float(np.average(best, weights=gt_count[gt_keep])) if len(best) else 0.)
