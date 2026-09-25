"""Scene-wide mask deduplication and integer instance IDs for both GIS and LAS."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import MultiPoint, Polygon


def merge_masks(arrays, raw, config):
    count = len(arrays['coord'])
    labels = np.zeros(count, np.uint32)
    confidence = np.zeros(count, np.float32)
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
    instances = []
    for ranking, indices, scores, object_score in sorted(candidates, key=lambda x: -x[0]):
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
