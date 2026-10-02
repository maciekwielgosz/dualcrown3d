"""Object-query calibration for small-tree-sensitive decoder fine-tuning."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def calibrated_query_loss(mask_output, truth, *, positive_weight=3.,
                          negative_weight=1., small_tree_weight=2.):
    """Punish duplicate queries on annotated support while ignoring unknown ALS.

    Hungarian matching gives at most one positive query per reference tree.
    Unmatched queries seeded on known points, or mostly covering known points,
    are explicit negatives. A query that only sees unannotated points is ignored.
    Tiny trees receive a higher positive weight instead of being removed by a
    global object-score threshold.
    """
    logits = mask_output['mask_logits'].float()
    objects = mask_output['object_logits'].float()
    if not len(objects):
        return objects.sum() * 0.
    valid = truth >= 0
    if not valid.any():
        return objects.sum() * 0.
    gt_ids, counts = torch.unique(truth[truth > 0], return_counts=True)
    keep = counts >= 4
    gt_ids, counts = gt_ids[keep], counts[keep]
    known_probability = logits[:, valid].sigmoid()
    targets = torch.zeros_like(objects)
    weights = torch.full_like(objects, float(negative_weight))
    supervised = truth[mask_output['seed_index']] >= 0
    if len(gt_ids):
        reference = (gt_ids[:, None] == truth[None, valid]).float()
        with torch.no_grad():
            dice_cost = 1. - (2. * (known_probability @ reference.T) + 1.) / (
                known_probability.sum(1)[:, None] + reference.sum(1)[None] + 1.)
            q, g = linear_sum_assignment(dice_cost.detach().cpu().numpy())
            q = torch.as_tensor(q, device=truth.device)
            g = torch.as_tensor(g, device=truth.device)
            intersection = (known_probability[q] * reference[g]).sum(1)
            union = known_probability[q].sum(1) + reference[g].sum(1) - intersection
            quality = (intersection / union.clamp_min(1e-6)).clamp(.05, .95)
            median_size = counts.float().median().clamp_min(1.)
            small_weight = (median_size / counts[g].float().clamp_min(1.)).sqrt().clamp(1., small_tree_weight)
        targets[q] = quality.detach()
        weights[q] = positive_weight * small_weight
        supervised[q] = True
    with torch.no_grad():
        all_probability = logits.sigmoid()
        known_fraction = known_probability.sum(1) / all_probability.sum(1).clamp_min(1e-6)
        supervised |= known_fraction >= .5
    if not supervised.any():
        return objects.sum() * 0.
    bce = F.binary_cross_entropy_with_logits(objects[supervised], targets[supervised],
                                             reduction='none')
    return (bce * weights[supervised]).sum() / weights[supervised].sum().clamp_min(1.)
