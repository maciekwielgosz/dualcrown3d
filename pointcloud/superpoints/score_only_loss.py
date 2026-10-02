"""Score-head-only supervision with strong small-tree protection."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def score_only_loss(output, truth, small_limit=300):
    masks = output['mask_logits'].float().detach()
    object_logits = output['object_logits'].float()
    valid = truth >= 0
    if not valid.any() or not len(object_logits):
        return object_logits.sum() * 0.
    ids, counts = torch.unique(truth[truth > 0], return_counts=True)
    keep = counts >= 4
    ids, counts = ids[keep], counts[keep]
    target = torch.zeros_like(object_logits)
    weights = torch.ones_like(object_logits)
    supervised = truth[output['seed_index']] >= 0
    p = masks[:, valid].sigmoid()
    if len(ids):
        gt = (ids[:, None] == truth[None, valid]).float()
        cost = 1. - (2. * (p @ gt.T) + 1.) / (
            p.sum(1)[:, None] + gt.sum(1)[None] + 1.)
        q, g = linear_sum_assignment(cost.cpu().numpy())
        q = torch.as_tensor(q, device=truth.device)
        g = torch.as_tensor(g, device=truth.device)
        intersection = (p[q] * gt[g]).sum(1)
        union = p[q].sum(1) + gt[g].sum(1) - intersection
        quality = intersection / union.clamp_min(1e-6)
        is_small = counts[g] <= small_limit
        # A matched tiny tree is not a negative just because its mask IoU is
        # initially lower than a large canopy. Mask geometry stays frozen.
        positive = quality >= .25
        small_positive = is_small & (quality >= .10)
        use = positive | small_positive
        q, quality, is_small = q[use], quality[use], is_small[use]
        target[q] = torch.where(is_small,
                                quality.clamp_min(.65), quality.clamp_min(.35)).detach()
        weights[q] = torch.where(is_small, 8., 3.)
        supervised[q] = True
    with torch.no_grad():
        all_p = masks.sigmoid()
        known_fraction = p.sum(1) / all_p.sum(1).clamp_min(1e-6)
        supervised |= known_fraction >= .5
    loss = F.binary_cross_entropy_with_logits(object_logits[supervised],
                                              target[supervised], reduction='none')
    return (loss * weights[supervised]).sum() / weights[supervised].sum().clamp_min(1.)
