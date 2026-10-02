"""Shared point and full-crown instance decoder for 0.25 m ALS voxels.

Both outputs are indexed by the same learned object queries. The old offset
branch initializes queries but has no veto over the final instances.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from shapely import intersects_xy

from pointcloud.decoder_v4 import HybridTreeMaskDecoder


CROWN_CELL_M = 0.5


def crown_grid(coord):
    """XY cell centers and point-to-cell indices, shared by training and inference."""
    xy = coord[:, :2]
    origin = torch.floor(xy.amin(0) / CROWN_CELL_M) * CROWN_CELL_M
    index = torch.floor((xy - origin) / CROWN_CELL_M).long()
    width = int(index[:, 0].amax()) + 1
    height = int(index[:, 1].amax()) + 1
    yy, xx = torch.meshgrid(torch.arange(height, device=coord.device),
                            torch.arange(width, device=coord.device), indexing='ij')
    centers = origin + CROWN_CELL_M * (torch.stack((xx, yy), -1).reshape(-1, 2) + 0.5)
    return origin, centers, index[:, 1] * width + index[:, 0], height, width


class CrownRasterHead(nn.Module):
    def __init__(self, hidden_dim=128, channels=64):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(hidden_dim, channels), nn.LayerNorm(channels), nn.GELU())
        self.smooth = nn.Sequential(nn.Conv2d(channels + 2, channels, 3, padding=1),
                                    nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1))
        self.query = nn.Linear(hidden_dim, channels)
        self.radius = nn.Linear(hidden_dim, 1)
        self.quality = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.radius.weight)
        nn.init.zeros_(self.radius.bias)

    def forward(self, point_features, query_features, coord, query_centers):
        origin, centers, cell, height, width = crown_grid(coord)
        pooled = self.point(point_features)
        total = pooled.new_zeros((height * width, pooled.shape[1]))
        total.index_add_(0, cell, pooled)
        counts = torch.bincount(cell, minlength=height * width).to(pooled.dtype)
        average = total / counts[:, None].clamp_min(1)
        high = coord.new_full((height * width,), -1e4)
        high.scatter_reduce_(0, cell, coord[:, 2], reduce='amax', include_self=True)
        high = torch.where(counts > 0, high, torch.zeros_like(high))
        raster = torch.cat((average, torch.log1p(counts)[:, None],
                            (high / 30.)[:, None]), 1)
        raster = raster.T.reshape(1, -1, height, width)
        features = self.smooth(raster)[0].flatten(1).T
        if not len(query_features):
            return dict(crown_logits=coord.new_empty((0, height, width)),
                        quality_logits=coord.new_empty(0), crown_xy=centers,
                        crown_grid_origin=origin)
        q = F.normalize(self.query(query_features), dim=1)
        m = F.normalize(features, dim=1)
        logits = 8. * (q @ m.T)
        radius = 1. + 5. * self.radius(query_features).sigmoid()
        distance2 = (centers[None] - query_centers[:, None]).square().sum(2)
        logits = logits + (2.5 - .5 * distance2 / radius.square()).clamp(min=-30.)
        return dict(crown_logits=logits.reshape(-1, height, width),
                    quality_logits=self.quality(query_features).squeeze(1),
                    crown_xy=centers, crown_grid_origin=origin)


class SharedCrownDecoder(HybridTreeMaskDecoder):
    """Every query predicts one point mask, one crown and calibrated object score."""

    def __init__(self, hidden_dim=128, queries=96, layers=3, memory_tokens=1024):
        super().__init__(hidden_dim=hidden_dim, queries=queries, layers=layers,
                         memory_tokens=memory_tokens)
        self.all_canopy_candidates = True
        self.teacher_probability = 0.
        self.crown = CrownRasterHead(hidden_dim)

    def forward(self, features, data, logits, offset):
        masks = super().forward(features, data, logits, offset)
        crown = self.crown(masks['point_features'], masks['query_features'],
                           data['coord'], masks['query_center_xy'])
        masks.update(crown)
        return masks


def polygon_targets(coord, semantic_target, tree_id, local_to_world, polygons):
    """Rasterize complete annotated crowns; unknown cells stay outside the loss."""
    with torch.no_grad():
        _, centers, cell, height, width = crown_grid(coord)
        ids = np.unique(tree_id[tree_id > 0].cpu().numpy())
        xy = centers.cpu().numpy().astype(np.float64)
        transform = local_to_world.cpu().numpy().astype(np.float64)
        world = xy @ transform[:2, :2].T + transform[:2, 2]
        targets, kept = [], []
        for identifier in ids:
            geometry = polygons.get(int(identifier))
            if geometry is None or geometry.is_empty:
                continue
            x0, y0, x1, y1 = geometry.bounds
            inside = ((world[:, 0] >= x0) & (world[:, 0] <= x1) &
                      (world[:, 1] >= y0) & (world[:, 1] <= y1))
            mask = np.zeros(len(world), np.bool_)
            if inside.any():
                mask[inside] = intersects_xy(geometry, world[inside, 0], world[inside, 1])
            if mask.any():
                targets.append(mask)
                kept.append(identifier)
        background = torch.zeros(height * width, dtype=torch.bool)
        background[cell[semantic_target == 0].cpu()] = True
        crowns = np.stack(targets).reshape(-1, height, width) if targets else np.zeros((0, height, width), np.bool_)
        positive = np.any(crowns, axis=0).reshape(-1) if len(crowns) else np.zeros(height * width, np.bool_)
        valid = np.logical_or(positive, background.numpy()).reshape(height, width)
        return (torch.tensor(kept, dtype=torch.long), torch.from_numpy(crowns),
                torch.from_numpy(valid))


def shared_instance_losses(prediction, batch):
    """One-to-one matching on 3-D masks and complete 2-D crown masks."""
    output = prediction['instance_masks']
    point_logits = output['mask_logits'].float()
    crown_logits = output['crown_logits'].float().flatten(1)
    ids = batch['tree_id']
    crowns = batch['crown_target'].bool().flatten(1)
    valid_cells = batch['crown_valid'].bool().flatten()
    target_ids = batch['crown_ids']
    known = (ids == 0) | torch.isin(ids, target_ids)
    zero = point_logits.sum() * 0. + crown_logits.sum() * 0.
    gt_points = (target_ids[:, None] == ids[None, known]).float()
    gt_crowns = crowns[:, valid_cells].float()
    pred_points = point_logits[:, known].sigmoid()
    pred_crowns = crown_logits[:, valid_cells].sigmoid()
    if len(target_ids) and len(point_logits):
        with torch.no_grad():
            point_cost = 1. - (2. * (pred_points @ gt_points.T) + 1.) / (
                pred_points.sum(1)[:, None] + gt_points.sum(1)[None] + 1.)
            crown_cost = 1. - (2. * (pred_crowns @ gt_crowns.T) + 1.) / (
                pred_crowns.sum(1)[:, None] + gt_crowns.sum(1)[None] + 1.)
            matched_q, matched_gt = linear_sum_assignment((point_cost + crown_cost).cpu().numpy())
            matched_q = torch.as_tensor(matched_q, device=ids.device)
            matched_gt = torch.as_tensor(matched_gt, device=ids.device)
    else:
        matched_q = torch.empty(0, dtype=torch.long, device=ids.device)
        matched_gt = matched_q
    def mask_loss(logits, target):
        if not len(logits) or not logits.numel():
            return zero
        probability = logits.sigmoid()
        dice = 1. - (2. * (probability * target).sum(1) + 1.) / (
            probability.sum(1) + target.sum(1) + 1.)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none').mean(1)
        return (dice + bce).mean()
    point_loss = mask_loss(point_logits[matched_q][:, known], gt_points[matched_gt])
    crown_loss = mask_loss(crown_logits[matched_q][:, valid_cells], gt_crowns[matched_gt])
    supervised = (ids[output['seed_index']] == 0) | torch.isin(ids[output['seed_index']], target_ids)
    supervised[matched_q] = True
    exists = torch.zeros(len(point_logits), device=ids.device)
    exists[matched_q] = 1.
    if supervised.any():
        weights = torch.where(exists > 0, 1., .25)
        object_loss = F.binary_cross_entropy_with_logits(
            output['object_logits'][supervised], exists[supervised],
            weight=weights[supervised])
    else:
        object_loss = zero
    with torch.no_grad():
        quality = torch.zeros_like(exists)
        if len(matched_q):
            p = (pred_points[matched_q] >= .5).float()
            t = gt_points[matched_gt]
            point_iou = (p*t).sum(1) / (p.sum(1)+t.sum(1)-(p*t).sum(1)).clamp_min(1e-6)
            p = (pred_crowns[matched_q] >= .5).float()
            t = gt_crowns[matched_gt]
            crown_iou = (p*t).sum(1) / (p.sum(1)+t.sum(1)-(p*t).sum(1)).clamp_min(1e-6)
            quality[matched_q] = .5 * (point_iou + crown_iou)
    quality_loss = F.mse_loss(output['quality_logits'][supervised].sigmoid(),
                              quality[supervised]) if supervised.any() else zero
    semantic = batch['semantic_target'] >= 0
    semantic_loss = F.cross_entropy(prediction['point_semantic_logits'][semantic],
                                    batch['semantic_target'][semantic].long()) if semantic.any() else zero
    legacy_loss = F.cross_entropy(prediction['semantic_logits'][semantic],
                                  batch['semantic_target'][semantic].long()) if semantic.any() else zero
    total = (2. * point_loss + crown_loss + .5 * object_loss + .5 * quality_loss +
             .5 * semantic_loss + .1 * legacy_loss)
    return dict(loss=total, point_mask=point_loss, crown_mask=crown_loss,
                objectness=object_loss, quality=quality_loss, semantic=semantic_loss,
                matched=torch.as_tensor(len(matched_q), device=ids.device))
