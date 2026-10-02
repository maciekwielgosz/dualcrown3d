"""Second pass (v4): learned tree-centre detection in Model20's 3-D vote space.

Diagnosis behind this module: with the true tree centres, nearest-centre
assignment of the frozen Model20 votes in (x, y, 0.33 * centroid height) finds
523 of 673 validation trees, against 299 for the stage-1 heuristic 2-D density
peaks. The votes are good; centre detection is the bottleneck. Stage 2 therefore
rasterises the votes of a window into a 3-D grid, adds stage-1 context channels,
and a small 3-D U-Net predicts a centre heatmap. Every tree is one peak, so small
and large trees weigh the same in the loss by construction.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint
from torch import nn
from torch.nn import functional as F

GRID = dict(resolution=.25, z_resolution=1., size=20., height=40., margin=4.)
CHANNEL_NAMES = ('log_votes', 'tree_probability', 'voter_height', 'offset_length', 'offset_height',
                 'stage1_assigned', 'stage1_centre', 'stage1_diversity', 'geometry')
CHANNELS = len(CHANNEL_NAMES)
# assignment_source codes written by the second pass (stage-1 codes 0..7 are unchanged)
SOURCE_STAGE1_CENTRE, SOURCE_RELOCATED, SOURCE_ADDED = 3, 8, 9
ASSIGN = dict(probability=.1, radius=3., height_weight=.33, min_voxels=12, min_height_m=2., min_area_m2=.75)


def grid_shape(cfg=GRID):
    side = int(round(cfg['size'] / cfg['resolution']))
    return side, side, int(round(cfg['height'] / cfg['z_resolution']))


def rasterize_votes(xyz, votes, probability, labels, origin, cfg=GRID, known=None):
    """Vote-space volume [C, X, Y, Z] for the window whose lower-left corner is ``origin``.

    ``known`` (optional bool per point) additionally returns the cells dominated by
    votes of unannotated points, which training must ignore.
    """
    X, Y, Z = grid_shape(cfg)
    res, zres = cfg['resolution'], cfg['z_resolution']
    xyz = np.asarray(xyz, np.float32)
    votes = np.asarray(votes, np.float32)
    canopy = xyz[:, 2] >= .5
    ix = np.floor((votes[:, 0] - origin[0]) / res).astype(np.int64)
    iy = np.floor((votes[:, 1] - origin[1]) / res).astype(np.int64)
    iz = np.clip(np.floor(votes[:, 2] / zres).astype(np.int64), 0, Z - 1)
    inside = canopy & (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y)
    flat = ((ix * Y + iy) * Z + iz)[inside]
    cells = X * Y * Z
    count = np.bincount(flat, minlength=cells).astype(np.float32)
    safe = np.maximum(count, 1.)

    def mean(values):
        return np.bincount(flat, weights=values[inside], minlength=cells).astype(np.float32) / safe

    labels = np.asarray(labels, np.int64)
    volume = np.zeros((CHANNELS, cells), np.float32)
    volume[0] = np.log1p(count) / 3.
    volume[1] = mean(np.asarray(probability, np.float32))
    volume[2] = mean(xyz[:, 2]) / 30.
    volume[3] = mean(np.linalg.norm(votes[:, :2] - xyz[:, :2], axis=1)) / 5.
    volume[4] = mean(votes[:, 2] - xyz[:, 2]) / 10.
    volume[5] = mean((labels > 0).astype(np.float32))
    assigned = inside & (labels > 0)
    if assigned.any():
        index = np.flatnonzero(assigned)
        index = index[np.argsort(labels[index], kind='stable')]
        for members in np.split(index, np.flatnonzero(np.diff(labels[index])) + 1):
            centre = np.median(votes[members], axis=0)
            cx = int(np.floor((centre[0] - origin[0]) / res))
            cy = int(np.floor((centre[1] - origin[1]) / res))
            if 0 <= cx < X and 0 <= cy < Y:
                volume[6, (cx * Y + cy) * Z + int(np.clip(np.floor(centre[2] / zres), 0, Z - 1))] = 1.
        pairs = np.unique(np.stack((((ix * Y + iy) * Z + iz)[assigned], labels[assigned])), axis=1)
        volume[7] = np.log1p(np.bincount(pairs[0], minlength=cells)) / 2.
    gx = np.floor((xyz[:, 0] - origin[0]) / res).astype(np.int64)
    gy = np.floor((xyz[:, 1] - origin[1]) / res).astype(np.int64)
    gz = np.clip(np.floor(xyz[:, 2] / zres).astype(np.int64), 0, Z - 1)
    present = canopy & (gx >= 0) & (gx < X) & (gy >= 0) & (gy < Y)
    volume[8] = np.log1p(np.bincount(((gx * Y + gy) * Z + gz)[present], minlength=cells)) / 3.
    volume = volume.reshape(CHANNELS, X, Y, Z)
    if known is None:
        return volume
    unknown = np.bincount(flat, weights=(~np.asarray(known, bool))[inside], minlength=cells).reshape(X, Y, Z)
    ignore = maximum_filter((unknown > count.reshape(X, Y, Z) - unknown).astype(np.uint8), size=3) > 0
    return volume, ignore


def centre_heatmap(centres, origin, cfg=GRID, sigma_xy=1.5, sigma_z=1.):
    """Gaussian targets (cells) with an exact 1 at each centre cell; returns heat and centre cells."""
    X, Y, Z = grid_shape(cfg)
    heat = np.zeros((X, Y, Z), np.float32)
    cells = []
    rx, rz = int(np.ceil(3 * sigma_xy)), int(np.ceil(3 * sigma_z))
    ax = np.exp(-np.arange(-rx, rx + 1) ** 2 / (2 * sigma_xy ** 2)).astype(np.float32)
    az = np.exp(-np.arange(-rz, rz + 1) ** 2 / (2 * sigma_z ** 2)).astype(np.float32)
    blob = ax[:, None, None] * ax[None, :, None] * az[None, None, :]
    for centre in np.asarray(centres, np.float32).reshape(-1, 3):
        cx = int(np.floor((centre[0] - origin[0]) / cfg['resolution']))
        cy = int(np.floor((centre[1] - origin[1]) / cfg['resolution']))
        cz = int(np.clip(np.floor(centre[2] / cfg['z_resolution']), 0, Z - 1))
        if not (0 <= cx < X and 0 <= cy < Y):
            continue
        x0, x1, y0, y1, z0, z1 = max(cx - rx, 0), min(cx + rx + 1, X), max(cy - rx, 0), min(cy + rx + 1, Y), max(cz - rz, 0), min(cz + rz + 1, Z)
        patch = blob[x0 - cx + rx:x1 - cx + rx, y0 - cy + rx:y1 - cy + rx, z0 - cz + rz:z1 - cz + rz]
        np.maximum(heat[x0:x1, y0:y1, z0:z1], patch, out=heat[x0:x1, y0:y1, z0:z1])
        cells.append((cx, cy, cz))
    return heat, np.asarray(cells, np.int64).reshape(-1, 3)


def tree_centroids(coord, tree_id, minimum_voxels=8):
    """Plot-level centroid of every annotated tree: ids, centres [K, 3], voxel counts."""
    positive = tree_id > 0
    ids, inverse, counts = np.unique(tree_id[positive], return_inverse=True, return_counts=True)
    centres = np.zeros((len(ids), 3), np.float64)
    np.add.at(centres, inverse, coord[positive])
    centres /= np.maximum(counts[:, None], 1)
    keep = counts >= minimum_voxels
    return ids[keep], centres[keep].astype(np.float32), counts[keep]


def block(inputs, outputs):
    return nn.Sequential(nn.Conv3d(inputs, outputs, 3, padding=1, bias=False), nn.GroupNorm(8, outputs), nn.ReLU(inplace=True),
                         nn.Conv3d(outputs, outputs, 3, padding=1, bias=False), nn.GroupNorm(8, outputs), nn.ReLU(inplace=True))


class VoteCenterNet(nn.Module):
    """Small 3-D U-Net over the vote volume; one logit per cell."""
    def __init__(self, channels=CHANNELS, width=24, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.stem = block(channels, width)
        self.down1 = nn.Sequential(nn.Conv3d(width, 2 * width, 2, stride=2), block(2 * width, 2 * width))
        self.down2 = nn.Sequential(nn.Conv3d(2 * width, 4 * width, 2, stride=2), block(4 * width, 4 * width))
        self.up1 = nn.ConvTranspose3d(4 * width, 2 * width, 2, stride=2)
        self.merge1 = block(4 * width, 2 * width)
        self.up2 = nn.ConvTranspose3d(2 * width, width, 2, stride=2)
        self.merge2 = block(2 * width, width)
        self.head = nn.Sequential(nn.Conv3d(width, width, 3, padding=1), nn.ReLU(inplace=True), nn.Conv3d(width, 1, 1))
        nn.init.constant_(self.head[-1].bias, -2.19)

    def forward(self, volume):
        a = self.stem(volume)
        b = self.down1(a)
        c = self.dropout(self.down2(b))
        b = self.merge1(torch.cat((self.up1(c), b), 1))
        a = self.merge2(torch.cat((self.up2(b), a), 1))
        return self.head(a).squeeze(1)


def centre_focal_loss(logits, heat, ignore=None):
    """CenterNet penalty-reduced focal loss; ``ignore`` cells carry no gradient."""
    probability = logits.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    positive = heat >= 1.
    weight = torch.ones_like(heat) if ignore is None else (~ignore | positive).float()
    positive_loss = -(torch.log(probability) * (1 - probability).square())[positive].sum()
    negative = -(torch.log(1 - probability) * probability.square() * (1 - heat).pow(4)) * weight
    return (positive_loss + negative[~positive].sum()) / positive.sum().clamp_min(1)


@torch.no_grad()
def extract_peaks(logits, threshold=.3, kernel=(5, 5, 3)):
    """Local maxima of one volume [X, Y, Z]: sub-cell positions [K, 3] (cells) and scores."""
    score = logits.float().sigmoid()
    padding = tuple(k // 2 for k in kernel)
    pooled = F.max_pool3d(score[None, None], kernel, stride=1, padding=padding)[0, 0]
    peaks = torch.nonzero((score == pooled) & (score >= threshold), as_tuple=False)
    if not len(peaks):
        return np.zeros((0, 3), np.float32), np.zeros(0, np.float32)
    padded = F.pad(score, (1, 1, 1, 1, 1, 1))
    offsets = torch.stack(torch.meshgrid(*[torch.arange(-1, 2, device=score.device)] * 3, indexing='ij'), -1).reshape(-1, 3)
    neighbours = peaks[:, None] + offsets[None]
    values = padded[neighbours[..., 0] + 1, neighbours[..., 1] + 1, neighbours[..., 2] + 1]
    refined = (neighbours.float() * values[..., None]).sum(1) / values.sum(1, keepdim=True).clamp_min(1e-6)
    return (refined + .5).cpu().numpy().astype(np.float32), score[peaks[:, 0], peaks[:, 1], peaks[:, 2]].cpu().numpy()


@torch.no_grad()
def detect_centres(network, arrays, votes, probability, labels, cfg=GRID, threshold=.1, overlap=8.,
                   batch_size=8, device='cuda:0', kernel=(5, 5, 3), flips=False):
    """Sweep a plot; returns centres [K, 3] in plot coordinates and their scores (>= threshold)."""
    from scripts.evaluate_pointcloud_litept import ownership_intervals, starts_for_axis
    xyz = arrays['coord']
    size, margin, res, zres = cfg['size'], cfg['margin'], cfg['resolution'], cfg['z_resolution']
    low, high = xyz[:, :2].min(0).astype(np.float64), xyz[:, :2].max(0).astype(np.float64)
    xs = starts_for_axis(low[0], high[0], size, overlap)
    ys = starts_for_axis(low[1], high[1], size, overlap)
    xo = ownership_intervals(xs, size, low[0], high[0] + 1e-3)
    yo = ownership_intervals(ys, size, low[1], high[1] + 1e-3)
    index = cKDTree(xyz[:, :2])
    network.eval()
    centres, scores, pending = [], [], []

    def flush():
        if not pending:
            return
        volumes = torch.from_numpy(np.stack([p[0] for p in pending])).to(device)
        logits = network(volumes)
        if flips:       # average the heatmap over the four axis flips of the window
            probability_sum = logits.float().sigmoid()
            for axes in ((2,), (3,), (2, 3)):
                flipped = network(volumes.flip(axes)).float().sigmoid()
                probability_sum = probability_sum + flipped.flip(tuple(a - 1 for a in axes))
            logits = torch.logit((probability_sum / 4.).clamp(1e-5, 1 - 1e-5))
        for volume_logits, (_, origin, bounds) in zip(logits, pending):
            cells, score = extract_peaks(volume_logits, threshold, kernel)
            if not len(cells):
                continue
            world = np.column_stack((origin[0] + cells[:, 0] * res, origin[1] + cells[:, 1] * res, cells[:, 2] * zres))
            a, b, c, d = bounds
            own = (world[:, 0] >= a) & (world[:, 0] < b) & (world[:, 1] >= c) & (world[:, 1] < d)
            centres.append(world[own])
            scores.append(score[own])
        pending.clear()

    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            members = np.asarray(index.query_ball_point([x + size / 2, y + size / 2], size / 2 + margin, p=np.inf), dtype=np.int64)
            if not len(members):
                continue
            origin = np.asarray([x, y], np.float64)
            volume = rasterize_votes(xyz[members], votes[members], probability[members], labels[members], origin, cfg)
            pending.append((volume, origin, (*xo[xi], *yo[yi])))
            if len(pending) == batch_size:
                flush()
    flush()
    if not centres:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.float32)
    return np.concatenate(centres).astype(np.float32), np.concatenate(scores).astype(np.float32)


def stage1_centres(votes, labels):
    """Median 3-D vote of every stage-1 instance (plot level)."""
    labels = np.asarray(labels, np.int64)
    index = np.flatnonzero(labels)
    if not len(index):
        return np.zeros((0, 3), np.float32)
    index = index[np.argsort(labels[index], kind='stable')]
    return np.asarray([np.median(votes[members], axis=0)
                       for members in np.split(index, np.flatnonzero(np.diff(labels[index])) + 1)], np.float32)


def merge_with_stage1(centres, anchors, xy=1.5, z=4.):
    """Corrective second pass: keep each stage-1 centre unless a detected centre replaces it.

    A stage-1 centre is replaced when a detected centre lies within ``xy`` metres
    horizontally and ``z`` metres vertically. Where the detector is silent, the
    stage-1 result therefore stands.
    """
    centres = np.asarray(centres, np.float32).reshape(-1, 3)
    anchors = np.asarray(anchors, np.float32).reshape(-1, 3)
    if not len(anchors) or not len(centres):
        return np.concatenate((centres, anchors)), np.zeros(len(anchors), bool) if len(centres) else np.ones(len(anchors), bool)
    tree = cKDTree(centres[:, :2])
    kept = np.ones(len(anchors), bool)
    for i, anchor in enumerate(anchors):
        near = tree.query_ball_point(anchor[:2], xy)
        if near and (np.abs(centres[near, 2] - anchor[2]) <= z).any():
            kept[i] = False
    return np.concatenate((centres, anchors[kept])), kept


def select_centres(centres, scores, anchors, replace_threshold, add_threshold, xy=1.5, z=4.):
    """Asymmetric corrective selection.

    A detected centre close to a stage-1 centre (within ``xy``/``z``) relocates or
    splits an existing tree and needs ``replace_threshold``. A centre with no
    stage-1 centre nearby creates a brand-new tree and needs ``add_threshold``.
    Stage-1 centres not replaced by a kept detection are retained.
    """
    centres = np.asarray(centres, np.float32).reshape(-1, 3)
    scores = np.asarray(scores, np.float32)
    anchors = np.asarray(anchors, np.float32).reshape(-1, 3)
    near = np.zeros(len(centres), bool)
    if len(anchors) and len(centres):
        tree = cKDTree(anchors[:, :2])
        for i, centre in enumerate(centres):
            candidates = tree.query_ball_point(centre[:2], xy)
            near[i] = bool(candidates) and bool((np.abs(anchors[candidates, 2] - centre[2]) <= z).any())
    keep = np.where(near, scores >= replace_threshold, scores >= add_threshold)
    merged, retained = merge_with_stage1(centres[keep], anchors, xy, z)
    kinds = np.concatenate((np.where(near[keep], SOURCE_RELOCATED, SOURCE_ADDED),
                            np.full(int(retained.sum()), SOURCE_STAGE1_CENTRE))).astype(np.uint8)
    return merged, dict(replacements=int((keep & near).sum()), additions=int((keep & ~near).sum()),
                        stage1_kept=int(retained.sum()), stage1_total=int(len(anchors)), kinds=kinds)


def assign_votes(arrays, votes, probability, centres, config=None, return_centre_index=False):
    """Nearest-centre assignment in (x, y, w * height) vote space with stage-1 size filters.

    With ``return_centre_index`` also returns, per final label (index 0 unused),
    the row of ``centres`` the instance was built from.
    """
    cfg = {**ASSIGN, **(config or {})}
    xyz = arrays['coord']
    labels = np.zeros(len(xyz), np.uint32)
    origin = [-1]
    if not len(centres):
        return (labels, np.asarray(origin)) if return_centre_index else labels
    scale = np.asarray([1., 1., cfg['height_weight']], np.float32)
    candidate = np.flatnonzero((probability >= cfg['probability']) & (xyz[:, 2] >= .5) & np.isfinite(votes).all(1))
    centres = np.asarray(centres, np.float32)
    limit = np.full(len(centres), cfg['radius'], np.float32)
    if cfg.get('radius_per_metre'):
        # Small trees have compact vote clouds; tall crowns need a wider reach.
        limit = np.clip(cfg['radius_base'] + cfg['radius_per_metre'] * centres[:, 2], cfg['radius_min'], cfg['radius'])
    distance, nearest = cKDTree(centres * scale).query(votes[candidate] * scale, distance_upper_bound=float(limit.max()))
    finite = np.isfinite(distance)
    finite[finite] = distance[finite] <= limit[nearest[finite]]
    temporary = np.zeros(len(xyz), np.int64)
    temporary[candidate[finite]] = nearest[finite] + 1
    index = np.flatnonzero(temporary)
    index = index[np.argsort(temporary[index], kind='stable')]
    identifier = 0
    voxel = float(arrays['voxel_size'])
    for members in (np.split(index, np.flatnonzero(np.diff(temporary[index])) + 1) if len(index) else []):
        if len(members) < cfg['min_voxels'] or xyz[members, 2].max() < cfg['min_height_m']:
            continue
        if MultiPoint(xyz[members, :2]).convex_hull.buffer(voxel / 2.).area < cfg['min_area_m2']:
            continue
        identifier += 1
        labels[members] = identifier
        origin.append(int(temporary[members[0]]) - 1)
    return (labels, np.asarray(origin)) if return_centre_index else labels


def second_pass_labels(arrays, raw, stage1_config, centres, assign_config=None, consensus=True, kinds=None):
    """Final labels from detected centres; optionally through Model20's unchanged consensus.

    ``kinds`` (per centre, from :func:`select_centres`) turns the per-point source
    into provenance: stage-1 centre kept, relocated/split by stage 2, or added.
    """
    from pointcloud.dual_fusion import fuse_heads, instance_records
    from pointcloud.instance_output import merge_masks
    anchors, origin = assign_votes(arrays, raw['shifted_center'], raw['tree_probability'], centres, assign_config,
                                   return_centre_index=True)
    if not consensus:
        confidence = np.where(anchors > 0, raw['tree_probability'], 0.).astype(np.float32)
        source = np.where(anchors > 0, SOURCE_STAGE1_CENTRE, 0).astype(np.uint8)
        if kinds is not None and len(origin) > 1:
            per_label = np.zeros(len(origin), np.uint8)
            per_label[1:] = np.asarray(kinds, np.uint8)[origin[1:]]
            source = per_label[anchors]
        return anchors, confidence, instance_records(arrays, anchors, confidence), source
    mask_labels, mask_confidence, _ = merge_masks(arrays, raw, {**stage1_config, 'merge_strategy': 'support_fusion_v2'})
    return fuse_heads(arrays, raw, mask_labels, mask_confidence, stage1_config, auxiliary_labels=anchors)
