"""Second pass v7: vote-field centre detector.

What changed against ``vote_centers`` (v4-v6), and why:

- **Targets are the per-tree median votes, not the geometric centroids.** The
  second pass assigns votes to the nearest centre, so the right centre of a tree
  is where *its votes* concentrate. That point is also what the input shows, so
  the detector is no longer asked to guess a 0.25 m cell the votes do not mark.
- **Fine input, coarse output.** The vote volume stays at 0.25 m, the heatmap is
  predicted at 0.5 m with a sub-cell offset head (CenterNet). A centre is four
  times easier to hit, which matters most when a tree casts a few dozen votes.
- **Extent head.** Each centre predicts the reach of its own votes (90th
  percentile of the distance to the median vote). With true extents the
  median-vote oracle on validation rises from 0.705 / 0.611 to 0.744 / 0.674
  point / crown SB-PQ and from 90 to 132 of 161 small crowns.
- **Inputs**: rotation-ensembled votes (``vote_ensemble``), the mean direction the
  votes of a cell arrive from, a canopy-height column, the window density and the
  ensemble spread, in addition to the v4 channels.
- **Stage-1 centres are scored too**, so the second pass can reject a stage-1
  tree instead of only relocating or adding.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint
from torch import nn
from torch.nn import functional as F

GRID = dict(resolution=.25, z_resolution=1., size=20., height=40., margin=4., stride=2)
CHANNEL_NAMES = ('log_votes', 'tree_probability', 'voter_height', 'offset_length', 'offset_height',
                 'stage1_assigned', 'stage1_centre', 'stage1_diversity', 'geometry',
                 'offset_x', 'offset_y', 'canopy_height', 'density', 'vote_spread')
CHANNELS = len(CHANNEL_NAMES)
FLIP_SIGN = {2: CHANNEL_NAMES.index('offset_x'), 3: CHANNEL_NAMES.index('offset_y')}   # volume dim -> channel to negate
HEIGHT_WEIGHT = .33
SCALE = np.asarray([1., 1., HEIGHT_WEIGHT], np.float32)
SOURCE_STAGE1_CENTRE, SOURCE_RELOCATED, SOURCE_ADDED, SOURCE_FALLBACK = 3, 8, 9, 10
ASSIGN = dict(probability=.1, min_voxels=12, min_height_m=2., min_area_m2=.75, neighbours=4,
              radius_base=1.5, radius_per_metre=.1, radius_min=2., radius_max=3.,
              extent_scale=0., extent_min=1., extent_max=4., normalised=False, grow_m=0.)


def fine_shape(cfg=GRID):
    side = int(round(cfg['size'] / cfg['resolution']))
    return side, side, int(round(cfg['height'] / cfg['z_resolution']))


def out_shape(cfg=GRID):
    X, Y, Z = fine_shape(cfg)
    return X // cfg['stride'], Y // cfg['stride'], Z


def cell_size(cfg=GRID):
    """Metres per output cell along x, y and centroid height."""
    return np.asarray([cfg['resolution'] * cfg['stride'], cfg['resolution'] * cfg['stride'], cfg['z_resolution']], np.float32)


def rasterize_field(xyz, votes, probability, labels, spread, origin, cfg=GRID, known=None):
    """Vote-space volume [C, X, Y, Z] at the fine grid for the window at ``origin``.

    With ``known`` (bool per point) also returns the output-grid cells dominated
    by votes of unannotated points, which training must ignore.
    """
    X, Y, Z = fine_shape(cfg)
    res, zres = cfg['resolution'], cfg['z_resolution']
    xyz = np.asarray(xyz, np.float32)
    votes = np.asarray(votes, np.float32)
    canopy = xyz[:, 2] >= .5
    ix = np.floor((votes[:, 0] - origin[0]) / res).astype(np.int64)
    iy = np.floor((votes[:, 1] - origin[1]) / res).astype(np.int64)
    iz = np.clip(np.floor(votes[:, 2] / zres).astype(np.int64), 0, Z - 1)
    inside = canopy & (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y)
    cell = (ix * Y + iy) * Z + iz
    flat = cell[inside]
    cells = X * Y * Z
    count = np.bincount(flat, minlength=cells).astype(np.float32)
    safe = np.maximum(count, 1.)

    def mean(values):
        return np.bincount(flat, weights=values[inside], minlength=cells).astype(np.float32) / safe

    labels = np.asarray(labels, np.int64)
    delta = votes - xyz
    volume = np.zeros((CHANNELS, cells), np.float32)
    volume[0] = np.log1p(count) / 3.
    volume[1] = mean(np.asarray(probability, np.float32))
    volume[2] = mean(xyz[:, 2]) / 30.
    volume[3] = mean(np.linalg.norm(delta[:, :2], axis=1)) / 5.
    volume[4] = mean(delta[:, 2]) / 10.
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
        pairs = np.unique(np.stack((cell[assigned], labels[assigned])), axis=1)
        volume[7] = np.log1p(np.bincount(pairs[0], minlength=cells)) / 2.
    gx = np.floor((xyz[:, 0] - origin[0]) / res).astype(np.int64)
    gy = np.floor((xyz[:, 1] - origin[1]) / res).astype(np.int64)
    gz = np.clip(np.floor(xyz[:, 2] / zres).astype(np.int64), 0, Z - 1)
    present = canopy & (gx >= 0) & (gx < X) & (gy >= 0) & (gy < Y)
    volume[8] = np.log1p(np.bincount(((gx * Y + gy) * Z + gz)[present], minlength=cells)) / 3.
    volume[9] = mean(delta[:, 0]) / 5.
    volume[10] = mean(delta[:, 1]) / 5.
    volume[13] = mean(np.asarray(spread, np.float32))
    volume = volume.reshape(CHANNELS, X, Y, Z)
    surface = np.zeros(X * Y, np.float32)
    np.maximum.at(surface, (gx * Y + gy)[present], xyz[present, 2])
    volume[11] = (maximum_filter(surface.reshape(X, Y), size=5) / 40.)[:, :, None]
    volume[12] = np.log10(1. + present.sum() / float(cfg['size']) ** 2) / 2.5
    if known is None:
        return volume
    unknown = np.bincount(flat, weights=(~np.asarray(known, bool))[inside], minlength=cells).reshape(X, Y, Z)
    ignore = maximum_filter((unknown > count.reshape(X, Y, Z) - unknown).astype(np.uint8), size=3) > 0
    s = cfg['stride']
    return volume, ignore.reshape(X // s, s, Y // s, s, Z).any(axis=(1, 3))


def tree_vote_centres(coord, tree_id, votes, minimum_voxels=8):
    """Per annotated tree: id, median vote [3], extent (90th percentile of the scaled
    distance of its votes to that median, metres) and voxel count."""
    use = np.flatnonzero((tree_id > 0) & (coord[:, 2] >= .5))
    use = use[np.argsort(tree_id[use], kind='stable')]
    ids, centres, extents, counts = [], [], [], []
    for members in (np.split(use, np.flatnonzero(np.diff(tree_id[use])) + 1) if len(use) else []):
        if len(members) < minimum_voxels:
            continue
        median = np.median(votes[members], axis=0)
        distance = np.linalg.norm((votes[members] - median) * SCALE, axis=1)
        ids.append(int(tree_id[members[0]]))
        centres.append(median)
        extents.append(np.percentile(distance, 90))
        counts.append(len(members))
    return (np.asarray(ids, np.int64), np.asarray(centres, np.float32).reshape(-1, 3),
            np.asarray(extents, np.float32), np.asarray(counts, np.int64))


def target_sigmas(extent, adaptive):
    """Gaussian widths of one tree's heat target in output cells (xy, z).

    Fixed (v7): 1.2 cells (0.6 m) and 1 cell (1 m). Adaptive (v8): proportional to the
    reach of the tree's votes, as CenterNet scales its Gaussians with object size. A
    wide crown scatters its votes over metres, its vote median is located less
    precisely, and a centre one metre off still collects the right votes.
    """
    if not adaptive:
        return 1.2, 1.
    return float(np.clip(.6 * extent, 1., 2.5)), float(np.clip(.6 * extent, 1., 2.5))


def field_targets(centres, extents, origin, cfg=GRID, adaptive=False):
    """Targets on the output grid: heat [X, Y, Z], offset [3, X, Y, Z] (cells, in
    [-0.5, 0.5)), log-extent [X, Y, Z] and the positive mask."""
    X, Y, Z = out_shape(cfg)
    size = cell_size(cfg)
    heat = np.zeros((X, Y, Z), np.float32)
    offset = np.zeros((3, X, Y, Z), np.float32)
    extent = np.zeros((X, Y, Z), np.float32)
    positive = np.zeros((X, Y, Z), bool)
    blobs = {}
    centres = np.asarray(centres, np.float32).reshape(-1, 3)
    for centre, reach in zip(centres, np.asarray(extents, np.float32).reshape(-1)):
        position = np.asarray([(centre[0] - origin[0]) / size[0], (centre[1] - origin[1]) / size[1], centre[2] / size[2]])
        cx, cy = int(np.floor(position[0])), int(np.floor(position[1]))
        cz = int(np.clip(np.floor(position[2]), 0, Z - 1))
        if not (0 <= cx < X and 0 <= cy < Y):
            continue
        sigma_xy, sigma_z = target_sigmas(float(reach), adaptive)
        key = (round(sigma_xy, 1), round(sigma_z, 1))
        if key not in blobs:
            rx, rz = int(np.ceil(3 * key[0])), int(np.ceil(3 * key[1]))
            ax = np.exp(-np.arange(-rx, rx + 1) ** 2 / (2 * key[0] ** 2)).astype(np.float32)
            az = np.exp(-np.arange(-rz, rz + 1) ** 2 / (2 * key[1] ** 2)).astype(np.float32)
            blobs[key] = (rx, rz, ax[:, None, None] * ax[None, :, None] * az[None, None, :])
        rx, rz, blob = blobs[key]
        x0, x1, y0, y1, z0, z1 = max(cx - rx, 0), min(cx + rx + 1, X), max(cy - rx, 0), min(cy + rx + 1, Y), max(cz - rz, 0), min(cz + rz + 1, Z)
        patch = blob[x0 - cx + rx:x1 - cx + rx, y0 - cy + rx:y1 - cy + rx, z0 - cz + rz:z1 - cz + rz]
        np.maximum(heat[x0:x1, y0:y1, z0:z1], patch, out=heat[x0:x1, y0:y1, z0:z1])
        positive[cx, cy, cz] = True
        offset[:, cx, cy, cz] = np.clip(position - np.asarray([cx, cy, cz]) - .5, -.5, .5)
        extent[cx, cy, cz] = np.log(max(float(reach), .25))
    heat[positive] = 1.           # a neighbour's wide Gaussian must not hide a centre
    return heat, offset, extent, positive


def block(inputs, outputs):
    return nn.Sequential(nn.Conv3d(inputs, outputs, 3, padding=1, bias=False), nn.GroupNorm(8, outputs), nn.ReLU(inplace=True),
                         nn.Conv3d(outputs, outputs, 3, padding=1, bias=False), nn.GroupNorm(8, outputs), nn.ReLU(inplace=True))


def head(width, outputs):
    return nn.Sequential(nn.Conv3d(width, width, 3, padding=1), nn.ReLU(inplace=True), nn.Conv3d(width, outputs, 1))


class VoteFieldNet(nn.Module):
    """3-D U-Net: 0.25 m input, three down levels, heads at 0.5 m (heat, offset, extent)."""
    def __init__(self, channels=CHANNELS, width=24, dropout=0.):
        super().__init__()
        w = width
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.stem = block(channels, w)
        self.down1 = nn.Sequential(nn.Conv3d(w, 2 * w, (2, 2, 1), stride=(2, 2, 1)), block(2 * w, 2 * w))
        self.down2 = nn.Sequential(nn.Conv3d(2 * w, 4 * w, 2, stride=2), block(4 * w, 4 * w))
        self.down3 = nn.Sequential(nn.Conv3d(4 * w, 8 * w, 2, stride=2), block(8 * w, 8 * w))
        self.up2 = nn.ConvTranspose3d(8 * w, 4 * w, 2, stride=2)
        self.merge2 = block(8 * w, 4 * w)
        self.up1 = nn.ConvTranspose3d(4 * w, 2 * w, 2, stride=2)
        self.merge1 = block(4 * w, 2 * w)
        self.heat = head(2 * w, 1)
        self.offset = head(2 * w, 3)
        self.extent = head(2 * w, 1)
        nn.init.constant_(self.heat[-1].bias, -2.19)
        nn.init.constant_(self.extent[-1].bias, .6)       # exp(0.6) = 1.8 m, the median extent

    def forward(self, volume):
        a = self.down1(self.stem(volume))
        b = self.down2(a)
        c = self.dropout(self.down3(b))
        b = self.merge2(torch.cat((self.up2(c), b), 1))
        a = self.merge1(torch.cat((self.up1(b), a), 1))
        return dict(heat=self.heat(a).squeeze(1), offset=self.offset(a), extent=self.extent(a).squeeze(1))


def field_loss(output, heat, ignore, offset, extent, positive, offset_weight=1., extent_weight=.5):
    """Penalty-reduced focal loss on the heatmap plus L1 on offset and log-extent at the centres."""
    probability = output['heat'].float().sigmoid().clamp(1e-4, 1 - 1e-4)
    weight = (~ignore | positive).float()
    positive_loss = -(torch.log(probability) * (1 - probability).square())[positive].sum()
    negative = -(torch.log(1 - probability) * probability.square() * (1 - heat).pow(4)) * weight
    count = positive.sum().clamp_min(1)
    focal = (positive_loss + negative[~positive].sum()) / count
    mask = positive[:, None].expand_as(offset)
    offset_loss = F.l1_loss(output['offset'].float()[mask], offset[mask], reduction='sum') / count
    extent_loss = F.l1_loss(output['extent'].float()[positive], extent[positive], reduction='sum') / count
    return dict(loss=focal + offset_weight * offset_loss + extent_weight * extent_loss,
                focal=focal, offset=offset_loss, extent=extent_loss)


@torch.no_grad()
def decode_peaks(probability, offset, extent, threshold=.2, kernel=(3, 3, 3)):
    """Local maxima of one window: positions [K, 3] in output cells, scores and extents (metres)."""
    padding = tuple(k // 2 for k in kernel)
    pooled = F.max_pool3d(probability[None, None], kernel, stride=1, padding=padding)[0, 0]
    peaks = torch.nonzero((probability == pooled) & (probability >= threshold), as_tuple=False)
    if not len(peaks):
        return np.zeros((0, 3), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)
    i, j, k = peaks[:, 0], peaks[:, 1], peaks[:, 2]
    position = peaks.float() + .5 + offset[:, i, j, k].T.clamp(-.5, .5)
    return (position.cpu().numpy().astype(np.float32), probability[i, j, k].cpu().numpy().astype(np.float32),
            extent[i, j, k].exp().cpu().numpy().astype(np.float32))


def _forward(network, volumes):
    with torch.autocast('cuda', dtype=torch.float16, enabled=volumes.is_cuda):
        output = network(volumes)
    return output['heat'].float().sigmoid(), output['offset'].float(), output['extent'].float()


@torch.no_grad()
def predict_windows(network, volumes, flips=False):
    """Heat probability, offset and log-extent for a batch of volumes.

    ``network`` may be a list of networks, whose outputs are averaged. With
    ``flips`` the outputs are also averaged over the four axis flips of the window.
    """
    networks = list(network) if isinstance(network, (list, tuple)) else [network]
    views = [()] + ([(2,), (3,), (2, 3)] if flips else [])
    probability = offset = extent = 0.
    for axes in views:
        view = volumes
        if axes:
            view = volumes.flip(axes).clone()
            for axis in axes:
                view[:, FLIP_SIGN[axis]] *= -1
        for member in networks:
            heat, shift, reach = _forward(member, view)
            if axes:
                heat = heat.flip(tuple(a - 1 for a in axes))
                shift = shift.flip(axes).clone()
                for axis in axes:
                    shift[:, axis - 2] *= -1
                reach = reach.flip(tuple(a - 1 for a in axes))
            probability, offset, extent = probability + heat, offset + shift, extent + reach
    count = float(len(views) * len(networks))
    return probability / count, offset / count, extent / count


@torch.no_grad()
def detect_field(network, arrays, votes, probability, labels, spread, cfg=GRID, threshold=.1, overlap=8.,
                 batch_size=8, device='cuda:0', kernel=(3, 3, 3), flips=False, queries=None):
    """Sweep a plot. Returns centres [K, 3] (plot coordinates), scores, extents and, for
    ``queries`` [M, 3] (e.g. stage-1 centres), the heat at each query (3x3x3 maximum)."""
    from scripts.evaluate_pointcloud_litept import ownership_intervals, starts_for_axis
    xyz = arrays['coord']
    size, margin = cfg['size'], cfg['margin']
    cells = cell_size(cfg)
    low, high = xyz[:, :2].min(0).astype(np.float64), xyz[:, :2].max(0).astype(np.float64)
    xs = starts_for_axis(low[0], high[0], size, overlap)
    ys = starts_for_axis(low[1], high[1], size, overlap)
    xo = ownership_intervals(xs, size, low[0], high[0] + 1e-3)
    yo = ownership_intervals(ys, size, low[1], high[1] + 1e-3)
    index = cKDTree(xyz[:, :2])
    for member in (network if isinstance(network, (list, tuple)) else [network]):
        member.eval()
    centres, scores, extents, pending = [], [], [], []
    queries = np.zeros((0, 3), np.float32) if queries is None else np.asarray(queries, np.float32).reshape(-1, 3)
    query_scores = np.zeros(len(queries), np.float32)
    X, Y, Z = out_shape(cfg)

    def flush():
        if not pending:
            return
        volumes = torch.from_numpy(np.stack([p[0] for p in pending])).to(device)
        heat, offset, extent = predict_windows(network, volumes, flips)
        for window, (_, origin, bounds) in enumerate(pending):
            a, b, c, d = bounds
            position, score, reach = decode_peaks(heat[window], offset[window], extent[window], threshold, kernel)
            if len(position):
                world = np.column_stack((origin[0] + position[:, 0] * cells[0], origin[1] + position[:, 1] * cells[1],
                                         position[:, 2] * cells[2]))
                own = (world[:, 0] >= a) & (world[:, 0] < b) & (world[:, 1] >= c) & (world[:, 1] < d)
                centres.append(world[own])
                scores.append(score[own])
                extents.append(reach[own])
            if len(queries):
                own = np.flatnonzero((queries[:, 0] >= a) & (queries[:, 0] < b) & (queries[:, 1] >= c) & (queries[:, 1] < d))
                if len(own):
                    pooled = F.max_pool3d(heat[window][None, None], 3, stride=1, padding=1)[0, 0]
                    qx = np.clip(np.floor((queries[own, 0] - origin[0]) / cells[0]).astype(np.int64), 0, X - 1)
                    qy = np.clip(np.floor((queries[own, 1] - origin[1]) / cells[1]).astype(np.int64), 0, Y - 1)
                    qz = np.clip(np.floor(queries[own, 2] / cells[2]).astype(np.int64), 0, Z - 1)
                    query_scores[own] = pooled[qx, qy, qz].cpu().numpy()
        pending.clear()

    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            members = np.asarray(index.query_ball_point([x + size / 2, y + size / 2], size / 2 + margin, p=np.inf), dtype=np.int64)
            if not len(members):
                continue
            origin = np.asarray([x, y], np.float64)
            volume = rasterize_field(xyz[members], votes[members], probability[members], labels[members],
                                     spread[members], origin, cfg)
            pending.append((volume, origin, (*xo[xi], *yo[yi])))
            if len(pending) == batch_size:
                flush()
    flush()
    if centres:
        centres, scores, extents = np.concatenate(centres), np.concatenate(scores), np.concatenate(extents)
    else:
        centres, scores, extents = np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    return dict(centres=centres.astype(np.float32), scores=scores.astype(np.float32),
                extents=extents.astype(np.float32), query_scores=query_scores)


def stage1_centres(votes, labels):
    """Median 3-D vote of every stage-1 instance."""
    labels = np.asarray(labels, np.int64)
    index = np.flatnonzero(labels)
    if not len(index):
        return np.zeros((0, 3), np.float32)
    index = index[np.argsort(labels[index], kind='stable')]
    return np.asarray([np.median(votes[members], axis=0)
                       for members in np.split(index, np.flatnonzero(np.diff(labels[index])) + 1)], np.float32)


def suppress_within_extent(centres, scores, extents, factor):
    """Extent-aware non-maximum suppression.

    A large crown scatters its votes over several metres and can raise more than
    one heat peak. Going from the strongest detection down, a detection is dropped
    when it lies within ``factor`` x the predicted extent of a stronger kept one
    (distance in scaled vote space, so an understory tree far below survives).
    Returns the boolean keep mask.
    """
    keep = np.ones(len(centres), bool)
    if factor <= 0 or len(centres) < 2:
        return keep
    scaled = np.asarray(centres, np.float32) * SCALE
    tree = cKDTree(scaled)
    for i in np.argsort(-np.asarray(scores)):
        if not keep[i]:
            continue
        for j in tree.query_ball_point(scaled[i], factor * float(extents[i])):
            if j != i and scores[j] <= scores[i]:
                keep[j] = False
    return keep


def select_field(detection, anchors, config, xy=1.5, z=4.):
    """Choose the final centres.

    ``suppress`` > 0 first applies extent-aware suppression to the detections.

    ``mode='replace'``: every centre comes from the detector (score >= ``threshold``).
    ``mode='corrective'``: a detection near a stage-1 centre (within ``xy``/``z``)
    needs ``threshold``, one elsewhere needs ``add_threshold``; a stage-1 centre no
    kept detection replaces is retained if the heat there reaches ``anchor_keep``
    (0 keeps all, as in v5/v6).

    Returns centres [K, 3], extents [K] (NaN for stage-1 centres) and provenance kinds.
    """
    centres, scores, extents = detection['centres'], detection['scores'], detection['extents']
    if config.get('suppress', 0.) > 0:
        floor = min(config['threshold'], config.get('add_threshold', config['threshold']))
        alive = scores >= floor
        alive[alive] = suppress_within_extent(centres[alive], scores[alive], extents[alive], config['suppress'])
        centres, scores, extents = centres[alive], scores[alive], extents[alive]
    anchors = np.asarray(anchors, np.float32).reshape(-1, 3)
    near = np.zeros(len(centres), bool)
    if len(anchors) and len(centres):
        tree = cKDTree(anchors[:, :2])
        for i, centre in enumerate(centres):
            candidates = tree.query_ball_point(centre[:2], xy)
            near[i] = bool(candidates) and bool((np.abs(anchors[candidates, 2] - centre[2]) <= z).any())
    if config.get('mode', 'corrective') == 'replace':
        keep = scores >= config['threshold']
        retained = np.zeros(len(anchors), bool)
    else:
        keep = np.where(near, scores >= config['threshold'], scores >= config.get('add_threshold', config['threshold']))
        retained = np.ones(len(anchors), bool)
        if keep.any() and len(anchors):
            tree = cKDTree(centres[keep, :2])
            heights = centres[keep, 2]
            for i, anchor in enumerate(anchors):
                candidates = tree.query_ball_point(anchor[:2], xy)
                if candidates and (np.abs(heights[candidates] - anchor[2]) <= z).any():
                    retained[i] = False
        if config.get('anchor_keep', 0.) > 0 and len(anchors):
            retained &= detection['query_scores'] >= config['anchor_keep']
    final = np.concatenate((centres[keep], anchors[retained])).astype(np.float32).reshape(-1, 3)
    reach = np.concatenate((extents[keep], np.full(int(retained.sum()), np.nan, np.float32)))
    kinds = np.concatenate((np.where(near[keep], SOURCE_RELOCATED, SOURCE_ADDED),
                            np.full(int(retained.sum()), SOURCE_STAGE1_CENTRE))).astype(np.uint8)
    info = dict(replacements=int((keep & near).sum()), additions=int((keep & ~near).sum()),
                stage1_kept=int(retained.sum()), stage1_total=int(len(anchors)))
    return final, reach, kinds, info


def centre_limits(centres, extents, config=None):
    """Reach of every centre in scaled vote space: predicted extent where available
    (``extent_scale`` > 0), otherwise the height rule of v5."""
    cfg = {**ASSIGN, **(config or {})}
    centres = np.asarray(centres, np.float32).reshape(-1, 3)
    limit = np.clip(cfg['radius_base'] + cfg['radius_per_metre'] * centres[:, 2], cfg['radius_min'], cfg['radius_max'])
    if cfg['extent_scale'] > 0:
        learned = np.clip(cfg['extent_scale'] * np.asarray(extents, np.float32), cfg['extent_min'], cfg['extent_max'])
        limit = np.where(np.isfinite(learned), learned, limit)
    return limit.astype(np.float32)


def assign_field(arrays, votes, probability, centres, limits, config=None):
    """Assign every vote to a centre whose reach covers it: the nearest such centre,
    or with ``normalised`` the one with the smallest distance / reach. Then Model20's
    size filters. Returns labels and, per label (index 0 unused), the centre row."""
    cfg = {**ASSIGN, **(config or {})}
    xyz = arrays['coord']
    labels = np.zeros(len(xyz), np.uint32)
    origin = [-1]
    centres = np.asarray(centres, np.float32).reshape(-1, 3)
    if not len(centres):
        return labels, np.asarray(origin)
    limits = np.asarray(limits, np.float32)
    candidate = np.flatnonzero((probability >= cfg['probability']) & (xyz[:, 2] >= .5) & np.isfinite(votes).all(1))
    k = int(min(cfg['neighbours'], len(centres)))
    distance, nearest = cKDTree(centres * SCALE).query(votes[candidate] * SCALE, k=k, distance_upper_bound=float(limits.max()))
    if k == 1:
        distance, nearest = distance[:, None], nearest[:, None]
    nearest = np.minimum(nearest, len(centres) - 1)
    reach = limits[nearest]
    cost = np.where(np.isfinite(distance) & (distance <= reach), distance / reach if cfg['normalised'] else distance, np.inf)
    best = cost.argmin(1)
    rows = np.arange(len(candidate))
    valid = np.isfinite(cost[rows, best])
    temporary = np.zeros(len(xyz), np.int64)
    temporary[candidate[valid]] = nearest[rows, best][valid] + 1
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
    return labels, np.asarray(origin)


def grow_labels(arrays, labels, probability, distance, minimum_probability=.1):
    """Attach unlabelled tree voxels to the instance of the nearest labelled voxel
    within ``distance`` metres (3-D). Votes that land outside every centre's reach
    leave crown voxels unlabelled; this hands them to the crown they touch."""
    xyz = arrays['coord']
    labelled = np.flatnonzero(labels > 0)
    free = np.flatnonzero((labels == 0) & (probability >= minimum_probability) & (xyz[:, 2] >= .5))
    if distance <= 0 or not len(labelled) or not len(free):
        return labels
    gap, nearest = cKDTree(xyz[labelled]).query(xyz[free], distance_upper_bound=distance)
    found = np.isfinite(gap)
    labels = labels.copy()
    labels[free[found]] = labels[labelled[nearest[found]]]
    return labels


def field_labels(arrays, votes, probability, centres, extents, kinds=None, config=None):
    """Final second-pass labels, confidence and per-point provenance."""
    labels, origin = assign_field(arrays, votes, probability, centres, centre_limits(centres, extents, config), config)
    labels = grow_labels(arrays, labels, probability, {**ASSIGN, **(config or {})}['grow_m'])
    confidence = np.where(labels > 0, probability, 0.).astype(np.float32)
    source = np.where(labels > 0, SOURCE_STAGE1_CENTRE, 0).astype(np.uint8)
    if kinds is not None and len(origin) > 1:
        per_label = np.zeros(len(origin), np.uint8)
        per_label[1:] = np.asarray(kinds, np.uint8)[origin[1:]]
        source = per_label[labels]
    return labels, confidence, source


def fallback_to_stage1(labels, source, stage1_labels, threshold=.7, minimum_voxels=12):
    """Coverage fallback: keep a stage-1 tree where the second pass claimed almost none of it.

    A stage-1 instance with at least ``threshold`` of its voxels left unlabelled by
    the second pass becomes an instance again, made of those unlabelled voxels.
    Where stage 2 found trees nothing changes. Returns labels, source and the count.
    """
    labels, source = labels.copy(), source.copy()
    stage1_labels = np.asarray(stage1_labels, np.int64)
    total = np.bincount(stage1_labels)
    free = np.bincount(stage1_labels[labels == 0], minlength=len(total))
    keep = np.flatnonzero((free >= threshold * np.maximum(total, 1)) & (free >= minimum_voxels))
    keep = keep[keep > 0]
    if not len(keep):
        return labels, source, 0
    renumber = np.zeros(len(total), np.int64)
    renumber[keep] = np.arange(len(keep)) + int(labels.max(initial=0)) + 1
    members = (labels == 0) & (renumber[stage1_labels] > 0)
    labels[members] = renumber[stage1_labels[members]].astype(labels.dtype)
    source[members] = SOURCE_FALLBACK
    return labels, source, int(len(keep))


def second_pass(network, arrays, ensemble, stage1_labels, config, device='cuda:0', flips=True, verifier=None,
                assign_network=None):
    """Complete v7 second pass for one plot.

    ``ensemble`` holds the rotation-ensembled ``shifted_center``, ``tree_probability``
    and ``vote_spread``; ``stage1_labels`` are Model20's unchanged labels; ``config``
    is a validated ``dict(selection=..., assign=...[, verifier=dict(rule=...)])``.
    With a verifier configuration, ``verifier`` (the fitted classifier) rescoring
    covers detections and stage-1 centres alike. With ``config['assign_net']``
    (``dict(pool_threshold=..., none_bias=...)``) the third stage replaces selection
    and nearest-centre assignment: ``assign_network`` chooses a candidate per voxel.
    Returns labels, confidence, per-point provenance and selection counts.
    """
    votes, probability = ensemble['shifted_center'], ensemble['tree_probability']
    stage1_labels = np.asarray(stage1_labels, np.int64)
    anchors = stage1_centres(votes, stage1_labels)
    detection = detect_field(network, arrays, votes, probability, stage1_labels, ensemble['vote_spread'],
                             threshold=.1, device=device, flips=flips, queries=anchors)
    detections = int(len(detection['centres']))
    if config.get('assign_net'):
        if assign_network is None:
            raise ValueError('The configuration needs the assignment network, none was given')
        from pointcloud.assign_net import third_pass
        labels, origin, pool, info = third_pass(assign_network, arrays, votes, probability, ensemble['vote_spread'],
                                                stage1_labels, detection, anchors, device=device, **config['assign_net'])
        confidence = np.where(labels > 0, probability, 0.).astype(np.float32)
        per_label = np.zeros(len(origin), np.uint8)
        if len(origin) > 1:
            chosen = origin[1:]
            near = np.zeros(len(chosen), bool)
            if len(anchors):
                tree = cKDTree(anchors[:, :2])
                for i, centre in enumerate(pool['centres'][chosen]):
                    hits = tree.query_ball_point(centre[:2], 1.5)
                    near[i] = bool(hits) and bool((np.abs(anchors[hits, 2] - centre[2]) <= 4.).any())
            per_label[1:] = np.where(pool['is_anchor'][chosen], SOURCE_STAGE1_CENTRE, np.where(near, SOURCE_RELOCATED, SOURCE_ADDED))
        info['detections'] = detections
        info['stage1_kept'] = int((per_label == SOURCE_STAGE1_CENTRE).sum())
        return labels, confidence, per_label[labels], info
    kept_stage1 = None
    if config.get('verifier'):
        if verifier is None:
            raise ValueError('The configuration needs a verifier, none was given')
        from pointcloud.centre_verifier import verify
        detection, is_anchor = verify(verifier, arrays, votes, probability, ensemble['vote_spread'], stage1_labels,
                                      detection, anchors, config['verifier'].get('rule', 'mean'))
        kept_stage1 = detection['centres'][is_anchor]
    centres, extents, kinds, info = select_field(detection, anchors, config['selection'])
    if kept_stage1 is not None and len(kept_stage1) and len(centres):
        gap, _ = cKDTree(kept_stage1).query(centres)
        kinds = np.where(gap < 1e-4, SOURCE_STAGE1_CENTRE, kinds).astype(np.uint8)
        info['stage1_kept'] = int((gap < 1e-4).sum())
    labels, confidence, source = field_labels(arrays, votes, probability, centres, extents, kinds, config['assign'])
    info['detections'] = detections
    if config.get('fallback'):
        labels, source, info['fallback_instances'] = fallback_to_stage1(labels, source, stage1_labels, **config['fallback'])
        confidence = np.where(labels > 0, probability, 0.).astype(np.float32)
    return labels, confidence, source, info
