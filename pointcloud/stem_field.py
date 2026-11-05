"""StemField: a stem-foot detector on raw points (no instances, no Model20).

The points of a window are rasterised into a 3-D grid (0.5 m x 0.5 m x 1 m, heights above ground
up to ``HEIGHT`` m); a small 3-D encoder reduces every level to a 2-D ground-plane map (max and
mean over height), a 2-D decoder brings it back to 0.5 m, and three heads predict, per cell, the
probability that a stem foot lies in it (CenterNet heat), the sub-cell position of the foot and the
log height of its tree. Peaks of the heat map are the stems; they are the queries of the
stem-seeded decoder (``pointcloud.stem_seeded``).

Training targets come from ``stems.csv`` (``scripts/prepare_stem_feet.py``). In partly annotated
plots, cells under unannotated vegetation (``tree_id`` -1) are ignored by the heat loss, so the
detector is not punished for finding trees nobody labelled.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

CELL = .5          # m, horizontal cell of input and output
Z_CELL = 1.        # m, vertical cell of the input
HEIGHT = 48        # layers (m); points higher go to the top layer
CHANNELS = 5       # log count, mean intensity, occupancy, layer height, window density
SIGMA_CELLS = 1.   # Gaussian of the heat target, cells
MATCH_RADIUS = 1.5  # m, a detected foot counts as a stem within this distance


def rasterise(coord, intensity, origin, shape, density=None):
    """Points (local x, y, height above ground) -> (CHANNELS, X, Y, Z) float32 volume.

    ``origin`` is the xy of the volume's lower corner, ``shape`` = (X, Y) cells. ``density``
    (points per m2 of the window) is broadcast as a channel so the network can tell dense
    from sparse scans; by default it is measured from the points inside the volume.
    """
    X, Y = shape
    ix = np.floor((coord[:, 0] - origin[0]) / CELL).astype(np.int64)
    iy = np.floor((coord[:, 1] - origin[1]) / CELL).astype(np.int64)
    iz = np.clip(np.floor(coord[:, 2] / Z_CELL).astype(np.int64), 0, HEIGHT - 1)
    inside = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (coord[:, 2] >= -.5)
    ix, iy, iz = ix[inside], iy[inside], iz[inside]
    flat = (ix * Y + iy) * HEIGHT + iz
    count = np.bincount(flat, minlength=X * Y * HEIGHT).astype(np.float32)
    inten = np.bincount(flat, weights=intensity[inside], minlength=X * Y * HEIGHT).astype(np.float32)
    volume = np.zeros((CHANNELS, X, Y, HEIGHT), np.float32)
    volume[0] = np.log1p(count).reshape(X, Y, HEIGHT)
    volume[1] = (inten / np.maximum(count, 1)).reshape(X, Y, HEIGHT)
    volume[2] = (count > 0).reshape(X, Y, HEIGHT)
    volume[3] = (np.arange(HEIGHT, dtype=np.float32) / HEIGHT)[None, None]
    if density is None:
        density = inside.sum() / max(X * Y * CELL * CELL, 1e-6)
    volume[4] = np.log1p(density) / 5.
    return volume


def heat_targets(feet, origin, shape, ignore=None):
    """Feet (local x, y, h; h <= 0 = unknown) -> heat (X, Y), offset (2, X, Y), log height (X, Y),
    positive mask (X, Y) and height-known mask. Feet outside the grid are skipped."""
    X, Y = shape
    heat = np.zeros((X, Y), np.float32)
    offset = np.zeros((2, X, Y), np.float32)
    log_h = np.zeros((X, Y), np.float32)
    positive = np.zeros((X, Y), bool)
    known = np.zeros((X, Y), bool)
    if len(feet):
        gx, gy = np.meshgrid(np.arange(X) + .5, np.arange(Y) + .5, indexing='ij')
        for x, y, h in np.asarray(feet, np.float64).reshape(-1, 3):
            cx, cy = (x - origin[0]) / CELL, (y - origin[1]) / CELL
            i, j = int(np.floor(cx)), int(np.floor(cy))
            if not (0 <= i < X and 0 <= j < Y):
                continue
            g = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * SIGMA_CELLS ** 2))
            heat = np.maximum(heat, g)
            heat[i, j] = 1.
            positive[i, j] = True
            offset[:, i, j] = (cx - i, cy - j)
            if h > 0:
                log_h[i, j] = np.log(max(h, 1.))
                known[i, j] = True
    if ignore is not None:
        ignore = ignore & ~(heat > .05)
    return heat, offset, log_h, positive, known, ignore


def ignore_mask(coord, tree_id, origin, shape, min_height=2.):
    """Cells whose column is mostly unannotated vegetation (``tree_id`` -1) above ``min_height``:
    a stem may stand there that nobody labelled. Dilated by one cell."""
    X, Y = shape
    ix = np.floor((coord[:, 0] - origin[0]) / CELL).astype(np.int64)
    iy = np.floor((coord[:, 1] - origin[1]) / CELL).astype(np.int64)
    veg = (coord[:, 2] >= min_height) & (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y)
    flat = ix[veg] * Y + iy[veg]
    unknown = np.bincount(flat[tree_id[veg] < 0], minlength=X * Y)
    labelled = np.bincount(flat[tree_id[veg] > 0], minlength=X * Y)
    mask = (unknown > labelled).reshape(X, Y)
    if mask.any():
        mask = F.max_pool2d(torch.from_numpy(mask[None, None].astype(np.float32)), 3, 1, 1)[0, 0].numpy() > 0
    return mask


def block3(i, o):
    return nn.Sequential(nn.Conv3d(i, o, 3, padding=1, bias=False), nn.GroupNorm(min(8, o), o), nn.ReLU(inplace=True),
                         nn.Conv3d(o, o, 3, padding=1, bias=False), nn.GroupNorm(min(8, o), o), nn.ReLU(inplace=True))


def block2(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.GroupNorm(min(8, o), o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.GroupNorm(min(8, o), o), nn.ReLU(inplace=True))


class StemFieldNet(nn.Module):
    """3-D encoder (three stride-2 levels), per-level height pooling to 2-D, 2-D U-Net decoder,
    heads at 0.5 m. Input (B, CHANNELS, X, Y, Z) with X, Y multiples of 8 and Z = HEIGHT."""

    def __init__(self, width=24):
        super().__init__()
        w = width
        self.enc0 = block3(CHANNELS, w)
        self.enc1 = nn.Sequential(nn.Conv3d(w, 2 * w, 2, stride=2), block3(2 * w, 2 * w))
        self.enc2 = nn.Sequential(nn.Conv3d(2 * w, 4 * w, 2, stride=2), block3(4 * w, 4 * w))
        self.enc3 = nn.Sequential(nn.Conv3d(4 * w, 8 * w, 2, stride=2), block3(8 * w, 8 * w))
        self.flat = nn.ModuleList(nn.Conv2d(2 * c, c, 1) for c in (w, 2 * w, 4 * w, 8 * w))
        self.up2 = nn.ConvTranspose2d(8 * w, 4 * w, 2, stride=2)
        self.dec2 = block2(8 * w, 4 * w)
        self.up1 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.dec1 = block2(4 * w, 2 * w)
        self.up0 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.dec0 = block2(2 * w, w)
        self.heat = nn.Sequential(nn.Conv2d(w, w, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(w, 1, 1))
        self.offset = nn.Sequential(nn.Conv2d(w, w, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(w, 2, 1))
        self.height = nn.Sequential(nn.Conv2d(w, w, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(w, 1, 1))
        nn.init.constant_(self.heat[-1].bias, -2.19)   # sigmoid = 0.1
        nn.init.constant_(self.height[-1].bias, 2.8)   # exp = 16 m

    @staticmethod
    def _flatten(conv, x):
        return conv(torch.cat((x.amax(-1), x.mean(-1)), 1))

    def forward(self, volume):
        a = self.enc0(volume)
        b = self.enc1(a)
        c = self.enc2(b)
        d = self.enc3(c)
        a2, b2, c2, d2 = (self._flatten(f, x) for f, x in zip(self.flat, (a, b, c, d)))
        x = self.dec2(torch.cat((self.up2(d2), c2), 1))
        x = self.dec1(torch.cat((self.up1(x), b2), 1))
        x = self.dec0(torch.cat((self.up0(x), a2), 1))
        return dict(heat=self.heat(x)[:, 0], offset=self.offset(x).sigmoid(), log_height=self.height(x)[:, 0])


def field_loss(output, heat, offset, log_h, positive, known, ignore, alpha=2., beta=4.):
    """CenterNet focal loss on the heat (ignored cells excluded), L1 on the sub-cell offset and
    on the log height at the foot cells. Tensors (B, X, Y) / (B, 2, X, Y)."""
    logits = output['heat'].float()
    p = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    valid = ~ignore
    pos = positive & valid
    pos_loss = -(F.logsigmoid(logits) * (1 - p) ** alpha)[pos].sum()
    neg = valid & ~positive
    neg_loss = -(F.logsigmoid(-logits) * p ** alpha * (1 - heat) ** beta)[neg].sum()
    n = pos.sum().clamp_min(1)
    heat_loss = (pos_loss + neg_loss) / n
    zero = logits.sum() * 0.
    if positive.any():
        off = F.l1_loss(output['offset'].permute(0, 2, 3, 1)[positive], offset.permute(0, 2, 3, 1)[positive])
    else:
        off = zero
    k = positive & known
    height = F.l1_loss(output['log_height'][k], log_h[k]) if k.any() else zero
    return dict(loss=heat_loss + off + .5 * height, heat=heat_loss, offset=off, height=height,
                positives=int(positive.sum()))


def decode(output, origin, threshold=.3, kernel=3):
    """Heat peaks of one window -> (K, 3) local x, y, height and (K,) scores."""
    output = {k: v.detach() for k, v in output.items()}
    heat = output['heat'].float().sigmoid()
    if heat.dim() == 3:
        heat = heat[0]
    peak = (heat == F.max_pool2d(heat[None, None], kernel, 1, kernel // 2)[0, 0]) & (heat >= threshold)
    i, j = torch.nonzero(peak, as_tuple=True)
    offset = output['offset'][0] if output['offset'].dim() == 4 else output['offset']
    log_h = output['log_height'][0] if output['log_height'].dim() == 3 else output['log_height']
    x = origin[0] + (i.float() + offset[0, i, j].float()) * CELL
    y = origin[1] + (j.float() + offset[1, i, j].float()) * CELL
    h = log_h[i, j].float().exp()
    feet = torch.stack((x, y, h), 1).cpu().numpy().astype(np.float64)
    return feet, heat[i, j].cpu().numpy().astype(np.float32)


@torch.no_grad()
def detect_plot(model, coord, intensity, threshold=.3, tile=32., keep=20., device=None, kernel=3):
    """Whole plot: tiles of ``tile`` m stepping ``keep`` m; a tile keeps the peaks in its central
    ``keep`` x ``keep`` m (every location belongs to exactly one tile). ``coord`` = local x, y,
    height above ground. ``kernel`` = peak suppression window in cells (3 = 1.5 m). Returns (K, 3)
    feet in the same frame and (K,) scores."""
    device = device or next(model.parameters()).device
    model.eval()
    lo, hi = coord[:, :2].min(0), coord[:, :2].max(0)
    margin = (tile - keep) / 2
    cells = int(round(tile / CELL))
    if cells % 8:
        raise ValueError('tile must be a multiple of 4 m')
    density = len(coord) / max(float(np.prod(hi - lo)), 1.)
    index = cKDTree(coord[:, :2])
    feet, scores = [], []
    xs = np.arange(lo[0], hi[0] + 1e-6, keep)
    ys = np.arange(lo[1], hi[1] + 1e-6, keep)
    for x0 in xs:
        for y0 in ys:
            origin = np.array([x0 - margin, y0 - margin])
            centre = origin + tile / 2
            pts = np.asarray(index.query_ball_point(centre, tile / 2 + 1e-4, p=np.inf), np.int64)
            if not len(pts):
                continue
            volume = rasterise(coord[pts], intensity[pts], origin, (cells, cells), density=density)
            output = model(torch.from_numpy(volume[None]).to(device))
            f, s = decode(output, origin, threshold, kernel)
            own = (f[:, 0] >= x0) & (f[:, 0] < x0 + keep) & (f[:, 1] >= y0) & (f[:, 1] < y0 + keep)
            feet.append(f[own]); scores.append(s[own])
    if not feet:
        return np.zeros((0, 3)), np.zeros(0, np.float32)
    return np.concatenate(feet), np.concatenate(scores)


def match_feet(predicted, truth, radius=MATCH_RADIUS, ignore_xy=None):
    """One-to-one matching of predicted to true feet within ``radius`` (Hungarian on distance).
    ``ignore_xy``: callable xy -> bool, predictions it flags (unannotated area) are not false
    positives. Returns tp, fp, fn, the matched distances and the matched truth indices."""
    predicted = np.asarray(predicted, np.float64).reshape(-1, 3)
    truth = np.asarray(truth, np.float64).reshape(-1, 3)
    if len(predicted) and len(truth):
        d = np.linalg.norm(predicted[:, None, :2] - truth[None, :, :2], axis=2)
        cost = np.where(d <= radius, d, 1e6)
        r, c = linear_sum_assignment(cost)
        ok = cost[r, c] < 1e6
        r, c = r[ok], c[ok]
    else:
        r = c = np.zeros(0, np.int64)
        d = np.zeros((len(predicted), len(truth)))
    unmatched = np.setdiff1d(np.arange(len(predicted)), r)
    if ignore_xy is not None and len(unmatched):
        unmatched = unmatched[~ignore_xy(predicted[unmatched, :2])]
    return dict(tp=int(len(r)), fp=int(len(unmatched)), fn=int(len(truth) - len(r)),
                distances=d[r, c] if len(r) else np.zeros(0), matched_truth=c, matched_pred=r)


def annotated_area(coord, tree_id, origin_xy=None, cell=1.):
    """xy -> True where the plot is mostly unannotated vegetation (used to excuse detections)."""
    xy = coord[:, :2] if origin_xy is None else coord[:, :2] + origin_xy
    veg = coord[:, 2] >= 2.
    lo = xy.min(0)
    ij = np.floor((xy[veg] - lo) / cell).astype(np.int64)
    shape = ij.max(0) + 1 if len(ij) else np.array([1, 1])
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    unknown = np.bincount(flat[tree_id[veg] < 0], minlength=int(np.prod(shape))).reshape(shape)
    labelled = np.bincount(flat[tree_id[veg] > 0], minlength=int(np.prod(shape))).reshape(shape)
    excuse = unknown > labelled

    def ignore_xy(points):
        k = np.floor((np.asarray(points)[:, :2] - lo) / cell).astype(np.int64)
        inside = (k >= 0).all(1) & (k[:, 0] < shape[0]) & (k[:, 1] < shape[1])
        out = np.zeros(len(points), bool)
        out[inside] = excuse[k[inside, 0], k[inside, 1]]
        return out
    return ignore_xy
