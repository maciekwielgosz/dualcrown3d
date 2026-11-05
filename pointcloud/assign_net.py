"""Stage 3 network: sparse 3-D U-Net that assigns voxels to candidate tree centres.

Every tree voxel carries its ``K`` nearest candidates as ordered slots (see
``candidate_assignment``). A shared MLP encodes each slot, a sparse U-Net over
the voxel grid (0.25 m) builds a context embedding from all slot encodings, and
a shared scorer rates each slot from (context, slot encoding); one extra logit
says "none of these". Voxels of one crown see each other through the U-Net, so
the choice is made consistently per crown rather than per voxel, which is what a
tabular model cannot do.
"""
from __future__ import annotations

import numpy as np
import spconv.pytorch as spconv
import torch
from torch import nn

from pointcloud.candidate_assignment import K, SLOT_FEATURES, VOXEL_FEATURES, labels_from_choice, slot_inputs
from pointcloud.centre_verifier import FEATURE_NAMES, build_candidates, candidate_features

CANDIDATE_COLUMNS = ('score', 'votes_extent', 'exclusive_votes', 'exclusive_share', 'covered_by_stronger',
                     'stronger_within_extent', 'n1_dist_over_extent', 'stage1_capture', 'stage1_purity',
                     'voter_offcentre_ratio', 'votes_1_share', 'voter_radius')
SLOT_DIM = len(SLOT_FEATURES) + len(CANDIDATE_COLUMNS)
GRID_SHAPE = (96, 96, 224)          # 24 m x 24 m x 56 m at 0.25 m
SIZE = 20.


def candidate_columns(feature_names):
    return np.asarray([feature_names.index(name) for name in CANDIDATE_COLUMNS], np.int64)


COLUMNS = candidate_columns(FEATURE_NAMES)


def slot_tensor(slot, slots, candidate_features, columns):
    """[M, K, SLOT_DIM]: slot geometry joined with the candidate-level features of that slot."""
    joined = candidate_features[np.maximum(slots, 0)][:, :, columns].astype(np.float32)
    joined = np.clip(joined, -9., 9.) / np.asarray([1., 9., 9., 1., 3., 3., 6., 1., 1., 3., 1., 4.], np.float32)
    joined[slots < 0] = 0.
    return np.concatenate((slot, joined), axis=2)


def subm(inputs, outputs, key):
    return spconv.SparseSequential(spconv.SubMConv3d(inputs, outputs, 3, padding=1, bias=False, indice_key=key),
                                   nn.BatchNorm1d(outputs), nn.ReLU(inplace=True))


class Level(nn.Module):
    def __init__(self, inputs, outputs, key):
        super().__init__()
        self.body = spconv.SparseSequential(subm(inputs, outputs, key), subm(outputs, outputs, key))

    def forward(self, x):
        return self.body(x)


class AssignNet(nn.Module):
    def __init__(self, k=K, width=48, slot_width=24, levels=4):
        super().__init__()
        self.k = k
        self.slot_encoder = nn.Sequential(nn.Linear(SLOT_DIM, 64), nn.ReLU(inplace=True), nn.Linear(64, slot_width), nn.ReLU(inplace=True))
        self.stem = nn.Sequential(nn.Linear(len(VOXEL_FEATURES) + k * slot_width, width), nn.ReLU(inplace=True))
        widths = [width * (2 ** i) for i in range(levels)]
        self.encoders = nn.ModuleList([Level(widths[0], widths[0], 'subm0')])
        self.downs, self.ups, self.decoders = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(1, levels):
            self.downs.append(spconv.SparseSequential(
                spconv.SparseConv3d(widths[i - 1], widths[i], 2, stride=2, bias=False, indice_key=f'down{i}'),
                nn.BatchNorm1d(widths[i]), nn.ReLU(inplace=True)))
            self.encoders.append(Level(widths[i], widths[i], f'subm{i}'))
            self.ups.append(spconv.SparseSequential(
                spconv.SparseInverseConv3d(widths[i], widths[i - 1], 2, bias=False, indice_key=f'down{i}'),
                nn.BatchNorm1d(widths[i - 1]), nn.ReLU(inplace=True)))
            self.decoders.append(Level(2 * widths[i - 1], widths[i - 1], f'subm{i - 1}'))
        self.scorer = nn.Sequential(nn.Linear(width + slot_width, 96), nn.ReLU(inplace=True), nn.Linear(96, 1))
        self.none = nn.Sequential(nn.Linear(width, 48), nn.ReLU(inplace=True), nn.Linear(48, 1))

    def forward(self, voxel, slot, valid, coords, batch_size):
        """voxel [N, V], slot [N, K, SLOT_DIM], valid [N, K] bool, coords [N, 4] int (batch, x, y, z).
        Returns logits [N, K + 1]; invalid slots get -inf."""
        encoded = self.slot_encoder(slot) * valid[..., None]
        features = self.stem(torch.cat((voxel, encoded.flatten(1)), 1))
        x = spconv.SparseConvTensor(features, coords, list(GRID_SHAPE), batch_size)
        skips = []
        x = self.encoders[0](x)
        for down, encoder in zip(self.downs, self.encoders[1:]):
            skips.append(x)
            x = encoder(down(x))
        for up, decoder, skip in zip(reversed(self.ups), reversed(self.decoders), reversed(skips)):
            x = up(x)
            x = decoder(x.replace_feature(torch.cat((x.features, skip.features), 1)))
        context = x.features
        scores = self.scorer(torch.cat((context[:, None, :].expand(-1, self.k, -1), encoded), 2)).squeeze(2)
        scores = scores.masked_fill(~valid, float('-inf'))
        return torch.cat((scores, self.none(context)), 1)


def window_coords(grid_coord, batch_index=0):
    """Integer voxel coordinates of one window, shifted to start at zero and clipped to the grid."""
    local = grid_coord - grid_coord.min(0, keepdims=True)
    local = np.minimum(local, np.asarray(GRID_SHAPE) - 1)
    return np.concatenate((np.full((len(local), 1), batch_index, np.int32), local.astype(np.int32)), 1)


def unique_rows(coords):
    """Indices keeping one row per voxel coordinate (clipping or symmetry can create collisions)."""
    key = ((coords[:, 0].astype(np.int64) * GRID_SHAPE[0] + coords[:, 1]) * GRID_SHAPE[1] + coords[:, 2]) * GRID_SHAPE[2] + coords[:, 3]
    _, first = np.unique(key, return_index=True)
    return np.sort(first)


def pool_mask(candidates, threshold):
    """Candidates admitted to the slots: detections at or above ``threshold`` and every stage-1 centre."""
    return (candidates['scores'] >= threshold) | candidates['is_anchor']



@torch.no_grad()
def assign_plot(network, arrays, cache, candidates, features, device='cuda:0', overlap=8., max_voxels=250000):
    """Plot-level inference: voxel index, slots and class probabilities [M, K + 1]."""
    index, slots, voxel, slot = slot_inputs(arrays, cache['votes'], cache['probability'], cache['spread'], candidates)
    probability = np.zeros((len(index), K + 1), np.float32)
    probability[:, K] = 1.
    if not len(index) or not len(candidates['centres']):
        return index, slots, probability
    tensor = slot_tensor(slot, slots, features, COLUMNS)
    xyz = arrays['coord'][index]
    grid = arrays['grid_coord'][index]
    low, high = xyz[:, :2].min(0).astype(np.float64), xyz[:, :2].max(0).astype(np.float64)
    from scripts.evaluate_pointcloud_litept import ownership_intervals, starts_for_axis
    xs, ys = starts_for_axis(low[0], high[0], SIZE, overlap), starts_for_axis(low[1], high[1], SIZE, overlap)
    xo = ownership_intervals(xs, SIZE, low[0], high[0] + 1e-3)
    yo = ownership_intervals(ys, SIZE, low[1], high[1] + 1e-3)
    network.eval()
    for xi, x in enumerate(xs):
        inside_x = (xyz[:, 0] >= x) & (xyz[:, 0] <= x + SIZE)
        for yi, y in enumerate(ys):
            members = np.flatnonzero(inside_x & (xyz[:, 1] >= y) & (xyz[:, 1] <= y + SIZE))
            if not len(members):
                continue
            (a, b), (c, d) = xo[xi], yo[yi]
            own = (xyz[members, 0] >= a) & (xyz[members, 0] < b) & (xyz[members, 1] >= c) & (xyz[members, 1] < d)
            if not own.any():
                continue
            if len(members) > max_voxels:        # keep every owned voxel, thin the context
                context = np.flatnonzero(~own)
                keep = np.concatenate((np.flatnonzero(own), context[np.linspace(0, len(context) - 1, max(max_voxels - int(own.sum()), 0)).astype(np.int64)] if len(context) and max_voxels > own.sum() else np.zeros(0, np.int64)))
                keep = np.sort(keep)
                members, own = members[keep], own[keep]
            coords = window_coords(grid[members])
            first = unique_rows(coords)
            rows = members[first]
            logits = network(torch.from_numpy(voxel[rows]).to(device), torch.from_numpy(tensor[rows]).to(device),
                             torch.from_numpy(slots[rows] >= 0).to(device), torch.from_numpy(coords[first]).to(device), 1)
            result = logits.float().softmax(1).cpu().numpy()
            owned = own[first]
            probability[rows[owned]] = result[owned]
    return index, slots, probability


def choose(probability, none_bias=0.):
    """Slot per voxel, or -1: 'none' wins when its probability times exp(bias) is the largest."""
    scores = probability.copy()
    scores[:, -1] *= np.exp(none_bias)
    best = scores.argmax(1)
    return np.where(best == probability.shape[1] - 1, -1, best)


def third_pass(network, arrays, votes, probability, spread, stage1_labels, detection, anchors, pool_threshold=.3,
               none_bias=0., device='cuda:0'):
    """Stage 3 for one plot: candidates from the detector and stage 1, learned assignment, labels.

    Returns labels, the candidate row behind every label (index 0 unused), the candidate set and counts.
    """
    candidates = build_candidates(detection, anchors)
    features = candidate_features(arrays, votes, probability, spread, stage1_labels, candidates, anchors)
    keep = pool_mask(candidates, pool_threshold)
    pool = {k: v[keep] for k, v in candidates.items()}
    index, slots, classes = assign_plot(network, arrays, dict(votes=votes, probability=probability, spread=spread), pool,
                                        features[keep], device)
    labels, origin = labels_from_choice(arrays, index, slots, choose(classes, none_bias))
    return labels, origin, pool, dict(candidates=int(len(candidates['centres'])), admitted=int(keep.sum()),
                                      instances=int(labels.max(initial=0)))
