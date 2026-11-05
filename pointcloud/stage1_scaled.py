"""Opt-in pieces of the scaled stage-1 fine-tune (``scripts/train_stage1_scaled.py``).

Nothing here changes the shared stage-1 code: the dataset subclasses
``DrawDataset`` and reproduces its crops exactly when thinning is off, the loss
adds terms on top of the unchanged ``joint_losses`` and the mixed-precision
forward is the same computation as ``DualHeadLitePT.forward`` with only the
backbone under autocast.
"""
from __future__ import annotations

from collections import OrderedDict
import contextlib

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.nn import functional as F

from pointcloud.data import load_npz, prepare_crop
from pointcloud.joint_training import DrawDataset, joint_losses

HEIGHT_WEIGHT = .33          # the second pass measures vote distances in (x, y, 0.33 z)
POINT_KEYS = ('coord', 'grid_coord', 'intensity', 'tree_id', 'instance_offset', 'semantic_target')


def thin_window(arrays, anchor, density, rng, size=20., minimum=64):
    """Thin the ``size`` m window around ``anchor`` to ``density`` voxels per occupied 1 m cell.

    The same uniform random voxel subsampling as ``scripts.cache_stage1_thinned.thin``,
    applied to the crop window only. Centroid targets (``instance_offset``) are kept,
    so a thinned tree still votes for the centroid of the complete tree. Returns the
    reduced arrays and the anchor's new index.
    """
    coord = arrays['coord']
    centre = coord[anchor, :2]
    window = np.flatnonzero((np.abs(coord[:, 0] - centre[0]) <= size / 2) & (np.abs(coord[:, 1] - centre[1]) <= size / 2))
    cells = len(np.unique(np.floor(coord[window, :2]).astype(np.int64), axis=0))
    count = max(minimum, int(round(density * cells)))
    if count < len(window):
        window = rng.choice(window, count, replace=False)
        window = np.unique(np.append(window, anchor))
    result = {k: (v[window] if k in POINT_KEYS else v) for k, v in arrays.items()}
    return result, int(np.searchsorted(window, anchor))


class ScaledDrawDataset(DrawDataset):
    """``DrawDataset`` with a plot cache, epoch-carrying indices and strong density thinning.

    Index ``(row, draw, epoch)``. With ``thin_probability == 0`` a crop is identical
    to ``DrawDataset``'s for the same seed, epoch, draw and row. Otherwise a real
    (non-simulated) plot is, with that probability, thinned to a log-uniform
    density from ``thin_range`` (voxels/m2) instead of the 0.5 / 0.75 / 1.0 keep
    fractions; simulated plots are already sparse and keep the original recipe
    unless ``thin_synthetic``.
    """
    def __init__(self, rows, seed, max_points, hard=False, difficulty=None, thin_probability=0.,
                 thin_range=(6., 25.), thin_synthetic=False, cache_size=96):
        super().__init__(rows, seed, max_points, hard=hard, difficulty=difficulty)
        self.thin_probability, self.thin_range, self.thin_synthetic = float(thin_probability), tuple(thin_range), thin_synthetic
        self.cache_size, self._plots = cache_size, OrderedDict()

    def plot(self, row_index):
        if row_index in self._plots:
            self._plots.move_to_end(row_index)
            return self._plots[row_index]
        row = self.rows[row_index]
        arrays = load_npz(row['output'])
        hard = None
        ids = arrays['tree_id']
        positive = ids > 0
        unique, inverse, count = np.unique(ids[positive], return_inverse=True, return_counts=True)
        if len(unique):                      # the hard-crop weights of DrawDataset, computed once per plot
            centres = np.zeros((len(unique), 2))
            np.add.at(centres, inverse, arrays['coord'][positive, :2])
            centres /= count[:, None]
            weight = np.clip(np.sqrt(np.median(count) / count), .5, 3.)
            if len(unique) > 1:
                distance = cKDTree(centres).query(centres, k=2)[0][:, 1]
                weight *= 1 + (distance < 3.)
            parent = row.get('parent_plot')
            key = f'treescan_helios__{parent}' if parent else row['dataset_id']
            missed = self.difficulty.get(key, {})
            weight *= [1 + 2 * float(missed.get(str(int(t)), 0)) for t in unique]
            hard = (unique, weight / weight.sum())
        self._plots[row_index] = (arrays, hard)
        if len(self._plots) > self.cache_size:
            self._plots.popitem(last=False)
        return self._plots[row_index]

    def __getitem__(self, index):
        row_index, draw, epoch = index
        row = self.rows[row_index]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, draw, row_index]))
        arrays, hard = self.plot(row_index)
        anchor = None
        if self.hard and rng.random() < .7 and hard is not None:
            selected = rng.choice(hard[0], p=hard[1])
            anchor = int(rng.choice(np.flatnonzero(arrays['tree_id'] == selected)))
        fractions = (.5, .75, 1.)
        if self.thin_probability > 0:
            real = row['source_dataset'] != 'TreeScanPL10k_HELIOS'
            # A separate stream, so the crop geometry does not depend on the thinning switch.
            extra = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, draw, row_index, 20261002]))
            if (real or self.thin_synthetic) and extra.random() < self.thin_probability:
                if anchor is None:
                    positive = np.flatnonzero(arrays['tree_id'] > 0)
                    anchor = int(extra.choice(positive if len(positive) else len(arrays['coord'])))
                density = float(np.exp(extra.uniform(*np.log(self.thin_range))))
                arrays, anchor = thin_window(arrays, anchor, density, extra)
                fractions = (1.,)
        return prepare_crop(arrays, rng, 20., self.max_points, True, density_keep_fractions=fractions,
                            preserve_height=bool(row.get('height_normalization')), anchor_index=anchor)


@contextlib.contextmanager
def float32_sparse_convolutions():
    """Run spconv layers in float32 inside an autocast region (the installed spconv build
    has no float16 kernels: ``can't find suitable algorithm``)."""
    from spconv.pytorch.conv import SparseConvolution
    original = SparseConvolution.forward

    def forward(self, input, *args, **kwargs):
        with torch.autocast('cuda', enabled=False):
            if input.features.dtype != torch.float32:
                input = input.replace_feature(input.features.float())
            return original(self, input, *args, **kwargs)
    SparseConvolution.forward = forward
    try:
        yield
    finally:
        SparseConvolution.forward = original


def forward_stage1(model, data, amp=False):
    """``model(data)``; with ``amp`` the LitePT backbone runs under float16 autocast (its sparse
    convolutions stay in float32) and the heads and the mask decoder, which sum over thousands
    of voxels, stay in float32."""
    if not amp:
        return model(data)
    if model.training_scope == 'mask':
        raise ValueError('Mixed precision is implemented for joint training only')
    with float32_sparse_convolutions(), torch.autocast('cuda', dtype=torch.float16):
        features = model.legacy.backbone(data).feat
    features = features.float()
    logits = model.legacy.semantic_head(features)
    offset = model.legacy.offset_head(features) * model.legacy.offset_scale_m
    dense = model.point_decoder(features, data, logits, offset)
    return dict(semantic_logits=logits, offset_m=offset, point_semantic_logits=dense['semantic_logits'],
                point_offset_m=dense['offset_m'], point_quality_logits=dense['quality_logits'], instance_masks=dense)


def per_tree_mean(values, inverse, count):
    """Mean over trees of the per-tree mean (small and large trees weigh the same)."""
    return (values / count[inverse]).sum() / len(count)


def scaled_losses(prediction, batch, z_weight=0., assign_weight=0., assign_temperature=.5):
    """``joint_losses`` plus optional terms that sharpen the centre votes.

    ``z_weight``: smooth-L1 on the vertical offset of both vote heads (Model20
    supervises XY only). ``assign_weight``: cross-entropy that asks every vote to be
    nearer to its own true centroid than to the other trees' centroids in the crop,
    in the scaled space of the second pass. With both weights 0 the result equals
    ``joint_losses`` exactly.
    """
    losses = joint_losses(prediction, batch)
    ids = batch['tree_id']
    positive = ids > 0
    if not positive.any() or (z_weight <= 0 and assign_weight <= 0):
        return losses
    unique, inverse, count = torch.unique(ids[positive], return_inverse=True, return_counts=True)
    target = batch['instance_offset'][positive]
    extra = 0.
    if z_weight > 0:
        for name, key in (('legacy_offset_z', 'offset_m'), ('offset_z', 'point_offset_m')):
            error = F.smooth_l1_loss(prediction[key][positive, 2], target[:, 2], reduction='none', beta=.5)
            losses[name] = per_tree_mean(error, inverse, count)
            extra = extra + z_weight * losses[name]
    if assign_weight > 0 and len(unique) > 1:
        scale = target.new_tensor([1., 1., HEIGHT_WEIGHT])
        truth = (batch['coord'][positive] + target) * scale
        centres = truth.new_zeros((len(unique), 3)).index_add_(0, inverse, truth) / count[:, None]
        for name, key in (('legacy_assign', 'offset_m'), ('assign', 'point_offset_m')):
            votes = (batch['coord'][positive] + prediction[key][positive]) * scale
            logits = -torch.cdist(votes, centres) / assign_temperature
            losses[name] = per_tree_mean(F.cross_entropy(logits, inverse, reduction='none'), inverse, count)
            extra = extra + assign_weight * losses[name]
    losses['loss'] = losses['loss'] + extra
    return losses
