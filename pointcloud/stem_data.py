"""Training crops with stem feet for the stem-seeded decoder.

Every DualCrown3D plot gets a ``stems.csv`` next to its ``points_0p25m.npz``
(``scripts/prepare_stem_feet.py``): one row per tree with the foot in the npz coordinate frame
(``x, y`` = ``coord + source_origin``), the tree height ``h``, a label ``quality`` (1 = measured or
TLS-traced foot, 0.5 = proxy from the lowest points) and the ``source``. ``StemDrawDataset`` cuts
the same crops as ``joint_training.DrawDataset``, moves the feet into the crop frame and, for
training, corrupts them (``stem_seeded.corrupt_stems``).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pointcloud.data import load_npz, prepare_crop
from pointcloud.paths import remap
from pointcloud.stem_seeded import corrupt_stems, stems_for_crop

STEM_FILE = 'stems.csv'


def stems_path(row):
    return Path(remap(row['output'])).with_name(STEM_FILE)


def read_stems(row):
    """(S, 3) world x, y, h and (S,) tree ids; rows without a usable foot are dropped."""
    frame = pd.read_csv(stems_path(row))
    frame = frame[np.isfinite(frame[['x', 'y']]).all(axis=1)]
    stems = np.column_stack((frame.x.values, frame.y.values, frame.h.fillna(0.).values)).astype(np.float64)
    return stems, frame.tree_id.values.astype(np.int64), frame.quality.values.astype(np.float32)


def read_stems_ignore(row):
    """As ``read_stems`` plus the ``ignore`` flag (1 = a stem the detector is neither rewarded nor
    punished for, e.g. Brandenburg understorey); all zeros when the column is absent."""
    frame = pd.read_csv(stems_path(row))
    frame = frame[np.isfinite(frame[['x', 'y']]).all(axis=1)]
    stems = np.column_stack((frame.x.values, frame.y.values, frame.h.fillna(0.).values)).astype(np.float64)
    ignore = frame['ignore'].values.astype(bool) if 'ignore' in frame else np.zeros(len(frame), bool)
    return stems, frame.tree_id.values.astype(np.int64), frame.quality.values.astype(np.float32), ignore


def source_weights(rows, weights):
    """Sampling weight per row: every named source group gets ``weights[group]`` of the draws,
    spread evenly over the group's plots (HELIOS flight variants count per parent plot)."""
    groups = [stem_group(r) for r in rows]
    counts = Counter(groups)
    total = sum(weights.get(g, 0.) for g in counts)
    if total <= 0:
        raise ValueError('No source group with positive weight')
    return np.array([weights.get(g, 0.) / total / counts[g] for g in groups], np.float64)


def stem_group(row):
    source = row['source_dataset']
    if source == 'TreeScanPL10k_HELIOS':
        return 'helios'
    if source == 'WEISER_2022_ALS':
        return 'weiser_real'
    if source == 'WEISER_2022_SIM_V2':
        return 'weiser_sim'
    if source == 'BRANDENBURG_ALS':
        return 'brandenburg'
    return 'for_instance'


class StemDrawDataset(Dataset):
    """Index = (row, draw); the draw number seeds a new crop of the same plot."""

    def __init__(self, rows, seed, max_points, corrupt=True, crop_size_m=20., min_quality=0., corruption=None,
                 detector=None, detector_probability=0., detector_noise=(.15, .1, .05)):
        if any(r['model_split'] != 'train' for r in rows):
            raise ValueError('StemDrawDataset accepts train rows only')
        self.rows, self.seed, self.max_points = rows, seed, max_points
        self.corrupt, self.crop_size_m, self.min_quality = corrupt, crop_size_m, min_quality
        self.corruption = dict(corruption or {})
        # Detector feet (``scripts/cache_detector_feet.py``: stems_<detector>.csv with matched tree ids):
        # with probability ``detector_probability`` a crop takes them instead of corrupted labelled feet,
        # plus noise (xy sigma m, log-height sigma, drop share), since the detector saw these plots.
        self.detector, self.detector_probability, self.detector_noise = detector, detector_probability, detector_noise
        self.epoch = 0
        self._stems = {}
        self._detected = {}

    def __len__(self):
        return len(self.rows)

    def stems(self, row):
        key = row['output']
        if key not in self._stems:
            self._stems[key] = read_stems(row)
        return self._stems[key]

    def detected(self, row):
        key = row['output']
        if key not in self._detected:
            path = stems_path(row).with_name(f'stems_{self.detector}.csv')
            frame = pd.read_csv(path) if path.exists() else None
            self._detected[key] = None if frame is None else (
                np.column_stack((frame.x.values, frame.y.values, frame.h.values)).astype(np.float64),
                frame.tree_id.values.astype(np.int64), frame.score.values.astype(np.float32))
        return self._detected[key]

    def __getitem__(self, index):
        row_index, draw = index
        row = self.rows[row_index]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, draw, row_index]))
        arrays = load_npz(row['output'])
        result = prepare_crop(arrays, rng, self.crop_size_m, self.max_points, True,
                              density_keep_fractions=(.5, .75, 1.),
                              preserve_height=bool(row.get('height_normalization')), return_transform=True)
        if self.detector and rng.random() < self.detector_probability and self.detected(row) is not None:
            return self.detector_sample(result, row, rng)
        world, ids, quality = self.stems(row)
        usable = quality >= self.min_quality
        coord = result['coord'].numpy()
        local, local_ids = stems_for_crop(world[usable], ids[usable], result['local_to_world'].numpy(), coord)
        # Feet of trees whose crop has no labelled points are not supervised, but stay as queries.
        present = np.isin(local_ids, np.unique(result['tree_id'].numpy()))
        local_ids = np.where(present, local_ids, -1)
        score = np.ones(len(local), np.float32)
        if self.corrupt:
            canopy = coord[coord[:, 2] >= 2., :2]
            real = local_ids > 0
            corrupted, corrupted_ids, score = corrupt_stems(local[real], local_ids[real], rng, xy_canopy=canopy,
                                                            **self.corruption)
            local = np.concatenate((corrupted, local[~real]))
            local_ids = np.concatenate((corrupted_ids, local_ids[~real]))
            score = np.concatenate((score, np.ones((~real).sum(), np.float32)))
        result.pop('local_to_world')
        result['stems'] = torch.from_numpy(local.astype(np.float32))
        result['stem_tree_id'] = torch.from_numpy(local_ids.astype(np.int64))
        result['stem_score'] = torch.from_numpy(score.astype(np.float32))
        return result

    def detector_sample(self, result, row, rng):
        """The detector's feet in the crop: matched feet keep their tree, false feet (0) own nothing,
        unknown ones (-1) are queries only; feet of trees with no labelled point in the crop become -1."""
        world, ids, score = self.detected(row)
        sigma, log_sigma, drop = self.detector_noise
        keep = rng.random(len(world)) >= drop
        world, ids, score = world[keep].copy(), ids[keep], score[keep]
        world[:, :2] += rng.normal(0., sigma, (len(world), 2))
        world[:, 2] *= np.exp(rng.normal(0., log_sigma, len(world)))
        coord = result['coord'].numpy()
        inside_ids = np.arange(len(world))
        local, local_index = stems_for_crop(world, inside_ids, result['local_to_world'].numpy(), coord)
        local_ids = ids[local_index]
        local_score = score[local_index]
        present = np.isin(local_ids, np.unique(result['tree_id'].numpy()))
        local_ids = np.where((local_ids > 0) & ~present, -1, local_ids)
        result.pop('local_to_world')
        result['stems'] = torch.from_numpy(local.astype(np.float32))
        result['stem_tree_id'] = torch.from_numpy(local_ids.astype(np.int64))
        result['stem_score'] = torch.from_numpy(local_score.astype(np.float32))
        return result
