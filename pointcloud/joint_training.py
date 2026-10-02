"""Joint fine-tuning utilities; no validation or test supervision is consumed."""
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from scipy.spatial import cKDTree

from pointcloud.data import load_npz, prepare_crop
from pointcloud.dual_head import dense_losses


def fixed_serialization(model):
    for module in model.modules():
        if hasattr(module, 'shuffle_orders'):
            module.shuffle_orders = False


def sampling_weights(rows, real_fraction):
    if not 0 < real_fraction < 1:
        raise ValueError('Both domains must receive positive sampling weight')
    real = [r['source_dataset'] != 'TreeScanPL10k_HELIOS' for r in rows]
    groups = [(r['source_dataset'], r['collection']) for r in rows]
    counts = Counter(g for g, flag in zip(groups, real) if flag)
    parents = [r.get('parent_plot') or r['dataset_id'].removeprefix('treescan_helios__') for r in rows]
    synthetic_counts = Counter(p for p, flag in zip(parents, real) if not flag)
    if not counts or not synthetic_counts:
        raise ValueError('Missing a training domain')
    return np.array([real_fraction / len(counts) / counts[g] if flag else
                     (1-real_fraction) / len(synthetic_counts) / synthetic_counts[p]
                     for g, p, flag in zip(groups, parents, real)], dtype=np.float64)


class DrawDataset(Dataset):
    """Tuple index includes draw number, so repeated plots produce new crops."""
    def __init__(self, rows, seed, max_points, hard=False, difficulty=None,
                 crown_supervision=False):
        if any(r['model_split'] != 'train' for r in rows):
            raise ValueError('DrawDataset accepts train rows only')
        self.rows, self.seed, self.max_points = rows, seed, max_points
        self.hard, self.difficulty, self.epoch = hard, difficulty or {}, 0
        self.crown_supervision, self._polygons = crown_supervision, {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row_index, draw = index
        row = self.rows[row_index]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, draw, row_index]))
        arrays = load_npz(row['output'])
        anchor = None
        if self.hard and rng.random() < .7:
            ids = arrays['tree_id']; pos = ids > 0
            unique, inv, count = np.unique(ids[pos], return_inverse=True, return_counts=True)
            if len(unique):
                centers = np.zeros((len(unique), 2))
                np.add.at(centers, inv, arrays['coord'][pos, :2])
                centers /= count[:, None]
                weight = np.clip(np.sqrt(np.median(count)/count), .5, 3.)
                if len(unique) > 1:
                    distance = cKDTree(centers).query(centers, k=2)[0][:, 1]
                    weight *= 1 + (distance < 3.)
                parent = row.get('parent_plot')
                key = f'treescan_helios__{parent}' if parent else row['dataset_id']
                missed = self.difficulty.get(key, {})
                weight *= [1 + 2*float(missed.get(str(int(t)), 0)) for t in unique]
                selected = rng.choice(unique, p=weight/weight.sum())
                anchor = int(rng.choice(np.flatnonzero(ids == selected)))
        result = prepare_crop(arrays, rng, 20., self.max_points, True,
                              density_keep_fractions=(.5, .75, 1.),
                              preserve_height=bool(row.get('height_normalization')),
                              anchor_index=anchor, return_transform=self.crown_supervision)
        if self.crown_supervision:
            import geopandas as gpd
            from pointcloud.shared_instance import polygon_targets
            path = row['gt_vector']
            if path not in self._polygons:
                frame = gpd.read_file(path)
                key = 'treeID' if 'treeID' in frame else 'tree_id'
                if 'complete' in frame:
                    frame = frame[frame['complete'].astype(bool)]
                if 'evaluation_eligible' in frame:
                    frame = frame[frame['evaluation_eligible'].astype(bool)]
                self._polygons[path] = dict(zip(frame[key].astype(int), frame.geometry))
            ids, masks, valid = polygon_targets(result['coord'], result['semantic_target'],
                                                result['tree_id'], result['local_to_world'],
                                                self._polygons[path])
            result.update(crown_ids=ids, crown_target=masks, crown_valid=valid)
        return result


def joint_losses(prediction, batch):
    losses = dense_losses(prediction, batch)
    ids = batch['tree_id']; positive = ids > 0
    target = batch.get('semantic_target', torch.where(ids < 0, -1, positive.long()))
    valid = target >= 0
    logits = prediction['semantic_logits']
    zero = logits.sum()*0
    semantic = F.cross_entropy(logits[valid], target[valid].long()) if valid.any() else zero
    offset = zero
    if positive.any():
        unique, inverse, count = torch.unique(ids[positive], return_inverse=True, return_counts=True)
        error = F.smooth_l1_loss(prediction['offset_m'][positive, :2],
                                batch['instance_offset'][positive, :2], reduction='none', beta=.5).mean(1)
        offset = (error / count[inverse]).sum() / len(unique)
    losses['legacy_semantic'] = semantic
    losses['legacy_offset'] = offset
    losses['loss'] = losses['loss'] + semantic + offset
    return losses
