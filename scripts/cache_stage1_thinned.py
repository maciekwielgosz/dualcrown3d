#!/usr/bin/env python3
"""Cache frozen Model20 predictions on density-thinned plots for the vote-centre detector.

The annotated plots hold roughly 70-380 voxels/m2, the reference scene about 10.
Random voxel thinning is a crude proxy for a sparser scan (it does not model
occlusion), but it puts Model20's votes in the sparse regime the detector must
handle. Train plots get two random densities, validation plots a fixed one.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz
from pointcloud.instance_output import merge_masks_with_sources
from pointcloud.two_pass import match_instances
from scripts.cache_stage1_predictions import DEFAULT_OUTPUT, eligible_rows, sha256, write_json
from scripts.predict_dual_head import predict
from scripts.train_supervision_v4 import build


def thin(arrays, density, rng):
    """Random voxel subset at ``density`` voxels per occupied square metre; returns indices."""
    cells = len(np.unique(np.floor(arrays['coord'][:, :2]).astype(np.int64), axis=0))
    count = min(len(arrays['coord']), max(64, int(round(density * cells))))
    return np.sort(rng.choice(len(arrays['coord']), count, replace=False)), count / cells


def subset(arrays, keep):
    size = len(arrays['coord'])
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == size else v) for k, v in arrays.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--train-variants', type=int, default=2)
    parser.add_argument('--train-range', type=float, nargs=2, default=(6., 40.))
    parser.add_argument('--val-density', type=float, default=10.)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.output.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    if sha256(protocol['stage1_checkpoint']) != protocol['stage1_sha256']:
        raise ValueError('Stage-1 checkpoint differs from the frozen protocol')
    model, _ = build(protocol['stage1_checkpoint'])
    model.eval()
    index = {}
    for split in ('val', 'train'):
        folder = root / 'stage1_thinned' / split
        folder.mkdir(parents=True, exist_ok=True)
        for number, row in enumerate(eligible_rows(split), 1):
            full = load_npz(row['output'])
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, number, split == 'train']))
            targets = ([args.val_density] if split == 'val' else
                       np.exp(rng.uniform(np.log(args.train_range[0]), np.log(args.train_range[1]), args.train_variants)).tolist())
            for variant, target in enumerate(targets):
                path = folder / f"{row['dataset_id']}__v{variant}.npz"
                if path.exists():
                    with np.load(path) as archive:
                        index.setdefault(split, {}).setdefault(row['dataset_id'], []).append(
                            dict(file=path.name, density=float(archive['density'])))
                    continue
                keep, density = thin(full, target, rng)
                arrays = subset(full, keep)
                raw = predict(model, arrays, owner_only=True, raw_object_threshold=.05)
                labels, confidence, _, _ = merge_masks_with_sources(arrays, raw, protocol['stage1_config'])
                gt_ids, matched, _ = match_instances(arrays['tree_id'], labels)
                temporary = path.with_name(path.stem + '.tmp.npz')
                np.savez_compressed(temporary, keep=keep.astype(np.int32), density=np.float32(density),
                                    shifted_center=raw['shifted_center'].astype(np.float32),
                                    tree_probability=raw['tree_probability'].astype(np.float32),
                                    labels=labels.astype(np.int64), confidence=confidence.astype(np.float32),
                                    missed_gt_ids=gt_ids[matched == 0].astype(np.int64),
                                    checkpoint_sha256=np.asarray(protocol['stage1_sha256']),
                                    input_sha256=np.asarray(sha256(row['output'])))
                os.replace(temporary, path)
                index.setdefault(split, {}).setdefault(row['dataset_id'], []).append(dict(file=path.name, density=float(density)))
            print(f'{split} {number} {row["dataset_id"]}: densities {[round(v["density"], 1) for v in index[split][row["dataset_id"]]]}', flush=True)
    write_json(root / 'stage1_thinned' / 'index.json',
               dict(index=index, seed=args.seed, train_range=list(args.train_range), val_density=args.val_density,
                    method='uniform random voxel subsampling to a target count per occupied 1 m cell; no occlusion model'))


if __name__ == '__main__':
    main()
