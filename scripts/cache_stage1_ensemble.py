#!/usr/bin/env python3
"""Cache rotation-ensembled Model20 votes for the v7 vote-field detector.

Three regimes, each self-contained per file (labels are Model20's unchanged
single-pass consensus; only the votes handed to stage 2 are ensembled):

- ``dense``   annotated plots at native density (labels reused from ``stage1/``);
- ``thinned`` the same plots randomly thinned (existing ``stage1_thinned`` draws are
  reused so earlier thinned-validation numbers stay comparable; extra train
  draws are added);
- ``helios``  TreeScanPL10k plots simulated with HELIOS++ at about 14 points/m2;
- ``weiser``  Weiser 2022 real sparse scans and simulated tiles (``prepare_weiser_dualcrown.py``),
  not part of the v7 protocol; opt in with ``--regimes weiser``.

The held-out test splits are not touched here.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz, read_manifest
from pointcloud.instance_output import merge_masks_with_sources
from pointcloud.two_pass import match_instances
from pointcloud.vote_ensemble import ensemble_votes
from scripts.cache_stage1_predictions import DEFAULT_OUTPUT, eligible_rows, sha256, write_json
from scripts.cache_stage1_thinned import subset, thin
from scripts.predict_dual_head import predict
from scripts.train_supervision_v4 import build

HELIOS = PROJECT.parent / 'dualcrown3d_treescan_helios_v1/manifest.csv'
HELIOS_FLIGHTS = PROJECT.parent / 'TreeScanPL10k_HELIOS_ALS_flight_augmentation_v2/manifest.csv'
WEISER = PROJECT.parent / 'dualcrown3d_weiser_v1/manifest.csv'


def helios_rows(split, flights=True):
    rows = read_manifest(HELIOS, split)
    if split == 'train' and flights and HELIOS_FLIGHTS.exists():
        rows = rows + read_manifest(HELIOS_FLIGHTS, 'train')
    return rows


def weiser_rows(split):
    """Weiser 2022 real scans and simulated tiles (``prepare_weiser_dualcrown.py``); the held-out
    plots are the ``test`` split and are never cached here."""
    return read_manifest(WEISER, split) if WEISER.exists() else []


def save(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def stage1(model, arrays, config):
    """Unchanged Model20: single-pass raw prediction and consensus labels."""
    raw = predict(model, arrays, owner_only=True, raw_object_threshold=.05)
    labels, confidence, _, _ = merge_masks_with_sources(arrays, raw, config)
    return raw, labels, confidence


def pack(arrays, labels, confidence, ensemble, **extra):
    gt_ids, matched, _ = match_instances(arrays['tree_id'], labels)
    return dict(labels=np.asarray(labels, np.int64), confidence=np.asarray(confidence, np.float32),
                missed_gt_ids=gt_ids[matched == 0].astype(np.int64),
                ensemble_center=ensemble['shifted_center'], ensemble_probability=ensemble['tree_probability'],
                vote_spread=ensemble['vote_spread'].astype(np.float16), **extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--symmetries', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--extra-train-variants', type=int, default=4)
    parser.add_argument('--extra-range', type=float, nargs=2, default=(5., 80.))
    parser.add_argument('--extra-val-density', type=float, nargs='*', default=[25.])
    parser.add_argument('--regimes', nargs='+', default=['dense', 'thinned', 'helios'])
    parser.add_argument('--splits', nargs='+', default=['val', 'train'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.output.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    if sha256(protocol['stage1_checkpoint']) != protocol['stage1_sha256']:
        raise ValueError('Stage-1 checkpoint differs from the frozen protocol')
    config = protocol['stage1_config']
    from pointcloud.paths import remap
    model, _ = build(remap(protocol['stage1_checkpoint']))  # protocol paths are the workstation's
    model.eval()
    target = root / 'stage1_ensemble'
    index_path = target / 'index.json'
    index = json.loads(index_path.read_text())['index'] if index_path.exists() else {}
    old_thin = json.loads((root / 'stage1_thinned/index.json').read_text())['index']
    started = time.monotonic()

    def register(regime, split, dataset_id, entry):
        entries = index.setdefault(regime, {}).setdefault(split, {}).setdefault(dataset_id, [])
        if not any(e['file'] == entry['file'] for e in entries):
            entries.append(entry)

    def flush():
        write_json(index_path, dict(index=index, symmetries=args.symmetries, seed=args.seed,
                                    stage1_sha256=protocol['stage1_sha256'],
                                    extra_range=list(args.extra_range),
                                    note='labels: unchanged single-pass Model20 consensus; ensemble_center: votes '
                                         'averaged over lattice rotations; thinning: uniform random voxel subset'))

    for split in args.splits:
        rows = eligible_rows(split)
        for number, row in enumerate(rows, 1):
            name = row['dataset_id']
            full = None
            if 'dense' in args.regimes:
                path = target / 'dense' / split / f'{name}.npz'
                if not path.exists():
                    full = load_npz(row['output'])
                    with np.load(root / 'stage1' / split / f'{name}.npz') as archive:
                        labels, confidence = archive['labels'], archive['confidence']
                    save(path, **pack(full, labels, confidence, ensemble_votes(model, full, args.symmetries)))
                register('dense', split, name, dict(file=path.name, density=None))
            if 'thinned' in args.regimes:
                for entry in old_thin[split][name]:        # earlier draws: keep, labels and single votes reused
                    path = target / 'thinned' / split / entry['file']
                    if not path.exists():
                        full = full or load_npz(row['output'])
                        with np.load(root / 'stage1_thinned' / split / entry['file']) as archive:
                            keep, labels, confidence = archive['keep'], archive['labels'], archive['confidence']
                        arrays = subset(full, keep)
                        save(path, keep=keep.astype(np.int32), density=np.float32(entry['density']),
                             **pack(arrays, labels, confidence, ensemble_votes(model, arrays, args.symmetries)))
                    register('thinned', split, name, dict(file=path.name, density=float(entry['density'])))
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, number, split == 'train']))
                densities = (np.exp(rng.uniform(*np.log(args.extra_range), args.extra_train_variants)).tolist()
                             if split == 'train' else list(args.extra_val_density))
                for variant, density in enumerate(densities):
                    path = target / 'thinned' / split / f'{name}__e{variant}.npz'
                    if path.exists():
                        with np.load(path) as archive:
                            actual = float(archive['density'])
                    else:
                        full = full or load_npz(row['output'])
                        keep, actual = thin(full, density, rng)
                        arrays = subset(full, keep)
                        _, labels, confidence = stage1(model, arrays, config)
                        save(path, keep=keep.astype(np.int32), density=np.float32(actual),
                             **pack(arrays, labels, confidence, ensemble_votes(model, arrays, args.symmetries)))
                    register('thinned', split, name, dict(file=path.name, density=float(actual)))
            if number % 5 == 0 or number == len(rows):
                flush()
                print(f'{split} real {number}/{len(rows)} {name} {time.monotonic() - started:.0f}s', flush=True)
        for regime, rows_of in (('helios', helios_rows), ('weiser', weiser_rows)):
            if regime not in args.regimes:
                continue
            rows = rows_of(split)
            for number, row in enumerate(rows, 1):
                name = row['dataset_id']
                path = target / regime / split / f'{name}.npz'
                if not path.exists():
                    arrays = load_npz(row['output'])
                    _, labels, confidence = stage1(model, arrays, config)
                    save(path, **pack(arrays, labels, confidence, ensemble_votes(model, arrays, args.symmetries)))
                register(regime, split, name, dict(file=path.name, density=None))
                if number % 20 == 0 or number == len(rows):
                    flush()
                    print(f'{split} {regime} {number}/{len(rows)} {name} {time.monotonic() - started:.0f}s', flush=True)
    flush()
    print(json.dumps({regime: {split: sum(len(v) for v in plots.values()) for split, plots in splits.items()}
                      for regime, splits in index.items()}), flush=True)


if __name__ == '__main__':
    main()
