#!/usr/bin/env python3
"""Stem feet from existing segmentations on the val plots, for the stem-seeded decoder and StemField.

    $PY scripts/cache_feet_sources.py [--out outputs/stem_seeded_v1/feet_cache]

For every val plot of ``train_stem_seeded.validation_rows`` (supervision-v4 point-eval plots,
HELIOS val, Weiser BR02/BR08 d14 s0) and every source, one foot per predicted instance (mean xy of
its lowest metre above 0.5 m, else of its points below 0.3 h, else its centroid; height = its top),
saved as ``<out>/<source>/<dataset_id>.npy`` (world x, y, h) with the instance labels in
``<dataset_id>_labels.npy``:

- ``model20``: Model20 with its production consensus;
- ``v7``: the two-pass v7 candidate ``ens4_b`` (4-rotation vote ensemble, 4 vote-field detectors,
  centre verifier, configuration selected on validation);
- ``v7fb``: v7 with the coverage fallback 0.7 (a Model20 tree is kept where v7 claimed less than
  30 % of its voxels), the variant used on the trunk plots.

Val splits only; existing files are kept.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz  # noqa: E402
from pointcloud.instance_output import merge_masks_with_sources  # noqa: E402
from pointcloud.vote_ensemble import ensemble_votes  # noqa: E402
from pointcloud.vote_field import fallback_to_stage1, second_pass  # noqa: E402
from scripts.build_two_pass_pipeline import load_assign_network  # noqa: E402
from scripts.predict_dual_head import predict  # noqa: E402
from scripts.prepare_stem_feet import proxy_feet  # noqa: E402
from scripts.train_centre_verifier import load_verifier  # noqa: E402
from scripts.train_stem_seeded import MODEL20, MODEL20_SELECTION, validation_rows  # noqa: E402
from scripts.train_supervision_v4 import build  # noqa: E402
from scripts.validate_vote_field import load_network  # noqa: E402

V7 = PROJECT / 'outputs/dualcrown3d_two_pass_v1/runs/ens4_b/weights/best.pt'


def feet_of(arrays, labels):
    feet = proxy_feet(arrays['coord'] + arrays['source_origin'], np.asarray(labels, np.int64))
    return np.array([[f['x'], f['y'], f['h']] for f in feet], np.float64).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=PROJECT / 'outputs/stem_seeded_v1/feet_cache')
    ap.add_argument('--fallback', type=float, default=.7)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(4)
    rows = validation_rows()[:args.limit] if args.limit else validation_rows()
    for source in ('model20', 'v7', 'v7fb'):
        (args.out / source).mkdir(parents=True, exist_ok=True)
    config1 = json.loads(MODEL20_SELECTION.read_text())['config']
    model, _ = build(MODEL20)
    model.eval()
    payload = torch.load(V7, map_location='cpu', weights_only=False)
    config2 = payload['selected_config']
    network, verifier, assign = load_network(payload), load_verifier(payload), load_assign_network(payload)
    symmetries = int(payload.get('symmetries', 4))
    report = []
    for number, row in enumerate(rows, 1):
        name = row['dataset_id']
        if all((args.out / s / f'{name}.npy').exists() for s in ('model20', 'v7', 'v7fb')):
            continue
        start = time.monotonic()
        arrays = load_npz(row['output'])
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(7001)
            raw = predict(model, arrays, owner_only=True, raw_object_threshold=.05)
            labels1, _, _, _ = merge_masks_with_sources(arrays, raw, config1)
            ensemble = ensemble_votes(model, arrays, symmetries)
            labels2, _, source2, info = second_pass(network, arrays, ensemble, labels1, config2,
                                                    verifier=verifier, assign_network=assign)
        labels3, _, fallback = fallback_to_stage1(labels2, source2, labels1, args.fallback)
        out = {}
        for source, labels in (('model20', labels1), ('v7', labels2), ('v7fb', labels3)):
            feet = feet_of(arrays, labels)
            np.save(args.out / source / f'{name}.npy', feet)
            np.save(args.out / source / f'{name}_labels.npy', np.asarray(labels, np.int32))
            out[source] = len(feet)
        report.append(dict(dataset_id=name, **out, fallback_instances=int(fallback), seconds=time.monotonic() - start))
        print(f"{number}/{len(rows)} {name}: feet {out}, {time.monotonic() - start:.0f}s", flush=True)
    (args.out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
