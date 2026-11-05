#!/usr/bin/env python3
"""Stem feet for paired scans (``trunk_reconstruction`` manifests), as input to ``predict_stem_seeded.py --stems <dir>``.

    $PY scripts/predict_stem_field.py --manifest ../trunk_reconstruction/data/paired/brandenburg_dc3d/manifest.csv \
        --source stemfield --checkpoint outputs/stem_field_v1/runs/sf_v1/weights/best.pt --out <dir> --densities d14

Writes ``<out>/<scan>.csv`` (x, y in the scan's CRS, h, score) per scan. Sources:

- ``stemfield``: StemField peaks on the scan, voxelised at 0.25 m as the DualCrown3D datasets
  (``predict_stem_seeded.scan_arrays``); threshold = the checkpoint's selected one unless
  ``--threshold`` is given;
- ``instances``: one foot per instance of the scan's ``tree_id`` dimension (e.g. Model20's or v7's
  trees), from the instance's lowest points as in ``prepare_stem_feet.proxy_feet``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.paths import remap  # noqa: E402
from pointcloud.stem_field import StemFieldNet, detect_plot  # noqa: E402
from scripts.predict_stem_seeded import scan_arrays  # noqa: E402
from scripts.prepare_stem_feet import proxy_feet  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--source', choices=('stemfield', 'instances'), required=True)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--threshold', type=float, default=None)
    ap.add_argument('--calibration', type=Path, default=None,
                    help='calibration_bb.json (scripts/calibrate_stem_field.py): threshold and peak window')
    ap.add_argument('--densities', nargs='+', default=['d14'])
    ap.add_argument('--plots', nargs='*', default=None)
    args = ap.parse_args()
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest.density.isin(args.densities)]
    if args.plots:
        manifest = manifest[manifest['plot'].isin(args.plots)]
    model, threshold, kernel = None, args.threshold, 3
    if args.calibration is not None and args.calibration.exists():
        calibration = json.loads(args.calibration.read_text())
        threshold = calibration['threshold'] if threshold is None else threshold
        kernel = int(calibration['kernel'])
    if args.source == 'stemfield':
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model = StemFieldNet(width=payload.get('width', 24))
        model.load_state_dict(payload['model'])
        model = model.cuda().eval()
        threshold = float(payload.get('threshold', .3)) if threshold is None else threshold
    args.out.mkdir(parents=True, exist_ok=True)
    report = []
    for r in manifest.itertuples():
        path = Path(remap(r.path))
        las = laspy.read(path)
        if args.source == 'stemfield':
            arrays, _, _ = scan_arrays(las)
            local, scores = detect_plot(model, arrays['coord'].astype(np.float64), arrays['intensity'].astype(np.float64),
                                        threshold=threshold, kernel=kernel)
            frame = pd.DataFrame(dict(x=local[:, 0] + arrays['source_origin'][0], y=local[:, 1] + arrays['source_origin'][1],
                                      h=local[:, 2], score=scores))
        else:
            hag = np.asarray(las.height_agl, np.float64)
            coord = np.column_stack((las.x, las.y, hag))
            ids = np.asarray(las.tree_id, np.int64)
            ids[(np.asarray(las.classification) == 2) | (hag < .3)] = 0
            frame = pd.DataFrame(proxy_feet(coord, ids))[['x', 'y', 'h']].assign(score=1.)
        name = path.stem.removeprefix('trees_')
        frame.to_csv(args.out / f'{name}.csv', index=False)
        report.append(dict(scan=name, feet=len(frame)))
        print(f'{name}: {len(frame)} feet', flush=True)
    record = args.out / 'feet.json'
    runs = json.loads(record.read_text()).get('runs', []) if record.exists() else []
    runs.append(dict(source=args.source, checkpoint=str(args.checkpoint), threshold=threshold, kernel=kernel,
                     calibration=str(args.calibration), scans=report))
    record.write_text(json.dumps(dict(runs=runs), indent=1) + '\n')


if __name__ == '__main__':
    main()
