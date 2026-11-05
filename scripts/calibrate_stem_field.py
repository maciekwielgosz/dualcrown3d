#!/usr/bin/env python3
"""Brandenburg threshold and peak spacing of a StemField detector, chosen on the detector's own training fold.

    $PY scripts/calibrate_stem_field.py --checkpoint outputs/stem_field_v1/runs/sf_v2A/weights/best.pt --fold A

On real leaf-off Brandenburg ALS the detector finds the measured stems but with low scores (median
0.19 against 0.3 elsewhere; the TLS stems lie about 0.9 m from the detections), so the threshold chosen
on the other val plots discards nearly all of them. This sweeps ``--thresholds`` x ``--kernels`` (peak
suppression window, cells of 0.5 m) on the d14 scans (3 seeds) of the Brandenburg plots the detector
was trained with (``--fold``) and keeps the pair with the best pooled F1 (measured canopy stems inside
the plot box; detections in the buffer or within 1.5 m of an understorey stem are not false
positives). The other fold is never read, so it stays a clean end-to-end test. Writes
``calibration_bb.json`` next to the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz, read_manifest  # noqa: E402
from pointcloud.stem_data import read_stems_ignore  # noqa: E402
from pointcloud.stem_field import annotated_area, detect_plot, match_feet  # noqa: E402
from scripts.evaluate_stem_seeded_feet import load_stem_field  # noqa: E402
from scripts.train_stem_field import BRANDENBURG, IGNORE_RADIUS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--fold', choices=('A', 'B'), required=True, help='the Brandenburg fold the detector was trained with')
    ap.add_argument('--thresholds', type=float, nargs='+', default=[.08, .1, .12, .14, .16, .18, .2, .25, .3])
    ap.add_argument('--kernels', type=int, nargs='+', default=[3, 5, 7])
    args = ap.parse_args()
    torch.set_num_threads(4)
    model, selected = load_stem_field(args.checkpoint)
    rows = [r for r in read_manifest(BRANDENBURG, 'train') if r['fold'] == args.fold and r['density'] == 'd14']
    counts = {(t, k): np.zeros(3, np.int64) for t in args.thresholds for k in args.kernels}
    distances = {key: [] for key in counts}
    for row in rows:
        arrays = load_npz(row['output'])
        shift = arrays['source_origin'][:2]
        labelled, _, _, flagged = read_stems_ignore(row)
        truth = np.column_stack((labelled[~flagged, :2] - shift, labelled[~flagged, 2]))
        area = annotated_area(arrays['coord'], arrays['tree_id'])
        hidden = labelled[flagged, :2] - shift
        near = cKDTree(hidden) if len(hidden) else None

        def excuse(xy):
            out = area(xy)
            if near is not None:
                out |= near.query(np.asarray(xy)[:, :2])[0] <= IGNORE_RADIUS
            return out
        for k in args.kernels:
            feet, scores = detect_plot(model, arrays['coord'].astype(np.float64), arrays['intensity'].astype(np.float64),
                                       threshold=min(args.thresholds), kernel=k)
            for t in args.thresholds:
                m = match_feet(feet[scores >= t], truth, ignore_xy=excuse)
                counts[(t, k)] += (m['tp'], m['fp'], m['fn'])
                distances[(t, k)].extend(m['distances'].tolist())
        print(row['dataset_id'], 'done', flush=True)
    table = []
    for (t, k), (tp, fp, fn) in counts.items():
        table.append(dict(threshold=t, kernel=k, tp=int(tp), fp=int(fp), fn=int(fn), recall=tp / max(tp + fn, 1),
                          precision=tp / max(tp + fp, 1), f1=2 * tp / max(2 * tp + fp + fn, 1),
                          median_distance_m=float(np.median(distances[(t, k)])) if distances[(t, k)] else None))
    best = max(table, key=lambda r: r['f1'])
    out = dict(checkpoint=str(args.checkpoint), fold=args.fold, scans=[r['dataset_id'] for r in rows],
               selected_threshold_val_plots=selected, threshold=best['threshold'], kernel=best['kernel'], best=best,
               table=sorted(table, key=lambda r: -r['f1']))
    path = args.checkpoint.parent / 'calibration_bb.json'
    path.write_text(json.dumps(out, indent=1) + '\n')
    for r in out['table'][:8]:
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    print(f'-> {path}: threshold {best["threshold"]}, kernel {best["kernel"]}')


if __name__ == '__main__':
    main()
