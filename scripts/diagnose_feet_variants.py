#!/usr/bin/env python3
"""Why do StemField feet segment worse than labelled feet although they lie 0.3 m from them?

    $PY scripts/diagnose_feet_variants.py --checkpoint <stem-seeded best.pt> --stem-field <StemField best.pt> --out <dir>

On the HELIOS val plots (fully annotated), the stem-seeded decoder with feet that differ in one
property at a time: labelled xy + labelled h; labelled xy + StemField h (nearest detection within
1.5 m); StemField xy + labelled h (nearest label within 1.5 m, else the predicted h); StemField xy +
StemField h; StemField xy + unknown h (0); StemField feet matched to a label only (no false feet).
Reports source-balanced point PQ, recall, precision and the share of tree points left unassigned.
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
from pointcloud.stem_data import read_stems  # noqa: E402
from pointcloud.stem_field import detect_plot, match_feet  # noqa: E402
from pointcloud.stem_seeded import predict_stems  # noqa: E402
from scripts.evaluate_combined_full_crowns import aggregate  # noqa: E402
from scripts.evaluate_stem_seeded_feet import load_stem_field, load_stem_seeded  # noqa: E402
from scripts.train_stem_seeded import HELIOS, metric_row  # noqa: E402


def nearest(values_from, xy_to, xy_from, radius=1.5):
    """For each point of xy_to, the value of the nearest xy_from within radius (nan if none)."""
    if not len(xy_from):
        return np.full(len(xy_to), np.nan)
    d, i = cKDTree(xy_from).query(xy_to)
    return np.where(d <= radius, values_from[np.minimum(i, len(values_from) - 1)], np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--stem-field', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    stem = load_stem_seeded(args.checkpoint)
    field, threshold = load_stem_field(args.stem_field)
    rows = read_manifest(HELIOS, 'val')
    names = ['label_xy_label_h', 'label_xy_sf_h', 'sf_xy_label_h', 'sf_xy_sf_h', 'sf_xy_unknown_h', 'sf_matched_only']
    results = {n: [] for n in names}
    unassigned = {n: [] for n in names}
    heights = []
    for row in rows:
        arrays = load_npz(row['output'])
        truth = arrays['tree_id']
        origin = arrays['source_origin'][:2]
        labels, _, _ = read_stems(row)
        local, _ = detect_plot(field, arrays['coord'].astype(np.float64), arrays['intensity'].astype(np.float64), threshold=threshold)
        sf = np.column_stack((local[:, :2] + origin, local[:, 2]))
        m = match_feet(sf, labels)
        matched_pred = np.zeros(len(sf), bool)
        if m['tp']:
            d = np.linalg.norm(sf[:, None, :2] - labels[None, m['matched_truth'], :2], axis=2)
            matched_pred[d.argmin(0)] = True
        label_h_at_sf = nearest(labels[:, 2], sf[:, :2], labels[:, :2])
        sf_h_at_label = nearest(sf[:, 2], labels[:, :2], sf[:, :2])
        heights += [dict(label=float(a), stemfield=float(b)) for a, b in zip(labels[:, 2], sf_h_at_label) if np.isfinite(b)]
        variants = {
            'label_xy_label_h': labels,
            'label_xy_sf_h': np.column_stack((labels[:, :2], np.where(np.isfinite(sf_h_at_label), sf_h_at_label, labels[:, 2]))),
            'sf_xy_label_h': np.column_stack((sf[:, :2], np.where(np.isfinite(label_h_at_sf), label_h_at_sf, sf[:, 2]))),
            'sf_xy_sf_h': sf,
            'sf_xy_unknown_h': np.column_stack((sf[:, :2], np.zeros(len(sf)))),
            'sf_matched_only': sf[matched_pred],
        }
        tree = truth > 0
        for n, feet in variants.items():
            lab, _, _ = predict_stems(stem, arrays, feet, max_points=40000)
            results[n].append(metric_row(row, truth, lab))
            unassigned[n].append(float((lab[tree] == 0).mean()))
        print(row['dataset_id'], {n: f"{results[n][-1]['tp']}/{results[n][-1]['fp']}/{results[n][-1]['fn']}" for n in names}, flush=True)
    h = np.array([[x['label'], x['stemfield']] for x in heights])
    summary = {n: dict(pq=aggregate(results[n])['source_balanced_pq'], recall=aggregate(results[n])['recall'],
                       precision=aggregate(results[n])['precision'], unassigned_tree_points=float(np.mean(unassigned[n])))
               for n in names}
    summary['height'] = dict(n=int(len(h)), label_median=float(np.median(h[:, 0])), stemfield_median=float(np.median(h[:, 1])),
                             mean_ratio=float(np.mean(h[:, 1] / np.maximum(h[:, 0], 1))),
                             mae=float(np.mean(np.abs(h[:, 1] - h[:, 0]))))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'diagnose.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
