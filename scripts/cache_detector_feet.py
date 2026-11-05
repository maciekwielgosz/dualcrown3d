#!/usr/bin/env python3
"""StemField's feet on every train and val plot, as realistic queries for training the stem-seeded decoder.

    $PY scripts/cache_detector_feet.py --checkpoint outputs/stem_field_v1/runs/sf_v1/weights/best.pt --name sf_v1

Per plot of the stem-seeded training and validation sets, writes ``stems_<name>.csv`` next to its
``points_0p25m.npz``: x, y (frame of ``coord + source_origin``), h (predicted tree height), score, and
``tree_id`` = the labelled tree the foot is matched to (one-to-one within 1.5 m), 0 = a false foot in
the annotated area, -1 = an unmatched foot in mostly unannotated vegetation (unknown). For the val
plots the feet also go to ``<feet-cache>/stemfield_<name>/<dataset_id>.npy`` (x, y, h), the format of
``cache_feet_sources.py``. On the training plots the detector has seen the data, so its feet are better
there than on new plots; the decoder adds noise on top (``StemDrawDataset``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz  # noqa: E402
from pointcloud.stem_data import read_stems, stem_group, stems_path  # noqa: E402
from pointcloud.stem_field import annotated_area, detect_plot, match_feet  # noqa: E402
from scripts.evaluate_stem_seeded_feet import load_stem_field  # noqa: E402
from scripts.train_stem_seeded import parse_weights, training_rows, validation_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--threshold', type=float, default=None, help='default: the threshold selected in training')
    ap.add_argument('--feet-cache', type=Path, default=PROJECT / 'outputs/stem_seeded_v1/feet_cache')
    ap.add_argument('--sources', default='for_instance=1,helios=1,weiser_real=1,weiser_sim=0.5')
    args = ap.parse_args()
    torch.set_num_threads(4)
    model, selected = load_stem_field(args.checkpoint)
    threshold = selected if args.threshold is None else args.threshold
    rows = training_rows(parse_weights(args.sources), True)
    val = validation_rows()
    cache = args.feet_cache / f'stemfield_{args.name}'
    cache.mkdir(parents=True, exist_ok=True)
    stats, started = [], time.monotonic()
    for number, row in enumerate(rows + val, 1):
        target = stems_path(row).with_name(f'stems_{args.name}.csv')
        if target.exists() and (row['model_split'] != 'val' or (cache / f"{row['dataset_id']}.npy").exists()):
            continue
        arrays = load_npz(row['output'])
        origin = arrays['source_origin'][:2]
        local, scores = detect_plot(model, arrays['coord'].astype(np.float64), arrays['intensity'].astype(np.float64),
                                    threshold=threshold)
        world = np.column_stack((local[:, :2] + origin, local[:, 2]))
        labelled, ids, _ = read_stems(row)
        area = annotated_area(arrays['coord'], arrays['tree_id'])
        m = match_feet(world, labelled)
        tree = np.where(area(local[:, :2]), -1, 0).astype(np.int64)
        tree[m['matched_pred']] = ids[m['matched_truth']]
        pd.DataFrame(dict(x=world[:, 0], y=world[:, 1], h=world[:, 2], score=scores, tree_id=tree)).to_csv(target, index=False)
        if row['model_split'] == 'val':
            np.save(cache / f"{row['dataset_id']}.npy", world)
        stats.append(dict(dataset_id=row['dataset_id'], split=row['model_split'], group=stem_group(row), feet=int(len(world)),
                          labelled=int(len(labelled)), matched=int(m['tp']), false=int((tree == 0).sum()),
                          unknown=int((tree == -1).sum())))
        if number % 25 == 0 or row['model_split'] == 'val':
            print(f"{number}/{len(rows) + len(val)} {row['dataset_id']}: {stats[-1]} {time.monotonic() - started:.0f}s", flush=True)
    frame = pd.DataFrame(stats)
    if len(frame):
        summary = frame.groupby(['split', 'group'])[['feet', 'labelled', 'matched', 'false', 'unknown']].sum()
        summary['recall'] = summary.matched / summary.labelled.clip(lower=1)
        print(summary.to_string())
        (cache / f'report_{args.name}.json').write_text(json.dumps(dict(threshold=threshold, checkpoint=str(args.checkpoint),
                                                                        plots=stats), indent=1) + '\n')


if __name__ == '__main__':
    main()
