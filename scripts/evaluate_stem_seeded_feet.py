#!/usr/bin/env python3
"""Stem feet from several sources on the val plots: detection quality, and the stem-seeded decoder on them.

    $PY scripts/evaluate_stem_seeded_feet.py --checkpoint outputs/stem_seeded_v1/runs/ss_v2/weights/best.pt \
        --feet-cache outputs/stem_seeded_v1/feet_cache --stem-field outputs/stem_field_v1/runs/sf_v1/weights/best.pt \
        --out outputs/stem_seeded_v1/runs/ss_v2/feet_eval_all

Sources (``--sources``):

- ``labels``: the labelled feet of ``stems.csv`` (the ceiling with perfect feet);
- ``model20``, ``v7``, ``v7fb``: one foot per instance of Model20, of two-pass v7 and of v7 with the
  coverage fallback, read from ``--feet-cache`` (``scripts/cache_feet_sources.py``; the cache also
  holds those segmentations, scored here as they are);
- ``stemfield``: StemField peaks (``--stem-field`` checkpoint, ``--stem-field-threshold``).

For each source: (1) feet against the labelled feet (one-to-one within 1.5 m; detections in mostly
unannotated vegetation are not false positives): recall, precision, F1, median distance;
(2) point PQ of the source's own segmentation where it has one; (3) point PQ of the stem-seeded
decoder on the source's feet, at each ``--object-thresholds``. Metric: source-balanced point PQ at
IoU 0.5 (Model20 protocol) and pooled recall / precision. Val splits only.
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
from pointcloud.dual_head import DualHeadLitePT  # noqa: E402
from pointcloud.joint_training import fixed_serialization  # noqa: E402
from pointcloud.stem_data import read_stems, stem_group  # noqa: E402
from pointcloud.stem_field import StemFieldNet, annotated_area, detect_plot, match_feet  # noqa: E402
from pointcloud.stem_seeded import predict_stems  # noqa: E402
from scripts.evaluate_combined_full_crowns import aggregate  # noqa: E402
from scripts.train_stem_seeded import metric_row, validation_rows  # noqa: E402

CACHED = ('model20', 'v7', 'v7fb')


def load_stem_seeded(checkpoint):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    kwargs = dict(payload.get('model_args', {}))
    kwargs.pop('decoder_policy', None)
    model = DualHeadLitePT(decoder_policy='stem_seeded', **kwargs)
    model.load_state_dict(payload['model'], strict=True)
    fixed_serialization(model)
    return model.cuda().eval()


def load_stem_field(checkpoint):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = StemFieldNet(width=payload.get('width', 24))
    model.load_state_dict(payload['model'])
    return model.cuda().eval(), float(payload.get('threshold', .3))


def pq_table(rows):
    if not rows:
        return {}
    out = {}
    for g in sorted({r['group'] for r in rows}) + ['all']:
        sub = rows if g == 'all' else [r for r in rows if r['group'] == g]
        a = aggregate(sub)
        out[g] = dict(source_balanced_pq=a['source_balanced_pq'], recall=a['recall'], precision=a['precision'],
                      tp=a['tp'], fp=a['fp'], fn=a['fn'])
    return out


def detection_table(rows):
    out = {}
    for g in sorted({r['group'] for r in rows}) + ['all']:
        sub = rows if g == 'all' else [r for r in rows if r['group'] == g]
        tp, fp, fn = (sum(r[k] for r in sub) for k in ('tp', 'fp', 'fn'))
        d = np.concatenate([r['distances'] for r in sub]) if sub else np.zeros(0)
        out[g] = dict(recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1), f1=2 * tp / max(2 * tp + fp + fn, 1),
                      median_distance_m=float(np.median(d)) if len(d) else None, tp=tp, fp=fp, fn=fn)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', type=Path, required=True, help='stem-seeded decoder')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sources', nargs='+', default=['labels', 'model20', 'v7', 'v7fb', 'stemfield'])
    ap.add_argument('--feet-cache', type=Path, default=PROJECT / 'outputs/stem_seeded_v1/feet_cache')
    ap.add_argument('--stem-field', type=Path, default=None)
    ap.add_argument('--stem-field-threshold', type=float, default=None, help='default: the threshold selected in training')
    ap.add_argument('--object-thresholds', type=float, nargs='+', default=[0., .5])
    ap.add_argument('--max-points', type=int, default=40000)
    ap.add_argument('--limit', type=int, default=None, help='first N val plots per group (quick checks)')
    args = ap.parse_args()
    torch.set_num_threads(4)
    sources = [s for s in args.sources if s != 'stemfield' or args.stem_field is not None]
    rows = validation_rows()
    if args.limit:
        rows = [r for g in sorted({stem_group(x) for x in rows}) for r in [x for x in rows if stem_group(x) == g][:args.limit]]
    stem = load_stem_seeded(args.checkpoint)
    field = None
    if 'stemfield' in sources:
        field, selected = load_stem_field(args.stem_field)
        if args.stem_field_threshold is None:
            args.stem_field_threshold = selected
    seg = {s: [] for s in sources if s in CACHED}
    detect = {s: [] for s in sources if s != 'labels'}
    seeded = {f'{s}_t{t:g}': [] for s in sources for t in args.object_thresholds}
    started = time.monotonic()
    for number, row in enumerate(rows, 1):
        arrays = load_npz(row['output'])
        truth = arrays['tree_id']
        origin = arrays['source_origin'][:2]
        labelled, _, _ = read_stems(row)
        excuse = annotated_area(arrays['coord'], truth)
        world_excuse = lambda xy: excuse(np.asarray(xy) - origin)  # noqa: E731
        feet = {}
        for s in sources:
            if s == 'labels':
                feet[s] = labelled
            elif s in CACHED:
                feet[s] = np.load(args.feet_cache / s / f"{row['dataset_id']}.npy")
                seg[s].append(metric_row(row, truth, np.load(args.feet_cache / s / f"{row['dataset_id']}_labels.npy")))
            elif s == 'stemfield':
                local, _ = detect_plot(field, arrays['coord'], arrays['intensity'], threshold=args.stem_field_threshold)
                feet[s] = np.column_stack((local[:, :2] + origin, local[:, 2]))
            if s != 'labels':
                m = match_feet(feet[s], labelled, ignore_xy=world_excuse)
                detect[s].append(dict(group=stem_group(row), tp=m['tp'], fp=m['fp'], fn=m['fn'], distances=m['distances']))
            for t in args.object_thresholds:
                labels, _, _ = predict_stems(stem, arrays, feet[s], max_points=args.max_points, object_threshold=t)
                seeded[f'{s}_t{t:g}'].append(metric_row(row, truth, labels))
        print(f"{number}/{len(rows)} {row['dataset_id']}: feet " + ', '.join(f'{s} {len(f)}' for s, f in feet.items())
              + f'; {time.monotonic() - started:.0f}s', flush=True)
    summary = dict(segmentation={s: pq_table(r) for s, r in seg.items()},
                   detection={s: detection_table(r) for s, r in detect.items()},
                   stem_seeded={s: pq_table(r) for s, r in seeded.items()})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'feet_eval.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint), stem_field=str(args.stem_field),
                                                             stem_field_threshold=args.stem_field_threshold, summary=summary),
                                                        indent=2, default=float) + '\n')
    groups = list(next(iter(summary['stem_seeded'].values())).keys())
    lines = ['## Point PQ (source-balanced, IoU 0.5)', '', '| method | ' + ' | '.join(groups) + ' | recall | precision |',
             '|---' * (len(groups) + 3) + '|']
    for title, table in [(f'{s} (own segmentation)', t) for s, t in summary['segmentation'].items()] + \
                        [(f'stem-seeded on {s}', t) for s, t in summary['stem_seeded'].items()]:
        lines.append(f'| {title} | ' + ' | '.join(f"{table[g]['source_balanced_pq']:.3f}" for g in groups)
                     + f" | {table['all']['recall']:.3f} | {table['all']['precision']:.3f} |")
    lines += ['', '## Feet against labelled feet (1.5 m)', '', '| source | group | recall | precision | F1 | median distance m |',
              '|---|---|---:|---:|---:|---:|']
    for s, table in summary['detection'].items():
        for g, v in table.items():
            dist = '-' if v['median_distance_m'] is None else f"{v['median_distance_m']:.2f}"
            lines.append(f"| {s} | {g} | {v['recall']:.3f} | {v['precision']:.3f} | {v['f1']:.3f} | {dist} |")
    (args.out / 'feet_eval.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
