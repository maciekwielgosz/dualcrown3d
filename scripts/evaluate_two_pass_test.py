#!/usr/bin/env python3
"""Held-out test of the two-pass model against Model20, with LAS/LAZ and GeoPackage exports.

Uses the frozen stage-1 checkpoint/config and a stage-2 checkpoint together with
the configuration selected on validation. Nothing is tuned here. By default the
script refuses checkpoints that did not pass the preregistered validation gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import geopandas as gpd
import numpy as np
import shapely
import torch
from openpyxl import Workbook

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz, read_manifest
from pointcloud.dual_fusion import instance_records
from pointcloud.instance_output import merge_masks_with_sources
from pointcloud.vote_centers import VoteCenterNet, detect_centres, second_pass_labels, stage1_centres
from scripts.cache_stage1_predictions import DEFAULT_OUTPUT, sha256, write_json
from scripts.cache_stage1_thinned import subset, thin
from scripts.evaluate_output20_output22_test import write_cloud, write_vectors
from scripts.predict_dual_head import predict
from scripts.train_supervision_v4 import REAL, build
from scripts.train_two_pass_refiner import evaluate_plot, summarize
from scripts.train_vote_centers import choose_centres

KEYS = ('point_sb_pq', 'crown_sb_pq', 'point_sb_f1', 'crown_sb_f1', 'point_precision', 'crown_precision',
        'point_recall', 'crown_recall', 'small_10_tp', 'small_10_gt', 'small_4_tp', 'small_4_gt', 'large_recall',
        'oversplit_gt', 'undersegmented_pred', 'predicted_instances')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run', required=True)
    parser.add_argument('--weights', default='best.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', default='test', choices=('test', 'val'))
    parser.add_argument('--allow-unselected', action='store_true',
                        help='Evaluate a checkpoint that did not pass the validation gate (reported as such)')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--thin-density', type=float, default=0.,
                        help='Randomly thin every plot to this many voxels per occupied m2 before stage 1 (0 = full density)')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is required')
    torch.set_num_threads(4)
    root = args.root.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    weights = root / 'runs' / args.run / 'weights' / args.weights
    payload = torch.load(weights, map_location='cpu', weights_only=False)
    config = payload.get('selected_config')
    gate_passed = args.weights == 'best.pt' and config is not None
    if config is None:
        raise ValueError('Checkpoint carries no validation-selected configuration')
    if not gate_passed and not args.allow_unselected:
        raise ValueError('Checkpoint did not pass the validation gate; use --allow-unselected to evaluate it anyway')
    if sha256(protocol['stage1_checkpoint']) != protocol['stage1_sha256']:
        raise ValueError('Stage-1 checkpoint differs from the frozen protocol')
    rows = [r for r in read_manifest(REAL, args.split) if r.get('point_eval_eligible') == 'true']
    if args.limit:
        rows = rows[:args.limit]
    run = dict(split=args.split, plots=len(rows), manifest_sha256=sha256(REAL), stage1_sha256=protocol['stage1_sha256'],
               stage1_config=protocol['stage1_config'], stage2_weights=str(weights), stage2_sha256=sha256(weights),
               stage2_epoch=payload['epoch'], stage2_config=config, validation_gate_passed=gate_passed,
               thin_density_voxels_per_m2=args.thin_density,
               threshold_selection='validation only; nothing tuned on this split',
               las_scope='voxelized model-input cloud; XY source coordinates; Z = height above ground')
    signature = hashlib.sha256(json.dumps(run, sort_keys=True, default=str).encode()).hexdigest()
    output = args.output.resolve()
    if (output / 'comparison.json').exists():
        raise FileExistsError('Completed comparison is protected')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'run.json', dict(**run, signature=signature))
    model, _ = build(protocol['stage1_checkpoint'])
    model.eval()
    network = VoteCenterNet(width=payload['width'], dropout=payload.get('dropout', 0.)).cuda()
    network.load_state_dict(payload['network'])
    results = {'Model20': [], 'TwoPass': []}
    timing = dict(stage1_forward=0., stage1_merge=0., stage2_detect=0., stage2_assign=0.)
    for number, row in enumerate(rows, 1):
        arrays = load_npz(row['output'])
        if args.thin_density > 0:
            keep, _ = thin(arrays, args.thin_density, np.random.default_rng(np.random.SeedSequence([20261002, number])))
            arrays = subset(arrays, keep)
        gt = gpd.read_file(row['gt_vector'])
        ignore = shapely.from_wkb(Path(row['ignore_vector']).read_bytes()) if row.get('ignore_vector') else None
        raw = predict(model, arrays, owner_only=True, raw_object_threshold=.05)
        timing['stage1_forward'] += float(raw['seconds'])
        start = time.monotonic()
        labels1, confidence1, _, _ = merge_masks_with_sources(arrays, raw, protocol['stage1_config'])
        timing['stage1_merge'] += time.monotonic() - start
        start = time.monotonic()
        centres, scores = detect_centres(network, arrays, raw['shifted_center'], raw['tree_probability'],
                                         labels1.astype(np.int64), threshold=.1, flips=True)
        timing['stage2_detect'] += time.monotonic() - start
        start = time.monotonic()
        chosen = choose_centres(centres, scores, stage1_centres(raw['shifted_center'], labels1), config)
        labels2, confidence2, _, _ = second_pass_labels(arrays, raw, protocol['stage1_config'], chosen,
                                                        config['assign'], config['consensus'])
        timing['stage2_assign'] += time.monotonic() - start
        for name, labels, confidence in (('Model20', labels1, confidence1), ('TwoPass', labels2, confidence2)):
            results[name].append(evaluate_plot(row, arrays, gt, ignore, labels, confidence))
            folder = output / name
            write_cloud(folder / 'PointClouds' / f"trees_{row['dataset_id']}", arrays, labels, confidence,
                        raw['point_probability'], gt.crs)
            write_vectors(folder / 'Segmentation3', row['dataset_id'], instance_records(arrays, labels, confidence), gt.crs)
        a, b = results['Model20'][-1], results['TwoPass'][-1]
        print(f"{args.split} {number}/{len(rows)} {row['dataset_id']}: crown TP {a['tp']}->{b['tp']} FP {a['fp']}->{b['fp']} "
              f"small {a['small_10_tp']}->{b['small_10_tp']}/{b['small_10_gt']}", flush=True)
    summaries = {name: summarize(records) for name, records in results.items()}
    write_json(output / 'comparison.json', dict(**run, signature=signature, timing_seconds=timing, summaries=summaries,
                                                per_plot=results))
    book = Workbook()
    sheet = book.active
    sheet.title = 'summary'
    sheet.append(('model', *KEYS))
    for name, summary in summaries.items():
        sheet.append((name, *[summary[k] for k in KEYS]))
    for name, records in results.items():
        tab = book.create_sheet(f'{name}_per_plot')
        tab.append(('dataset_id', 'collection', 'crown_tp', 'crown_fp', 'crown_fn', 'point_tp', 'point_fp', 'point_fn',
                    'small_10_tp', 'small_10_gt', 'oversplit_gt', 'undersegmented_pred', 'predicted_instances'))
        for r in records:
            tab.append((r['dataset_id'], r['collection'], r['tp'], r['fp'], r['fn'], r['point_metrics']['tp'],
                        r['point_metrics']['fp'], r['point_metrics']['fn'], r['small_10_tp'], r['small_10_gt'],
                        r['diagnostics']['oversplit_gt'], r['diagnostics']['undersegmented_pred'], r['predicted_instances']))
        tab.freeze_panes = 'A2'
    book.save(output / 'comparison.xlsx')
    (output / 'README.md').write_text(
        '# Model20 versus two-pass vote-centre model\n\n'
        f'Split: {args.split}, {len(rows)} labelled plots'
        + (f', randomly thinned to {args.thin_density:g} voxels per occupied m2 before inference' if args.thin_density > 0 else '')
        + '. Stage 1 is the frozen Model20; stage 2 detects tree centres in '
        'its vote space and reassigns votes. Thresholds were selected on validation only. '
        f'Validation gate passed: {gate_passed}.\n\n'
        'Model20/ and TwoPass/ each contain PointClouds (LAS and LAZ of the voxelized model-input cloud; XY in source '
        'coordinates, Z = height above ground, tree_id = prediction, reference_tree_id = ground truth for viewing only) '
        'and Segmentation3 (crowns_*.gpkg, ttops_*.gpkg). See comparison.xlsx and comparison.json for metrics.\n')
    for name, summary in summaries.items():
        print(name, {k: round(summary[k], 4) if isinstance(summary[k], float) else summary[k] for k in KEYS}, flush=True)
    print('timing', {k: round(v, 1) for k, v in timing.items()})


if __name__ == '__main__':
    main()
