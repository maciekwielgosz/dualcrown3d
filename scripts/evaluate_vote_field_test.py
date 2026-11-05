#!/usr/bin/env python3
"""Held-out test of the v7 two-pass model (vote-field detector) against Model20.

Three suites: ``dense`` (24 native test plots), ``thin10`` (the same plots thinned
to 10 voxels/m2 with the draw used for output_30) and ``helios`` (7 simulated test
plots). Stage 1 is the frozen Model20; the stage-2 checkpoint and configuration
must be the ones selected on validation. Nothing is tuned here.
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
from pointcloud.vote_ensemble import ensemble_votes
from pointcloud.vote_field import second_pass
from scripts.cache_stage1_ensemble import HELIOS
from scripts.cache_stage1_predictions import DEFAULT_OUTPUT, sha256, write_json
from scripts.cache_stage1_thinned import subset, thin
from scripts.evaluate_output20_output22_test import write_cloud, write_vectors
from scripts.evaluate_two_pass_test import KEYS
from scripts.predict_dual_head import predict
from scripts.train_supervision_v4 import REAL, build
from scripts.train_two_pass_refiner import evaluate_plot, summarize
from scripts.build_two_pass_pipeline import load_assign_network
from scripts.train_centre_verifier import load_verifier
from scripts.validate_vote_field import load_network


def suite_rows(suite, split):
    if suite == 'helios':
        return read_manifest(HELIOS, split)
    return [r for r in read_manifest(REAL, split) if r.get('point_eval_eligible') == 'true']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run', required=True)
    parser.add_argument('--weights', default='best.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', required=True, choices=('dense', 'thin10', 'helios'))
    parser.add_argument('--split', default='test', choices=('test', 'val'))
    parser.add_argument('--allow-unselected', action='store_true')
    parser.add_argument('--no-flips', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is required')
    torch.set_num_threads(4)
    root = args.root.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    v7 = json.loads((root / 'protocol_v7.json').read_text())
    weights = root / 'runs' / args.run / 'weights' / args.weights
    payload = torch.load(weights, map_location='cpu', weights_only=False)
    config = payload.get('selected_config')
    gate_passed = args.weights == 'best.pt' and config is not None
    if config is None:
        raise ValueError('Checkpoint carries no validation-selected configuration')
    if not gate_passed and not args.allow_unselected:
        raise ValueError('Checkpoint did not pass the validation gates; use --allow-unselected to evaluate it anyway')
    if sha256(protocol['stage1_checkpoint']) != protocol['stage1_sha256']:
        raise ValueError('Stage-1 checkpoint differs from the frozen protocol')
    rows = suite_rows(args.suite, args.split)
    if args.limit:
        rows = rows[:args.limit]
    symmetries = int(payload.get('symmetries', v7['ensemble_symmetries']))
    run = dict(suite=args.suite, split=args.split, plots=len(rows), stage1_sha256=protocol['stage1_sha256'],
               stage1_config=protocol['stage1_config'], stage2_weights=str(weights), stage2_sha256=sha256(weights),
               stage2_epoch=payload['epoch'], stage2_config=config, stage2_config_name=payload.get('selected_config_name'),
               vote_ensemble_symmetries=symmetries, flips=not args.no_flips, validation_gate_passed=gate_passed,
               thin_density_voxels_per_m2=10. if args.suite == 'thin10' else 0.,
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
    network = load_network(payload)
    verifier = load_verifier(payload)
    assign_network = load_assign_network(payload)
    results = {'Model20': [], 'TwoPassV7': []}
    timing = dict(stage1_forward=0., stage1_merge=0., vote_ensemble=0., stage2=0.)
    operations = {}
    for number, row in enumerate(rows, 1):
        arrays = load_npz(row['output'])
        if args.suite == 'thin10':        # the same draw as scripts/evaluate_two_pass_test.py --thin-density 10
            keep, _ = thin(arrays, 10., np.random.default_rng(np.random.SeedSequence([20261002, number])))
            arrays = subset(arrays, keep)
        gt = gpd.read_file(row['gt_vector'])
        ignore = shapely.from_wkb(Path(row['ignore_vector']).read_bytes()) if row.get('ignore_vector') else None
        raw = predict(model, arrays, owner_only=True, raw_object_threshold=.05)
        timing['stage1_forward'] += float(raw['seconds'])
        start = time.monotonic()
        labels1, confidence1, _, _ = merge_masks_with_sources(arrays, raw, protocol['stage1_config'])
        timing['stage1_merge'] += time.monotonic() - start
        start = time.monotonic()
        ensemble = ensemble_votes(model, arrays, symmetries)
        timing['vote_ensemble'] += time.monotonic() - start
        start = time.monotonic()
        labels2, confidence2, source2, info = second_pass(network, arrays, ensemble, labels1, config, flips=not args.no_flips,
                                                          verifier=verifier, assign_network=assign_network)
        timing['stage2'] += time.monotonic() - start
        operations[row['dataset_id']] = info
        for name, labels, confidence in (('Model20', labels1, confidence1), ('TwoPassV7', labels2, confidence2)):
            results[name].append(evaluate_plot(row, arrays, gt, ignore, labels, confidence))
            folder = output / name
            write_cloud(folder / 'PointClouds' / f"trees_{row['dataset_id']}", arrays, labels, confidence,
                        raw['point_probability'], gt.crs)
            write_vectors(folder / 'Segmentation3', row['dataset_id'], instance_records(arrays, labels, confidence), gt.crs)
        a, b = results['Model20'][-1], results['TwoPassV7'][-1]
        print(f"{args.suite} {number}/{len(rows)} {row['dataset_id']}: crown TP {a['tp']}->{b['tp']} FP {a['fp']}->{b['fp']} "
              f"small {a['small_10_tp']}->{b['small_10_tp']}/{b['small_10_gt']}", flush=True)
    summaries = {name: summarize(records) for name, records in results.items()}
    write_json(output / 'comparison.json', dict(**run, signature=signature, timing_seconds=timing, summaries=summaries,
                                                per_plot=results, operations=operations))
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
    described = dict(dense='native density', thin10='randomly thinned to 10 voxels per occupied m2 before inference',
                     helios='HELIOS++ simulated ALS, about 14 points/m2')[args.suite]
    (output / 'README.md').write_text(
        '# Model20 versus the v7 two-pass model (vote-field detector)\n\n'
        f'Suite: {args.suite} ({described}), split {args.split}, {len(rows)} labelled plots. Stage 1 is the frozen '
        'Model20. Stage 2 averages its votes over four rotations, detects tree centres and their reach in the vote '
        'space and reassigns the votes. The configuration was selected on validation only. '
        f'Validation gates passed: {gate_passed}.\n\n'
        'Model20/ and TwoPassV7/ each contain PointClouds (LAS and LAZ of the voxelized model-input cloud; XY in source '
        'coordinates, Z = height above ground, tree_id = prediction, reference_tree_id = ground truth for viewing only) '
        'and Segmentation3 (crowns_*.gpkg, ttops_*.gpkg). See comparison.xlsx and comparison.json for metrics.\n')
    for name, summary in summaries.items():
        print(name, {k: round(summary[k], 4) if isinstance(summary[k], float) else summary[k] for k in KEYS}, flush=True)
    print('timing', {k: round(v, 1) for k, v in timing.items()})


if __name__ == '__main__':
    main()
