#!/usr/bin/env python3
"""Geometry-first verification of Model22's low-score small-tree proposals.

The validation diagnosis found that true tiny trees have low raw object scores,
while confident Model22 fragments tend to be parts of major Model20 anchors.
This experiment tests that hypothesis without using held-out test annotations.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import geopandas as gpd
import numpy as np
import shapely

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz
from pointcloud.small_tree_split import internal_split_features, guarded_internal_splits
from scripts.benchmark_superpoint_algorithms import save_json
from scripts.calibrate_guarded_small_crowns import (eligible, cached_val_anchor,
    test_anchor_and_proposal, inputs_and_signature, DEFAULT as PRIOR)
from scripts.evaluate_output20_output22_test import (score, summary, write_cloud,
    write_vectors, excel)
from scripts.train_superpoint_decoder_pilot import crown_records

DEFAULT = PROJECT / 'output_26_low_confidence_small_split_verification'


def configurations():
    common = dict(min_area_m2=.25, min_voxels=8, min_height_m=1.5,
                  min_anchor_dominance=.60, min_anchor_area_ratio=1.5,
                  min_anchor_remaining_voxels=16, min_confidence=.05)
    return {f'c{conf:g}_h{gap:g}_f{fraction:g}_a{area:g}_d{distance:g}':
            dict(**common, max_confidence=conf, min_height_gap_m=gap,
                 max_anchor_fraction=fraction, max_area_m2=area,
                 min_top_distance_m=distance)
            for conf, gap, fraction, area, distance in itertools.product(
                (.13, .16, .20), (2., 4.), (.08, .15), (10., 15.), (1.5, 2.5))}


def predicted(arrays, anchor, proposal, features, config):
    selected = [f for f in features if f['record']['confidence'] <= config['max_confidence']]
    l20, c20, r20 = anchor
    l22, c22 = proposal
    return guarded_internal_splits(l20, c20, l22, c22, r20, selected, config)


def choose(results, options):
    baseline = summary(results['Model20'])
    pq_point = baseline['point']['source_balanced_pq']
    pq_crown = baseline['crown']['source_balanced_pq']
    precision = baseline['point']['precision']
    trials, safe = {}, []
    for name, cfg in options.items():
        scores = summary(results[name])
        trials[name] = dict(config=cfg, summary=scores)
        if (scores['point']['source_balanced_pq'] >= pq_point - .005 and
            scores['crown']['source_balanced_pq'] >= pq_crown - .005 and
            scores['point']['precision'] >= precision - .01):
            safe.append(name)
    better = [x for x in safe if trials[x]['summary']['small_crowns']['up_to_10_m2']['tp']
              > baseline['small_crowns']['up_to_10_m2']['tp']]
    winner = max(better, key=lambda x:(
        trials[x]['summary']['small_crowns']['up_to_10_m2']['tp'],
        trials[x]['summary']['small_crowns']['up_to_4_m2']['tp'],
        trials[x]['summary']['point']['source_balanced_pq'] +
        trials[x]['summary']['crown']['source_balanced_pq'])) if better else None
    return dict(selected=winner, baseline=baseline, trials=trials, safe=safe, better=better,
                quality_gate=dict(point_pq_min=pq_point-.005,
                                  crown_pq_min=pq_crown-.005,
                                  point_precision_min=precision-.01))


def validation(output, old, signature, protocol):
    path = output/'selection.json'
    if path.exists():
        answer = json.loads(path.read_text())
        if answer['signature'] != signature:
            raise ValueError('Validation signature mismatch')
        return answer
    rows = eligible('val')
    options = configurations()
    records = {key: [] for key in ('Model20', *options)}
    started = time.monotonic()
    for number, row in enumerate(rows, 1):
        arrays = load_npz(row['output'])
        anchor = cached_val_anchor(row, arrays, old['mask_config'], old['checkpoint_sha256'])
        with np.load(PRIOR/'work/val_proposals'/(row['dataset_id']+'.npz')) as file:
            if str(file['signature']) != signature:
                raise ValueError('Validation proposal cache mismatch')
            proposal = np.asarray(file['labels']), np.asarray(file['confidence'], np.float32)
        gt = gpd.read_file(row['gt_vector'])
        ignore = shapely.from_wkb(Path(row['ignore_vector']).read_bytes()) if row.get('ignore_vector') else None
        l20, c20, r20 = anchor
        records['Model20'].append(score(row, arrays, gt, ignore, l20,
                                        [r['geometry'] for r in r20]))
        features = internal_split_features(arrays, l20, proposal[0], proposal[1],
                                           r20, crown_records)
        for name, cfg in options.items():
            labels, _, crowns, additions = predicted(arrays, anchor, proposal, features, cfg)
            result = score(row, arrays, gt, ignore, labels,
                           [r['geometry'] for r in crowns])
            result['accepted_splits'] = len(additions)
            records[name].append(result)
        print(f'validation {number}/{len(rows)} {row["dataset_id"]}: '
              f'{len(features)} proposals', flush=True)
    answer = choose(records, options)
    answer.update(signature=signature, protocol=protocol, split='val', plots=len(rows),
                  elapsed_seconds=time.monotonic()-started)
    save_json(path, answer)
    return answer


def test(output, selection, signature):
    if selection['selected'] is None:
        return dict(status='No validation improvement under quality gate; test not touched')
    destination = output/'test_comparison.json'
    if destination.exists():
        raise FileExistsError(f'Completed test is protected: {destination}')
    config = selection['trials'][selection['selected']]['config']
    original = json.loads((PROJECT/'output_23_labeled_test_output20_vs_output22/comparison.json').read_text())
    if (original['manifest_sha256'] != selection['protocol']['manifest_sha256'] or
        original['output20_sha256'] != selection['protocol']['old_checkpoint_sha256'] or
        original['output22_sha256'] != selection['protocol']['new_checkpoint_sha256']):
        raise ValueError('Test source changed since validation')
    rows = eligible('test')
    metrics = []
    started = time.monotonic()
    for number, row in enumerate(rows, 1):
        arrays = load_npz(row['output'])
        anchor, proposal, semantic = test_anchor_and_proposal(row, arrays)
        l20, c20, r20 = anchor
        features = internal_split_features(arrays, l20, proposal[0], proposal[1],
                                           r20, crown_records)
        labels, confidence, crowns, added = predicted(arrays, anchor, proposal, features, config)
        gt = gpd.read_file(row['gt_vector'])
        ignore = shapely.from_wkb(Path(row['ignore_vector']).read_bytes()) if row.get('ignore_vector') else None
        result = score(row, arrays, gt, ignore, labels,
                       [r['geometry'] for r in crowns])
        result['accepted_splits'] = len(added)
        folder = output/'Model26'
        cloud = write_cloud(folder/'PointClouds'/f"trees_{row['dataset_id']}",
                            arrays, labels, confidence, semantic, gt.crs)
        vectors = write_vectors(folder/'Segmentation3', row['dataset_id'], crowns, gt.crs)
        metrics.append(result)
        save_json(output/'metrics/per_plot'/f"{row['dataset_id']}.json",
                  dict(signature=signature, metrics=result, accepted=len(added),
                       cloud=cloud, vectors=vectors))
        print(f'test {number}/{len(rows)} {row["dataset_id"]}: '
              f'anchor={len(r20)}, small_splits={len(added)}', flush=True)
    report = dict(signature=signature, config=config,
                  selected_on='validation; test labels used for scoring only',
                  split='test', plots=len(rows), elapsed_seconds=time.monotonic()-started,
                  results={'Model20':original['results']['Model20'],
                           'Model22':original['results']['Model22'],
                           'Model26':dict(summary=summary(metrics), per_plot=metrics)})
    save_json(destination, report)
    excel(output/'test_comparison.xlsx', report['results'])
    (output/'README.md').write_text(
        '# Low-confidence small-tree proposal verification\n\n'
        'Validation-selected geometric gate for the small low-score Model22 proposals. '
        'A proposal may split at most one Model20 anchor; all original anchor crown '
        'outlines remain. Model26/PointClouds contains paired LAS/LAZ; '
        'Model26/Segmentation3 contains crowns and treetops. LAS Z is height AGL. '
        'The reference_tree_id is included only for visual scoring and was not used '
        'to make predictions. See selection.json and test_comparison.xlsx.\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--phase', choices=('all', 'val', 'test'), default='all')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    old, _, _, _, protocol, signature = inputs_and_signature()
    if args.phase in ('all', 'val'):
        selection = validation(output, old, signature, protocol)
        print('selected', selection['selected'], 'safe', len(selection['safe']),
              'better', len(selection['better']), flush=True)
    else:
        selection = json.loads((output/'selection.json').read_text())
        if selection['signature'] != signature:
            raise ValueError('Selection signature mismatch')
    if args.phase in ('all', 'test'):
        result = test(output, selection, signature)
        print(json.dumps(result.get('results', result), indent=2)[:20000], flush=True)


if __name__ == '__main__':
    main()
