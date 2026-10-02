#!/usr/bin/env python3
"""Validation-calibrated, bounded small-tree splits of stable Model20 anchors."""
from __future__ import annotations

import argparse
import hashlib
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

DEFAULT = PROJECT / 'output_25_guarded_internal_small_splits'


def configurations():
    common = dict(min_area_m2=.25, min_voxels=8, min_height_m=1.5,
                  min_anchor_dominance=.85, min_anchor_area_ratio=2.,
                  min_top_distance_m=1.5, min_anchor_remaining_voxels=16)
    return {f'c{conf:g}_a{area:g}_h{gap:g}_f{fraction:g}':
            dict(**common, min_confidence=conf, max_area_m2=area,
                 min_height_gap_m=gap, max_anchor_fraction=fraction)
            for conf, area, gap, fraction in itertools.product(
                (.10, .20, .30), (8., 15.), (1.5, 3.), (.30, .50))}


def variants(row, arrays, gt, ignore, anchor, proposal, options):
    l20, c20, r20 = anchor
    l22, c22 = proposal
    features = internal_split_features(arrays, l20, l22, c22, r20, crown_records)
    entries = {'Model20': score(row, arrays, gt, ignore, l20,
                                [r['geometry'] for r in r20])}
    for name, config in options.items():
        labels, _, crowns, accepted = guarded_internal_splits(
            l20, c20, l22, c22, r20, features, config)
        entry = score(row, arrays, gt, ignore, labels,
                      [r['geometry'] for r in crowns])
        entry['accepted_splits'] = len(accepted)
        entries[name] = entry
    return entries, len(features)


def select(results, options):
    base = summary(results['Model20'])
    point = base['point']['source_balanced_pq']
    crown = base['crown']['source_balanced_pq']
    precision = base['point']['precision']
    trials, passed = {}, []
    for name, config in options.items():
        s = summary(results[name])
        trials[name] = dict(config=config, summary=s)
        if (s['point']['source_balanced_pq'] >= point - .005 and
            s['crown']['source_balanced_pq'] >= crown - .005 and
            s['point']['precision'] >= precision - .01):
            passed.append(name)
    # No promotion if the small-crown recall on validation is unchanged.
    improved = [name for name in passed if trials[name]['summary']['small_crowns']['up_to_10_m2']['tp']
                > base['small_crowns']['up_to_10_m2']['tp']]
    winner = max(improved, key=lambda name: (
        trials[name]['summary']['small_crowns']['up_to_10_m2']['tp'],
        trials[name]['summary']['small_crowns']['up_to_4_m2']['tp'],
        trials[name]['summary']['point']['source_balanced_pq'] +
        trials[name]['summary']['crown']['source_balanced_pq'])) if improved else None
    return dict(selected=winner, base=base, trials=trials, passed=passed,
                improved=improved, quality_gate=dict(point_pq_min=point-.005,
                    crown_pq_min=crown-.005, point_precision_min=precision-.01))


def validation(output, old, signature, protocol):
    target = output/'selection.json'
    if target.exists():
        result = json.loads(target.read_text())
        if result['signature'] != signature:
            raise ValueError('Selection signature differs')
        return result
    rows = eligible('val')
    options = configurations()
    results = {name: [] for name in ('Model20', *options)}
    diagnostics = []
    started = time.monotonic()
    for number, row in enumerate(rows, 1):
        arrays = load_npz(row['output'])
        anchor = cached_val_anchor(row, arrays, old['mask_config'], old['checkpoint_sha256'])
        path = PRIOR/'work/val_proposals'/(row['dataset_id']+'.npz')
        with np.load(path) as file:
            if str(file['signature']) != signature:
                raise ValueError(f'Model22 validation cache differs: {path}')
            proposal = np.asarray(file['labels']), np.asarray(file['confidence'], np.float32)
        gt = gpd.read_file(row['gt_vector'])
        ignore = shapely.from_wkb(Path(row['ignore_vector']).read_bytes()) if row.get('ignore_vector') else None
        entries, candidates = variants(row, arrays, gt, ignore, anchor, proposal, options)
        for key, entry in entries.items():
            results[key].append(entry)
        diagnostics.append(dict(dataset_id=row['dataset_id'], candidates=candidates))
        print(f'validation {number}/{len(rows)} {row["dataset_id"]}: '
              f'anchor={entries["Model20"]["predicted_instances"]}, '
              f'internal_candidates={candidates}', flush=True)
    selection = select(results, options)
    selection.update(signature=signature, protocol=protocol, split='val',
                     plots=len(rows), elapsed_seconds=time.monotonic()-started,
                     diagnostics=diagnostics)
    save_json(target, selection)
    return selection


def test(output, selection, signature):
    if selection['selected'] is None:
        return dict(status='No safe validation improvement; no test export or promotion')
    target = output/'test_comparison.json'
    if target.exists():
        raise FileExistsError(f'Completed test is protected: {target}')
    config = selection['trials'][selection['selected']]['config']
    previous = json.loads((PRIOR.parent/'output_23_labeled_test_output20_vs_output22/comparison.json').read_text())
    if (previous['manifest_sha256'] != selection['protocol']['manifest_sha256'] or
        previous['output20_sha256'] != selection['protocol']['old_checkpoint_sha256'] or
        previous['output22_sha256'] != selection['protocol']['new_checkpoint_sha256']):
        raise ValueError('Held-out source output differs')
    rows = eligible('test')
    metrics = []
    started = time.monotonic()
    for number, row in enumerate(rows, 1):
        arrays = load_npz(row['output'])
        anchor, proposal, semantic = test_anchor_and_proposal(row, arrays)
        l20, c20, r20 = anchor
        l22, c22 = proposal
        features = internal_split_features(arrays, l20, l22, c22, r20, crown_records)
        labels, confidence, crowns, accepted = guarded_internal_splits(
            l20, c20, l22, c22, r20, features, config)
        gt = gpd.read_file(row['gt_vector'])
        ignore = shapely.from_wkb(Path(row['ignore_vector']).read_bytes()) if row.get('ignore_vector') else None
        result = score(row, arrays, gt, ignore, labels,
                       [r['geometry'] for r in crowns])
        result['accepted_splits'] = len(accepted)
        folder = output/'Model25'
        cloud = write_cloud(folder/'PointClouds'/f"trees_{row['dataset_id']}",
                            arrays, labels, confidence, semantic, gt.crs)
        vectors = write_vectors(folder/'Segmentation3', row['dataset_id'], crowns, gt.crs)
        metrics.append(result)
        save_json(output/'metrics/per_plot'/f"{row['dataset_id']}.json",
                  dict(signature=signature, metrics=result, accepted=len(accepted),
                       cloud=cloud, vectors=vectors))
        print(f'test {number}/{len(rows)} {row["dataset_id"]}: '
              f'anchor={len(r20)}, new_small={len(accepted)}', flush=True)
    report = dict(signature=signature, config=config,
                  selection='frozen on validation before test scoring',
                  split='test', plots=len(rows), elapsed_seconds=time.monotonic()-started,
                  results={'Model20':previous['results']['Model20'],
                           'Model22':previous['results']['Model22'],
                           'Model25':dict(summary=summary(metrics), per_plot=metrics)})
    save_json(target, report)
    excel(output/'test_comparison.xlsx', report['results'])
    (output/'README.md').write_text(
        '# Guarded internal small-tree splits\n\n'
        'A validation-calibrated compact Model22 proposal may split at most one '
        'lower-height small crown from each Model20 anchor. Other Model20 IDs and '
        'all existing anchor polygon outlines are retained. Output Model25 has '
        'paired LAS/LAZ and crown/treetop GeoPackages for held-out test plots. '
        'LAS XY retains source coordinates; Z is height above ground. '
        'reference_tree_id is for inspection only. See selection.json and '
        'test_comparison.xlsx for the actual quality tradeoff.\n')
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
        print('selected', selection['selected'], 'passed', len(selection['passed']),
              'improved', len(selection['improved']), flush=True)
    else:
        selection = json.loads((output/'selection.json').read_text())
        if selection['signature'] != signature:
            raise ValueError('Validation signature differs')
    if args.phase in ('all', 'test'):
        result = test(output, selection, signature)
        print(json.dumps(result.get('results', result), indent=2)[:20000], flush=True)


if __name__ == '__main__':
    main()
