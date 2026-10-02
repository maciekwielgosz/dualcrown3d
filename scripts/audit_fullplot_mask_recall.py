#!/usr/bin/env python3
"""Audit small-crown potential of whole masks on validation plots only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz
from pointcloud.superpoints.fullplot_graph import DEFAULT_GRAPH, reconcile
from pointcloud.superpoints.graph_geometry import graph_crown_matches
from scripts.calibrate_legacy_small_trees import small_hits
from scripts.train_fullplot_mask_verifier import DEFAULT, eligible
from scripts.train_superpoint_decoder_pilot import crown_records
from scripts.benchmark_superpoint_algorithms import save_json


def audit(root, ratios):
    result = {'split': 'val', 'protocol': 'full-mask polygon IoU >= 0.5; '
              'potential hits are not deployable detections',
              'small_area_m2': 10, 'ratios': ratios, 'plots': []}
    for row in eligible('val'):
        arrays = load_npz(row['output'])
        gt = gpd.read_file(row['gt_vector'])
        id_column = 'treeID' if 'treeID' in gt else 'tree_id'
        small = set(gt.loc[gt.geometry.area <= 10, id_column].astype(int))
        raw_path = root / 'raw/val' / (row['dataset_id'] + '.npz')
        with np.load(raw_path) as archive:
            raw = {key: archive[key] for key in archive.files}
        base_path = PROJECT / 'output_24_guarded_small_crown_fusion/work/val_proposals' / (row['dataset_id'] + '.npz')
        with np.load(base_path) as archive:
            labels, confidence = archive['labels'], archive['confidence']
        world = arrays['coord'][:, :2].astype(np.float64) + arrays['source_origin'][:2]
        crowns = crown_records({**arrays, 'world_xy': world}, labels, confidence)
        base = small_hits(gt.geometry, [c['geometry'] for c in crowns])
        entry = {'dataset_id': row['dataset_id'], 'small_gt': len(small),
                 'baseline22_small_hits': int(base['small_10_tp']), 'full_masks': {}}
        for ratio in ratios:
            graph = reconcile(arrays, raw, {**DEFAULT_GRAPH,
                                          'link_min_size_ratio': ratio})
            best, matched, _ = graph_crown_matches(arrays, graph, gt)
            possible = {int(identifier) for identifier in matched[best >= .5]
                        if int(identifier) in small}
            entry['full_masks'][str(ratio)] = {'potential_small_hits': len(possible),
                                               'candidate_clusters': len(graph['feature'])}
        result['plots'].append(entry)
        print(row['dataset_id'], 'baseline', entry['baseline22_small_hits'],
              'potential', {k: v['potential_small_hits'] for k, v in entry['full_masks'].items()},
              flush=True)
    result['total'] = {'small_gt': sum(p['small_gt'] for p in result['plots']),
                       'baseline22_small_hits': sum(p['baseline22_small_hits']
                                                    for p in result['plots']),
                       'full_masks': {str(r): sum(p['full_masks'][str(r)]['potential_small_hits']
                                                  for p in result['plots']) for r in ratios}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--ratios', type=float, nargs='+', default=[0., .85, 1.01])
    args = parser.parse_args()
    root = args.root.resolve()
    report = audit(root, args.ratios)
    save_json(root / 'validation_full_mask_audit.json', report)
    print('TOTAL', json.dumps(report['total']), flush=True)


if __name__ == '__main__':
    main()
