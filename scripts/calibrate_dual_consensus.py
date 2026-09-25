#!/usr/bin/env python3
"""Paired validation-only ablation of dual-head recovery; no network retraining."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import geopandas as gpd
import numpy as np
import torch
from openpyxl import Workbook
from pointcloud.data import load_npz, read_manifest
from pointcloud.instance_output import merge_masks, point_instance_metrics
from pointcloud.dual_fusion import vote_instances, fuse_heads, instance_records
from scripts.calibrate_dual_support_fusion import checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PROJECT/'outputs/dual_head_consensus_v3')
    parser.add_argument('--baseline', type=Path, default=PROJECT/'configs/dual_head_inference.json')
    parser.add_argument('--checkpoint', type=Path, default=PROJECT/'outputs/dual_head_satv2_litept_v3/weights/best.pt')
    parser.add_argument('--manifest', type=Path, default=PROJECT.parent/'combined_als_crowns_no_rectangles_v2/manifest.csv')
    parser.add_argument('--cache-dir', type=Path, default=PROJECT/'outputs/dual_head_support_fusion_v2/raw_val')
    parser.add_argument('--limit-plots', type=int)
    parser.add_argument('--anchor', choices=('mask', 'vote', 'both'), default='both')
    parser.add_argument('--growth-study', action='store_true', help='Also test recovery beyond the original 2 m vote-assignment radius')
    parser.add_argument('--selection-objective', choices=('point', 'balanced'), default='balanced',
                        help='Balanced uses geometric mean of point and crown PQ, keeping both tasks in scope')
    parser.add_argument('--select-only', action='store_true', help='Re-select from a complete, verified validation ablation table')
    parser.add_argument('--fallback-study', action='store_true', help='Test independent mask-only trees using that head\'s own foreground evidence')
    parser.add_argument('--joint-pq-tolerance', type=float, default=0.,
                        help='Optional absolute joint-PQ tolerance: maximize coverage among near-best eligible trials')
    args = parser.parse_args()
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    os.environ['SEGMENTATION_EXPERIMENT_ROOT'] = str(folder)
    from scripts.evaluate_combined_full_crowns import plot_metrics, aggregate
    torch.set_num_threads(4)
    baseline = json.loads(args.baseline.read_text())
    ckpt_hash = checksum(args.checkpoint)
    if ckpt_hash != baseline['checkpoint_sha256']:
        raise ValueError('Unexpected checkpoint')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    vote_config = checkpoint['config']['legacy_cluster_config']
    base = baseline['config']
    configs = [base, {**base, 'merge_strategy': 'vote_control', 'vote_cluster_config': vote_config}]
    for distance in (1.5, 3.):
        for add in (False, True):
            for growth in (0., 1.5):
                configs.append({**base, 'merge_strategy': 'dual_consensus_v3', 'vote_cluster_config': vote_config,
                                'dual_dominance': .8, 'dual_min_intersection': 4, 'dual_min_probability': .5,
                                'dual_max_distance_m': distance, 'dual_vote_distance_m': 2.,
                                'dual_add_instances': add, 'dual_new_max_overlap': .05, 'dual_new_separation_m': 1.,
                                'dual_growth_distance_m': growth, 'dual_growth_vote_distance_m': 1.,
                                'dual_growth_margin_m': .25})
    forward = configs[2:]
    reverse = [{**c, 'dual_anchor': 'vote'} for c in forward]
    configs = configs[:2] + ([] if args.anchor == 'vote' else forward) + ([] if args.anchor == 'mask' else reverse)
    if args.growth_study:
        for anchor in (('mask', 'vote') if args.anchor == 'both' else (args.anchor,)):
            for distance in (1.5, 3.):
                for vote_distance in (3., 4.):
                    for support_only in (False, True):
                        configs.append({**forward[0], 'dual_anchor': anchor,
                                        'dual_growth_distance_m': distance, 'dual_growth_vote_distance_m': vote_distance,
                                        'dual_growth_support_only': support_only})
    if args.fallback_study:
        current = json.loads((PROJECT/'configs/dual_head_consensus.json').read_text())
        if current['checkpoint_sha256'] != ckpt_hash:
            raise ValueError('Fallback control has different weights')
        configs = configs[:2] + [current['config']]
        for probability in (0., .3):
            for overlap in (.05, .2):
                for guard in (False, True):
                    configs.append({**current['config'], 'dual_add_instances': True,
                        'dual_complement_own_semantic': True, 'dual_complement_min_probability': probability,
                        'dual_new_max_overlap': overlap, 'dual_new_use_vote_guard': guard})
    rows = read_manifest(args.manifest, 'val')
    if args.limit_plots:
        rows = rows[:args.limit_plots]
    trials = [dict(config=c, per_plot=[]) for c in configs]
    if args.select_only:
        if args.limit_plots:
            raise ValueError('Cannot select using a validation subset')
        previous = json.loads((folder/'selected.json').read_text())
        if (previous['checkpoint_sha256'] != ckpt_hash or previous['manifest_sha256'] != checksum(args.manifest)
                or previous['split'] != 'val' or previous['validation_plots'] != len(rows)):
            raise ValueError('Selection metadata does not match this experiment')
        trials = json.loads((folder/'configuration_trials.json').read_text())
        if [t['config'] for t in trials] != configs or any(
                [r['dataset_id'] for r in t['per_plot']] != [r['dataset_id'] for r in rows] for t in trials):
            raise ValueError('Trial configurations or validation plots differ')
    for number, row in enumerate([] if args.select_only else rows, 1):
        arrays = load_npz(row['output'])
        with np.load(args.cache_dir/(row['dataset_id']+'.npz')) as archive:
            raw = {k: archive[k] for k in archive.files}
        if (str(raw['checkpoint_sha256']) != ckpt_hash or str(raw['input_sha256']) != checksum(row['output'])
                or str(raw['cache_protocol']) != 'dual_predict_owner_true_20m_8m_40000_v1'):
            raise ValueError('Stale validation cache')
        started = time.monotonic()
        base_labels, base_confidence, base_instances = merge_masks(arrays, raw, base)
        base_seconds = time.monotonic()-started
        started = time.monotonic()
        auxiliary = vote_instances(arrays, raw, vote_config)
        vote_seconds = time.monotonic()-started
        gt = gpd.read_file(row['gt_vector'])
        metadata = {key: row[key] for key in ('dataset_id', 'source_dataset', 'collection', 'annotation_method')}
        foreground = arrays['tree_id'] > 0
        for i, trial in enumerate(trials):
            started = time.monotonic()
            if i == 0:
                labels, instances = base_labels, base_instances
                stages = {}
            elif i == 1:
                labels = auxiliary
                instances = instance_records(arrays, labels, raw['tree_probability'])
                stages = {}
            else:
                labels, _, instances, source = fuse_heads(arrays, raw, base_labels, base_confidence, trial['config'], auxiliary)
                stages = {str(k): int((source == k).sum()) for k in range(6)}
            merge_seconds = time.monotonic()-started + (base_seconds if i != 1 else 0) + (vote_seconds if i else 0)
            pm = point_instance_metrics(arrays['tree_id'], labels)
            trial['per_plot'].append({**metadata, **plot_metrics(row, gt.geometry, [p['geometry'] for p in instances]),
                'point_metrics': pm, 'foreground': int(foreground.sum()),
                'labelled_foreground': int((labels[foreground] > 0).sum()), 'assignment_source': stages,
                'merge_seconds': merge_seconds})
        print(f'Validation {number}/{len(rows)}: {row["dataset_id"]}', flush=True)
    for trial in trials:
        results = trial['per_plot']
        point_rows = [{**{k: r[k] for k in ('dataset_id','source_dataset','collection','annotation_method')},
                       **r['point_metrics']} for r in results]
        trial.update(point_metrics=aggregate(point_rows), metrics=aggregate(results),
                     labelled_gt_foreground_fraction=sum(r['labelled_foreground'] for r in results)/max(1, sum(r['foreground'] for r in results)),
                     merge_seconds=sum(r['merge_seconds'] for r in results))
        for key in ('mucov', 'mwcov'):
            trial['point_metrics'][key] = float(np.mean([r['point_metrics'][key] for r in results]))
    name = 'smoke_trials.json' if args.limit_plots else 'configuration_trials.json'
    (folder/name).write_text(json.dumps(trials, indent=2)+'\n')
    for i, trial in enumerate(trials):
        print(json.dumps(dict(trial=i, point_pq=trial['point_metrics']['source_balanced_pq'], point_f1=trial['point_metrics']['f1'],
            crown_pq=trial['metrics']['source_balanced_pq'], crown_f1=trial['metrics']['f1'], coverage=trial['labelled_gt_foreground_fraction'])), flush=True)
    if args.limit_plots:
        return
    def improves(t):
        return all(t[branch][metric]+1e-10 >= trials[0][branch][metric]
                   for branch in ('point_metrics', 'metrics') for metric in ('source_balanced_pq', 'f1'))
    eligible = [t for t in trials[2:] if improves(t) and t['labelled_gt_foreground_fraction'] > trials[0]['labelled_gt_foreground_fraction']]
    def selection_score(t):
        point, crown = t['point_metrics']['source_balanced_pq'], t['metrics']['source_balanced_pq']
        return (float(np.sqrt(point*crown)) if args.selection_objective == 'balanced' else point, crown)
    best = max(eligible or trials[2:], key=selection_score)
    if args.joint_pq_tolerance < 0:
        raise ValueError('Joint-PQ tolerance must be nonnegative')
    if args.joint_pq_tolerance and eligible:
        near = [t for t in eligible if selection_score(t)[0] >= selection_score(best)[0]-args.joint_pq_tolerance]
        best = max(near, key=lambda t: (t['labelled_gt_foreground_fraction'], selection_score(t)))
    result = dict(epoch=baseline['epoch'], checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=ckpt_hash,
        manifest_sha256=checksum(args.manifest), config=best['config'], validation=best, baseline=trials[0],
        vote_control=trials[1], split='val', validation_plots=len(rows), acceptance_passed=bool(eligible),
        selection_metric=('Geometric mean of point/crown source-balanced PQ' if args.selection_objective == 'balanced' else 'Point source-balanced PQ')
                         + f'; within {args.joint_pq_tolerance:g} absolute score choose highest foreground coverage'
                         + '; point/crown PQ and F1 non-regression against output_17; increased coverage',
        note='Validation-only dual-head postprocessing, unchanged weights. Test set untouched.')
    (folder/'selected.json').write_text(json.dumps(result, indent=2)+'\n')
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'Validation ablations'
    sheet.append(['trial','architecture','parameters','point_PQ','point_F1','crown_PQ','crown_F1','GT_coverage','seconds','checkpoint','sha256'])
    for i,t in enumerate(trials):
        sheet.append([i,'Frozen LitePT-S + mask decoder + centre-vote consensus',json.dumps(t['config']),
            t['point_metrics']['source_balanced_pq'],t['point_metrics']['f1'],t['metrics']['source_balanced_pq'],
            t['metrics']['f1'],t['labelled_gt_foreground_fraction'],t['merge_seconds'],str(args.checkpoint.resolve()),ckpt_hash])
    sheet.freeze_panes='A2'
    sheet.auto_filter.ref=sheet.dimensions
    protocol = workbook.create_sheet('Protocol')
    for key in ('checkpoint','checkpoint_sha256','manifest_sha256','split','validation_plots','acceptance_passed','selection_metric','note'):
        protocol.append([key, result[key]])
    protocol.append(['selected_parameters',json.dumps(best['config'])])
    workbook.save(folder/'experiments.xlsx')
    print('Acceptance passed:', bool(eligible), flush=True)


if __name__ == '__main__':
    main()
