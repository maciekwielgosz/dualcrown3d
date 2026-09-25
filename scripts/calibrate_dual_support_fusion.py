#!/usr/bin/env python3
"""Validate support-preserving mask fusion using cached validation predictions.

Only validation is read. A single checkpoint and identical raw proposals are
used for the legacy control and all fusion variants. No training is performed.
"""
import argparse
import hashlib
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
from pointcloud.dual_head import DualHeadLitePT
from pointcloud.instance_output import merge_masks, point_instance_metrics
from scripts.predict_dual_head import predict


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('cache', 'evaluate', 'all'))
    parser.add_argument('--checkpoint', type=Path, default=PROJECT / 'outputs/dual_head_satv2_litept_v3/weights/best.pt')
    parser.add_argument('--selection', type=Path, default=PROJECT / 'outputs/dual_head_satv2_litept_v3/selected.json')
    parser.add_argument('--manifest', type=Path, default=PROJECT.parent / 'combined_als_crowns_no_rectangles_v2/manifest.csv')
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'outputs/dual_head_support_fusion_v2')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit-plots', type=int)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--family', choices=('boundary', 'interior', 'vote'), default='boundary')
    args = parser.parse_args()
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    os.environ['SEGMENTATION_EXPERIMENT_ROOT'] = str(folder)
    from scripts.evaluate_combined_full_crowns import plot_metrics, aggregate
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    selected = json.loads(args.selection.read_text())
    ckpt_hash = checksum(args.checkpoint)
    if ckpt_hash != selected['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the archived selection')
    rows = read_manifest(args.manifest, 'val')
    if args.limit_plots:
        rows = rows[:args.limit_plots]
    cache = args.cache_dir.resolve() if args.cache_dir else folder / 'raw_val'
    cache.mkdir(exist_ok=True)
    if args.stage in ('cache', 'all'):
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        model = DualHeadLitePT().to(args.device)
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
        for row in rows:
            target = cache / (row['dataset_id'] + '.npz')
            if target.exists():
                continue
            arrays = load_npz(row['output'])
            raw = predict(model, arrays, owner_only=True)
            raw['checkpoint_sha256'] = np.asarray(ckpt_hash)
            raw['input_sha256'] = np.asarray(checksum(row['output']))
            raw['cache_protocol'] = np.asarray('dual_predict_owner_true_20m_8m_40000_v1')
            np.savez_compressed(target, **raw)
            print(f'Cached validation {row["dataset_id"]}: {int(raw["windows"])} windows', flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.stage == 'cache':
        return

    base = {**selected['config'], 'merge_strategy': 'legacy'}
    configurations = [base] + [
        {**base, 'merge_strategy': 'support_fusion_v2', 'fusion_min_iou': overlap,
         'fusion_dominance': .8, 'fusion_max_distance_m': distance, 'fusion_min_probability': .5,
         'fusion_inside_hull': args.family == 'interior'}
        for overlap in ((.3,) if args.family == 'interior' else (.15, .3, .5))
        for distance in (.75, 1.5, 3.)]
    if args.family == 'vote':
        configurations = [base] + [
            {**base, 'merge_strategy': 'support_fusion_v2', 'fusion_min_iou': .3,
             'fusion_dominance': .8, 'fusion_max_distance_m': 1.5, 'fusion_min_probability': .5,
             'fusion_inside_hull': interior, 'fusion_max_vote_distance_m': vote_distance}
            for interior in (False, True) for vote_distance in (.5, 1., 2.)]
    all_rows = [[] for _ in configurations]
    for number, row in enumerate(rows, 1):
        arrays = load_npz(row['output'])
        with np.load(cache / (row['dataset_id'] + '.npz')) as data:
            raw = {key: data[key] for key in data.files}
        if (str(raw['checkpoint_sha256']) != ckpt_hash
                or str(raw['input_sha256']) != checksum(row['output'])
                or str(raw['cache_protocol']) != 'dual_predict_owner_true_20m_8m_40000_v1'):
            raise ValueError('Stale validation prediction cache')
        gt = gpd.read_file(row['gt_vector'])
        metadata = {key: row[key] for key in ('dataset_id', 'source_dataset', 'collection', 'annotation_method')}
        for config, results in zip(configurations, all_rows):
            start = time.monotonic()
            labels, _, instances = merge_masks(arrays, raw, config)
            merge_seconds = time.monotonic()-start
            point = point_instance_metrics(arrays['tree_id'], labels)
            foreground = arrays['tree_id'] > 0
            results.append({**metadata, **plot_metrics(row, gt.geometry, [p['geometry'] for p in instances]),
                            'point_metrics': point, 'labelled_foreground': int(np.count_nonzero(labels[foreground])),
                            'foreground': int(foreground.sum()), 'merge_seconds': merge_seconds})
        print(f'Evaluated validation {number}/{len(rows)}: {row["dataset_id"]}', flush=True)
    trials = []
    for config, results in zip(configurations, all_rows):
        point_rows = [{**{k: r[k] for k in ('dataset_id', 'source_dataset', 'collection', 'annotation_method')},
                       **r['point_metrics']} for r in results]
        point = aggregate(point_rows)
        point['mucov'] = float(np.mean([r['point_metrics']['mucov'] for r in results]))
        point['mwcov'] = float(np.mean([r['point_metrics']['mwcov'] for r in results]))
        trials.append(dict(config=config, metrics=aggregate(results), point_metrics=point, per_plot=results,
                           labelled_gt_foreground_fraction=sum(r['labelled_foreground'] for r in results)/max(1, sum(r['foreground'] for r in results)),
                           merge_seconds=sum(r['merge_seconds'] for r in results)))
    output_name = 'smoke_trials.json' if args.limit_plots else 'configuration_trials.json'
    (folder / output_name).write_text(json.dumps(trials, indent=2)+'\n')
    for i, trial in enumerate(trials):
        print(json.dumps(dict(trial=i, config=trial['config'], point_pq=trial['point_metrics']['source_balanced_pq'],
                              crown_pq=trial['metrics']['source_balanced_pq'], coverage=trial['labelled_gt_foreground_fraction'])), flush=True)
    if args.limit_plots:
        return  # A smoke subset must never select the production configuration.
    eligible = [trial for trial in trials[1:]
                if trial['point_metrics']['source_balanced_pq'] >= trials[0]['point_metrics']['source_balanced_pq']
                and trial['metrics']['source_balanced_pq'] + 1e-10 >= trials[0]['metrics']['source_balanced_pq']]
    best = max(eligible or trials[1:], key=lambda r: (r['point_metrics']['source_balanced_pq'], r['metrics']['source_balanced_pq']))
    passed = bool(eligible)
    result = dict(epoch=selected['epoch'], checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=ckpt_hash,
                  manifest_sha256=checksum(args.manifest), config=best['config'], validation=best,
                  baseline=trials[0], split='val', validation_plots=len(rows), acceptance_passed=passed,
                  selection_metric='source-balanced point-instance PQ@0.5; crown PQ non-regression check',
                  note='Same trained weights; validation-only support-fusion correction. No test evaluation.')
    (folder / 'selected.json').write_text(json.dumps(result, indent=2)+'\n')
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'Validation trials'
    sheet.append(['trial', 'parameters', 'point_PQ_balanced', 'point_F1', 'point_precision', 'point_recall',
                  'crown_PQ_balanced', 'crown_F1', 'GT_foreground_coverage', 'merge_seconds', 'checkpoint', 'SHA256'])
    for i, trial in enumerate(trials):
        pm, cm = trial['point_metrics'], trial['metrics']
        sheet.append([i, json.dumps(trial['config']), pm['source_balanced_pq'], pm['f1'], pm['precision'], pm['recall'],
                      cm['source_balanced_pq'], cm['f1'], trial['labelled_gt_foreground_fraction'], trial['merge_seconds'], str(args.checkpoint.resolve()), ckpt_hash])
    sheet.freeze_panes = 'A2'
    sheet.auto_filter.ref = sheet.dimensions
    workbook.save(folder / 'experiments.xlsx')
    print('Acceptance passed:', passed, flush=True)


if __name__ == '__main__':
    main()
