#!/usr/bin/env python3
"""Compare 96-query EZ-SP pilot with 128-query, wider-graph/decoder ablation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import geopandas as gpd
import laspy
import numpy as np
from openpyxl import load_workbook
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.superpoints.model import CachedInstanceModel
from scripts.benchmark_superpoint_algorithms import save_json, sha256
from scripts.report_superpoint_decoder_pilot import paired_interval
from scripts.train_superpoint_decoder_pilot import DEFAULT, export_predictions


def chosen(path):
    trials = json.loads(path.read_text())
    key = max(trials, key=lambda name: trials[name]['score'])
    return key, trials[key]


def row(label, epoch, key, result, checkpoint, parameters, history):
    return dict(variant=label, epoch=epoch, thresholds=key,
                point_pq=result['point']['source_balanced_pq'],
                point_f1=result['point']['source_balanced_f1'],
                point_precision=result['point']['precision'],
                point_recall=result['point']['recall'],
                crown_pq=result['crown']['source_balanced_pq'],
                crown_f1=result['crown']['source_balanced_f1'],
                sparse_tree_recall=result['sparse_tree_recall_le100_voxels'],
                annotated_point_coverage=result['annotated_tree_point_coverage'],
                joint_score=result['score'], decoder_parameters=parameters,
                peak_train_vram_mib=max((item.get('peak_vram_mb', 0) for item in history), default=0),
                mean_cached_decoder_seconds=float(np.mean([item['decoder_seconds'] for item in result['per_plot']])),
                checkpoint=str(checkpoint))


def write_sheet(book, title, rows):
    if title in book:
        del book[title]
    sheet = book.create_sheet(title)
    if not rows:
        return
    columns = list(dict.fromkeys(key for record in rows for key in record))
    sheet.append(columns)
    for record in rows:
        sheet.append([record.get(key) for key in columns])
    sheet.freeze_panes = 'A2'
    sheet.auto_filter.ref = sheet.dimensions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--large-run', default='ezsp_large_w192_q128')
    args = parser.parse_args()
    root = args.output.resolve()
    small = root / 'runs/ezsp'
    large = root / 'runs' / args.large_run
    if not (large / 'DONE.json').exists():
        raise RuntimeError(f'Large run incomplete: {large}')
    prepared = json.loads((root / 'prepared.json').read_text())
    large_config = json.loads((large / 'configuration.json').read_text())
    if large_config['initial_sha256'] != json.loads((small / 'configuration.json').read_text())['initial_sha256']:
        raise ValueError('Different initial checkpoint')
    if large_config['train_crops'] != 79 or large_config['val_crops'] != 14 or large_config['seed'] != 20261001:
        raise ValueError('Not the matched protocol')
    outputs = {}
    summaries = []
    for name, folder in [('small', small), ('large', large)]:
        history = json.loads((folder / 'history.json').read_text())['epochs']
        for stage, path, checkpoint in (
            ('initial', folder / 'validation/epoch_000.json', folder / 'weights/best.pt'),
            ('trained', folder / 'validation/epoch_012.json', folder / 'weights/last.pt')):
            threshold, result = chosen(path)
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            parameters = sum(weight.numel() for key, weight in payload['model'].items()
                             if key.startswith('decoder.'))
            summaries.append(row(f'{name}_{stage}', 0 if stage == 'initial' else 12,
                                 threshold, result, '' if stage == 'initial' else checkpoint,
                                 parameters, history))
            outputs[(name, stage)] = result
    selected = json.loads((large / 'selected.json').read_text())
    selected_row = row('large_selected', selected['epoch'],
                       next(key for key, value in json.loads((large / 'validation' /
                           f"epoch_{selected['epoch']:03d}.json").read_text()).items()
                           if value['score'] == selected['validation']['score']),
                       selected['validation'], selected['checkpoint'],
                       next(record['decoder_parameters'] for record in summaries if record['variant'] == 'large_trained'),
                       json.loads((large / 'history.json').read_text())['epochs'])
    summaries.append(selected_row)
    intervals = {}
    for stage in ('initial', 'trained'):
        intervals[stage] = {metric: paired_interval(outputs[('small', stage)]['per_plot'],
                                                   outputs[('large', stage)]['per_plot'], metric)
                            for metric in ('point', 'crown')}
    # Save final trained predictions separately if validation selects epoch zero.
    export_root = large / 'last_epoch_validation_exports'
    validation = [item for item in prepared['entries'] if item['split'] == 'val']
    if not export_root.exists():
        torch.set_num_threads(4)
        model = CachedInstanceModel('ezsp', queries=large_config['queries'],
                                    memory_tokens=large_config['memory_tokens'],
                                    graph_width=large_config['graph_width'],
                                    graph_layers=large_config['graph_layers'],
                                    wide_dim=large_config['wide_dim'],
                                    wide_layers=large_config['wide_layers']).cuda().eval()
        model.load_state_dict(torch.load(large / 'weights/last.pt', map_location='cpu', weights_only=False)['model'])
        selection = dict(epoch=12, checkpoint=str(large / 'weights/last.pt'),
                         validation=outputs[('large', 'trained')])
        export_predictions(model, validation, selection, export_root)
    checks = []
    for item in validation:
        stem = item['dataset_id']
        cloud = laspy.read(export_root / 'PointClouds' / f'trees_{stem}.laz')
        ids = set(np.unique(cloud.tree_id).tolist()) - {0}
        if ids:
            frame = gpd.read_file(export_root / 'Segmentation3' / f'crowns_{stem}.gpkg')
            tops = gpd.read_file(export_root / 'Segmentation3' / f'ttops_{stem}.gpkg')
            if set(frame.tree_id) != ids or set(tops.tree_id) != ids or not frame.geometry.is_valid.all():
                raise ValueError(f'Invalid or mismatched output: {stem}')
        checks.append(dict(dataset_id=stem, point_and_crown_ids_match=True,
                           instances=len(ids), points=len(cloud.points)))
    report = dict(comparison=summaries, paired_large_minus_small=intervals,
                  large_configuration=large_config, exports=checks,
                  training_scope='frozen LitePT encoder; same 79/14 precomputed ALS crops, 12 epochs, seed and thresholds',
                  interpretation='Exploratory validation size ablation; cannot attribute a change separately to graph width, query count or wide refinement.',
                  speed_scope='cached decoder forward only; full inference includes encoder, grouping and export')
    suffix = args.large_run.removeprefix('ezsp_')
    save_json(root / f'size_ablation_{suffix}.json', report)
    previous = json.loads((root / 'trained_checkpoint_report.json').read_text())
    prior_selected = json.loads((root / 'final_report.json').read_text())
    book = load_workbook(root / 'experiments.xlsx')
    write_sheet(book, f'size_{suffix}', summaries)
    write_sheet(book, f'size_ci_{suffix}', [dict(stage=stage, metric=metric, **numbers)
        for stage, values in intervals.items() for metric, numbers in values.items()])
    write_sheet(book, f'size_export_{suffix}', checks)
    write_sheet(book, 'final_epoch', previous['final_epoch'])
    write_sheet(book, 'final_epoch_intervals', [dict(comparison=name, metric=metric, **numbers)
        for name, values in previous['paired_intervals'].items() for metric, numbers in values.items()])
    write_sheet(book, 'final_epoch_export_checks', previous['export_checks'])
    write_sheet(book, 'timings', prior_selected['times'])
    write_sheet(book, 'paired_intervals', [dict(comparison=name, metric=metric, **numbers)
        for name, values in prior_selected['paired_intervals'].items() for metric, numbers in values.items()])
    write_sheet(book, 'export_checks', prior_selected['artifacts_verified'])
    book.save(root / 'experiments.xlsx')
    print(json.dumps(dict(comparison=summaries, paired_large_minus_small=intervals), indent=2), flush=True)


if __name__ == '__main__':
    main()
