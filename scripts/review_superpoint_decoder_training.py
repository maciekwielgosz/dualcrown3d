#!/usr/bin/env python3
"""Expose final trained checkpoints even when epoch zero wins joint selection."""
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
from scripts.train_superpoint_decoder_pilot import DEFAULT, export_predictions
from scripts.report_superpoint_decoder_pilot import paired_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    root = args.output.resolve()
    torch.set_num_threads(4)
    prepared = json.loads((root / 'prepared.json').read_text())
    validation = [row for row in prepared['entries'] if row['split'] == 'val']
    table, selected_values, checks = [], {}, []
    for method in ('control', 'fixed', 'ezsp'):
        run = root / 'runs' / method
        payload = torch.load(run / 'weights/last.pt', map_location='cpu', weights_only=False)
        epoch = payload['epoch']
        trials = json.loads((run / 'validation' / f'epoch_{epoch:03d}.json').read_text())
        key = max(trials, key=lambda key: trials[key]['score'])
        value = trials[key]
        selected_values[method] = value
        row = dict(method=method, epoch=epoch, thresholds=key,
                   point_pq=value['point']['source_balanced_pq'], point_f1=value['point']['source_balanced_f1'],
                   crown_pq=value['crown']['source_balanced_pq'], crown_f1=value['crown']['source_balanced_f1'],
                   score=value['score'], coverage=value['annotated_tree_point_coverage'],
                   sparse_recall=value['sparse_tree_recall_le100_voxels'],
                   checkpoint=str(run / 'weights/last.pt'))
        if method != 'control':
            row['graph_weight_norm'] = float(payload['model']['decoder.graph.point_update.2.weight'].norm())
            row['groups_split_by_final_prediction'] = sum(r['groups_split_by_final_prediction'] for r in value['per_plot'])
        table.append(row)
        model = CachedInstanceModel(method).cuda().eval()
        model.load_state_dict(payload['model'])
        selection = dict(epoch=epoch, validation=value, checkpoint=row['checkpoint'], checkpoint_sha256=sha256(Path(row['checkpoint'])))
        save_json(run / 'last_epoch_review.json', selection)
        destination = run / 'last_epoch_validation_exports'
        export_predictions(model, validation, selection, destination)
        for entry in validation:
            name = entry['dataset_id']
            cloud = laspy.read(destination / 'PointClouds' / f'trees_{name}.laz')
            identifiers = set(np.unique(cloud.tree_id).tolist()) - {0}
            if identifiers:
                polygons = gpd.read_file(destination / 'Segmentation3' / f'crowns_{name}.gpkg')
                tops = gpd.read_file(destination / 'Segmentation3' / f'ttops_{name}.gpkg')
                if set(polygons.tree_id) != identifiers or set(tops.tree_id) != identifiers:
                    raise AssertionError(f'Mismatched point/crown IDs: {method}/{name}')
                if not polygons.geometry.is_valid.all():
                    raise AssertionError(f'Invalid crown geometry: {method}/{name}')
            checks.append(dict(method=method, dataset_id=name, matching_ids=True, instances=len(identifiers)))
        del model, payload
        torch.cuda.empty_cache()
    intervals = {f'{second}_minus_{first}': {metric: paired_interval(selected_values[first]['per_plot'],
        selected_values[second]['per_plot'], metric) for metric in ('point', 'crown')}
        for first, second in [('control', 'fixed'), ('control', 'ezsp'), ('fixed', 'ezsp')]}
    report = dict(final_epoch=table, paired_intervals=intervals, export_checks=checks,
                  selection='All comparisons use epoch12; thresholds chosen on the same validation grid.',
                  caveat='Exploratory validation results, not held-out-test performance.')
    save_json(root / 'trained_checkpoint_report.json', report)
    book = load_workbook(root / 'experiments.xlsx')
    for title, rows in [('final_epoch', table), ('final_epoch_intervals',
        [dict(comparison=c, metric=m, **v) for c, item in intervals.items() for m, v in item.items()]),
        ('final_epoch_export_checks', checks)]:
        if title in book:
            del book[title]
        sheet = book.create_sheet(title)
        keys = list(dict.fromkeys(key for row in rows for key in row))
        sheet.append(keys)
        for row in rows:
            sheet.append([row.get(key) for key in keys])
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
    book.save(root / 'experiments.xlsx')
    print(json.dumps(dict(final_epoch=table, paired_intervals=intervals), indent=2), flush=True)


if __name__ == '__main__':
    main()
