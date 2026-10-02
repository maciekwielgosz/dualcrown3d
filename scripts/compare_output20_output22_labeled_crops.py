#!/usr/bin/env python3
"""Matched 20 m labelled-crop comparison of output_20 and output_22 checkpoints.

Both models receive the exact same retained ALS voxels and are scored with the
same point matcher and 0.5 m support-polygon exporter. This is not an evaluation
of either stitched whole-scene output directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import geopandas as gpd
import laspy
import numpy as np
from openpyxl import Workbook
import shapely
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.instance_output import merge_masks_with_sources
from scripts.benchmark_superpoint_algorithms import save_json, sha256
from scripts.calibrate_legacy_small_trees import small_hits
from scripts.evaluate_combined_full_crowns import aggregate, metrics
from scripts.predict_dual_head import predict
from scripts.train_superpoint_decoder_pilot import crown_records, load_crop, point_diagnostics
from scripts.train_supervision_v4 import build

PILOT = PROJECT / 'outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot'
OUTPUT20 = PROJECT / 'output_20_dualcrown3d_joint_finetune'
DEFAULT = PROJECT / 'outputs/dualcrown3d_output20_output22_matched_crops_v1'


def crop_scene(arrays):
    coord = arrays['coord'].astype(np.float32)
    xy = arrays['world_xy'].astype(np.float64)
    origin = xy[0] - coord[0, :2]
    if np.max(np.abs((coord[:, :2].astype(np.float64) + origin) - xy)) > .001:
        raise ValueError('Crop-to-world XY transform is not a constant translation')
    grid = np.floor((coord - coord.min(0)) / .25).astype(np.int32)
    if len(np.unique(grid, axis=0)) != len(coord):
        raise ValueError('Training crop has duplicate sparse-grid coordinates')
    return dict(coord=coord, grid_coord=grid,
                intensity=arrays['feat'][:, 3].astype(np.float32),
                source_origin=np.asarray([*origin, 0.], np.float64),
                voxel_size=np.float32(.25), world_xy=xy)


def score(entry, arrays, labels, confidence, crowns):
    gt = [shapely.from_wkb(bytes.fromhex(value)) for value in entry['gt_wkb']]
    ignore = shapely.from_wkb(bytes.fromhex(entry['ignore_wkb']))
    polygons = list(crowns)
    return dict(dataset_id=entry['dataset_id'],
                source_dataset=entry['source_dataset'], collection=entry['collection'],
                point=point_diagnostics(arrays['tree_id'], labels),
                crown=metrics(gt, polygons, ignore),
                small_crown=small_hits(gt, polygons),
                assigned_gt_points=int(((arrays['tree_id'] > 0) & (labels > 0)).sum()),
                annotated_gt_points=int((arrays['tree_id'] > 0).sum()),
                predicted_instances=int(len(polygons)))


def summarize(rows):
    point = aggregate([{**{k: row[k] for k in ('dataset_id', 'source_dataset', 'collection')},
                        **row['point']} for row in rows])
    crown = aggregate([{**{k: row[k] for k in ('dataset_id', 'source_dataset', 'collection')},
                        **row['crown']} for row in rows])
    small = {f'up_to_{int(limit)}_m2': dict(
        gt=sum(row['small_crown'][f'small_{limit:g}_gt'] for row in rows),
        tp=sum(row['small_crown'][f'small_{limit:g}_tp'] for row in rows))
        for limit in (4., 10.)}
    for value in small.values():
        value['recall'] = value['tp'] / max(value['gt'], 1)
    sparse_gt = sum(row['point']['sparse_tree_count'] for row in rows)
    sparse_tp = sum(row['point']['sparse_tree_tp'] for row in rows)
    return dict(point=point, crown=crown, small_crowns=small,
                sparse_points=dict(gt=sparse_gt, tp=sparse_tp,
                                   recall=sparse_tp / max(sparse_gt, 1)),
                annotated_point_coverage=sum(row['assigned_gt_points'] for row in rows) /
                    max(sum(row['annotated_gt_points'] for row in rows), 1),
                predicted_instances=sum(row['predicted_instances'] for row in rows))


def write_excel(path, reports):
    book = Workbook()
    summary = book.active
    summary.title = 'comparison'
    fields = ('model', 'point_sb_pq', 'point_sb_f1', 'crown_sb_pq', 'crown_sb_f1',
              'point_small_tp', 'point_small_gt', 'crown_le4_tp', 'crown_le4_gt',
              'crown_le10_tp', 'crown_le10_gt', 'predicted_instances')
    summary.append(fields)
    for model, report in reports.items():
        result = report['summary']
        summary.append((model, result['point']['source_balanced_pq'],
            result['point']['source_balanced_f1'], result['crown']['source_balanced_pq'],
            result['crown']['source_balanced_f1'], result['sparse_points']['tp'],
            result['sparse_points']['gt'], result['small_crowns']['up_to_4_m2']['tp'],
            result['small_crowns']['up_to_4_m2']['gt'], result['small_crowns']['up_to_10_m2']['tp'],
            result['small_crowns']['up_to_10_m2']['gt'], result['predicted_instances']))
    summary.freeze_panes = 'A2'
    for model, report in reports.items():
        sheet = book.create_sheet(f'plots_{model}')
        fields = ('dataset_id', 'source_dataset', 'collection', 'point_tp', 'point_fp',
                  'point_fn', 'crown_tp', 'crown_fp', 'crown_fn', 'small_point_tp',
                  'small_point_gt', 'small_4_tp', 'small_4_gt', 'small_10_tp', 'small_10_gt')
        sheet.append(fields)
        for row in report['per_plot']:
            sheet.append((row['dataset_id'], row['source_dataset'], row['collection'],
                row['point']['tp'], row['point']['fp'], row['point']['fn'],
                row['crown']['tp'], row['crown']['fp'], row['crown']['fn'],
                row['point']['sparse_tree_tp'], row['point']['sparse_tree_count'],
                row['small_crown']['small_4_gt'] and row['small_crown']['small_4_tp'] or 0,
                row['small_crown']['small_4_gt'], row['small_crown']['small_10_tp'],
                row['small_crown']['small_10_gt']))
        sheet.freeze_panes = 'A2'
    book.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--limit', type=int, default=0, help='Smoke-test first N crops only')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Protected comparison output: {output}')
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the output_20 inference')
    prepared = json.loads((PILOT / 'prepared.json').read_text())
    entries = [entry for entry in prepared['entries'] if entry['split'] == 'val']
    if args.limit:
        entries = entries[:args.limit]
    old_report = json.loads((OUTPUT20 / 'inference_report.json').read_text())
    checkpoint = Path(old_report['checkpoint'])
    if sha256(checkpoint) != old_report['checkpoint_sha256']:
        raise ValueError('Output_20 checkpoint digest mismatch')
    model, _ = build(checkpoint)
    model.eval()
    new_report = json.loads((PILOT / 'size_ablation_large_w192_q128.json').read_text())
    chosen = next(row for row in new_report['comparison'] if row['variant'] == 'large_selected')
    new_checkpoint = Path(chosen['checkpoint'])
    if chosen['epoch'] != 12 or chosen['thresholds'] != 'o0.1_m0.5':
        raise ValueError('Unexpected output_22 selected checkpoint or threshold')
    exports = PILOT / 'runs/ezsp_large_w192_q128/last_epoch_validation_exports'
    rows20, rows22 = [], []
    began = time.monotonic()
    for number, entry in enumerate(entries, 1):
        arrays = load_crop(entry['cache'])
        scene = crop_scene(arrays)
        raw = predict(model, scene, max_points=12000, owner_only=True,
                      raw_object_threshold=.05)
        labels20, confidence20, _, _ = merge_masks_with_sources(scene, raw, old_report['mask_config'])
        crowns20 = crown_records(arrays, labels20, confidence20)
        rows20.append(score(entry, arrays, labels20, confidence20,
                            [record['geometry'] for record in crowns20]))
        cloud = laspy.read(exports / 'PointClouds' / f"trees_{entry['dataset_id']}.laz")
        labels22 = np.asarray(cloud.tree_id).astype(np.int32)
        confidence22 = np.asarray(cloud.confidence)
        if len(labels22) != len(arrays['coord']) or not np.array_equal(
                np.asarray(cloud.reference_tree_id), arrays['tree_id']):
            raise ValueError(f'Output_22 export/truth misalignment: {entry["dataset_id"]}')
        crown22 = gpd.read_file(exports / 'Segmentation3' / f"crowns_{entry['dataset_id']}.gpkg")
        rows22.append(score(entry, arrays, labels22, confidence22, crown22.geometry))
        print(f'matched {number}/{len(entries)} {entry["dataset_id"]} '
              f'20={len(crowns20)} 22={len(crown22)}', flush=True)
    if not args.limit:
        stored = json.loads((PILOT / 'runs/ezsp_large_w192_q128/validation/epoch_012.json').read_text())['o0.1_m0.5']
        control = summarize(rows22)
        for group, key in (('point', 'source_balanced_pq'), ('point', 'source_balanced_f1'),
                           ('crown', 'source_balanced_pq'), ('crown', 'source_balanced_f1')):
            if abs(control[group][key] - stored[group][key]) > 1e-8:
                raise AssertionError(f'Output_22 export metric mismatch: {group}/{key}')
    reports = dict(output20=dict(summary=summarize(rows20), per_plot=rows20),
                   output22=dict(summary=summarize(rows22), per_plot=rows22))
    output.mkdir(parents=True)
    payload = dict(protocol='same 14 native ALS validation crops, same voxels and point/0.5m support-polygon IoU@0.5 evaluator',
                   scope='crop-level comparison; not stitched output_20/output_22 full-scene GPKG metrics',
                   output20_checkpoint=str(checkpoint), output20_sha256=sha256(checkpoint),
                   output20_merge_config=old_report['mask_config'],
                   output22_checkpoint=str(new_checkpoint), output22_sha256=sha256(new_checkpoint),
                   output22_thresholds=chosen['thresholds'], plots=len(entries),
                   seconds=time.monotonic() - began, results=reports,
                   caveats=['Output_20 production dual-consensus is rerun on the same 20m crops, not cut out of the archived full-scene export.',
                            'Both crown sets are reconstructed from labelled point supports by the same 0.5m raster method.',
                            'Validation split and thresholds have already been explored; this is not a held-out test.'])
    save_json(output / 'comparison.json', payload)
    write_excel(output / 'comparison.xlsx', reports)
    print(json.dumps({model: report['summary'] for model, report in reports.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
