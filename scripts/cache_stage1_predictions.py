#!/usr/bin/env python3
"""Cache frozen Model20 (stage-1) predictions and second-pass diagnostics.

Per plot: raw window masks (same protocol as the legacy calibration cache), the
consensus labels, the label-free conditioning arrays and GT-based diagnostics
(missed, absorbed and over-split trees). GT is used for diagnostics only.
"""
import argparse
import hashlib
import json
import os
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
from pointcloud.two_pass import match_instances
from pointcloud.two_pass_inference import STAGE1_KEYS, stage1_arrays
from pointcloud.two_pass_reconcile import segmentation_diagnostics
from scripts.calibrate_legacy_small_trees import small_hits
from scripts.evaluate_combined_full_crowns import metrics
from scripts.predict_dual_head import predict
from scripts.train_supervision_v4 import REAL, build

DEFAULT_OUTPUT = PROJECT / 'outputs/dualcrown3d_two_pass_v1'
MODEL20_REPORT = PROJECT / 'output_20_dualcrown3d_joint_finetune/inference_report.json'
BASELINE = PROJECT / 'outputs/dualcrown3d_supervision_v4/baseline/val/metrics.json'
LEGACY_RAW = PROJECT / 'outputs/dualcrown3d_legacy_small_tree_calibration/raw_val'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')
    os.replace(temporary, path)


def eligible_rows(split):
    rows = [r for r in read_manifest(REAL, split) if r.get('point_eval_eligible') == 'true']
    if split == 'train':
        rows = [r for r in rows if r.get('train_eligible') == 'true']
    return rows


def plot_diagnostics(arrays, labels, confidence, gt, ignore):
    truth = arrays['tree_id']
    gt_ids, matched, best_iou = match_instances(truth, labels)
    counts = np.bincount(np.searchsorted(gt_ids, truth[truth > 0]), minlength=len(gt_ids)) if len(gt_ids) else np.zeros(0, int)
    missed = matched == 0
    absorbed = np.zeros(len(gt_ids), bool)
    absorbed_by_found = np.zeros(len(gt_ids), bool)
    found_ids = set(int(v) for v in matched[matched > 0])
    for k, identifier in enumerate(gt_ids):
        if not missed[k]:
            continue
        owners = labels[truth == identifier]
        assigned = owners > 0
        if assigned.mean() >= .8:
            absorbed[k] = True
            host = np.bincount(owners[assigned]).argmax()
            absorbed_by_found[k] = int(host) in found_ids
    small = counts <= 100
    geometries = [r['geometry'] for r in instance_records(arrays, labels, confidence)]
    crown = metrics(gt.geometry, geometries, ignore)
    result = dict(segmentation_diagnostics(truth, labels))
    result.update(gt_total=int(len(gt_ids)), found=int((~missed).sum()), missed=int(missed.sum()),
                  missed_ids=[int(v) for v in gt_ids[missed]], missed_counts=[int(v) for v in counts[missed]],
                  absorbed=int(absorbed.sum()), absorbed_by_found=int(absorbed_by_found.sum()),
                  small_gt=int(small.sum()), small_missed=int((small & missed).sum()),
                  small_absorbed=int((small & absorbed).sum()),
                  mean_best_iou=float(best_iou.mean()) if len(best_iou) else 0.,
                  predicted_instances=int(len(geometries)), crown=crown,
                  small_crowns=small_hits(gt.geometry, geometries))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--splits', nargs='+', default=['val', 'train'])
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = json.loads(MODEL20_REPORT.read_text())
    checkpoint = Path(report['checkpoint'])
    digest = sha256(checkpoint)
    if digest != report['checkpoint_sha256']:
        raise ValueError('Model20 checkpoint differs from the frozen output_20 report')
    config = report['mask_config']
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(BASELINE.read_text())['legacy_consensus']
    protocol_path = output / 'protocol.json'
    protocol = dict(
        stage1_checkpoint=str(checkpoint), stage1_sha256=digest, stage1_config=config, stage1_config_sha256=config_sha,
        manifest=str(REAL), manifest_sha256=sha256(REAL), window_m=20., overlap_m=8., max_points_eval=40000,
        raw_object_threshold=.05, validation_plots=len(eligible_rows('val')), train_plots=len(eligible_rows('train')),
        baseline=dict(point_sb_pq=baseline['point_metrics']['source_balanced_pq'],
                      crown_sb_pq=baseline['metrics']['source_balanced_pq'],
                      point_sb_f1=baseline['point_metrics']['source_balanced_f1'],
                      crown_sb_f1=baseline['metrics']['source_balanced_f1'],
                      point_precision=baseline['point_metrics']['precision'],
                      crown_precision=baseline['metrics']['precision'],
                      small_10_tp=6, small_10_gt=161, small_4_tp=0, small_4_gt=39),
        gate=dict(point_sb_pq_tolerance=.005, crown_sb_pq_tolerance=.005, precision_tolerance=.01,
                  large_recall_tolerance=.01, small_10_minimum_tp=16,
                  description='Preregistered before training: keep Model20 point/crown SB-PQ (-0.005), pooled '
                              'precision (-0.01) and large-tree recall (-0.01) while reaching >=16/161 small '
                              'crowns (<=10 m2) on all 14 validation plots. Test is never used for selection.'),
        polygonizer='convex hull of instance voxels buffered by half a voxel (Model20 exporter)',
        test_used=False)
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text())
        for key in ('stage1_sha256', 'stage1_config_sha256', 'manifest_sha256'):
            if existing[key] != protocol[key]:
                raise ValueError(f'Existing protocol differs in {key}; use a new output folder')
    else:
        write_json(protocol_path, protocol)
    model = None
    diagnostics = {}
    for split in args.splits:
        rows = eligible_rows(split)
        if args.limit:
            rows = rows[:args.limit]
        folder = output / 'stage1' / split
        folder.mkdir(parents=True, exist_ok=True)
        for number, row in enumerate(rows, 1):
            path = folder / f"{row['dataset_id']}.npz"
            input_hash = sha256(row['output'])
            started = time.monotonic()
            if path.exists():
                with np.load(path) as archive:
                    if (str(archive['checkpoint_sha256']) != digest or str(archive['input_sha256']) != input_hash
                            or str(archive['config_sha256']) != config_sha):
                        raise ValueError(f'Stale stage-1 cache: {path}')
                    diagnostics[row['dataset_id']] = json.loads(str(archive['diagnostics']))
                print(f'{split} {number}/{len(rows)} cached {row["dataset_id"]}', flush=True)
                continue
            arrays = load_npz(row['output'])
            raw = None
            legacy = LEGACY_RAW / f"{row['dataset_id']}.npz"
            if split == 'val' and legacy.exists():
                with np.load(legacy) as archive:
                    if str(archive['checkpoint_sha256']) == digest and str(archive['input_sha256']) == input_hash:
                        raw = {k: archive[k] for k in archive.files if k not in ('checkpoint_sha256', 'input_sha256')}
            if raw is None:
                if model is None:
                    model, _ = build(checkpoint)
                    model.eval()
                raw = predict(model, arrays, owner_only=True, raw_object_threshold=.05)
            labels, confidence, _, source = merge_masks_with_sources(arrays, raw, config)
            stage1 = stage1_arrays(arrays, raw, labels, confidence, source)
            gt = gpd.read_file(row['gt_vector'])
            ignore = shapely.from_wkb(Path(row['ignore_vector']).read_bytes()) if row.get('ignore_vector') else None
            diagnostic = plot_diagnostics(arrays, labels, confidence, gt, ignore)
            diagnostic.update(dataset_id=row['dataset_id'], split=split, source_dataset=row['source_dataset'],
                              collection=row['collection'], voxels=int(len(labels)),
                              stage1_seconds=float(raw['seconds']))
            diagnostics[row['dataset_id']] = diagnostic
            payload = {**raw, **{k: stage1[k] for k in STAGE1_KEYS},
                       'missed_gt_ids': np.asarray(diagnostic['missed_ids'], np.int64),
                       'missed_gt_counts': np.asarray(diagnostic['missed_counts'], np.int64),
                       'checkpoint_sha256': np.asarray(digest), 'input_sha256': np.asarray(input_hash),
                       'config_sha256': np.asarray(config_sha), 'diagnostics': np.asarray(json.dumps(diagnostic))}
            temporary = path.with_name(path.stem + '.tmp.npz')
            np.savez_compressed(temporary, **payload)
            os.replace(temporary, path)
            print(f'{split} {number}/{len(rows)} {row["dataset_id"]}: GT {diagnostic["gt_total"]} '
                  f'missed {diagnostic["missed"]} absorbed {diagnostic["absorbed"]} '
                  f'oversplit {diagnostic["oversplit_gt"]} {time.monotonic()-started:.1f}s', flush=True)
    summary = {}
    for split in args.splits:
        rows = [d for d in diagnostics.values() if d['split'] == split]
        if not rows:
            continue
        keys = ('gt_total', 'found', 'missed', 'absorbed', 'absorbed_by_found', 'small_gt', 'small_missed',
                'small_absorbed', 'oversplit_gt', 'undersegmented_pred', 'large_gt', 'large_tp', 'predicted_instances')
        summary[split] = {k: int(sum(d[k] for d in rows)) for k in keys}
        summary[split]['plots'] = len(rows)
        summary[split]['small_10_tp'] = int(sum(d['small_crowns']['small_10_tp'] for d in rows))
        summary[split]['small_10_gt'] = int(sum(d['small_crowns']['small_10_gt'] for d in rows))
    write_json(output / 'stage1' / 'diagnostics.json', dict(summary=summary, per_plot=diagnostics))
    book = Workbook()
    sheet = book.active
    sheet.title = 'per_plot'
    fields = ['dataset_id', 'split', 'source_dataset', 'collection', 'voxels', 'gt_total', 'found', 'missed',
              'absorbed', 'absorbed_by_found', 'small_gt', 'small_missed', 'small_absorbed', 'oversplit_gt',
              'undersegmented_pred', 'large_gt', 'large_tp', 'predicted_instances', 'mean_best_iou', 'stage1_seconds']
    sheet.append(fields + ['crown_tp', 'crown_fp', 'crown_fn', 'small_10_tp', 'small_10_gt'])
    for d in diagnostics.values():
        sheet.append([d[k] for k in fields] + [d['crown']['tp'], d['crown']['fp'], d['crown']['fn'],
                                               d['small_crowns']['small_10_tp'], d['small_crowns']['small_10_gt']])
    sheet.freeze_panes = 'A2'
    total = book.create_sheet('summary')
    for split, values in summary.items():
        total.append([split] + [f'{k}={v}' for k, v in values.items()])
    book.save(output / 'stage1' / 'diagnostics.xlsx')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
