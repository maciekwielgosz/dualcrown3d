#!/usr/bin/env python3
"""Frozen, matched full-plot held-out test with inspectable LAS/LAZ exports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import geopandas as gpd
import laspy
import numpy as np
from openpyxl import Workbook
from pyproj import CRS
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import Point
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz, read_manifest
from pointcloud.instance_output import merge_masks, merge_masks_with_sources
from scripts.benchmark_superpoint_algorithms import load_methods, save_json, sha256
from scripts.calibrate_legacy_small_trees import small_hits
from scripts.evaluate_combined_full_crowns import aggregate, metrics
from scripts.evaluate_pointcloud_litept import starts_for_axis, ownership_intervals, dbh_naslund
from scripts.predict_dual_head import predict, color_ids
from scripts.predict_superpoint_wide_scene import MERGE, load_models, predict_window
from scripts.train_superpoint_decoder_pilot import crown_records, point_diagnostics
from scripts.train_supervision_v4 import REAL, build

OLD = PROJECT / 'output_20_dualcrown3d_joint_finetune'
PILOT = PROJECT / 'outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot'
NEW = PILOT / 'runs/ezsp_large_w192_q128/weights/best.pt'
DEFAULT = PROJECT / 'output_23_labeled_test_output20_vs_output22'


@torch.no_grad()
def wide_predict_plot(arrays, backbone, decoder, embedding, csr, ezsp, device,
                      seed, max_points=12000):
    xyz = arrays['coord']
    xs = starts_for_axis(float(xyz[:, 0].min()), float(xyz[:, 0].max()), 20., 8.)
    ys = starts_for_axis(float(xyz[:, 1].min()), float(xyz[:, 1].max()), 20., 8.)
    xmax, ymax = float(xyz[:, 0].max()), float(xyz[:, 1].max())
    xo = ownership_intervals(xs, 20., float(xyz[:, 0].min()),
                             float(np.nextafter(xyz[:, 0].max(), np.float32(np.inf))))
    yo = ownership_intervals(ys, 20., float(xyz[:, 1].min()),
                             float(np.nextafter(xyz[:, 1].max(), np.float32(np.inf))))
    tree = cKDTree(xyz[:, :2])
    visited = np.zeros(len(xyz), np.uint8)
    semantic = np.zeros(len(xyz), np.float32)
    members, probabilities, objects, offsets = [], [], [], [0]
    windows = 0
    start = time.monotonic()
    for xi, x in enumerate(xs):
        rng = np.random.default_rng(seed + xi)
        for yi, y in enumerate(ys):
            context = np.asarray(sorted(tree.query_ball_point([x + 10., y + 10.],
                                  10.0001, p=np.inf)), dtype=np.int64)
            if not len(context):
                continue
            xend = xmax if xi == len(xs) - 1 else x + 20.
            yend = ymax if yi == len(ys) - 1 else y + 20.
            context = context[(xyz[context, 0] >= x) & (xyz[context, 0] <= xend)
                              & (xyz[context, 1] >= y) & (xyz[context, 1] <= yend)]
            a, b = xo[xi]
            c, d = yo[yi]
            owner = context[(xyz[context, 0] >= a) & (xyz[context, 0] < b)
                            & (xyz[context, 1] >= c) & (xyz[context, 1] < d)]
            if not len(owner):
                continue
            for owned in np.array_split(owner, max(1, math.ceil(len(owner) / max_points))):
                extra = np.setdiff1d(context, owned, assume_unique=True)
                capacity = max_points - len(owned)
                if len(extra) > capacity:
                    extra = rng.choice(extra, capacity, replace=False)
                chosen = np.sort(np.concatenate((owned, extra)))
                own_local = np.flatnonzero(np.isin(chosen, owned))
                masks, object_score, point_semantic = predict_window(
                    arrays, chosen, backbone, decoder, embedding, csr, ezsp, device)
                if np.any(visited[owned]):
                    raise AssertionError('Repeated owner voxel')
                visited[owned] = 1
                semantic[owned] = point_semantic[own_local]
                for q in np.flatnonzero(object_score >= .1):
                    retained = np.flatnonzero(masks[q] >= .2)
                    if len(retained) < 4:
                        continue
                    center = np.average(xyz[chosen[retained], :2], axis=0,
                                        weights=masks[q, retained])
                    if not (a <= center[0] < b and c <= center[1] < d):
                        continue
                    members.append(chosen[retained].astype(np.int32))
                    probabilities.append(masks[q, retained].astype(np.float16))
                    objects.append(float(object_score[q]))
                    offsets.append(offsets[-1] + len(retained))
                windows += 1
    if not visited.all():
        raise AssertionError(f'{int((visited == 0).sum())} plot voxels lack owner predictions')
    raw = dict(candidate_offset=np.asarray(offsets, np.int64),
               point_index=np.concatenate(members) if members else np.empty(0, np.int32),
               point_score=np.concatenate(probabilities) if probabilities else np.empty(0, np.float16),
               object_score=np.asarray(objects, np.float32), point_probability=semantic,
               seconds=time.monotonic() - start, windows=windows)
    return raw


def score(row, arrays, gt, ignore, labels, polygons):
    names = {k: row[k] for k in ('dataset_id', 'source_dataset', 'collection')}
    point = point_diagnostics(arrays['tree_id'], labels)
    crown = metrics(gt.geometry, polygons, ignore)
    small = small_hits(gt.geometry, polygons)
    return dict(**names, point=point, crown=crown, small_crowns=small,
                predicted_instances=len(polygons),
                known_tree_voxel_coverage=point['assigned_tree_points'] /
                    max(point['annotated_tree_points'], 1))


def summary(rows):
    point = aggregate([{**{key: row[key] for key in ('dataset_id','source_dataset','collection')},
                        **row['point']} for row in rows])
    crown = aggregate([{**{key: row[key] for key in ('dataset_id','source_dataset','collection')},
                        **row['crown']} for row in rows])
    small = {}
    for limit in (4., 10.):
        gt = sum(row['small_crowns'][f'small_{limit:g}_gt'] for row in rows)
        tp = sum(row['small_crowns'][f'small_{limit:g}_tp'] for row in rows)
        small[f'up_to_{limit:g}_m2'] = dict(gt=gt, tp=tp, recall=tp/max(gt,1))
    sparse_gt = sum(row['point']['sparse_tree_count'] for row in rows)
    sparse_tp = sum(row['point']['sparse_tree_tp'] for row in rows)
    return dict(point=point, crown=crown, small_crowns=small,
                sparse_tree_le100_voxels=dict(gt=sparse_gt, tp=sparse_tp,
                                              recall=sparse_tp/max(sparse_gt,1)),
                predicted_instances=sum(row['predicted_instances'] for row in rows),
                known_tree_voxel_coverage=sum(row['point']['assigned_tree_points'] for row in rows) /
                    max(sum(row['point']['annotated_tree_points'] for row in rows), 1))


def write_cloud(path, arrays, labels, confidence, semantic, crs):
    xyz = arrays['coord'].astype(np.float64).copy()
    xyz[:, :2] += arrays['source_origin'][:2]
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.offsets = xyz.min(0)
    header.scales = [.001, .001, .001]
    if crs is not None:
        header.add_crs(CRS.from_user_input(crs))
    for name, dtype in [('tree_id',np.uint32),('reference_tree_id',np.int32),
                        ('height_agl',np.float32),('confidence',np.float32),
                        ('pred_semantic',np.uint8)]:
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = xyz.T
    cloud.tree_id = labels.astype(np.uint32)
    cloud.reference_tree_id = arrays['tree_id'].astype(np.int32)
    cloud.height_agl = arrays['coord'][:, 2].astype(np.float32)
    cloud.confidence = confidence.astype(np.float32)
    cloud.pred_semantic = (semantic >= .3).astype(np.uint8)
    cloud.red, cloud.green, cloud.blue = color_ids(labels).T
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud.write(path.with_suffix('.las'))
    cloud.write(path.with_suffix('.laz'))
    check = laspy.read(path.with_suffix('.laz'))
    if (not np.array_equal(np.asarray(check.tree_id), labels.astype(np.uint32)) or
        not np.array_equal(np.asarray(check.reference_tree_id), arrays['tree_id'].astype(np.int32))):
        raise AssertionError(f'LAS/LAZ ID roundtrip failed: {path}')
    return dict(las=str(path.with_suffix('.las')), laz=str(path.with_suffix('.laz')),
                points=len(labels), labelled_points=int((labels>0).sum()))


def write_vectors(folder, stem, records, crs):
    folder.mkdir(parents=True, exist_ok=True)
    crowns = gpd.GeoDataFrame(dict(treeID=[r['tree_id'] for r in records],
                                   area_m2=[r['geometry'].area for r in records]),
                              geometry=[r['geometry'] for r in records], crs=crs)
    tops = gpd.GeoDataFrame(dict(treeID=[r['tree_id'] for r in records],
                                 Z=[r['height'] for r in records],
                                 dbh=[round(dbh_naslund(r['height']),2) for r in records]),
                            geometry=[Point(r['top_x'],r['top_y'],r['height']) for r in records], crs=crs)
    crowns.to_file(folder/f'crowns_{stem}.gpkg', driver='GPKG', index=False)
    tops.to_file(folder/f'ttops_{stem}.gpkg', driver='GPKG', index=False)
    if not crowns.geometry.is_valid.all() or set(crowns.treeID) != set(tops.treeID):
        raise AssertionError(f'Invalid vectors for {stem}')
    return dict(crowns=str(folder/f'crowns_{stem}.gpkg'),
                treetops=str(folder/f'ttops_{stem}.gpkg'), instances=len(records))


def excel(path, reports):
    book = Workbook()
    sheet = book.active
    sheet.title = 'test_comparison'
    sheet.append(('model','plots','point_SB_PQ','point_SB_F1','crown_SB_PQ','crown_SB_F1',
                  'small_point_TP','small_point_GT','crown_le4_TP','crown_le4_GT',
                  'crown_le10_TP','crown_le10_GT','predicted_instances'))
    for model, result in reports.items():
        s = result['summary']
        sheet.append((model,len(result['per_plot']),s['point']['source_balanced_pq'],
                      s['point']['source_balanced_f1'],s['crown']['source_balanced_pq'],
                      s['crown']['source_balanced_f1'],s['sparse_tree_le100_voxels']['tp'],
                      s['sparse_tree_le100_voxels']['gt'],s['small_crowns']['up_to_4_m2']['tp'],
                      s['small_crowns']['up_to_4_m2']['gt'],s['small_crowns']['up_to_10_m2']['tp'],
                      s['small_crowns']['up_to_10_m2']['gt'],s['predicted_instances']))
    for model, result in reports.items():
        tab = book.create_sheet(f'{model}_per_plot')
        tab.append(('dataset_id','source_dataset','collection','point_TP','point_FP','point_FN',
                    'crown_TP','crown_FP','crown_FN','small_point_TP','small_point_GT',
                    'crown_le4_TP','crown_le4_GT','crown_le10_TP','crown_le10_GT'))
        for row in result['per_plot']:
            tab.append((row['dataset_id'],row['source_dataset'],row['collection'],
                        row['point']['tp'],row['point']['fp'],row['point']['fn'],
                        row['crown']['tp'],row['crown']['fp'],row['crown']['fn'],
                        row['point']['sparse_tree_tp'],row['point']['sparse_tree_count'],
                        row['small_crowns']['small_4_gt'] and row['small_crowns']['small_4_tp'] or 0,
                        row['small_crowns']['small_4_gt'],row['small_crowns']['small_10_tp'],
                        row['small_crowns']['small_10_gt']))
        tab.freeze_panes='A2'
    book.save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--limit',type=int,default=0,
                        help='Limit to first N eligible plots for a smoke test')
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for both test models')
    torch.set_num_threads(4)
    rows=[row for row in read_manifest(REAL,'test') if row.get('point_eval_eligible')=='true']
    if args.limit:
        rows=rows[:args.limit]
    output=args.output.resolve()
    old_report=json.loads((OLD/'inference_report.json').read_text())
    old_checkpoint=Path(old_report['checkpoint'])
    if sha256(old_checkpoint)!=old_report['checkpoint_sha256']:
        raise ValueError('Output_20 checkpoint differs from frozen selection')
    new_payload=torch.load(NEW,map_location='cpu',weights_only=False)
    if new_payload['epoch']!=12 or new_payload['config']['run_name']!='ezsp_large_w192_q128':
        raise ValueError('Unexpected output_22 checkpoint')
    initial=Path(new_payload['initial_checkpoint'])
    embedding_path=PROJECT/'outputs/dualcrown3d_superpoint_v1/stage2_instance_embedding/embedding.pt'
    if sha256(initial)!=new_payload['config']['initial_sha256']:
        raise ValueError('Output_22 frozen backbone differs from training')
    run=dict(split='test', test_eligible_plots=len(rows), manifest=str(REAL),
             manifest_sha256=sha256(REAL), output20_checkpoint=str(old_checkpoint),
             output20_sha256=sha256(old_checkpoint), output20_config=old_report['mask_config'],
             output22_checkpoint=str(NEW), output22_sha256=sha256(NEW),
             output22_backbone=str(initial), output22_embedding=str(embedding_path),
             output22_config=MERGE, test_threshold_selection='none; use previously frozen configurations',
             protocol='same full native ALS plots; point IoU>=0.5; crown IoU>=0.5; source-balanced PQ/F1; explicit annotation ignore',
             las_scope='voxelized model-input cloud; XY source coordinates; Z=height AGL, not original absolute elevation')
    signature=hashlib.sha256(json.dumps(run,sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    run_path=output/'run.json'
    if run_path.exists():
        if json.loads(run_path.read_text())['signature']!=signature:
            raise ValueError('Existing output uses another checkpoint/configuration')
    else:
        save_json(run_path,dict(**run,signature=signature))
    if (output/'comparison.json').exists():
        raise FileExistsError('Completed test comparison is protected')
    model20,_=build(old_checkpoint)
    model20.eval()
    backbone,decoder,embedding=load_models(new_payload,initial,embedding_path,'cuda:0')
    _,csr,ezsp,_=load_methods()
    results={'Model20':[],'Model22':[]}
    started=time.monotonic()
    for number,row in enumerate(rows,1):
        marker=output/'metrics/per_plot'/f"{row['dataset_id']}.json"
        if marker.exists():
            done=json.loads(marker.read_text())
            if done['signature']!=signature:
                raise ValueError(f'Existing plot has another run signature: {marker}')
            for model in ('Model20','Model22'):
                results[model].append(done['metrics'][model])
            print(f'skip verified {number}/{len(rows)} {row["dataset_id"]}',flush=True)
            continue
        for model in ('Model20','Model22'):
            folder=output/model
            if (folder/'PointClouds'/f"trees_{row['dataset_id']}.las").exists():
                raise RuntimeError(f'Incomplete existing plot output, protect and inspect: {row["dataset_id"]}')
        arrays=load_npz(row['output'])
        gt=gpd.read_file(row['gt_vector'])
        ignore=shapely.from_wkb(Path(row['ignore_vector']).read_bytes()) if row.get('ignore_vector') else None
        stage=time.monotonic()
        raw20=predict(model20,arrays,owner_only=True,raw_object_threshold=.05)
        labels20,confidence20,instances20,_=merge_masks_with_sources(arrays,raw20,old_report['mask_config'])
        result20=score(row,arrays,gt,ignore,labels20,[item['geometry'] for item in instances20])
        raw22=wide_predict_plot(arrays,backbone,decoder,embedding,csr,ezsp,'cuda:0',
                                seed=20261001+number*1000)
        labels22,confidence22,_=merge_masks(arrays,raw22,MERGE)
        polygon_arrays={**arrays,'world_xy':arrays['coord'][:,:2].astype(np.float64)+arrays['source_origin'][:2]}
        instances22=crown_records(polygon_arrays,labels22,confidence22)
        result22=score(row,arrays,gt,ignore,labels22,[item['geometry'] for item in instances22])
        artifacts={}
        for model,labels,confidence,semantic,records in (
            ('Model20',labels20,confidence20,raw20['point_probability'],instances20),
            ('Model22',labels22,confidence22,raw22['point_probability'],instances22)):
            folder=output/model
            cloud=write_cloud(folder/'PointClouds'/f"trees_{row['dataset_id']}",
                              arrays,labels,confidence,semantic,gt.crs)
            vectors=write_vectors(folder/'Segmentation3',row['dataset_id'],records,gt.crs)
            artifacts[model]=dict(cloud=cloud,vectors=vectors)
        marker.parent.mkdir(parents=True,exist_ok=True)
        save_json(marker,dict(signature=signature,metrics={'Model20':result20,'Model22':result22},
                              artifacts=artifacts,seconds=time.monotonic()-stage,
                              windows={'Model20':int(raw20['windows']),'Model22':int(raw22['windows'])}))
        results['Model20'].append(result20)
        results['Model22'].append(result22)
        print(f'test {number}/{len(rows)} {row["dataset_id"]}: '
              f"20={len(instances20)} 22={len(instances22)} "
              f"PQcrown={result20['crown']['tp']}/{result22['crown']['tp']} TP "
              f'{time.monotonic()-stage:.1f}s',flush=True)
    reports={model:dict(summary=summary(records),per_plot=records)
             for model,records in results.items()}
    payload=dict(**run,signature=signature,plots=len(rows),elapsed_seconds=time.monotonic()-started,
                 results=reports,caveats=['Held-out split was not used for threshold selection.',
                    'LAS/LAZ are voxelized model inputs, not original full-resolution returns.',
                    'Model20 uses its original dual-consensus crown geometry; Model22 uses point-supported 0.5m filled footprints.'])
    save_json(output/'comparison.json',payload)
    excel(output/'comparison.xlsx',reports)
    (output/'README.md').write_text(
        '# Held-out ALS test: Model20 versus Model22\n\n'
        'Model20 and Model22 used the same labelled test plots, with previously frozen checkpoints and thresholds. '
        'Model20/ and Model22/ each contain LAS and LAZ voxelized clouds plus crown/treetop GeoPackages. '
        'LAS XY is in the source coordinate system; Z is height above ground, not absolute elevation. '
        'tree_id is the prediction, reference_tree_id is the ground truth for inspection only and was never used by inference. '
        'Unknown-reference points use negative reference_tree_id. See comparison.xlsx and comparison.json for metrics.\n')
    print(json.dumps({model:report['summary'] for model,report in reports.items()},indent=2),flush=True)


if __name__=='__main__':
    main()
