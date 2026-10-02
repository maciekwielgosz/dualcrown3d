#!/usr/bin/env python3
"""Calibrate the retained DualCrown3D postprocessor for small trees on native validation."""
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
from scipy.optimize import linear_sum_assignment

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz, read_manifest
from pointcloud.instance_output import merge_masks_with_sources, point_instance_metrics
from scripts.evaluate_combined_full_crowns import aggregate, metrics
from scripts.predict_dual_head import predict
from scripts.train_supervision_v4 import INITIAL, REAL, build, configurations

DEFAULT_OUTPUT = PROJECT/'outputs/dualcrown3d_legacy_small_tree_calibration'
BASELINE = PROJECT/'outputs/dualcrown3d_supervision_v4/baseline/val/metrics.json'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def variants():
    base=configurations()['legacy_consensus']
    def size(name, minimum_voxels, height, area, mask=True, vote=True, **changes):
        cfg=json.loads(json.dumps(base))
        if mask:
            cfg.update(minimum_voxels=minimum_voxels,
                       minimum_height_m=height, minimum_area_m2=area)
        if vote:
            cfg['vote_cluster_config'].update(min_voxels=minimum_voxels,
                                               min_height_m=height, min_area_m2=area)
        cfg.update(changes)
        return name,cfg
    result=dict([('baseline',base),
        size('mask_8',8,1.5,.4,vote=False),
        size('vote_8',8,1.5,.4,mask=False),
        size('both_10',10,1.75,.5),
        size('both_8',8,1.5,.4),
        size('both_6',6,1.,.25),
        size('both_8_object_05',8,1.5,.4,object_threshold=.05),
        size('both_8_mask_04',8,1.5,.4,mask_threshold=.4),
        size('both_8_vote_02',8,1.5,.4,
             vote_cluster_config={**base['vote_cluster_config'],
                                  'probability':.2,'min_voxels':8,
                                  'min_height_m':1.5,'min_area_m2':.4}),
        size('both_8_object_05_mask_04',8,1.5,.4,
             object_threshold=.05,mask_threshold=.4),
        size('both_6_mask_03',6,1.,.25,mask_threshold=.3),
        size('both_6_object_05_mask_03',6,1.,.25,
             object_threshold=.05,mask_threshold=.3),
        size('both_6_vote_015',6,1.,.25,
             vote_cluster_config={**base['vote_cluster_config'],
                                  'probability':.15,'min_voxels':6,
                                  'min_height_m':1.,'min_area_m2':.25}),
        size('both_6_vote_01',6,1.,.25,
             vote_cluster_config={**base['vote_cluster_config'],
                                  'probability':.1,'min_voxels':6,
                                  'min_height_m':1.,'min_area_m2':.25}),
        size('both_6_vote_01_peaks_01',6,1.,.25,
             vote_cluster_config={**base['vote_cluster_config'],
                                  'probability':.1,'peak_threshold_fraction':.01,
                                  'min_voxels':6,'min_height_m':1.,
                                  'min_area_m2':.25})])
    return result


def small_hits(ground_truth, predictions):
    """Cardinality-first IoU=.5 matching, stratified by reference crown area."""
    gt=list(ground_truth);pred=list(predictions)
    area=np.asarray([g.area for g in gt])
    matched=np.zeros(len(gt),bool)
    if gt and pred:
        overlap=np.zeros((len(gt),len(pred)),np.float64)
        tree=shapely.STRtree(pred)
        for i,geometry in enumerate(gt):
            for j in tree.query(geometry,predicate='intersects'):
                intersection=geometry.intersection(pred[j]).area
                overlap[i,j]=intersection/max(geometry.area+pred[j].area-intersection,1e-9)
        valid=overlap>=.5
        score=valid*(min(overlap.shape)+1.+overlap)
        rows,columns=linear_sum_assignment(-score)
        matched[rows[overlap[rows,columns]>=.5]]=True
    return {f'small_{limit:g}_gt':int((area<=limit).sum()) for limit in (4.,10.)} | {
            f'small_{limit:g}_tp':int(matched[area<=limit].sum()) for limit in (4.,10.)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--checkpoint',type=Path,default=INITIAL)
    parser.add_argument('--raw-cache-dir',type=Path,help='Reuse hashed validation predictions')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for uncached validation predictions')
    output=args.output.resolve()
    if (output/'selected.json').exists():
        raise FileExistsError('Completed calibration is protected')
    output.mkdir(parents=True,exist_ok=True)
    cache=(args.raw_cache_dir or output/'raw_val').resolve();cache.mkdir(parents=True,exist_ok=True)
    rows=[r for r in read_manifest(REAL,'val') if r.get('point_eval_eligible','true')=='true']
    expected=json.loads(BASELINE.read_text())['legacy_consensus']
    settings=variants()
    digest=sha256(args.checkpoint)
    results={name:[] for name in settings}
    forward_seconds=0.
    model=None
    for number,row in enumerate(rows,1):
        arrays=load_npz(row['output'])
        input_hash=sha256(row['output'])
        path=cache/(row['dataset_id']+'.npz')
        if path.exists():
            with np.load(path) as archive:
                raw={k:archive[k] for k in archive.files}
            if str(raw['checkpoint_sha256'])!=digest or str(raw['input_sha256'])!=input_hash:
                raise ValueError(f'Stale raw cache: {path}')
        else:
            if model is None:
                model,_=build(args.checkpoint)
            raw=predict(model,arrays,owner_only=True,raw_object_threshold=.05)
            raw['checkpoint_sha256']=np.asarray(digest)
            raw['input_sha256']=np.asarray(input_hash)
            temporary=path.with_name(path.stem+'.tmp.npz')
            np.savez_compressed(temporary,**raw)
            os.replace(temporary,path)
        forward_seconds+=float(raw['seconds'])
        gt=gpd.read_file(row['gt_vector'])
        ignore=shapely.from_wkb(Path(row['ignore_vector']).read_bytes())
        common={k:row[k] for k in ('dataset_id','source_dataset','collection','annotation_method')}
        for name,config in settings.items():
            started=time.monotonic()
            labels,confidence,instances,source=merge_masks_with_sources(arrays,raw,config)
            geometries=[instance['geometry'] for instance in instances]
            record={**common,**metrics(gt.geometry,geometries,ignore),
                    **small_hits(gt.geometry,geometries)}
            record['point_metrics']=point_instance_metrics(arrays['tree_id'],labels)
            known=arrays['tree_id']>0
            record['known_tree_voxel_coverage']=float((labels[known]>0).mean()) if known.any() else 0.
            record['merge_seconds']=time.monotonic()-started
            results[name].append(record)
        print(f'legacy calibration {number}/{len(rows)} {row["dataset_id"]}',flush=True)
    trials={}
    for name,records in results.items():
        point_rows=[{**{k:r[k] for k in common},**r['point_metrics']} for r in records]
        crown=aggregate(records);point=aggregate(point_rows)
        small={}
        for limit in (4.,10.):
            gt=sum(r[f'small_{limit:g}_gt'] for r in records)
            tp=sum(r[f'small_{limit:g}_tp'] for r in records)
            small[f'up_to_{limit:g}_m2']=dict(gt=gt,tp=tp,recall=tp/max(gt,1))
        trials[name]=dict(config=settings[name],metrics=crown,point_metrics=point,
                          small_crowns=small,per_plot=records,
                          mean_known_tree_voxel_coverage=float(np.mean([
                              r['known_tree_voxel_coverage'] for r in records])),
                          merge_seconds=sum(r['merge_seconds'] for r in records),
                          forward_seconds=forward_seconds)
    control=trials['baseline']
    for branch in ('metrics','point_metrics'):
        for metric in ('source_balanced_pq','source_balanced_f1'):
            if abs(control[branch][metric]-expected[branch][metric])>1e-6:
                write_json(output/'trials_mismatch.json',trials)
                raise AssertionError(f'Frozen baseline changed: {branch}/{metric}: '
                                     f'{control[branch][metric]} vs {expected[branch][metric]}')
    def accepted(value):
        return all(value[branch][metric]+1e-10>=control[branch][metric]
                   for branch in ('metrics','point_metrics')
                   for metric in ('source_balanced_pq','source_balanced_f1'))
    eligible=[name for name,value in trials.items() if name!='baseline' and accepted(value)
              and value['small_crowns']['up_to_10_m2']['recall'] >
                  control['small_crowns']['up_to_10_m2']['recall']]
    chosen=max(eligible,key=lambda name:(trials[name]['small_crowns']['up_to_10_m2']['recall'],
                        (trials[name]['point_metrics']['source_balanced_pq']*
                         trials[name]['metrics']['source_balanced_pq'])**.5)) if eligible else 'baseline'
    summary=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=digest,
                 manifest=str(REAL.resolve()),manifest_sha256=sha256(REAL),
                 split='val',plots=len(rows),selection=chosen,accepted=bool(eligible),
                 baseline_reproduced=True,small_area_limits_m2=[4.,10.],
                 test_used_for_selection=False,validation=trials[chosen],
                 baseline=control,all_trials=trials)
    write_json(output/'selected.json',summary)
    book=Workbook();sheet=book.active;sheet.title='threshold_sweep'
    fields=['name','accepted','minimum_voxels','minimum_height_m','minimum_area_m2',
            'object_threshold','mask_threshold','vote_min_voxels','vote_probability',
            'point_sb_pq','crown_sb_pq','point_sb_f1','crown_sb_f1',
            'small_4_recall','small_10_recall','point_precision','crown_precision',
            'known_tree_coverage','merge_seconds']
    sheet.append(fields)
    for name,value in trials.items():
        config=value['config']
        sheet.append([name,accepted(value),config['minimum_voxels'],
                      config.get('minimum_height_m',2.),config.get('minimum_area_m2',.75),
                      config['object_threshold'],config['mask_threshold'],
                      config['vote_cluster_config']['min_voxels'],
                      config['vote_cluster_config']['probability'],
                      value['point_metrics']['source_balanced_pq'],
                      value['metrics']['source_balanced_pq'],
                      value['point_metrics']['source_balanced_f1'],
                      value['metrics']['source_balanced_f1'],
                      value['small_crowns']['up_to_4_m2']['recall'],
                      value['small_crowns']['up_to_10_m2']['recall'],
                      value['point_metrics']['precision'],value['metrics']['precision'],
                      value['mean_known_tree_voxel_coverage'],value['merge_seconds']])
    sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
    book.save(output/'calibration.xlsx')
    print(json.dumps(dict(selection=chosen,accepted=bool(eligible),
                          baseline_small=control['small_crowns'],
                          selected_small=trials[chosen]['small_crowns'],
                          point_sb_pq=trials[chosen]['point_metrics']['source_balanced_pq'],
                          crown_sb_pq=trials[chosen]['metrics']['source_balanced_pq']),indent=2),flush=True)


if __name__=='__main__':
    main()
