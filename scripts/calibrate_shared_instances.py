#!/usr/bin/env python3
"""Calibrate shared instance proposals on validation without touching the test split."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from openpyxl import Workbook

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT.parent
sys.path.insert(0,str(PROJECT))
from pointcloud.dual_head import DualHeadLitePT
from scripts.train_shared_instances import REAL, BASELINE, evaluate, quality, write_json


def configurations():
    base=dict(mask_threshold=.5,crown_threshold=.5,minimum_voxels=12,
              minimum_height_m=2.,minimum_area_m2=.75,duplicate_iou=.5,
              duplicate_containment=.8,duplicate_distance_m=2.,
              crown_output='head_union_support')
    result={}
    for object_threshold in (.1,.2,.4,.6):
        for quality_threshold in (0.,.05,.15,.3):
            for unique_fraction in ((.25,.4) if object_threshold in (.2,.4)
                                    and quality_threshold in (.05,.15) else (.25,)):
                name=f'o{object_threshold:.2f}_q{quality_threshold:.2f}_u{unique_fraction:.2f}'
                result[name]=dict(base,object_threshold=object_threshold,
                                  quality_threshold=quality_threshold,
                                  minimum_unique_fraction=unique_fraction)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=PROJECT/'outputs/dualcrown3d_shared_v5_calibration')
    parser.add_argument('--val-limit',type=int,default=0)
    args=parser.parse_args()
    torch.set_num_threads(4)
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=DualHeadLitePT(**payload['model_args']).cuda().eval()
    model.load_state_dict(payload['model'],strict=True)
    trials=evaluate(model,REAL,'val',output,configurations(),limit=args.val_limit)
    baseline=json.loads(BASELINE.read_text())['legacy_consensus']
    def accepted(trial):
        if args.val_limit:
            return False
        return all(trial[head][metric] >= baseline[head][metric]
                   for head in ('metrics','point_metrics')
                   for metric in ('source_balanced_pq','source_balanced_f1'))
    winners=[name for name,value in trials.items() if accepted(value)]
    pool=winners or list(trials)
    best=max(pool,key=lambda name:quality(trials[name]))
    rows=[]
    for name,trial in trials.items():
        row=dict(name=name,**trial['config'],accepted=accepted(trial),
                 point_sb_pq=trial['point_metrics']['source_balanced_pq'],
                 crown_sb_pq=trial['metrics']['source_balanced_pq'],
                 point_sb_f1=trial['point_metrics']['source_balanced_f1'],
                 crown_sb_f1=trial['metrics']['source_balanced_f1'],
                 point_precision=trial['point_metrics']['precision'],
                 crown_precision=trial['metrics']['precision'],
                 known_tree_voxel_coverage=trial['known_tree_voxel_coverage'],
                 merge_seconds=trial['merge_seconds'])
        rows.append(row)
    rows.sort(key=lambda row:-(row['point_sb_pq']*row['crown_sb_pq'])**.5)
    book=Workbook();sheet=book.active;sheet.title='validation_grid'
    fields=list(rows[0]);sheet.append(fields)
    for row in rows:sheet.append([row[field] for field in fields])
    sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
    book.save(output/'calibration.xlsx')
    selection=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=digest,
                   epoch=payload.get('epoch'),config_name=best,config=trials[best]['config'],
                   validation=trials[best],acceptance_passed=bool(winners),
                   diagnostic_subset=bool(args.val_limit),test_used_for_selection=False,
                   baseline={head:{metric:baseline[head][metric]
                           for metric in ('source_balanced_pq','source_balanced_f1')}
                           for head in ('metrics','point_metrics')})
    write_json(output/'selected.json',selection)
    write_json(output/'ranking.json',rows)
    print(json.dumps(dict(best=best,accepted=bool(winners),best_point_sb_pq=rows[0]['point_sb_pq'],
                          best_crown_sb_pq=rows[0]['crown_sb_pq'],baseline=selection['baseline'],
                          checkpoint_sha256=digest),indent=2),flush=True)


if __name__=='__main__':
    main()
