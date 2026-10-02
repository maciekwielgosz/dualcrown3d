#!/usr/bin/env python3
"""Train shared point/crown queries and select all thresholds on native validation."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import geopandas as gpd
import numpy as np
import shapely
import torch
from openpyxl import Workbook
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))
from pointcloud.data import read_manifest, load_npz, move_to_device
from pointcloud.dual_head import DualHeadLitePT
from pointcloud.joint_training import DrawDataset, sampling_weights, fixed_serialization
from pointcloud.shared_instance import shared_instance_losses
from pointcloud.shared_merge import merge_shared_queries
from pointcloud.instance_output import point_instance_metrics
from scripts.evaluate_combined_full_crowns import aggregate, metrics
from scripts.predict_dual_head import predict

REAL = ROOT/'combined_als_crowns_supervision_v4/manifest.csv'
SYNTHETIC = ROOT/'dualcrown3d_treescan_helios_v1/manifest.csv'
AUGMENTED = ROOT/'TreeScanPL10k_HELIOS_ALS_flight_augmentation_v2/manifest.csv'
INITIAL = PROJECT/'outputs/dualcrown3d_joint_campaign_v1/runs/repeat_20260930/weights/best.pt'
BASELINE = PROJECT/'outputs/dualcrown3d_supervision_v4/baseline/val/metrics.json'
DEFAULT_OUTPUT = PROJECT/'outputs/dualcrown3d_shared_v5'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')
    os.replace(temporary,path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(initial=INITIAL,queries=128,memory_tokens=1536):
    model_args=dict(queries=queries,decoder_layers=3,memory_tokens=memory_tokens,
                    decoder_policy='shared_v5')
    model=DualHeadLitePT(**model_args)
    checkpoint=torch.load(initial,map_location='cpu',weights_only=False)
    missing,extra=model.load_state_dict(checkpoint['model'],strict=False)
    if extra or any(not key.startswith('point_decoder.crown.') for key in missing):
        raise ValueError(f'Unexpected initializer mismatch: missing={missing}, extra={extra}')
    fixed_serialization(model)
    return model.cuda(),model_args


def configurations():
    base=dict(mask_threshold=.5,crown_threshold=.5,minimum_voxels=12,
              minimum_height_m=2.,minimum_area_m2=.75,duplicate_iou=.5,
              duplicate_containment=.8,duplicate_distance_m=2.)
    return {
        'balanced':dict(base,object_threshold=.25,quality_threshold=.2,
                        minimum_unique_fraction=.25,crown_output='head_union_support'),
        'strict':dict(base,object_threshold=.4,quality_threshold=.35,
                      minimum_unique_fraction=.35,crown_output='head_union_support'),
        'recall':dict(base,object_threshold=.1,quality_threshold=.1,
                      minimum_unique_fraction=.15,crown_output='head_union_support'),
        'support_ablation':dict(base,object_threshold=.25,quality_threshold=.2,
                                minimum_unique_fraction=.25,crown_output='support_hull'),
    }


def evaluate(model, manifest, split, folder, configs, limit=0):
    rows=[r for r in read_manifest(manifest,split) if r.get('point_eval_eligible','true')=='true']
    if limit:
        rows=rows[:limit]
    per_config={name:[] for name in configs}
    forward_seconds=0.
    merge_seconds={name:0. for name in configs}
    model.eval()
    fixed_serialization(model)
    folder.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(rows,1):
        arrays=load_npz(row['output'])
        raw=predict(model,arrays,owner_only=True,collect_crowns=True,
                    raw_object_threshold=.005)
        forward_seconds+=float(raw['seconds'])
        gt=gpd.read_file(row['gt_vector'])
        ignore=shapely.from_wkb(Path(row['ignore_vector']).read_bytes()) if row.get('ignore_vector') else None
        for name,config in configs.items():
            started=time.monotonic()
            labels,confidence,instances,sources=merge_shared_queries(arrays,raw,config)
            merge_seconds[name]+=time.monotonic()-started
            record={k:row[k] for k in ('dataset_id','source_dataset','collection','annotation_method')}
            record.update(metrics(gt.geometry,[item['geometry'] for item in instances],ignore))
            record['point_metrics']=point_instance_metrics(arrays['tree_id'],labels)
            known=arrays['tree_id']>0
            record['known_tree_voxel_coverage']=float((labels[known]>0).mean()) if known.any() else 0.
            record['predicted_instances']=len(instances)
            record['unknown_assigned_voxels']=int(((arrays['tree_id']<0)&(labels>0)).sum())
            per_config[name].append(record)
        print(f'shared_v5 {split} {i}/{len(rows)} {row["dataset_id"]}',flush=True)
    result={}
    for name,plot_rows in per_config.items():
        points=[{**{k:r[k] for k in ('dataset_id','source_dataset','collection','annotation_method')},
                 **r['point_metrics']} for r in plot_rows]
        result[name]=dict(config=configs[name],metrics=aggregate(plot_rows),
                          point_metrics=aggregate(points),
                          known_tree_voxel_coverage=float(np.mean([r['known_tree_voxel_coverage'] for r in plot_rows])),
                          per_plot=plot_rows,forward_seconds=forward_seconds,
                          merge_seconds=merge_seconds[name],split=split,
                          protocol='native_v4; complete crown polygons; unknown point and cells ignored')
    write_json(folder/'metrics.json',result)
    return result


def quality(trial):
    return math.sqrt(trial['point_metrics']['source_balanced_pq']*
                     trial['metrics']['source_balanced_pq'])


def write_log(path,rows):
    if not rows:
        return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields)
        writer.writeheader()
        writer.writerows(rows)


def excel(path,rows):
    book=Workbook()
    sheet=book.active
    sheet.title='epochs'
    if rows:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        sheet.append(fields)
        for row in rows:
            sheet.append([row.get(key) for key in fields])
        sheet.freeze_panes='A2'
        sheet.auto_filter.ref=sheet.dimensions
    temporary=path.with_suffix('.tmp.xlsx')
    book.save(temporary)
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--samples',type=int,default=120)
    parser.add_argument('--eval-every',type=int,default=3)
    parser.add_argument('--seed',type=int,default=20260930)
    parser.add_argument('--real-fraction',type=float,default=.6)
    parser.add_argument('--max-points',type=int,default=16000)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--initial-checkpoint',type=Path,default=INITIAL)
    parser.add_argument('--val-limit',type=int,default=0,
                        help='Diagnostic subset; cannot make a production selection')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    output=args.output.resolve()
    if output.exists() and not args.smoke:
        raise FileExistsError(f'Run already exists: {output}')
    output.mkdir(parents=True,exist_ok=True)
    if not REAL.with_name('READY.json').exists():
        raise FileNotFoundError('v4 native dataset preparation is incomplete')
    real=[r for r in read_manifest(REAL,'train') if r['train_eligible']=='true']
    synthetic=read_manifest(SYNTHETIC,'train')
    augmented=read_manifest(AUGMENTED,'train')
    parents={r['dataset_id'].removeprefix('treescan_helios__') for r in synthetic}
    if any(r['parent_plot'] not in parents for r in augmented):
        raise ValueError('Synthetic training augmentation crosses the split')
    rows=real+synthetic+augmented
    weights=sampling_weights(rows,args.real_fraction)
    dataset=DrawDataset(rows,args.seed,args.max_points,hard=True,crown_supervision=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model,model_args=build(args.initial_checkpoint)
    config=dict(model_args=model_args,initial_checkpoint=str(args.initial_checkpoint),
                initial_sha256=sha256(args.initial_checkpoint),real_manifest_sha256=sha256(REAL),
                epochs=args.epochs,samples_per_epoch=args.samples,seed=args.seed,
                real_fraction=args.real_fraction,real_train_plots=len(real),
                synthetic_parents=len(parents),synthetic_variants=len(synthetic)+len(augmented),
                train_crown_target='complete native polygons transformed with the crop',
                unknown_policy='ignore unknown points and cells',
                quality_target='point/crown IoU of thresholded masks at 0.5',
                selection_split='val',test_used_for_selection=False)
    write_json(output/'configuration.json',config)
    if args.smoke:
        batch=move_to_device(dataset[(0,0)],torch.device('cuda:0'))
        model.configure_training('heads')
        result=model(batch)
        loss=shared_instance_losses(result,batch)
        loss['loss'].backward()
        if not torch.isfinite(loss['loss']):
            raise FloatingPointError('Nonfinite smoke loss')
        write_json(output/'smoke.json',dict(loss={k:float(v.detach()) for k,v in loss.items()},
                                            points=len(batch['coord']),
                                            crown_cells=int(batch['crown_valid'].sum()),
                                            vram_mb=torch.cuda.max_memory_allocated()/1024**2,
                                            gpu=torch.cuda.get_device_name()))
        return
    run=output/'run'
    (run/'weights').mkdir(parents=True)
    groups=[dict(params=model.point_decoder.parameters(),lr=1e-4),
            dict(params=model.legacy.semantic_head.parameters(),lr=3e-5),
            dict(params=model.legacy.offset_head.parameters(),lr=3e-5),
            dict(params=model.backbone.parameters(),lr=5e-6)]
    optimizer=torch.optim.AdamW(groups,weight_decay=.02)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=5e-7)
    history=[]
    best=-1.
    best_epoch=0
    for epoch in range(1,args.epochs+1):
        model.configure_training('mask' if epoch<=2 else 'partial')
        if args.initial_checkpoint != INITIAL and epoch<=2:
            for name,parameter in model.point_decoder.named_parameters():
                parameter.requires_grad_(name.startswith('crown.quality.') or name.startswith('score.'))
        model.train()
        model.point_decoder.epoch=100+epoch
        dataset.epoch=epoch
        rng=np.random.default_rng(args.seed+epoch*100003)
        draws=[(int(i),draw) for draw,i in enumerate(
            rng.choice(len(rows),args.samples,p=weights/weights.sum()))]
        loader=DataLoader(dataset,batch_size=None,sampler=draws,num_workers=2,pin_memory=True)
        totals={}
        optimizer.zero_grad(set_to_none=True)
        started=time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for step,batch in enumerate(loader,1):
            batch=move_to_device(batch,torch.device('cuda:0'))
            prediction=model(batch)
            losses=shared_instance_losses(prediction,batch)
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError(f'Nonfinite training loss at epoch {epoch} step {step}')
            (losses['loss']/2).backward()
            if step%2==0 or step==len(draws):
                torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for key,value in losses.items():
                totals[key]=totals.get(key,0.)+float(value.detach())
            if step%20==0:
                print(f'epoch {epoch} step {step}/{len(draws)} loss {totals["loss"]/step:.4f}',flush=True)
        scheduler.step()
        record=dict(epoch=epoch,train_seconds=time.monotonic()-started,
                    peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2,
                    **{f'train_{key}':value/len(draws) for key,value in totals.items()})
        checkpoint=dict(model=model.state_dict(),model_args=model_args,epoch=epoch,
                        training_config=config,optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict())
        temporary=run/'weights/last.tmp.pt'
        torch.save(checkpoint,temporary)
        os.replace(temporary,run/'weights/last.pt')
        if epoch%args.eval_every==0 or epoch==args.epochs:
            trials=evaluate(model,REAL,'val',run/f'val_epoch_{epoch:03d}',
                            configurations(),limit=args.val_limit)
            winner=max(trials,key=lambda name:quality(trials[name]))
            score=quality(trials[winner])
            record.update(val_score=score,val_config=winner,
                          val_point_sb_pq=trials[winner]['point_metrics']['source_balanced_pq'],
                          val_crown_sb_pq=trials[winner]['metrics']['source_balanced_pq'],
                          val_point_f1=trials[winner]['point_metrics']['f1'],
                          val_crown_f1=trials[winner]['metrics']['f1'])
            if score>best:
                best,best_epoch=score,epoch
                temporary=run/'weights/best.tmp.pt'
                torch.save(checkpoint,temporary)
                os.replace(temporary,run/'weights/best.pt')
                selected=dict(epoch=epoch,score=score,config=trials[winner]['config'],
                              config_name=winner,validation=trials[winner],
                              checkpoint=str((run/'weights/best.pt').resolve()),
                              checkpoint_sha256=sha256(run/'weights/best.pt'),
                              model_args=model_args,test_used_for_selection=False,
                              diagnostic_subset=bool(args.val_limit))
                write_json(output/'selected.json',selected)
        history.append(record)
        write_log(output/'training_log.csv',history)
        excel(output/'experiments.xlsx',history)
        write_json(output/'status.json',dict(stage='training',epoch=epoch,
                                             best_epoch=best_epoch,best_score=best))
        print(json.dumps(record),flush=True)
    baseline=json.loads(BASELINE.read_text())['legacy_consensus']
    selected=json.loads((output/'selected.json').read_text())
    acceptance=(not args.val_limit and
                selected['validation']['point_metrics']['source_balanced_pq'] >= baseline['point_metrics']['source_balanced_pq'] and
                selected['validation']['metrics']['source_balanced_pq'] >= baseline['metrics']['source_balanced_pq'] and
                selected['validation']['point_metrics']['source_balanced_f1'] >= baseline['point_metrics']['source_balanced_f1'] and
                selected['validation']['metrics']['source_balanced_f1'] >= baseline['metrics']['source_balanced_f1'])
    report=dict(checkpoint=selected['checkpoint'],checkpoint_sha256=selected['checkpoint_sha256'],
                best_epoch=best_epoch,validation=selected['validation'],
                baseline_validation=dict(point_sb_pq=baseline['point_metrics']['source_balanced_pq'],
                                         crown_sb_pq=baseline['metrics']['source_balanced_pq'],
                                         point_sb_f1=baseline['point_metrics']['source_balanced_f1'],
                                         crown_sb_f1=baseline['metrics']['source_balanced_f1']),
                acceptance_passed=acceptance,test_used_for_selection=False,
                note='Held-out test remains untouched until a full-validation selection passes.')
    write_json(output/'final_report.json',report)
    write_json(output/'status.json',dict(stage='complete',best_epoch=best_epoch,
                                         acceptance_passed=acceptance))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
