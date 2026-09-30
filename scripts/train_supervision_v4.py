#!/usr/bin/env python3
"""Controlled v4 campaign: repaired supervision, hybrid queries, revisable fusion.

Same spatial splits. Old checkpoint is re-evaluated on the corrected protocol.
ECODSE contributes supplemental 2D recall only, never 3D training or selection.
"""
import argparse
from collections import Counter
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
from torch.utils.data import DataLoader
from openpyxl import Workbook

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT.parent
sys.path.insert(0,str(PROJECT))
from pointcloud.data import read_manifest,load_npz,move_to_device
from pointcloud.dual_head import DualHeadLitePT
from pointcloud.joint_training import DrawDataset,sampling_weights,joint_losses,fixed_serialization
from pointcloud.instance_output import merge_masks_with_sources,point_instance_metrics
from scripts.evaluate_combined_full_crowns import aggregate,metrics,annotation_ignore
from scripts.predict_dual_head import predict
from scripts.prepare_supervision_v4 import write_csv

REAL=ROOT/'combined_als_crowns_supervision_v4/manifest.csv'
INITIAL=PROJECT/'outputs/dualcrown3d_joint_campaign_v1/runs/repeat_20260930/weights/best.pt'
DEFAULT_OUTPUT=PROJECT/'outputs/dualcrown3d_supervision_v4'


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')
    os.replace(tmp,path)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(path,model_args=None):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    model=DualHeadLitePT(**(model_args or payload.get('model_args',{})))
    model.load_state_dict(payload['model'],strict=True)
    fixed_serialization(model)
    return model.cuda(),payload


def quality(result):
    return math.sqrt(result['point_metrics']['source_balanced_pq']*result['metrics']['source_balanced_pq'])


def configurations():
    base=json.loads((PROJECT/'configs/dual_head_complete_consensus.json').read_text())['config']
    return {
        'legacy_consensus':base,
        'revisable_split':{**base,'merge_strategy':'adaptive_consensus_v4','revision_mode':'split','revision_confidence':.4},
        'revisable_both':{**base,'merge_strategy':'adaptive_consensus_v4','revision_mode':'both','revision_confidence':.4},
        'revisable_strict':{**base,'merge_strategy':'adaptive_consensus_v4','revision_mode':'both','revision_confidence':.6},
        'mask_anchor':{**base,'dual_anchor':'mask'},
    }


def evaluate(model,manifest,split,folder,configs,*,cache=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    results={name:[] for name in configs};seconds=0.;merge_seconds={name:0. for name in configs}
    rows=[r for r in read_manifest(manifest,split) if r.get('point_eval_eligible','true')=='true']
    model.eval();fixed_serialization(model)
    for i,row in enumerate(rows,1):
        arrays=load_npz(row['output']);torch.cuda.synchronize();start=time.monotonic()
        raw=predict(model,arrays,owner_only=True);torch.cuda.synchronize();seconds+=time.monotonic()-start
        if cache:np.savez_compressed(folder/(row['dataset_id']+'_raw.npz'),**raw)
        gt=gpd.read_file(row['gt_vector'])
        ignore=shapely.from_wkb(Path(row['ignore_vector']).read_bytes()) if row.get('ignore_vector') else annotation_ignore(row['source_las'],row['chm'])
        for name,config in configs.items():
            start=time.monotonic();labels,confidence,instances,sources=merge_masks_with_sources(arrays,raw,config)
            merge_seconds[name]+=time.monotonic()-start
            result={k:row[k] for k in ('dataset_id','source_dataset','collection','annotation_method')}
            result.update(metrics(gt.geometry,[p['geometry'] for p in instances],ignore))
            result['point_metrics']=point_instance_metrics(arrays['tree_id'],labels)
            result['assigned_known_tree_fraction']=float((labels[arrays['tree_id']>0]>0).mean()) if (arrays['tree_id']>0).any() else 0.
            result['revised_split_points']=int((sources==6).sum());result['revised_merge_points']=int((sources==7).sum())
            results[name].append(result)
        print(f'v4 {split} {i}/{len(rows)} {row["dataset_id"]}',flush=True)
    trials={}
    for name,per_plot in results.items():
        point_rows=[{**{k:r[k] for k in ('dataset_id','source_dataset','collection','annotation_method')},
                     **r['point_metrics']} for r in per_plot]
        trials[name]=dict(config=configs[name],metrics=aggregate(per_plot),point_metrics=aggregate(point_rows),
                          per_plot=per_plot,split=split,inference_seconds=seconds,merge_seconds=merge_seconds[name],
                          protocol='v4 native 3D, unknown ignored, bushes are non-tree; crowns use explicit unknown-cell mask')
    dump(folder/'metrics.json',trials)
    return trials


def excel(root):
    book=Workbook();book.remove(book.active)
    experiments=[];epochs=[];comparison=[]
    for p in sorted(root.glob('runs/*/configuration.json')):
        row=json.loads(p.read_text());selection=p.with_name('selected.json')
        if selection.exists():
            selected=json.loads(selection.read_text());row.update(epoch=selected['epoch'],score=selected['score'],
                checkpoint=str(p.parent/'weights/best.pt'),fusion=selected['fusion'])
        experiments.append(row)
        log=p.with_name('training_log.csv')
        if log.exists():
            with log.open() as f:epochs.extend(dict(run=p.parent.name,**r) for r in csv.DictReader(f))
    final=root/'final_report.json'
    if final.exists():comparison=json.loads(final.read_text())['comparison']
    for name,rows in [('experiments',experiments),('epochs',epochs),('comparison',comparison)]:
        sheet=book.create_sheet(name);fields=list(dict.fromkeys(k for r in rows for k in r))
        if fields:
            sheet.append(fields)
            for r in rows:sheet.append([json.dumps(r.get(k),default=str) if isinstance(r.get(k),(dict,list)) else r.get(k) for k in fields])
            sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
    tmp=root/'experiments.tmp.xlsx';book.save(tmp);os.replace(tmp,root/'experiments.xlsx')


def train(root,name,options,baseline):
    run=root/'runs'/name
    if (run/'DONE.json').exists():return json.loads((run/'selected.json').read_text())
    if run.exists():raise FileExistsError(f'Incomplete run protected: {run}')
    run.mkdir(parents=True);(run/'weights').mkdir()
    seed=options.seed;torch.manual_seed(seed);np.random.seed(seed)
    model_args=dict(queries=96,decoder_layers=3,memory_tokens=1024)
    if name=='hybrid_queries':model_args.update(queries=128,memory_tokens=1536,decoder_policy='hybrid_v4')
    model,_=build(INITIAL,model_args)
    real=[r for r in read_manifest(REAL,'train') if r['train_eligible']=='true']
    synthetic=read_manifest(ROOT/'dualcrown3d_treescan_helios_v1/manifest.csv','train')
    extra=read_manifest(ROOT/'TreeScanPL10k_HELIOS_ALS_flight_augmentation_v2/manifest.csv','train')
    parents={r['dataset_id'].removeprefix('treescan_helios__') for r in synthetic}
    if any(r['parent_plot'] not in parents for r in extra):raise ValueError('Synthetic split leakage')
    rows=real+synthetic+extra;write_csv(run/'training_manifest.csv',rows)
    weights=sampling_weights(rows,.5);dataset=DrawDataset(rows,seed,16000,hard=True)
    model.configure_training('partial');fixed_serialization(model)
    params=[p for p in model.parameters() if p.requires_grad]
    backbone=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('legacy.backbone.')]
    heads=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('legacy.backbone.')]
    optimizer=torch.optim.AdamW([dict(params=heads,lr=1e-4),dict(params=backbone,lr=1e-5)],weight_decay=.02)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=options.epochs,eta_min=5e-7)
    cfg=dict(name=name,seed=seed,model_args=model_args,initial_checkpoint=str(INITIAL),initial_sha256=digest(INITIAL),
             manifest_sha256=digest(REAL),epochs_max=options.epochs,samples_per_epoch=options.samples,
             max_points_train=16000,max_points_eval=40000,real_fraction=.5,real_plots=len(real),
             synthetic_parent_plots=len(parents),synthetic_variants=len(synthetic+extra),scope='partial',
             lr_heads=1e-4,lr_backbone=1e-5,accumulation=2,teacher_warmup_epochs=8,
             architecture='LitePT-S + centre votes + '+('hybrid spatial/embedding queries, soft support, Hungarian masks' if name=='hybrid_queries' else 'legacy seeded masks'),
             serialization='fixed at every pooling stage',test_used_for_selection=False)
    dump(run/'configuration.json',cfg)
    def save(path,epoch,score,config):
        temp=path.with_suffix('.tmp.pt')
        torch.save(dict(model=model.state_dict(),model_args=model_args,epoch=epoch,score=score,
                        config={**cfg,'legacy_cluster_config':config['vote_cluster_config']},
                        optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),temp)
        os.replace(temp,path)
    initial=evaluate(model,REAL,'val',run/'validation/epoch_000',configurations()) if name=='hybrid_queries' else baseline
    best_fusion=max(initial,key=lambda k:quality(initial[k]));best=quality(initial[best_fusion]);best_epoch=0
    selected=dict(epoch=0,score=best,fusion=best_fusion,config=initial[best_fusion]['config'],validation=initial[best_fusion],model_args=model_args)
    save(run/'weights/best.pt',0,best,selected['config']);dump(run/'selected.json',selected)
    history=[];started=time.monotonic()
    for epoch in range(1,options.epochs+1):
        torch.manual_seed(seed+epoch);model.configure_training('heads' if epoch<=4 else 'partial');model.train()
        model.point_decoder.epoch=100+epoch
        if name=='hybrid_queries':model.point_decoder.teacher_probability=max(0.,1.-epoch/8.)
        dataset.epoch=epoch
        rng=np.random.default_rng(seed+epoch*100003)
        draws=[(int(i),step) for step,i in enumerate(rng.choice(len(rows),options.samples,p=weights/weights.sum()))]
        loader=DataLoader(dataset,batch_size=None,sampler=draws,num_workers=2,pin_memory=True)
        totals=Counter();optimizer.zero_grad(set_to_none=True);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for step,batch in enumerate(loader,1):
            batch=move_to_device(batch,torch.device('cuda:0'))
            inputs={k:batch[k] for k in ('coord','grid_coord','feat','offset','tree_id')}
            losses=joint_losses(model(inputs),batch)
            if not torch.isfinite(losses['loss']):raise FloatingPointError('Non-finite loss')
            (losses['loss']/2).backward()
            if step%2==0 or step==len(draws):
                torch.nn.utils.clip_grad_norm_(params,2.,error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            totals.update({k:float(v.detach()) for k,v in losses.items()})
        scheduler.step()
        record=dict(epoch=epoch,seconds=time.monotonic()-start,peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2,
                    **{f'train_{k}':v/len(draws) for k,v in totals.items()})
        if epoch%4==0 or epoch==options.epochs:
            validation=evaluate(model,REAL,'val',run/f'validation/epoch_{epoch:03d}',configurations())
            fusion=max(validation,key=lambda k:quality(validation[k]));value=quality(validation[fusion])
            record.update(val_score=value,val_point_PQ=validation[fusion]['point_metrics']['source_balanced_pq'],
                          val_crown_PQ=validation[fusion]['metrics']['source_balanced_pq'],fusion=fusion)
            if value>best:
                best,best_epoch=value,epoch
                selected=dict(epoch=epoch,score=value,fusion=fusion,config=validation[fusion]['config'],
                              validation=validation[fusion],model_args=model_args)
                save(run/'weights/best.pt',epoch,best,selected['config']);dump(run/'selected.json',selected)
        save(run/'weights/last.pt',epoch,best,selected['config'])
        history.append(record);write_csv(run/'training_log.csv',history);excel(root)
        dump(root/'status.json',dict(stage='training',run=name,epoch=epoch,best_epoch=best_epoch,best_score=best))
        print(json.dumps(dict(run=name,**record)),flush=True)
        if epoch>=16 and epoch-best_epoch>=12:break
    dump(run/'DONE.json',dict(best_epoch=best_epoch,best_score=best,epochs=epoch,seconds=time.monotonic()-started))
    del model;torch.cuda.empty_cache()
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--samples',type=int,default=160)
    p.add_argument('--seed',type=int,default=20261002)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('GPU is required')
    readiness=json.loads(REAL.with_name('READY.json').read_text())
    if readiness['plots']!=147:raise ValueError('Incomplete data preparation')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if args.smoke:
        row=next(r for r in read_manifest(REAL,'train') if r['train_eligible']=='true')
        dataset=DrawDataset([row],args.seed,16000)
        model,_=build(INITIAL,dict(queries=128,decoder_layers=3,memory_tokens=1536,decoder_policy='hybrid_v4'))
        model.configure_training('partial');model.train()
        batch=move_to_device(dataset[(0,0)],torch.device('cuda:0'))
        losses=joint_losses(model(batch),batch);losses['loss'].backward()
        if not torch.isfinite(losses['loss']):raise AssertionError('Nonfinite smoke loss')
        dump(root/'smoke.json',dict(loss=float(losses['loss'].detach()),device=torch.cuda.get_device_name(),
            gradient_norm=float(sum(p.grad.square().sum() for p in model.point_decoder.parameters() if p.grad is not None).sqrt())))
        return
    baseline_path=root/'baseline/val/metrics.json'
    if baseline_path.exists():baseline=json.loads(baseline_path.read_text())
    else:
        model,_=build(INITIAL);baseline=evaluate(model,REAL,'val',baseline_path.parent,configurations());del model;torch.cuda.empty_cache()
    selections={name:train(root,name,args,baseline) for name in ('data_only','hybrid_queries')}
    winner=max(selections,key=lambda k:selections[k]['score']);selected=selections[winner]
    checkpoint=root/'runs'/winner/'weights/best.pt'
    old=baseline['legacy_consensus']
    acceptance=all(selected['validation'][m]['source_balanced_pq']>=old[m]['source_balanced_pq'] for m in ('metrics','point_metrics'))
    frozen=dict(**selected,run=winner,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
                acceptance_passed=acceptance,test_used_for_selection=False)
    dump(root/'selected.json',frozen)
    # The historically exposed test is evaluated only after this selection.
    comparison=[]
    for name,path,config in [('baseline',INITIAL,configurations()['legacy_consensus']),('selected',checkpoint,selected['config'])]:
        model,_=build(path)
        for domain,manifest in [('real',REAL),('helios',ROOT/'dualcrown3d_treescan_helios_v1/manifest.csv')]:
            target=root/'final_test'/name/domain/'metrics.json'
            result=json.loads(target.read_text())['fixed'] if target.exists() else evaluate(model,manifest,'test',target.parent,{'fixed':config})['fixed']
            comparison.append(dict(model=name,domain=domain,point_SB_PQ=result['point_metrics']['source_balanced_pq'],
                crown_SB_PQ=result['metrics']['source_balanced_pq'],point_F1=result['point_metrics']['f1'],
                crown_F1=result['metrics']['f1'],inference_seconds=result['inference_seconds'],merge_seconds=result['merge_seconds']))
        del model;torch.cuda.empty_cache()
    report=dict(winner=winner,checkpoint=str(checkpoint),acceptance_passed=acceptance,comparison=comparison,
                protocol=readiness['protocol'],test_used_for_selection=False,seeds=1,
                caveat='Same spatial split; corrected annotation protocol and native-only primary metrics. Historically exposed test. One training seed.')
    dump(root/'final_report.json',report);excel(root);dump(root/'status.json',dict(stage='complete',winner=winner))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
