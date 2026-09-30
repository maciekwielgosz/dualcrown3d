#!/usr/bin/env python3
"""Five-step DualCrown3D campaign: joint heads, partial backbone, data, decoder.

Hyperparameters and architecture are selected on real validation only. Three
seeds of the winning setting are tested after architecture selection is frozen.
"""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from openpyxl import Workbook

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import read_manifest, load_npz, move_to_device
from pointcloud.dual_head import DualHeadLitePT
from pointcloud.joint_training import fixed_serialization, sampling_weights, DrawDataset, joint_losses
from scripts.finetune_dualcrown3d import evaluate, quality, summary_row, write_csv
from scripts.predict_dual_head import predict
from pointcloud.instance_output import merge_masks

WORKSPACE = PROJECT.parent
DEFAULT_ROOT = PROJECT/'outputs/dualcrown3d_joint_campaign_v1'
REAL = WORKSPACE/'combined_als_crowns_no_rectangles_v2/manifest.csv'
HELIOS = WORKSPACE/'dualcrown3d_treescan_helios_v1/manifest.csv'
AUGMENT = WORKSPACE/'TreeScanPL10k_HELIOS_ALS_flight_augmentation_v2/manifest.csv'
INITIAL = PROJECT/'outputs/dualcrown3d_treescan_helios_finetune_v2/weights/best.pt'
SELECTION = PROJECT/'configs/dual_head_complete_consensus.json'
SMALL = dict(queries=96, decoder_layers=3, memory_tokens=1024)
LARGE = dict(queries=128, decoder_layers=5, memory_tokens=1536)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str, allow_nan=False)+'\n')
    os.replace(tmp, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(payload, model_args=None):
    kwargs = model_args or payload.get('model_args', SMALL)
    model = DualHeadLitePT(**kwargs)
    state = dict(payload['model'])
    # Transfer extra attention layers from the last trained layer explicitly.
    # All other weights keep their exact shape and strict loading is enforced.
    trained_layers = max(int(k.split('.')[2]) for k in state if k.startswith('point_decoder.layers.'))+1
    for i in range(trained_layers, kwargs['decoder_layers']):
        prefix = f'point_decoder.layers.{trained_layers-1}.'
        for key, value in list(state.items()):
            if key.startswith(prefix):
                state[f'point_decoder.layers.{i}.'+key[len(prefix):]] = value.clone()
    model.load_state_dict(state, strict=True)
    fixed_serialization(model)
    return model


def evaluate_fixed(model, manifest, folder, config, split):
    # Fork RNG also prevents the length of evaluation from affecting training.
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(7001); torch.cuda.manual_seed_all(7001)
        fixed_serialization(model)
        return evaluate(model, manifest, folder, config, split)


def workbook(root):
    book = Workbook(); book.remove(book.active)
    records = []
    for p in sorted(root.glob('runs/*/configuration.json')):
        cfg = json.loads(p.read_text()); run = p.parent
        row = {**cfg, 'run': run.name, 'checkpoint': str(run/'weights/best.pt')}
        architecture = cfg['model_args']
        row['architecture_description'] = (
            'LitePT-S backbone; semantic foreground + XY centre-offset head; '
            f'point-mask transformer: {architecture["decoder_layers"]} layers, '
            f'{architecture["queries"]} queries, {architecture["memory_tokens"]} memory tokens, '
            '128 hidden features, 4 attention heads; dual-consensus instance merging'
        )
        if (run/'selected.json').exists():
            selected = json.loads((run/'selected.json').read_text())
            row.update(best_epoch=selected['epoch'], validation_score=selected['score'])
            for branch in ('metrics','point_metrics'):
                for metric in ('pq','f1','source_balanced_pq','source_balanced_f1'):
                    row[f'val_{branch}_{metric}'] = selected['validation'][branch][metric]
        row['completed'] = (run/'DONE.json').exists(); records.append(row)
    tables = {'experiments': records, 'epochs': [], 'test_metrics': []}
    for path in sorted(root.glob('runs/*/training_log.csv')):
        with path.open() as f:
            tables['epochs'].extend(dict(run=path.parent.name, **row) for row in csv.DictReader(f))
    report = root/'final_report.json'
    if report.exists(): tables['test_metrics'] = json.loads(report.read_text())['test_rows']
    for name, rows in tables.items():
        sheet = book.create_sheet(name)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        if fields:
            sheet.append(fields)
            for r in rows:
                sheet.append([json.dumps(r.get(k), default=str) if isinstance(r.get(k),(dict,list)) else r.get(k) for k in fields])
            sheet.freeze_panes='A2'; sheet.auto_filter.ref=sheet.dimensions
    tmp = root/'experiments.tmp.xlsx'; book.save(tmp); os.replace(tmp, root/'experiments.xlsx')


def mine_difficulty(root, payload, merge_config):
    path = root/'training_difficulty.json'
    if path.exists(): return json.loads(path.read_text())
    model = build(payload).cuda().eval(); rows = read_manifest(REAL,'train') + read_manifest(HELIOS,'train')
    progress = root/'training_difficulty.progress.json'
    result = json.loads(progress.read_text()) if progress.exists() else {}
    for i,row in enumerate(rows):
        if row['dataset_id'] in result: continue
        a = load_npz(row['output'])
        raw = predict(model, a)
        labels,_,_ = merge_masks(a,raw,merge_config)
        valid = a['tree_id'] >= 0; gt = a['tree_id'][valid]; pred = labels[valid]
        g, gi, gc = np.unique(gt,return_inverse=True,return_counts=True)
        p, pi, pc = np.unique(pred,return_inverse=True,return_counts=True)
        table = np.bincount(gi*len(p)+pi,minlength=len(g)*len(p)).reshape(len(g),len(p))
        iou = table / np.maximum(gc[:,None]+pc[None,:]-table,1)
        best = iou[:,p>0].max(1) if (p>0).any() else np.zeros(len(g))
        result[row['dataset_id']] = {str(int(t)): float(b < .5) for t,b in zip(g,best) if t>0}
        dump(progress,result)
        print(f'train-only mining {i+1}/{len(rows)} {row["dataset_id"]}', flush=True)
    dump(path,result); del model; torch.cuda.empty_cache(); return result


def train_run(root, spec, payload, baseline, merge_config, options, difficulty=None):
    run = root/'runs'/spec['name']; done = run/'DONE.json'
    if done.exists(): return json.loads(done.read_text())
    if run.exists():
        raise FileExistsError(f'Incomplete run protected: {run}; inspect before restarting')
    run.mkdir(parents=True); (run/'weights').mkdir()
    seed = spec['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model_args = LARGE if spec.get('large') else SMALL
    model = build(payload,model_args).cuda()
    rows = read_manifest(REAL,'train') + read_manifest(HELIOS,'train')
    if spec.get('augment'):
        readiness = json.loads(AUGMENT.with_name('READY.json').read_text())
        if readiness['simulations'] != 86:
            raise ValueError('The complete 86-flight augmentation is required')
        extra = read_manifest(AUGMENT,'train')
        parents = {r['dataset_id'].removeprefix('treescan_helios__') for r in read_manifest(HELIOS,'train')}
        if any(r['parent_plot'] not in parents for r in extra): raise ValueError('Augmentation split leakage')
        rows += extra
    write_csv(run/'training_manifest.csv', rows)
    weights = sampling_weights(rows,spec['real_fraction'])
    dataset = DrawDataset(rows,seed,options.max_points,hard=spec.get('hard',False),difficulty=difficulty)
    model.configure_training(spec['scope']); fixed_serialization(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    backbone_params = [p for n,p in model.named_parameters() if p.requires_grad and n.startswith('legacy.backbone.')]
    head_params = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('legacy.backbone.')]
    groups = [dict(params=head_params,lr=5e-5,name='heads')]
    if backbone_params: groups.append(dict(params=backbone_params,lr=5e-6,name='backbone'))
    optimizer = torch.optim.AdamW(groups,weight_decay=.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=options.epochs,eta_min=5e-7)
    original = payload['model']
    config = {**spec, 'model_args':model_args, 'epochs_max':options.epochs,
              'samples_per_epoch':options.samples, 'max_points':options.max_points,
              'accumulation':2, 'warmup_heads_epochs':4, 'validation_every':4,
              'patience_epochs':12, 'lr_heads':5e-5,'lr_backbone':5e-6,
              'initial_checkpoint':str(INITIAL),'initial_sha256':digest(INITIAL),
              'manifest_sha256':digest(run/'training_manifest.csv'),
              'trainable_parameters':sum(p.numel() for p in parameters),
              'total_parameters':sum(p.numel() for p in model.parameters()),
              'serialization':'fixed at every pooling stage',
              'selection':'real validation sqrt(source-balanced point PQ * crown PQ)',
              'test_used_for_selection':False}
    dump(run/'configuration.json',config)
    def save(path,epoch,score):
        torch.save(dict(model=model.state_dict(),model_args=model_args,epoch=epoch,score=score,
                        format='dualcrown3d_joint_v1',config={**config,'legacy_cluster_config':merge_config['vote_cluster_config']},
                        optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),path)
    initial_val = evaluate_fixed(model,REAL,run/'validation/epoch_000',merge_config,'val') if spec.get('large') else baseline
    best, best_epoch, history = quality(initial_val),0,[]
    save(run/'weights/best.pt',0,best)
    selected = dict(epoch=0,score=best,validation=initial_val,config=merge_config,model_args=model_args,test_used_for_selection=False)
    dump(run/'selected.json',selected)
    trainable_names = {n for n,p in model.named_parameters() if p.requires_grad}
    started=time.monotonic()
    for epoch in range(1,options.epochs+1):
        model.configure_training('heads' if spec['scope']=='partial' and epoch<=4 else spec['scope'])
        fixed_serialization(model); model.train(); model.point_decoder.epoch=100+epoch; dataset.epoch=epoch
        rng=np.random.default_rng(seed+epoch*100003)
        draws=[(int(i),step) for step,i in enumerate(rng.choice(len(rows),options.samples,p=weights/weights.sum()))]
        loader=DataLoader(dataset,batch_size=None,sampler=draws,num_workers=2,pin_memory=True)
        total=Counter(); optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
        epoch_start=time.monotonic()
        for step,batch in enumerate(loader,1):
            batch=move_to_device(batch,torch.device('cuda:0'))
            data={k:batch[k] for k in ('coord','grid_coord','feat','offset','tree_id')}
            losses=joint_losses(model(data),batch)
            if not torch.isfinite(losses['loss']): raise FloatingPointError('Non-finite joint loss')
            (losses['loss']/2).backward()
            if step%2==0 or step==len(draws):
                torch.nn.utils.clip_grad_norm_(parameters,2.,error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            total.update({k:float(v.detach()) for k,v in losses.items()})
        scheduler.step()
        log=dict(epoch=epoch,scope=model.training_scope,seconds=time.monotonic()-epoch_start,
                 peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2,
                 **{f'train_{k}':v/len(draws) for k,v in total.items()})
        if epoch%4==0 or epoch==options.epochs:
            val=evaluate_fixed(model,REAL,run/f'validation/epoch_{epoch:03d}',merge_config,'val')
            score=quality(val)
            log.update(val_score=score,val_point_PQ=val['point_metrics']['source_balanced_pq'],
                       val_crown_PQ=val['metrics']['source_balanced_pq'])
            if score>best:
                best,best_epoch=score,epoch
                save(run/'weights/best.pt',epoch,best)
                selected=dict(epoch=epoch,score=score,validation=val,config=merge_config,
                              model_args=model_args,test_used_for_selection=False)
                dump(run/'selected.json',selected)
        save(run/'weights/last.pt',epoch,best)
        history.append(log); write_csv(run/'training_log.csv',history)
        dump(root/'status.json',dict(stage='training',run=spec['name'],epoch=epoch,best_epoch=best_epoch,best_score=best))
        workbook(root); print(json.dumps(dict(run=spec['name'],**log)),flush=True)
        if epoch>=16 and epoch-best_epoch>=12: break
    best_payload=torch.load(run/'weights/best.pt',map_location='cpu',weights_only=False)
    state=best_payload['model']
    changed=[n for n,p in state.items() if n in original and not torch.equal(p,original[n])]
    illegal=[n for n in changed if n not in trainable_names]
    if illegal: raise AssertionError(f'Frozen parameters/buffers changed: {illegal[:8]}')
    summary=dict(name=spec['name'],spec=spec,score=best,best_epoch=best_epoch,
                 epochs_completed=epoch,seconds=time.monotonic()-started,
                 checkpoint=str(run/'weights/best.pt'),checkpoint_sha256=digest(run/'weights/best.pt'),
                 changed_parameters=len(changed),frozen_parameters_unchanged=True)
    selected.update(checkpoint_sha256=summary['checkpoint_sha256'],acceptance_passed=True)
    dump(run/'selected.json',selected); dump(done,summary); workbook(root)
    del model,optimizer; torch.cuda.empty_cache(); return summary


def smoke(root,payload,options):
    model=build(payload).cuda(); model.configure_training('partial'); fixed_serialization(model); model.train()
    row=read_manifest(REAL,'train')[0]
    dataset=DrawDataset([row],123,options.max_points,hard=True)
    batch=move_to_device(dataset[(0,0)],torch.device('cuda:0'))
    torch.cuda.reset_peak_memory_stats()
    loss=joint_losses(model({k:batch[k] for k in ('coord','grid_coord','feat','offset','tree_id')}),batch)['loss']
    loss.backward()
    groups={prefix:sum(p.grad is not None and bool(p.grad.abs().sum()>0) for n,p in model.named_parameters() if n.startswith(prefix))
            for prefix in ('legacy.backbone.enc.enc4.','legacy.backbone.dec.','legacy.semantic_head.','legacy.offset_head.','point_decoder.')}
    assert all(groups.values()), groups
    assert all(p.grad is None for n,p in model.named_parameters() if n.startswith('legacy.backbone.enc.enc0.'))
    model.eval(); data={k:batch[k] for k in ('coord','grid_coord','feat','offset')}
    with torch.no_grad():
        a=model(data)['semantic_logits']; b=model(data)['semantic_logits']
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    report=dict(loss=float(loss.detach()),gradients=groups,peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2,
                repeated_inference_exact=True)
    dump(root/'smoke.json',report); print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--samples',type=int,default=160)
    parser.add_argument('--max-points',type=int,default=16000)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); root=args.root.resolve(); root.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    payload=torch.load(INITIAL,map_location='cpu',weights_only=False)
    cfg=json.loads(SELECTION.read_text())['config']
    if args.smoke: smoke(root,payload,args); return
    model=build(payload).cuda()
    baseline_path=root/'baseline/real_val/val_metrics.json'
    baseline=json.loads(baseline_path.read_text()) if baseline_path.exists() else evaluate_fixed(model,REAL,baseline_path.parent,cfg,'val')
    del model; torch.cuda.empty_cache()
    plan=[dict(name='heads75',scope='heads',real_fraction=.75,seed=20260929),
          dict(name='partial50',scope='partial',real_fraction=.5,seed=20260929),
          dict(name='partial75',scope='partial',real_fraction=.75,seed=20260929)]
    dump(root/'plan.json',dict(steps=5,initial=str(INITIAL),initial_sha256=digest(INITIAL),
         initial_runs=plan,epochs_max=args.epochs,early_stop_patience=12,
         real_manifest_sha256=digest(REAL),helios_manifest_sha256=digest(HELIOS),
         later_runs=['best partial setting + HELIOS flight augmentation + hard crops',
                     'best setting + larger transformer decoder', 'winning setting: two additional seeds'],
         selection='real validation only; baseline is eligible; test after final selection'))
    results=[train_run(root,s,payload,baseline,cfg,args) for s in plan]
    partial=max([r for r in results if r['spec']['scope']=='partial'],key=lambda r:r['score'])
    difficulty=mine_difficulty(root,payload,cfg)
    ready=AUGMENT.with_name('READY.json')
    while not ready.exists() or json.loads(ready.read_text()).get('simulations')!=86:
        dump(root/'status.json',dict(stage='waiting_for_HELIOS_86_simulations'))
        print('Waiting for complete HELIOS flight variants',flush=True); time.sleep(20)
    augmented={**partial['spec'],'name':'partial_aug_hard','augment':True,'hard':True}
    results.append(train_run(root,augmented,payload,baseline,cfg,args,difficulty))
    best=max(results,key=lambda r:r['score'])
    larger={**best['spec'],'name':'larger_decoder','large':True}
    results.append(train_run(root,larger,payload,baseline,cfg,args,difficulty))
    winner=max(results,key=lambda r:r['score'])
    dump(root/'frozen_architecture_selection.json',dict(winner=winner,all_runs=results,test_used=False))
    seed_results=[winner]
    for seed in (20260930,20261001):
        spec={**winner['spec'],'name':f'repeat_{seed}','seed':seed}
        seed_results.append(train_run(root,spec,payload,baseline,cfg,args,difficulty))
    # Freeze representative by validation, not by the forthcoming test metrics.
    representative=max(seed_results,key=lambda r:r['score'])
    dump(root/'frozen_checkpoint_selection.json',dict(representative=representative,seeds=seed_results,test_used=False))
    evaluations=[]; test_rows=[]
    for name,checkpoint in [('baseline',str(INITIAL))]+[(r['name'],r['checkpoint']) for r in seed_results]:
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False); model=build(cp).cuda()
        for domain,manifest in [('real',REAL),('helios',HELIOS)]:
            destination=root/'final_test'/name/domain
            path=destination/'test_metrics.json'
            result=json.loads(path.read_text()) if path.exists() else evaluate_fixed(model,manifest,destination,cfg,'test')
            test_rows.append({**summary_row(name,domain,result),'checkpoint':checkpoint})
            evaluations.append(dict(run=name,domain=domain,result=result))
        del model; torch.cuda.empty_cache()
    statistics={}
    for domain in ('real','helios'):
        rows=[r for r in test_rows if r['domain']==domain and r['stage']!='baseline']
        statistics[domain]={k:dict(mean=float(np.mean([r[k] for r in rows])),std=float(np.std([r[k] for r in rows],ddof=1)))
                            for k in ('point_source_balanced_PQ','crown_source_balanced_PQ','point_F1','crown_F1')}
    report=dict(completed=True,steps_completed=[1,2,3,4,5],architecture_winner=winner,
                representative=representative,seeds=seed_results,test_rows=test_rows,three_seed_statistics=statistics,
                test_used_for_selection=False,
                caveat='Real test is historically exposed; paired regression benchmark, not pristine holdout. HELIOS test has 7 plots.')
    dump(root/'final_report.json',report); write_csv(root/'test_comparison.csv',test_rows); workbook(root)
    dump(root/'status.json',dict(stage='complete',report=str(root/'final_report.json')))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
