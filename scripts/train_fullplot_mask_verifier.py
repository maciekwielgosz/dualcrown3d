#!/usr/bin/env python3
"""Train mask-graph verifier on full ALS plots, including fragment negatives."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import geopandas as gpd
import numpy as np
import shapely
from sklearn.metrics import average_precision_score
import torch
from torch.nn import functional as F

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from pointcloud.data import load_npz,read_manifest
from pointcloud.superpoints.fullplot_graph import (DEFAULT_GRAPH, FEATURES,
    reconcile,target_labels)
from pointcloud.superpoints.fullplot_verifier import ProposalVerifier,assign_complete_masks
from scripts.benchmark_superpoint_algorithms import save_json,sha256
from scripts.evaluate_output20_output22_test import score,summary
from scripts.train_superpoint_decoder_pilot import crown_records
from scripts.train_supervision_v4 import REAL

DEFAULT=PROJECT/'outputs/dualcrown3d_fullplot_verifier_v1'
PRIOR=PROJECT/'output_24_guarded_small_crown_fusion/selection.json'


def eligible(split):
    return [r for r in read_manifest(REAL,split) if r.get('point_eval_eligible')=='true']


def signature(root):
    cache=json.loads((root/'configuration.json').read_text())
    protocol=dict(raw_signature=cache['signature'],manifest_sha256=sha256(REAL),
                  graph=DEFAULT_GRAPH,features=FEATURES,
                  target='GT point IoU>=0.5; unknown-dominated candidates ignored',
                  hard_negative='0.10<=best GT IoU<0.50')
    value=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    return protocol,value


def graph_for(row,split,root,graph_signature):
    path=root/'graph'/split/(row['dataset_id']+'.npz')
    if path.exists():
        with np.load(path) as file:
            if str(file['signature'])!=graph_signature:
                raise ValueError(f'Graph cache signature mismatch: {path}')
            return {k:file[k] for k in file.files if k!='signature'}
    arrays=load_npz(row['output'])
    raw_path=root/'raw'/split/(row['dataset_id']+'.npz')
    with np.load(raw_path) as file:
        if int(file['point_count'])!=len(arrays['coord']):
            raise ValueError(f'Raw mask point count mismatch: {raw_path}')
        raw={key:file[key] for key in file.files}
    graph=reconcile(arrays,raw,DEFAULT_GRAPH)
    targets,iou,gt=target_labels(graph,arrays['tree_id'])
    graph.update(target=targets,best_iou=iou,best_gt=gt)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(temporary,**graph,signature=np.asarray(graph_signature))
    os.replace(temporary,path)
    return graph


def prepare(root,protocol,graph_signature):
    meta=root/'graph_protocol.json'
    if meta.exists() and json.loads(meta.read_text())['signature']!=graph_signature:
        raise ValueError('Graph protocol differs from existing cache')
    save_json(meta,dict(**protocol,signature=graph_signature))
    report={}
    for split in ('train','val'):
        rows=eligible(split)
        diagnostics=[]
        for number,row in enumerate(rows,1):
            graph=graph_for(row,split,root,graph_signature)
            target=graph['target']
            diagnostics.append(dict(dataset_id=row['dataset_id'],source=row['source_dataset']+':'+row['collection'],
                                    proposals=len(target),positive=int((target==1).sum()),
                                    negative=int((target==0).sum()),ignored=int((target<0).sum()),
                                    hard_negative=int(((target==0)&(graph['best_iou']>=.1)).sum()),
                                    graph_edges=int(graph['graph_edges'])))
            print(f'graph {split} {number}/{len(rows)} {row["dataset_id"]}: '
                  f'{len(target)} clusters, {(target==1).sum()} matched trees',flush=True)
        report[split]=diagnostics
    save_json(root/'graph_summary.json',report)
    return report


def matrix(root,split,graph_signature):
    feature,target,iou,source=[],[],[],[]
    for row in eligible(split):
        graph=graph_for(row,split,root,graph_signature)
        keep=graph['target']>=0
        feature.append(graph['feature'][keep])
        target.append(graph['target'][keep])
        iou.append(graph['best_iou'][keep])
        source.extend([row['source_dataset']+':'+row['collection']]*int(keep.sum()))
    return (np.concatenate(feature),np.concatenate(target),
            np.concatenate(iou),np.asarray(source))


def train(root,graph_signature,epochs=50):
    selected=root/'verifier_selected.json'
    if selected.exists():
        answer=json.loads(selected.read_text())
        if answer['graph_signature']!=graph_signature:
            raise ValueError('Existing model is from another graph protocol')
        return answer
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for proposal verifier training')
    torch.set_num_threads(4)
    np.random.seed(20261001);torch.manual_seed(20261001)
    x,y,iou,source=matrix(root,'train',graph_signature)
    xv,yv,_,_=matrix(root,'val',graph_signature)
    mean=x.mean(0);std=x.std(0).clip(min=1e-3)
    x=np.clip((x-mean)/std,-5,5).astype(np.float32)
    xv=np.clip((xv-mean)/std,-5,5).astype(np.float32)
    counts=Counter(source.tolist())
    sample=np.asarray([1./np.sqrt(counts[s]) for s in source],np.float32)
    sample/=sample.mean()
    # Fragment negatives matter; positives are balanced without discarding hard cases.
    sample*=np.where(y==1, min(max((y==0).sum()/max((y==1).sum(),1),1.),8.),
                     np.where(iou>=.1,2.,1.)).astype(np.float32)
    model=ProposalVerifier(len(FEATURES)).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    tensor_x=torch.from_numpy(x).cuda();tensor_y=torch.from_numpy(y.astype(np.float32)).cuda()
    tensor_w=torch.from_numpy(sample).cuda()
    tensor_v=torch.from_numpy(xv).cuda()
    history=[];best=-1.;checkpoint=root/'weights/best.pt'
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for indices in np.array_split(np.random.default_rng(20261001+epoch).permutation(len(x)),
                                      max(1,int(np.ceil(len(x)/512)))):
            indices=torch.as_tensor(indices,device='cuda')
            output=model(tensor_x[indices])
            loss=(F.binary_cross_entropy_with_logits(output,tensor_y[indices],reduction='none')*
                  tensor_w[indices]).sum()/tensor_w[indices].sum().clamp_min(1.)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():prob=torch.sigmoid(model(tensor_v)).cpu().numpy()
        ap=float(average_precision_score(yv,prob)) if len(np.unique(yv))>1 else 0.
        record=dict(epoch=epoch,loss=float(np.mean(losses)),val_average_precision=ap)
        history.append(record)
        if ap>best:
            best=ap
            torch.save(dict(model=model.state_dict(),mean=mean,std=std,
                            graph_signature=graph_signature,epoch=epoch,
                            feature_names=FEATURES),checkpoint)
            save_json(selected,dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                                    graph_signature=graph_signature,epoch=epoch,
                                    val_average_precision=ap,train_candidates=len(x),
                                    train_positives=int(y.sum()),val_candidates=len(xv),
                                    val_positives=int(yv.sum()),gpu=torch.cuda.get_device_name()))
        if epoch==1 or epoch%10==0 or epoch==epochs:
            print('verifier',record,flush=True)
    save_json(root/'training_history.json',dict(epochs=history))
    return json.loads(selected.read_text())


def load_verifier(root,graph_signature):
    chosen=json.loads((root/'verifier_selected.json').read_text())
    path=Path(chosen['checkpoint'])
    if sha256(path)!=chosen['checkpoint_sha256']:
        raise ValueError('Verifier checkpoint checksum mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload['graph_signature']!=graph_signature:
        raise ValueError('Verifier graph signature mismatch')
    model=ProposalVerifier(len(FEATURES)).eval()
    model.load_state_dict(payload['model'])
    return model,payload


def probability(model,payload,feature):
    if not len(feature):return np.empty(0,np.float32)
    values=np.clip((feature-payload['mean'])/payload['std'],-5,5).astype(np.float32)
    with torch.no_grad():return model(torch.from_numpy(values)).sigmoid().numpy()


def validate(root,graph_signature):
    target=root/'validation_selection.json'
    if target.exists():
        answer=json.loads(target.read_text())
        if answer['graph_signature']!=graph_signature:
            raise ValueError('Validation selection graph mismatch')
        return answer
    model,payload=load_verifier(root,graph_signature)
    configs={f'p{p:g}_n{n:g}':dict(object_threshold=p,claimed_limit=n)
             for p in (.15,.3,.45,.6,.75) for n in (.15,.30,.50)}
    results={name:[] for name in configs}
    diagnostics=[]
    rows=eligible('val')
    for number,row in enumerate(rows,1):
        arrays=load_npz(row['output'])
        graph=graph_for(row,'val',root,graph_signature)
        p=probability(model,payload,graph['feature'])
        gt=gpd.read_file(row['gt_vector'])
        ignore=shapely.from_wkb(Path(row['ignore_vector']).read_bytes()) if row.get('ignore_vector') else None
        world_xy=arrays['coord'][:,:2].astype(np.float64)+arrays['source_origin'][:2]
        for name,cfg in configs.items():
            labels,confidence,accepted=assign_complete_masks(arrays,graph,p,**cfg)
            crowns=crown_records({**arrays,'world_xy':world_xy},labels,confidence)
            entry=score(row,arrays,gt,ignore,labels,[r['geometry'] for r in crowns])
            entry['accepted_clusters']=len(accepted)
            results[name].append(entry)
        diagnostics.append(dict(dataset_id=row['dataset_id'],clusters=len(p),
                                probability_quantiles=np.quantile(p,[0,.1,.5,.9,1]).tolist() if len(p) else []))
        print(f'validation {number}/{len(rows)} {row["dataset_id"]}: {len(p)} graph clusters',flush=True)
    trials={name:dict(config=configs[name],summary=summary(entries),per_plot=entries)
            for name,entries in results.items()}
    baseline22=json.loads(PRIOR.read_text())['raw22']
    small_floor=int(np.ceil(.8*baseline22['small_crowns']['up_to_10_m2']['tp']))
    # Minimum viable improvement: no catastrophic loss of small crowns while
    # gaining both point and crown PQ over the fragmented Model22 baseline.
    passed=[name for name,t in trials.items()
            if (t['summary']['point']['source_balanced_pq']>baseline22['point']['source_balanced_pq'] and
                t['summary']['crown']['source_balanced_pq']>baseline22['crown']['source_balanced_pq'] and
                t['summary']['small_crowns']['up_to_10_m2']['tp']>=small_floor)]
    winner=max(passed,key=lambda name:(
        trials[name]['summary']['point']['source_balanced_pq']+
        trials[name]['summary']['crown']['source_balanced_pq'],
        trials[name]['summary']['small_crowns']['up_to_10_m2']['tp'])) if passed else None
    answer=dict(graph_signature=graph_signature,verifier_checkpoint=str(root/'weights/best.pt'),
                split='val',plots=len(rows),baseline22_validation_summary=baseline22,
                selected=winner,passed=passed,trials=trials,diagnostics=diagnostics,
                gate=f'improve both PQ over Model22 validation and retain >={small_floor} small crowns')
    save_json(target,answer)
    return answer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--phase',choices=('prepare','train','validate','all'),default='all')
    parser.add_argument('--epochs',type=int,default=50)
    args=parser.parse_args();root=args.root.resolve()
    protocol,graph_signature=signature(root)
    if args.phase in ('prepare','all'):prepare(root,protocol,graph_signature)
    if args.phase in ('train','all'):train(root,graph_signature,args.epochs)
    if args.phase in ('validate','all'):
        answer=validate(root,graph_signature)
        print('selected',answer['selected'],'passed',len(answer['passed']),flush=True)


if __name__=='__main__':main()
