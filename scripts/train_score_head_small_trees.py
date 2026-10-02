#!/usr/bin/env python3
"""Fine-tune only object-score heads of frozen EZ-SP Q128 decoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.superpoints.model import CachedInstanceModel
from pointcloud.superpoints.score_only_loss import score_only_loss
from scripts.benchmark_superpoint_algorithms import save_json, sha256
from scripts.train_superpoint_decoder_pilot import (DEFAULT as PILOT, load_crop,
    batch_for, evaluate)

INITIAL = PILOT/'runs/ezsp_large_w192_q128/weights/best.pt'
DEFAULT = PROJECT/'outputs/dualcrown3d_score_head_small_trees_v1'


def validation_options():
    return {f'o{o:g}_m{m:g}':dict(object_threshold=o, mask_threshold=m,
                 minimum_points=8, duplicate_iou=.6)
            for o in (.05,.1,.2,.3,.4) for m in (.4,.5,.6)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--seed',type=int,default=20261001)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Protected run exists: {output}')
    (output/'weights').mkdir(parents=True)
    (output/'validation').mkdir()
    prepared=json.loads((PILOT/'prepared.json').read_text())
    train=[x for x in prepared['entries'] if x['split']=='train']
    val=[x for x in prepared['entries'] if x['split']=='val']
    if {x['group_id'] for x in train}&{x['group_id'] for x in val}:
        raise ValueError('Spatial split leakage')
    source=torch.load(INITIAL,map_location='cpu',weights_only=False)
    model=CachedInstanceModel('ezsp',queries=128,memory_tokens=1536,
                              graph_width=192,graph_layers=4,
                              wide_dim=192,wide_layers=2).cuda()
    model.load_state_dict(source['model'],strict=True)
    model.decoder.epoch=100
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.decoder.score,model.decoder.wide.score):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    params=[p for p in model.parameters() if p.requires_grad]
    config=dict(architecture='Frozen EZ-SP Q128 geometry and masks; train decoder.score and decoder.wide.score only',
                initialization=str(INITIAL),initialization_sha256=sha256(INITIAL),
                prepared_sha256=sha256(PILOT/'prepared.json'),train_crops=len(train),
                val_crops=len(val),epochs=args.epochs,seed=args.seed,
                learning_rate=1e-4,small_limit_voxels=300,
                loss='Hungarian score BCE, tiny matched trees x8, other matched trees x3, duplicates x1',
                selected_by='validation crop PQ subject to sparse-tree recall >= initial',
                test_used=False,gpu=torch.cuda.get_device_name())
    save_json(output/'configuration.json',config)
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(args.epochs,1),eta_min=1e-5)
    history=[];best=-1.;initial_sparse=None
    for epoch in range(args.epochs+1):
        record=dict(epoch=epoch)
        if epoch:
            model.train()
            model.semantic_head.eval()
            model.offset_head.eval()
            started=time.monotonic();values=[]
            for index in np.random.default_rng(args.seed+epoch).permutation(len(train)):
                batch=batch_for(load_crop(train[index]['cache']),'ezsp')
                result=model(batch)['instance_masks']
                loss=score_only_loss(result,batch['tree_id'])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite score loss: epoch={epoch} sample={index}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params,2.,error_if_nonfinite=True)
                optimizer.step()
                values.append(float(loss.detach()))
            scheduler.step()
            record.update(seconds=time.monotonic()-started,score_loss=float(np.mean(values)))
        if epoch==0 or epoch%2==0 or epoch==args.epochs:
            result=evaluate(model,val,validation_options())
            save_json(output/'validation'/f'epoch_{epoch:03d}.json',result)
            if initial_sparse is None:
                initial_sparse=max(x['sparse_tree_recall_le100_voxels'] for x in result.values())
            eligible={k:v for k,v in result.items()
                      if v['sparse_tree_recall_le100_voxels']>=initial_sparse}
            if eligible:
                name=max(eligible,key=lambda k:eligible[k]['score'])
                choice=eligible[name]
                record.update(validation_score=choice['score'],
                              point_pq=choice['point']['source_balanced_pq'],
                              crown_pq=choice['crown']['source_balanced_pq'],
                              sparse_recall=choice['sparse_tree_recall_le100_voxels'],
                              thresholds=name)
                if choice['score']>best:
                    best=choice['score']
                    checkpoint=output/'weights/best.pt'
                    torch.save(dict(model=model.state_dict(),config=config,
                                    epoch=epoch,initial_checkpoint=str(INITIAL)),checkpoint)
                    save_json(output/'selected.json',dict(epoch=epoch,validation=choice,
                              checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                              initial_sparse_recall=initial_sparse))
            else:
                record.update(validation_score=None,thresholds=None,
                              reason='sparse-tree recall below initial')
        history.append(record)
        save_json(output/'history.json',dict(epochs=history))
        print(json.dumps(record),flush=True)
    save_json(output/'DONE.json',dict(completed=True,best_validation_score=best,
              selected_epoch=json.loads((output/'selected.json').read_text())['epoch']))


if __name__=='__main__':
    main()
