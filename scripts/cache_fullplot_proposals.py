#!/usr/bin/env python3
"""Cache full-plot Q128 mask proposals before assigning any tree IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz, read_manifest
from scripts.benchmark_superpoint_algorithms import load_methods, save_json, sha256
from scripts.evaluate_output20_output22_test import NEW, wide_predict_plot
from scripts.predict_superpoint_wide_scene import load_models
from scripts.train_supervision_v4 import REAL

DEFAULT = PROJECT/'outputs/dualcrown3d_fullplot_verifier_v1'


def setup():
    payload=torch.load(NEW,map_location='cpu',weights_only=False)
    initial=Path(payload['initial_checkpoint'])
    embedding=PROJECT/'outputs/dualcrown3d_superpoint_v1/stage2_instance_embedding/embedding.pt'
    if payload['epoch']!=12 or payload['config']['run_name']!='ezsp_large_w192_q128':
        raise ValueError('Unexpected Q128 checkpoint')
    if sha256(initial)!=payload['config']['initial_sha256']:
        raise ValueError('Q128 backbone checksum differs')
    config=dict(checkpoint=str(NEW),checkpoint_sha256=sha256(NEW),
                initial_checkpoint=str(initial),initial_sha256=sha256(initial),
                embedding=str(embedding),embedding_sha256=sha256(embedding),
                manifest_sha256=sha256(REAL),window_m=20.,overlap_m=8.,
                query_candidate_object_min=.1,query_mask_min=.2,
                protocol='whole ALS plot; raw overlapping-window point masks before any tree IDs')
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    return payload,initial,embedding,config,signature


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=('train','val'),required=True)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required to build uncached full-plot proposals')
    torch.set_num_threads(4)
    payload,initial,embedding,config,signature=setup()
    output=args.output.resolve()
    rows=[r for r in read_manifest(REAL,args.split) if r.get('point_eval_eligible')=='true']
    if args.limit:rows=rows[:args.limit]
    metadata=output/'configuration.json'
    if metadata.exists():
        if json.loads(metadata.read_text())['signature']!=signature:
            raise ValueError('Existing cache has another checkpoint or protocol')
    else:
        output.mkdir(parents=True,exist_ok=True)
        save_json(metadata,dict(**config,signature=signature))
    models=None;methods=None
    for number,row in enumerate(rows,1):
        path=output/'raw'/args.split/(row['dataset_id']+'.npz')
        input_hash=sha256(Path(row['output']))
        if path.exists():
            with np.load(path) as file:
                if str(file['signature'])!=signature or str(file['input_sha256'])!=input_hash:
                    raise ValueError(f'Stale raw proposal cache: {path}')
                count=len(file['object_score'])
            print(f'skip {number}/{len(rows)} {row["dataset_id"]}: {count} masks',flush=True)
            continue
        if models is None:
            models=load_models(payload,initial,embedding,'cuda:0')
            _,csr,ezsp,_=load_methods()
            methods=(csr,ezsp)
        arrays=load_npz(row['output'])
        started=time.monotonic()
        backbone,decoder,embed=models
        csr,ezsp=methods
        seed=20261001+int(hashlib.sha256(row['dataset_id'].encode()).hexdigest()[:8],16)
        raw=wide_predict_plot(arrays,backbone,decoder,embed,csr,ezsp,'cuda:0',seed)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.stem+'.tmp.npz')
        np.savez_compressed(temporary,**raw,signature=np.asarray(signature),
                            input_sha256=np.asarray(input_hash),
                            point_count=np.int64(len(arrays['coord'])))
        os.replace(temporary,path)
        print(f'cached {number}/{len(rows)} {row["dataset_id"]}: '
              f'{len(raw["object_score"])} masks, '
              f'{int(raw["candidate_offset"][-1])} memberships, '
              f'{time.monotonic()-started:.1f}s',flush=True)


if __name__=='__main__':main()
