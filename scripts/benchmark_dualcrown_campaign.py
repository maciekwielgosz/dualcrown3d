#!/usr/bin/env python3
"""Paired speed check of baseline and selected architecture, including merging."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts.train_dualcrown_campaign import build, dump, INITIAL, REAL, SELECTION
from scripts.predict_dual_head import predict
from pointcloud.data import read_manifest,load_npz
from pointcloud.instance_output import merge_masks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=PROJECT/'outputs/dualcrown3d_joint_campaign_v1')
    args=parser.parse_args(); root=args.root
    report=json.loads((root/'final_report.json').read_text())
    config=json.loads(SELECTION.read_text())['config']
    all_rows=read_manifest(REAL,'val')
    rows=[next(r for r in all_rows if r['collection']==c) for c in ('CULS','FGI_EMIT','WILDFOREST3D')]
    arrays=[load_npz(r['output']) for r in rows]
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    models={name:build(torch.load(path,map_location='cpu',weights_only=False)).cuda().eval()
            for name,path in [('baseline',INITIAL),('selected',report['representative']['checkpoint'])]}
    for model in models.values(): predict(model,arrays[0])
    results=[]
    for repeat in range(3):
        for name in (('baseline','selected') if repeat%2==0 else ('selected','baseline')):
            model=models[name]
            for row,a in zip(rows,arrays):
                torch.cuda.synchronize(); t=time.perf_counter()
                raw=predict(model,a); torch.cuda.synchronize(); inference=time.perf_counter()-t
                start=time.perf_counter(); labels,_,instances=merge_masks(a,raw,config)
                merge=time.perf_counter()-start
                results.append(dict(model=name,repeat=repeat,plot=row['dataset_id'],voxels=len(labels),
                                    network_seconds=inference,merge_seconds=merge,total_seconds=inference+merge,
                                    instances=len(instances)))
    summary={}
    for name in models:
        totals=[sum(r['total_seconds'] for r in results if r['model']==name and r['repeat']==i) for i in range(3)]
        summary[name]=dict(median_seconds=float(np.median(totals)),repeat_seconds=totals)
    summary['selected_to_baseline_ratio']=summary['selected']['median_seconds']/summary['baseline']['median_seconds']
    dump(root/'speed_benchmark.json',dict(summary=summary,measurements=results,
         protocol='Same three real validation plots, warmed models, alternating model order, three repeats; network+CPU merging; excludes loading and export'))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
