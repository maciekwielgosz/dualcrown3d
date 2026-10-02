#!/usr/bin/env python3
"""Export matched 3-D point IDs and full 2-D crowns from shared v5 queries."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.dual_head import DualHeadLitePT
from pointcloud.shared_merge import merge_shared_queries
from scripts.predict_dual_head import predict, write_vectors, export_laz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--prepared-dir', type=Path,
                        default=PROJECT/'output_15_litept_v2_no_rectangles_pointcloud/work')
    parser.add_argument('--output-dir', type=Path,
                        default=PROJECT/'output_21_shared_instances_v5')
    parser.add_argument('--stage', choices=('predict','export','all'), default='all')
    parser.add_argument('--raw-predictions', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.raw_predictions is not None and args.stage != 'export':
        parser.error('--raw-predictions requires --stage export')
    torch.set_num_threads(4)
    output = args.output_dir.resolve()
    if (output/'inference_report.json').exists():
        raise FileExistsError('Completed inference output is protected')
    work = output/'work'
    work.mkdir(parents=True,exist_ok=True)
    with np.load(args.prepared_dir/'benchmark_pointcloud.npz') as f:
        arrays={k:f[k] for k in f.files}
    metadata=json.loads((args.prepared_dir/'preparation.json').read_text())
    selection=json.loads(args.selection.read_text())
    digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest!=selection['checkpoint_sha256']:
        raise ValueError('Checkpoint hash differs from validation selection')
    path = args.raw_predictions.resolve() if args.raw_predictions else work/'shared_predictions.npz'
    if args.stage in ('predict','all'):
        if path.exists():
            raise FileExistsError(path)
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        model=DualHeadLitePT(**checkpoint['model_args']).to(args.device)
        model.load_state_dict(checkpoint['model'],strict=True)
        raw=predict(model,arrays,owner_only=True,collect_crowns=True,
                    raw_object_threshold=.005)
        raw['checkpoint_sha256']=np.asarray(digest)
        np.savez_compressed(path,**raw)
        del model
        torch.cuda.empty_cache()
    if args.stage in ('export','all'):
        started=time.monotonic()
        with np.load(path) as f:raw={k:f[k] for k in f.files}
        if str(raw['checkpoint_sha256'])!=digest or len(raw['point_probability'])!=len(arrays['coord']):
            raise ValueError('Raw predictions do not belong to this checkpoint or scene')
        ids,confidence,instances,sources=merge_shared_queries(arrays,raw,selection['config'])
        core=[item for item in instances if any(
            tile['bounds'][0] <= item['top_x'] < tile['bounds'][2] and
            tile['bounds'][1] <= item['top_y'] < tile['bounds'][3] for tile in metadata['tiles'])]
        renumber=np.zeros(int(ids.max(initial=0))+1,np.uint32)
        for number,item in enumerate(core,1):
            renumber[item['tree_id']]=number
            item['tree_id']=number
        ids=renumber[ids]
        confidence[ids==0]=0.
        sources[ids==0]=0
        point_report=write_vectors(core,metadata,output/'PointHead/Segmentation3')
        benchmark_report=write_vectors(core,metadata,output/'Segmentation3')
        clouds=export_laz(output,arrays,metadata,ids,np.zeros_like(ids),confidence,
                          (raw['point_probability']>=.3).astype(np.uint8),sources)
        report=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=digest,
                    selection=str(args.selection.resolve()),raw_predictions=str(path.resolve()),
                    merge_config=selection['config'],inference_seconds=float(raw['seconds']),
                    export_seconds=time.monotonic()-started,windows=int(raw['windows']),
                    instances=len(core),point_vectors=point_report,
                    benchmark_vectors=benchmark_report,clouds=clouds,
                    protocol='shared_v5 matched point and full-crown queries')
        (output/'inference_report.json').write_text(json.dumps(report,indent=2)+'\n')
        (output/'README.md').write_text(
            '# Shared point and crown instances\n\n'
            'PointHead/Segmentation3 and Segmentation3 contain the same tree IDs and polygons. '
            'PointClouds contains original ALS points, original XYZ and a global tree_id. '
            'tree_id=0 is unassigned, pred_semantic=1 is a predicted tree and '
            'segmentation_status=2 is a predicted tree lacking an accepted instance. '
            'legacy_tree_id=0 because the vote branch is auxiliary, not an exported instance branch. '
            'Crowns come from the shared query raster head and its matched point support. '
            'See inference_report.json for checkpoint hash and thresholds.\n')
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
