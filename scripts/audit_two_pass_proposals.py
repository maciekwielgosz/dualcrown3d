#!/usr/bin/env python3
"""Funnel audit of second-pass proposals on validation plots (diagnostic only).

For each proposal: best ground-truth point IoU, whether that tree was missed by
stage 1, decision/quality scores and the reconciliation outcome. Ground truth is
used for the audit only. Nothing is selected and the test split is not read.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz
from pointcloud.joint_training import fixed_serialization
from pointcloud.two_pass import build_refiner, match_instances
from pointcloud.two_pass_inference import stage2_sweep
from pointcloud.two_pass_reconcile import DEFAULT_RECONCILE, reconcile
from scripts.cache_stage1_predictions import DEFAULT_OUTPUT, eligible_rows, write_json
from scripts.train_supervision_v4 import build
from scripts.train_two_pass_refiner import load_stage1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run', default='residual_v1')
    parser.add_argument('--weights', default='last.pt')
    parser.add_argument('--new-probability', type=float, default=.5)
    parser.add_argument('--min-quality', type=float, default=.2)
    parser.add_argument('--max-take-fraction', type=float, default=.5)
    args = parser.parse_args()
    root = args.output.resolve()
    payload = torch.load(root / 'runs' / args.run / 'weights' / args.weights, map_location='cpu', weights_only=False)
    model, _ = build(payload['stage1_checkpoint'])
    model.requires_grad_(False)
    model.eval()
    fixed_serialization(model)
    refiner = build_refiner(payload).cuda()
    config = {**DEFAULT_RECONCILE, 'new_probability': args.new_probability, 'min_quality': args.min_quality,
              'max_take_fraction': args.max_take_fraction}
    totals, per_plot = Counter(), []
    for row in eligible_rows('val'):
        arrays = load_npz(row['output'])
        stage1, _, _ = load_stage1(root / 'stage1/val' / f"{row['dataset_id']}.npz")
        truth = arrays['tree_id']
        gt_ids, matched, _ = match_instances(truth, stage1['labels'])
        gt_size = np.bincount(np.searchsorted(gt_ids, truth[truth > 0]), minlength=len(gt_ids))
        proposals = stage2_sweep(model, refiner, arrays, stage1)
        trace = {}
        reconcile(arrays, stage1['labels'], stage1['confidence'], stage1['source'], proposals, config, trace)
        offsets = proposals['candidate_offset']
        decision, quality = proposals['decision'], proposals['quality']
        known = truth >= 0
        counter = Counter(proposals=len(quality), missed_gt=int((matched == 0).sum()))
        recoverable, recovered = set(), set()
        for k in range(len(quality)):
            a, b = offsets[k:k + 2]
            idx = proposals['point_index'][a:b][proposals['point_score'][a:b] >= .5]
            idx = idx[known[idx]]
            owners = truth[idx]
            owners = owners[owners > 0]
            best, target = 0., -1
            if len(owners):
                values, counts = np.unique(owners, return_counts=True)
                g = np.searchsorted(gt_ids, values)
                iou = counts / (len(idx) + gt_size[g] - counts)
                best, target = float(iou.max()), int(g[iou.argmax()])
            good = best >= .5
            missed = good and matched[target] == 0
            kind = 'good_missed' if missed else 'good_found' if good else 'poor'
            counter[kind] += 1
            if decision[k, 1] >= args.new_probability:
                counter[f'{kind}__p_new'] += 1
                if quality[k] >= args.min_quality:
                    counter[f'{kind}__p_new_quality'] += 1
            counter[f'{kind}__{trace.get(k, "below_threshold")}'] += 1
            if missed:
                recoverable.add(target)
                if trace.get(k) == 'accepted':
                    recovered.add(target)
        counter['unique_missed_gt_with_good_mask'] = len(recoverable)
        counter['unique_missed_gt_accepted'] = len(recovered)
        totals.update(counter)
        per_plot.append(dict(dataset_id=row['dataset_id'], **counter))
        print(row['dataset_id'], dict(missed=counter['missed_gt'], good_mask=len(recoverable), accepted=len(recovered),
                                      poor_accepted=counter['poor__accepted'], found_accepted=counter['good_found__accepted']), flush=True)
    result = dict(run=args.run, weights=args.weights, epoch=payload['epoch'], config=config,
                  totals=dict(sorted(totals.items())), per_plot=per_plot,
                  note='Diagnostic on validation; good = point IoU >= 0.5 of the mask (>=0.5) with a GT tree on known voxels.')
    write_json(root / 'runs' / args.run / f'audit_epoch_{payload["epoch"]:03d}.json', result)
    print(json.dumps(result['totals'], indent=1))


if __name__ == '__main__':
    main()
