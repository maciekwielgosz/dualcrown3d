#!/usr/bin/env python3
"""Re-run the validation sweep of a saved vote-centre checkpoint (validation plots only)."""
import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.vote_centers import VoteCenterNet
from scripts.cache_stage1_predictions import DEFAULT_OUTPUT, eligible_rows, write_json
from scripts.train_two_pass_refiner import gate_checks, stage1_baseline
from scripts.train_vote_centers import sweep, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run', required=True)
    parser.add_argument('--weights', default='last.pt')
    parser.add_argument('--tag', default='revalidation')
    parser.add_argument('--only', default='', help='comma-separated substrings; keep sweep entries containing any')
    args = parser.parse_args()
    root = args.output.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    payload = torch.load(root / 'runs' / args.run / 'weights' / args.weights, map_location='cpu', weights_only=False)
    network = VoteCenterNet(width=payload['width'], dropout=payload.get('dropout', 0.)).cuda()
    network.load_state_dict(payload['network'])
    rows = eligible_rows('val')
    context = {}
    base, _ = stage1_baseline(rows, root / 'stage1/val', context, protocol)
    configs = sweep()
    if args.only:
        configs = {k: v for k, v in configs.items() if any(part in k for part in args.only.split(','))}
    summaries, records, seconds, detection = validate(network, rows, root, protocol, configs, context, torch.device('cuda:0'))
    checks = {name: gate_checks(s, base, protocol['gate']) for name, s in summaries.items()}
    target = root / 'runs' / args.run / f'validation/{args.tag}_epoch_{payload["epoch"]:03d}/metrics.json'
    write_json(target, dict(summaries=summaries, per_plot=records, seconds=seconds, detection=detection,
                            checks=checks, stage1_only=base, weights=args.weights, epoch=payload['epoch']))
    for name, s in summaries.items():
        print(f"{name:26s} pPQ {s['point_sb_pq']:.4f} cPQ {s['crown_sb_pq']:.4f} pP {s['point_precision']:.3f} "
              f"cP {s['crown_precision']:.3f} small10 {s['small_10_tp']:3d} small4 {s['small_4_tp']:2d} "
              f"large {s['large_recall']:.3f} overs {s['oversplit_gt']:3d} unders {s['undersegmented_pred']:3d} "
              f"n {s['predicted_instances']:4d} gate {'PASS' if all(checks[name].values()) else [k for k, v in checks[name].items() if not v]}")
    print('stage1', {k: round(base[k], 4) if isinstance(base[k], float) else base[k]
                     for k in ('point_sb_pq', 'crown_sb_pq', 'point_precision', 'crown_precision', 'small_10_tp', 'large_recall')})


if __name__ == '__main__':
    main()
