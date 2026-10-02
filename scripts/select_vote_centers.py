#!/usr/bin/env python3
"""Freeze the validation-selected vote-centre checkpoint and configuration.

Reads saved revalidation sweeps (validation plots only), keeps configurations
that pass the preregistered gate and writes ``weights/best.pt`` carrying the
selected configuration. If nothing passes, nothing is written.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.cache_stage1_predictions import DEFAULT_OUTPUT, sha256, write_json
from scripts.train_two_pass_refiner import rank
from scripts.train_vote_centers import sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run', required=True)
    parser.add_argument('--tag', default='adaptive')
    args = parser.parse_args()
    run = args.output.resolve() / 'runs' / args.run
    configs = sweep()
    candidates = []
    for path in sorted(run.glob(f'validation/{args.tag}_epoch_*/metrics.json')):
        data = json.loads(path.read_text())
        for name, summary in data['summaries'].items():
            if all(data['checks'][name].values()):
                candidates.append(dict(epoch=data['epoch'], weights=data['weights'], config_name=name,
                                       config=configs[name], validation=summary, checks=data['checks'][name],
                                       source=str(path)))
    if not candidates:
        print('No configuration passed the gate; nothing selected')
        return
    best = max(candidates, key=lambda c: rank(c['validation']))
    payload = torch.load(run / 'weights' / best['weights'], map_location='cpu', weights_only=False)
    payload['selected_config'] = best['config']
    payload['selected_config_name'] = best['config_name']
    torch.save(payload, run / 'weights/best.pt')
    selection = json.loads((run / 'selected.json').read_text())
    selection['training_sweep_selected'] = selection.get('selected')
    selection['selected'] = best
    selection['all_gate_passing'] = [{k: c[k] for k in ('epoch', 'config_name')} | {
        m: c['validation'][m] for m in ('point_sb_pq', 'crown_sb_pq', 'point_precision', 'crown_precision',
                                         'small_10_tp', 'large_recall')} for c in candidates]
    selection['selection_note'] = ('Selected from revalidation sweeps on the 14 validation plots (height-dependent '
                                   'assignment radius and asymmetric thresholds, added after the training-time sweep). '
                                   'Rank: small crowns, then geometric mean of point/crown SB-PQ. Test split not used.')
    selection['best_sha256'] = sha256(run / 'weights/best.pt')
    write_json(run / 'selected.json', selection)
    print(json.dumps({k: best[k] for k in ('epoch', 'config_name', 'config')}, indent=2))
    print(len(candidates), 'gate-passing configurations')


if __name__ == '__main__':
    main()
