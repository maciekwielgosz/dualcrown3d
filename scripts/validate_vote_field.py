#!/usr/bin/env python3
"""Full validation sweep of a saved vote-field checkpoint on every validation set
(dense, thinned, HELIOS). With ``--select`` the best configuration passing the
frozen v7 gates is written into ``weights/best.pt``. Validation plots only."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.vote_field import VoteFieldNet
from scripts.cache_stage1_predictions import DEFAULT_OUTPUT, sha256, write_json
from scripts.train_vote_field import (GATED_SETS, VALIDATION_SETS, gate_checks, load_context, quality, stage1_summary,
                                      sweep, validate, validation_sets)


def load_network(payload, device='cuda:0'):
    """One detector, or a list of them when the checkpoint is an ensemble (``members``)."""
    def build(item):
        network = VoteFieldNet(width=item['width'], dropout=item.get('dropout', 0.)).to(device)
        network.load_state_dict(item['network'])
        return network.eval()
    if 'members' in payload:
        return [build(item) for item in payload['members']]
    return build(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run', required=True)
    parser.add_argument('--weights', default='last.pt')
    parser.add_argument('--tag', default='full')
    parser.add_argument('--only', default='', help='comma-separated substrings; keep sweep entries containing any')
    parser.add_argument('--sets', nargs='+', default=list(VALIDATION_SETS))
    parser.add_argument('--no-flips', action='store_true')
    parser.add_argument('--jobs', type=int, default=10, help='processes for the configuration sweep')
    parser.add_argument('--select', action='store_true', help='freeze the best gate-passing configuration into best.pt')
    args = parser.parse_args()
    root = args.output.resolve()
    protocol = json.loads((root / 'protocol_v7.json').read_text())
    run = root / 'runs' / args.run
    weights = run / 'weights' / args.weights
    payload = torch.load(weights, map_location='cpu', weights_only=False)
    network = load_network(payload)
    configs = sweep(full=True)
    if args.only:
        configs = {k: v for k, v in configs.items() if any(part in k for part in args.only.split(','))}
    results, bases = {}, {}
    folder = run / f'validation/{args.tag}_epoch_{payload["epoch"]:03d}'
    for name, items in validation_sets(root).items():
        if name not in args.sets:
            continue
        context = load_context(items)
        bases[name], _ = stage1_summary(context)
        summaries, records, seconds, detection = validate(network, context, configs, torch.device('cuda:0'),
                                                          flips=not args.no_flips, jobs=args.jobs)
        write_json(folder / f'{name}.json', dict(summaries=summaries, per_plot=records, seconds=seconds, detection=detection,
                                                 stage1_only=bases[name], weights=args.weights, epoch=payload['epoch'],
                                                 flips=not args.no_flips))
        results[name] = summaries
        truth = max(detection.get('truth', 0), 1)
        print(f"== {name}: Model20 pPQ {bases[name]['point_sb_pq']:.4f} cPQ {bases[name]['crown_sb_pq']:.4f} "
              f"pP {bases[name]['point_precision']:.3f} cP {bases[name]['crown_precision']:.3f} "
              f"annotated pPQ {bases[name]['annotated_point_sb_pq']:.4f} pP {bases[name]['annotated_point_precision']:.3f} "
              f"fp-unann {bases[name]['fp_unannotated']} fp-in-tree {bases[name]['fp_inside_one_tree']} "
              f"small {bases[name]['small_10_tp']} large {bases[name]['large_recall']:.3f}; detector "
              + ' '.join(f"t{t}: R {detection.get(f'hits_{t}', 0) / truth:.2f} P {detection.get(f'hits_{t}', 0) / max(detection.get(f'predicted_{t}', 0), 1):.2f}"
                         for t in ('0.2', '0.3', '0.5')) + f" detect {seconds.get('detect', 0):.0f}s", flush=True)
        for config, s in summaries.items():
            checks = gate_checks(s, bases[name], protocol['gate'][name]) if name in protocol['gate'] else {}
            print(f"{config:34s} pPQ {s['point_sb_pq']:.4f} cPQ {s['crown_sb_pq']:.4f} pP {s['point_precision']:.3f} "
                  f"cP {s['crown_precision']:.3f} small {s['small_10_tp']:3d} large {s['large_recall']:.3f} "
                  f"over {s['oversplit_gt']:3d} under {s['undersegmented_pred']:3d} n {s['predicted_instances']:4d} "
                  f"| annotated pPQ {s['annotated_point_sb_pq']:.4f} pP {s['annotated_point_precision']:.3f} "
                  f"fp-unann {s['fp_unannotated']:3d} fp-in-tree {s['fp_inside_one_tree']:3d} "
                  f"{'PASS' if all(checks.values()) else [k for k, v in checks.items() if not v]}", flush=True)
    if all(s in results for s in GATED_SETS):
        checks = {c: {s: gate_checks(results[s][c], bases[s], protocol['gate'][s]) for s in GATED_SETS} for c in configs}
        passing = [c for c in configs if all(all(v.values()) for v in checks[c].values())]

        def gain(c):
            return sum(quality(results[s][c]) - quality(bases[s]) for s in ('dense', 'thin10'))
        ranked = sorted(configs, key=gain, reverse=True)
        print('passing all gates:', sorted(passing, key=gain, reverse=True)[:8])
        print('best gain regardless of gates:', [(c, round(gain(c), 4)) for c in ranked[:5]])
        selection = None
        if passing:
            best = max(passing, key=gain)
            selection = dict(epoch=payload['epoch'], weights=args.weights, config_name=best, config=configs[best],
                             gain=gain(best), validation={s: results[s][best] for s in results}, checks=checks[best],
                             source_weights_sha256=sha256(weights))
        write_json(folder / 'selection.json', dict(selected=selection, passing=passing, stage1_only=bases, gate=protocol['gate']))
        if args.select and selection:
            target = run / 'weights/best.pt'
            temporary = target.with_suffix('.tmp.pt')
            torch.save({**payload, 'selected_config': configs[best], 'selected_config_name': best}, temporary)
            os.replace(temporary, target)
            write_json(run / 'selected.json', dict(selected=selection, stage1_only=bases, gate=protocol['gate'],
                                                   test_used_for_selection=False, best_sha256=sha256(target)))
            print(f'selected {best} (epoch {payload["epoch"]}) -> {target}')


if __name__ == '__main__':
    main()
