#!/usr/bin/env python3
"""One line per vote-field run: best gate-passing configuration of its full validation sweep
and a fixed reference configuration, for comparing runs (validation only)."""
import argparse
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT = PROJECT / 'outputs/dualcrown3d_two_pass_v1'
REFERENCE = 'replace_t0.5_nms0_extent1.5_norm'
SETS = ('dense', 'thin10', 'helios')


def quality(summary):
    return math.sqrt(max(summary['point_sb_pq'] * summary['crown_sb_pq'], 0.))


def describe(results, name):
    cells = []
    for s in SETS:
        r = results[s]['summaries'][name]
        cells.append(f"{s} {r['point_sb_pq']:.3f}/{r['crown_sb_pq']:.3f} ann {r.get('annotated_point_sb_pq', float('nan')):.3f} "
                     f"sm {r['small_10_tp']:2d} lg {r['large_recall']:.2f} pP {r['point_precision']:.2f}")
    return ' | '.join(cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--match', default='')
    parser.add_argument('--tag', default='full')
    args = parser.parse_args()
    rows = []
    for folder in sorted((args.output / 'runs').glob(f'*{args.match}*/validation/{args.tag}_epoch_*')):
        try:
            results = {s: json.loads((folder / f'{s}.json').read_text()) for s in SETS}
            selection = json.loads((folder / 'selection.json').read_text())
        except FileNotFoundError:
            continue
        bases = {s: results[s]['stage1_only'] for s in SETS}

        def gain(name):
            return sum(quality(results[s]['summaries'][name]) - quality(bases[s]) for s in ('dense', 'thin10'))

        def annotated_gain(name):
            return sum(math.sqrt(results[s]['summaries'][name]['annotated_point_sb_pq'] * results[s]['summaries'][name]['crown_sb_pq'])
                       - math.sqrt(bases[s]['annotated_point_sb_pq'] * bases[s]['crown_sb_pq']) for s in ('dense', 'thin10'))
        best = selection['selected']['config_name'] if selection.get('selected') else None
        rows.append((folder.parts[-3], folder.name, best, gain(best) if best else float('nan'), gain(REFERENCE), results, annotated_gain))
    for run, tag, best, best_gain, reference_gain, results, annotated_gain in sorted(rows, key=lambda r: -(r[3] if r[3] == r[3] else -1)):
        d = results['dense']['detection']
        truth = max(d.get('truth', 1), 1)
        print(f"== {run} [{tag}]  detector dense R/P@0.5 {d.get('hits_0.5', 0) / truth:.2f}/{d.get('hits_0.5', 0) / max(d.get('predicted_0.5', 1), 1):.2f}")
        if best:
            print(f"   best {best:36s} gain {best_gain:.4f} ann-gain {annotated_gain(best):.4f} | {describe(results, best)}")
        print(f"   ref  {REFERENCE:36s} gain {reference_gain:.4f} ann-gain {annotated_gain(REFERENCE):.4f} | {describe(results, REFERENCE)}")


if __name__ == '__main__':
    main()
