#!/usr/bin/env python3
"""Freeze the v7 protocol: Model20 baselines on the dense, thinned and HELIOS
validation sets and the gates a v7 candidate must pass. Written before any v7
detector is trained; refuses to overwrite."""
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.cache_stage1_predictions import DEFAULT_OUTPUT, sha256, write_json
from scripts.train_vote_field import KEYS, load_context, stage1_summary, validation_sets


def main():
    root = DEFAULT_OUTPUT.resolve()
    path = root / 'protocol_v7.json'
    if path.exists():
        raise FileExistsError(f'{path} is frozen')
    v1 = json.loads((root / 'protocol.json').read_text())
    index = json.loads((root / 'stage1_ensemble/index.json').read_text())
    baselines = {}
    for name, items in validation_sets(root).items():
        summary, _ = stage1_summary(load_context(items))
        baselines[name] = {**{k: summary[k] for k in KEYS}, 'plots': len(items)}
        print(name, json.dumps(baselines[name]), flush=True)
    for key in ('point_sb_pq', 'crown_sb_pq'):
        if abs(baselines['dense'][key] - v1['baseline'][key]) > 1e-6:
            raise AssertionError('Dense Model20 baseline not reproduced')
    dense = v1['gate']
    protocol = dict(
        protocol_v1_sha256=sha256(root / 'protocol.json'), stage1_sha256=v1['stage1_sha256'],
        ensemble_symmetries=index['symmetries'],
        validation_sets=dict(dense='14 native validation plots', thin10='the same plots thinned to 10 voxels/m2 (draw used since v6)',
                             thin25='the same plots thinned to 25 voxels/m2; reported, not gated',
                             helios='12 HELIOS++ simulated validation plots, about 14 points/m2'),
        baselines=baselines,
        gate=dict(
            dense=dict(point_sb_pq_margin=-dense['point_sb_pq_tolerance'], crown_sb_pq_margin=-dense['crown_sb_pq_tolerance'],
                       precision_tolerance=dense['precision_tolerance'], large_recall_tolerance=dense['large_recall_tolerance'],
                       small_10_minimum_tp=dense['small_10_minimum_tp']),
            thin10=dict(point_sb_pq_margin=.02, crown_sb_pq_margin=.02, precision_tolerance=.01,
                        large_recall_tolerance=.01, small_10_minimum_tp=baselines['thin10']['small_10_tp']),
            helios=dict(point_sb_pq_margin=-.005, crown_sb_pq_margin=-.005, precision_tolerance=.01,
                        large_recall_tolerance=.01, small_10_minimum_tp=None)),
        selection='Among (epoch, configuration) pairs passing all three gates with ONE configuration for every '
                  'density: maximise the summed gain over Model20 in sqrt(point SB-PQ x crown SB-PQ) on dense and thin10.',
        fallback='If no single configuration passes, a density switch is allowed: one configuration for plots with at '
                 'least 40 voxels/m2 and one below, each passing the gates of its regime (dense; thin10 and helios).',
        description='Dense: the v1 non-inferiority gate plus 16 of 161 small crowns. Thinned to 10 voxels/m2 (the '
                    'reference-scene density): a demonstrated benefit is required, +0.02 point and crown SB-PQ over '
                    'Model20, no loss of precision, large-tree recall or small crowns. HELIOS: non-inferiority.',
        test='Read once for the selected candidate: 24 native test plots, the same plots thinned to 10 voxels/m2, '
             '7 HELIOS test plots.',
        test_used=False)
    write_json(path, protocol)
    print(f'frozen {path}')


if __name__ == '__main__':
    main()
