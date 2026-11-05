#!/usr/bin/env python3
"""RESULTS.md of the scaled stage-1 fine-tune: Model20 against every validated checkpoint.

Reads ``validation/<name>/summary.json`` (written by ``validate_stage1_votes.py``) and the
run folders. Validation numbers only; no test data is involved.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
DEFAULT_ROOT = PROJECT / 'outputs/dualcrown3d_stage1_scaled_v1'
SETS = (('dense', 'dense'), ('thin10', 'thinned to 10 voxels/m2'), ('helios', 'HELIOS'))
MODES = (('single', 'single pass'), ('ens4', '4-rotation ensemble'))


def table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(header) - 1)) + '|']
    lines += ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows]
    return '\n'.join(lines)


def number(value, digits=3):
    return '' if value is None else f'{value:.{digits}f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--baseline', default='model20')
    parser.add_argument('--recommended', default=None, help='validation name of the recommended checkpoint')
    parser.add_argument('--notes', type=Path, default=None, help='markdown file appended verbatim')
    args = parser.parse_args()
    root = args.root.resolve()
    summaries = {p.parent.name: json.loads(p.read_text()) for p in sorted(root.glob('validation/*/summary.json'))}
    summaries = {k: v for k, v in summaries.items() if not v.get('from_cache') and not v.get('limit')}
    if args.baseline not in summaries:
        raise FileNotFoundError(f'Baseline validation {args.baseline} is missing')
    runs = {}
    for path in sorted(root.glob('runs/*/configuration.json')):
        run = path.parent
        config = json.loads(path.read_text())
        log = list(csv.DictReader((run / 'training_log.csv').open())) if (run / 'training_log.csv').exists() else []
        done = json.loads((run / 'DONE.json').read_text()) if (run / 'DONE.json').exists() else None
        selected = json.loads((run / 'selected.json').read_text()) if (run / 'selected.json').exists() else None
        runs[run.name] = dict(config=config, log=log, done=done, selected=selected)
    order = [args.baseline] + [k for k in summaries if k != args.baseline]

    def label(name):
        if name == args.baseline:
            return 'Model20'
        for which in ('best', 'last'):
            run = name.removesuffix('_' + which)
            if name.endswith('_' + which) and run in runs:
                info = runs[run]
                epoch = info['selected']['epoch'] if which == 'best' and info['selected'] else (info['log'][-1]['epoch'] if info['log'] else '?')
                return f'{run} ({which}, epoch {epoch})'
        return name

    text = ['# Scaled stage-1 fine-tune: vote quality against Model20', '',
            'All numbers are on validation plots only (14 real plots at native density, the same plots thinned to '
            '10 voxels/m2 with the frozen draw of the two-pass protocol, 12 HELIOS simulations). No test split was read, '
            'copied or evaluated. Produced by `scripts/report_stage1_scaled.py` from `validation/*/summary.json`.', '']
    if args.recommended:
        text += [f'**Recommended checkpoint: `{args.recommended}`** (see the end of this file).', '']
    text += ['## True-centre oracle (point / crown source-balanced PQ)', '',
             'Every voxel goes to the nearest per-tree median vote within the tree\'s true vote extent x 1.25 '
             '(`field_labels`, `extent_scale=1.25`). Independent of any centre detector: it is the ceiling a perfect '
             'second pass could reach with these votes.', '']
    for mode, mode_name in MODES:
        header = ['model'] + [f'{set_name} point' if i == 0 else f'{set_name} crown' for _, set_name in SETS for i in (0, 1)]
        rows = []
        for name in order:
            row = [label(name)]
            for key, _ in SETS:
                value = summaries[name]['summaries'].get(key, {}).get(mode)
                row += [number(value['oracle']['point_sb_pq']), number(value['oracle']['crown_sb_pq'])] if value else ['', '']
            rows.append(row)
        text += [f'### {mode_name}', '', table(header, rows), '']
    text += ['## Vote error against the true tree centroid (metres)', '',
             'Per tree: median over its voxels, then the median over trees (`tree`). Per voxel: pooled over all annotated '
             'tree voxels, median and mean (`voxel`). xy = horizontal, z = vertical.', '']
    for mode, mode_name in MODES:
        for key, set_name in SETS:
            header = ['model', 'tree xy', 'tree z', 'voxel xy median', 'voxel xy mean', 'voxel z median', 'voxel z mean',
                      'vote scatter', 'vote extent p90']
            rows = []
            for name in order:
                value = summaries[name]['summaries'].get(key, {}).get(mode)
                if value:
                    rows.append([label(name), number(value['tree_horizontal']['median']), number(value['tree_vertical']['median']),
                                 number(value['voxel_horizontal']['median']), number(value['voxel_horizontal']['mean']),
                                 number(value['voxel_vertical']['median']), number(value['voxel_vertical']['mean']),
                                 number(value['tree_scatter']['median']), number(value['tree_extent']['median'])])
            text += [f'### {mode_name}, {set_name}', '', table(header, rows), '']
    text += ['`vote scatter` / `vote extent p90`: median / 90th percentile of the distance of a tree\'s votes from their own '
             'median in (x, y, 0.33 z), median over trees. This is the size of the blob a centre detector has to separate.', '',
             '## Oracle with geometric centroids as centres (point / crown SB-PQ)', '',
             'Same assignment, but the centres are the true centroids instead of the median votes; this also charges a '
             'systematic bias of the votes.', '']
    for mode, mode_name in MODES:
        header = ['model'] + [f'{set_name} point' if i == 0 else f'{set_name} crown' for _, set_name in SETS for i in (0, 1)]
        rows = []
        for name in order:
            row = [label(name)]
            for key, _ in SETS:
                value = summaries[name]['summaries'].get(key, {}).get(mode)
                row += [number(value['oracle_centroid']['point_sb_pq']), number(value['oracle_centroid']['crown_sb_pq'])] if value else ['', '']
            rows.append(row)
        text += [f'### {mode_name}', '', table(header, rows), '']
    text += ['## Nearest-centre purity', '',
             'Share of annotated tree voxels whose vote is nearer to its own tree\'s median vote than to any other '
             'tree\'s, in (x, y, 0.33 z): the clustering-free separability of the votes (pooled over voxels / mean over trees).', '']
    header = ['model'] + [f'{set_name} {mode}' for _, set_name in SETS for mode, _ in MODES]
    rows = []
    for name in order:
        row = [label(name)]
        for key, _ in SETS:
            for mode, _ in MODES:
                value = summaries[name]['summaries'].get(key, {}).get(mode, {}).get('nearest_centre_purity')
                row.append(f"{value['voxel']:.3f} / {value['tree']:.3f}" if value else '')
        rows.append(row)
    text += [table(header, rows), '']
    text += ['## Semantic head and mask decoder', '',
             'Tree / non-tree accuracy and tree IoU of the semantic head on voxels with a known class (single pass), and '
             'the unchanged stage-1 consensus pipeline (masks + 2-D vote clustering, `merge_masks_with_sources` with the '
             'frozen Model20 configuration), which is the existing validation score and uses the mask decoder.', '']
    header = ['model'] + [f'{set_name} {what}' for _, set_name in SETS for what in ('acc', 'tree IoU', 'consensus point', 'consensus crown')]
    rows = []
    for name in order:
        row = [label(name)]
        for key, _ in SETS:
            value = summaries[name]['summaries'].get(key, {})
            single, consensus = value.get('single'), value.get('consensus')
            row += [number(single['semantic']['accuracy'], 4), number(single['semantic']['tree_iou'], 4)] if single else ['', '']
            row += [number(consensus['point_sb_pq']), number(consensus['crown_sb_pq'])] if consensus else ['', '']
        rows.append(row)
    text += [table(header, rows), '']
    if any('point_ens4' in summaries[n]['summaries'].get('dense', {}) for n in order):
        text += ['## Votes of the point decoder branch (`point_offset_m`) instead of the legacy vote head', '']
        header = ['model', 'mode'] + [f'{set_name} {what}' for _, set_name in SETS for what in ('point', 'crown', 'tree xy', 'tree z')]
        rows = []
        for name in order:
            for mode in ('point_single', 'point_ens4'):
                if mode not in summaries[name]['summaries'].get('dense', {}):
                    continue
                row = [label(name), mode]
                for key, _ in SETS:
                    value = summaries[name]['summaries'][key][mode]
                    row += [number(value['oracle']['point_sb_pq']), number(value['oracle']['crown_sb_pq']),
                            number(value['tree_horizontal']['median']), number(value['tree_vertical']['median'])]
                rows.append(row)
        text += [table(header, rows), '']
    text += ['## Runs', '']
    header = ['run', 'trainable', 'lr heads / backbone', 'crops done', 'voxels per crop', 'accumulation', 'z weight',
              'assign weight', 'thin p (range)', 'real data', 'best epoch', 'hours', 'state']
    rows = []
    for name, info in runs.items():
        c, log = info['config'], info['log']
        hours = sum(float(r['seconds']) + float(r.get('val_seconds') or 0) for r in log if r.get('seconds')) / 3600
        rows.append([name, f"{c['scope']} ({c['trainable_parameters'] / 1e6:.1f} M)", f"{c['lr_heads']:g} / {c['lr_backbone']:g}",
                     f"{int(log[-1]['crops']) if log and log[-1].get('crops') else 0:,} of {c['crops_total']:,}", f"{c['max_points']:,}",
                     c['accumulation'], c['z_weight'], c['assign_weight'],
                     f"{c['thin_probability']} ({c['thin_range'][0]:g}-{c['thin_range'][1]:g})" if c['thin_probability'] else '0',
                     c['real_source'], info['selected']['epoch'] if info['selected'] else '', f'{hours:.1f}',
                     'done' if info['done'] else 'incomplete'])
    text += [table(header, rows), '',
             'Model20 for comparison: partial (7.6 M trainable), 5e-05 / 5e-06, 6,400 crops of 16,000 voxels, accumulation 2, '
             'XY offset loss only, keep fractions 0.5 / 0.75 / 1.0, real data v2.', '',
             'In-training selection (`best`): single-pass votes, mean over dense and thin10 of sqrt(point x crown oracle PQ), '
             'validated every few epochs (`runs/<run>/training_log.csv`, `runs/<run>/validation/`).', '']
    if args.notes and args.notes.exists():
        text += [args.notes.read_text().rstrip(), '']
    (root / 'RESULTS.md').write_text('\n'.join(text))
    print('\n'.join(text))


if __name__ == '__main__':
    main()
