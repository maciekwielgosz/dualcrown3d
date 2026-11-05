#!/usr/bin/env python3
"""Train the v7 vote-field centre detector on rotation-ensembled Model20 votes.

Training mixes three regimes (dense plots, the same plots thinned, HELIOS
simulations). Validation covers the same three regimes and is judged against the
gates frozen in ``protocol_v7.json`` before the first v7 run. The held-out test
splits are never read here.
"""
import argparse
import csv
import itertools
import json
import math
import os
from collections import Counter, OrderedDict
from pathlib import Path
import sys
import time

import geopandas as gpd
import numpy as np
import shapely
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz
from pointcloud.vote_field import (ASSIGN, CHANNEL_NAMES, GRID, VoteFieldNet, detect_field, field_labels, field_loss,
                                   field_targets, rasterize_field, select_field, stage1_centres, tree_vote_centres)
from scripts.cache_stage1_ensemble import helios_rows, weiser_rows
from scripts.cache_stage1_predictions import DEFAULT_OUTPUT, eligible_rows, sha256, write_json
from scripts.cache_stage1_thinned import subset
from scripts.train_two_pass_refiner import balanced_weights, evaluate_plot, summarize

VALIDATION_SETS = ('dense', 'thin10', 'thin25', 'helios', 'weiser')
GATED_SETS = ('dense', 'thin10', 'helios')
KEYS = ('point_sb_pq', 'crown_sb_pq', 'point_precision', 'crown_precision', 'point_recall', 'crown_recall',
        'small_10_tp', 'small_10_gt', 'small_4_tp', 'large_recall', 'oversplit_gt', 'undersegmented_pred',
        'predicted_instances')
REPORT_KEYS = KEYS + ('annotated_point_sb_pq', 'annotated_point_precision', 'fp_unannotated', 'fp_inside_one_tree')


def load_variant(row, path):
    """Arrays of one plot variant plus its cached stage-1 labels and ensembled votes."""
    arrays = load_npz(row['output'])
    with np.load(path) as archive:
        if 'keep' in archive.files:
            arrays = subset(arrays, archive['keep'])
        cache = dict(votes=archive['ensemble_center'].astype(np.float32),
                     probability=archive['ensemble_probability'].astype(np.float32),
                     spread=archive['vote_spread'].astype(np.float32), labels=archive['labels'].astype(np.int64),
                     confidence=archive['confidence'].astype(np.float32), missed=archive['missed_gt_ids'])
    return arrays, cache


def validation_sets(root):
    """name -> list of (row, cache path); thin10 is the draw used since v6, thin25 is reported only."""
    base = root / 'stage1_ensemble'
    real = eligible_rows('val')
    return dict(dense=[(r, base / 'dense/val' / f"{r['dataset_id']}.npz") for r in real],
                thin10=[(r, base / 'thinned/val' / f"{r['dataset_id']}__v0.npz") for r in real],
                thin25=[(r, base / 'thinned/val' / f"{r['dataset_id']}__e0.npz") for r in real],
                helios=[(r, base / 'helios/val' / f"{r['dataset_id']}.npz") for r in helios_rows('val')],
                weiser=[(r, base / 'weiser/val' / f"{r['dataset_id']}.npz") for r in weiser_rows('val')
                        if (base / 'weiser/val' / f"{r['dataset_id']}.npz").exists()])


def load_context(items):
    context = []
    for row, path in items:
        arrays, cache = load_variant(row, path)
        gt = gpd.read_file(row['gt_vector'])
        ignore = shapely.from_wkb(Path(row['ignore_vector']).read_bytes()) if row.get('ignore_vector') else None
        ids, centres, extents, counts = tree_vote_centres(arrays['coord'], arrays['tree_id'], cache['votes'])
        cache.update(centres=centres, extents=extents, tree_ids=ids, anchors=stage1_centres(cache['votes'], cache['labels']))
        context.append((row, arrays, cache, gt, ignore))
    return context


def stage1_summary(context):
    records = [evaluate_field(row, arrays, gt, ignore, cache['labels'].astype(np.uint32), cache['confidence'])
               for row, arrays, cache, gt, ignore in context]
    return summarize_field(records), records


def annotated_point_metrics(truth, labels, minimum_known=.5):
    """Point metrics that do not charge predictions lying mostly on unannotated voxels.

    Half of a typical real plot carries ``tree_id == -1``. The protocol metric
    counts every unmatched predicted instance with at least one annotated voxel as
    a false positive, which punishes a model for finding trees nobody labelled.
    Here an unmatched prediction is a false positive only if at least
    ``minimum_known`` of its voxels are annotated. Matches are unchanged.
    """
    from scipy.optimize import linear_sum_assignment
    truth, labels = np.asarray(truth), np.asarray(labels).astype(np.int64)
    known = truth >= 0
    total = np.bincount(labels, minlength=int(labels.max(initial=0)) + 1)
    inside = np.bincount(labels[known], minlength=len(total))
    gt_ids, gt_inverse, gt_count = np.unique(truth[known], return_inverse=True, return_counts=True)
    pr_ids, pr_inverse, pr_count = np.unique(labels[known], return_inverse=True, return_counts=True)
    table = np.bincount(gt_inverse * len(pr_ids) + pr_inverse, minlength=len(gt_ids) * len(pr_ids)).reshape(len(gt_ids), len(pr_ids))
    gt_keep, pr_keep = gt_ids > 0, pr_ids > 0
    intersection = table[gt_keep][:, pr_keep]
    iou = intersection / np.maximum(gt_count[gt_keep, None] + pr_count[None, pr_keep] - intersection, 1)
    a, b = linear_sum_assignment(-((iou >= .5) * (min(iou.shape, default=0) + 1 + iou)))
    good = iou[a, b] >= .5
    matched = np.zeros(int(pr_keep.sum()), bool)
    matched[b[good]] = True
    share = inside[pr_ids[pr_keep]] / np.maximum(total[pr_ids[pr_keep]], 1)
    duplicate = (intersection.max(0) / np.maximum(pr_count[pr_keep], 1) >= .5) if intersection.shape[0] else np.zeros(len(matched), bool)
    charged = ~matched & (share >= minimum_known)
    tp = int(good.sum())
    return dict(tp=tp, fp=int(charged.sum()), fn=int(gt_keep.sum()) - tp, iou_sum=float(iou[a, b][good].sum()),
                fp_unannotated=int((~matched & (share < minimum_known)).sum()),
                fp_inside_one_tree=int((charged & duplicate).sum()))


def summarize_field(records):
    """Protocol summary plus the annotated-area point metrics."""
    from scripts.evaluate_combined_full_crowns import aggregate
    summary = summarize(records)
    names = ('dataset_id', 'source_dataset', 'collection')
    annotated = aggregate([{**{k: r[k] for k in names}, **r['annotated']} for r in records])
    summary.update(annotated_point_sb_pq=annotated['source_balanced_pq'], annotated_point_sb_f1=annotated['source_balanced_f1'],
                   annotated_point_precision=annotated['precision'],
                   fp_unannotated=int(sum(r['annotated']['fp_unannotated'] for r in records)),
                   fp_inside_one_tree=int(sum(r['annotated']['fp_inside_one_tree'] for r in records)),
                   crown_ignored=int(sum(r.get('ignored_predictions', 0) for r in records)))
    return summary


def evaluate_field(row, arrays, gt, ignore, labels, confidence):
    record = evaluate_plot(row, arrays, gt, ignore, labels, confidence)
    record['annotated'] = annotated_point_metrics(arrays['tree_id'], labels)
    return record


def gate_checks(summary, base, gate):
    checks = dict(point_sb_pq=summary['point_sb_pq'] >= base['point_sb_pq'] + gate['point_sb_pq_margin'],
                  crown_sb_pq=summary['crown_sb_pq'] >= base['crown_sb_pq'] + gate['crown_sb_pq_margin'],
                  point_precision=summary['point_precision'] >= base['point_precision'] - gate['precision_tolerance'],
                  crown_precision=summary['crown_precision'] >= base['crown_precision'] - gate['precision_tolerance'],
                  large_recall=summary['large_recall'] >= base['large_recall'] - gate['large_recall_tolerance'])
    if gate.get('small_10_minimum_tp') is not None:
        checks['small_crowns'] = summary['small_10_tp'] >= gate['small_10_minimum_tp']
    return checks


def quality(summary):
    return math.sqrt(max(summary['point_sb_pq'] * summary['crown_sb_pq'], 0.))


ASSIGNMENTS = {'height': dict(ASSIGN), 'extent': {**ASSIGN, 'extent_scale': 1.25},
               'extent1.5': {**ASSIGN, 'extent_scale': 1.5},
               'extent_min2': {**ASSIGN, 'extent_scale': 1.25, 'extent_min': 2.},
               'extent1.5_norm': {**ASSIGN, 'extent_scale': 1.5, 'normalised': True},
               # tall stands (Weiser): a 45 m Douglas fir's votes reach 6-8 m, the 4 m cap drops it
               'extent1.5_norm_max8': {**ASSIGN, 'extent_scale': 1.5, 'normalised': True, 'extent_max': 8., 'radius_max': 5.}}


def sweep(full=False):
    """Selection x assignment configurations. The short list runs during training."""
    configs = {}

    def add(name, selection, assign):
        configs[name] = dict(selection=selection, assign=assign)
    if not full:
        for threshold in (.2, .3):
            for tag in ('height', 'extent'):
                add(f'replace_t{threshold:g}_{tag}', dict(mode='replace', threshold=threshold), ASSIGNMENTS[tag])
        for near, far in ((.2, .4), (.3, .55)):
            for tag in ('height', 'extent'):
                add(f'corr_r{near:g}_a{far:g}_k0_{tag}',
                    dict(mode='corrective', threshold=near, add_threshold=far, anchor_keep=0.), ASSIGNMENTS[tag])
        return configs
    for tag, assign in ASSIGNMENTS.items():
        for threshold, factor in itertools.product((.3, .4, .5, .6), (0., 1., 1.5)):
            add(f'replace_t{threshold:g}_nms{factor:g}_{tag}', dict(mode='replace', threshold=threshold, suppress=factor), assign)
        for (near, far), factor in itertools.product(((.3, .55), (.4, .55), (.4, .7), (.5, .7), (.5, .8)), (0., 1., 1.5)):
            add(f'corr_r{near:g}_a{far:g}_nms{factor:g}_{tag}',
                dict(mode='corrective', threshold=near, add_threshold=far, anchor_keep=0., suppress=factor), assign)
    for near, far in ((.3, .55), (.4, .55)):      # let stage 2 reject unsupported stage-1 centres
        for keep in (.05, .1):
            add(f'corr_r{near:g}_a{far:g}_nms1_keep{keep:g}_extent',
                dict(mode='corrective', threshold=near, add_threshold=far, anchor_keep=keep, suppress=1.), ASSIGNMENTS['extent'])
    return configs


def centre_statistics(detection, cache, arrays, thresholds=(.1, .2, .3, .4, .5)):
    """Centre recall, and precision restricted to annotated area, per score threshold."""
    truth = cache['centres']
    result = Counter(truth=len(truth))
    votes = cKDTree(cache['votes'][:, :2])
    known = arrays['tree_id'] >= 0
    tree = cKDTree(truth[:, :2]) if len(truth) else None
    order = np.argsort(-detection['scores'])
    centres, scores = detection['centres'][order], detection['scores'][order]
    annotated = np.ones(len(centres), bool)
    for i, centre in enumerate(centres):
        voters = votes.query_ball_point(centre[:2], 1.)
        annotated[i] = bool(voters) and known[voters].mean() >= .5
    for threshold in thresholds:
        used = set()
        hits = 0
        keep = scores >= threshold
        for centre in centres[keep]:
            for j in (tree.query_ball_point(centre[:2], 1.) if tree is not None else ()):
                if j not in used and abs(truth[j, 2] - centre[2]) <= 3.:
                    used.add(j)
                    hits += 1
                    break
        result[f'hits_{threshold:g}'] += hits
        result[f'predicted_{threshold:g}'] += int((keep & annotated).sum())
    return result


def detect_context(network, context, device, flips=True):
    """Run the detector once per plot; returns detections, centre statistics and seconds."""
    detections, totals, seconds = [], Counter(), 0.
    for row, arrays, cache, gt, ignore in context:
        start = time.monotonic()
        detection = detect_field(network, arrays, cache['votes'], cache['probability'], cache['labels'], cache['spread'],
                                 threshold=.1, device=device, flips=flips, queries=cache['anchors'])
        seconds += time.monotonic() - start
        totals.update(centre_statistics(detection, cache, arrays))
        if len(detection['centres']) and len(cache['centres']):       # extent error at matched centres
            distance, nearest = cKDTree(cache['centres']).query(detection['centres'])
            close = (distance <= 1.) & (detection['scores'] >= .3)
            totals['extent_abs_log_error'] += float(np.abs(np.log(detection['extents'][close] / cache['extents'][nearest[close]])).sum())
            totals['extent_log_bias'] += float(np.log(detection['extents'][close] / cache['extents'][nearest[close]]).sum())
            totals['extent_matched'] += int(close.sum())
        detections.append(detection)
    return detections, dict(totals), seconds


def evaluate_configs(item, detection, configs):
    """All configurations for one plot: list of (record, selection info)."""
    row, arrays, cache, gt, ignore = item
    results = []
    for config in configs.values():
        centres, extents, kinds, info = select_field(detection, cache['anchors'], config['selection'])
        labels, confidence, _ = field_labels(arrays, cache['votes'], cache['probability'], centres, extents, kinds,
                                             config['assign'])
        results.append((evaluate_field(row, arrays, gt, ignore, labels, confidence), info))
    return results


_SHARED = {}


def _evaluate_shared(index):
    return evaluate_configs(_SHARED['context'][index], _SHARED['detections'][index], _SHARED['configs'])


def validate(network, context, configs, device, flips=True, jobs=1):
    detections, detection_totals, detect_seconds = detect_context(network, context, device, flips)
    start = time.monotonic()
    if jobs > 1:
        import multiprocessing
        _SHARED.update(context=context, detections=detections, configs=configs)
        with multiprocessing.get_context('fork').Pool(min(jobs, len(context))) as pool:
            per_plot = pool.map(_evaluate_shared, range(len(context)))
        _SHARED.clear()
    else:
        per_plot = [evaluate_configs(item, detection, configs) for item, detection in zip(context, detections)]
    records = {name: [plot[i][0] for plot in per_plot] for i, name in enumerate(configs)}
    operations = {name: Counter() for name in configs}
    for plot in per_plot:
        for name, (_, info) in zip(configs, plot):
            operations[name].update(info)
    summaries = {name: {**summarize_field(value), 'operations': dict(operations[name])} for name, value in records.items()}
    return summaries, records, dict(detect=detect_seconds, assign=time.monotonic() - start), detection_totals


class FieldCrops(Dataset):
    """Random rotated windows from dense, thinned and HELIOS plot variants."""
    def __init__(self, sources, seed, augment=True, cache_size=400, adaptive_sigma=False):
        self.sources, self.seed, self.augment, self.cache_size = sources, seed, augment, cache_size
        self.adaptive_sigma = adaptive_sigma
        self._plots = OrderedDict()

    def __len__(self):
        return sum(len(v) for v in self.sources.values())

    def plot(self, regime, index, file):
        key = (regime, index, str(file))
        if key in self._plots:
            self._plots.move_to_end(key)
            return self._plots[key]
        arrays, cache = load_variant(self.sources[regime][index]['row'], file)
        ids, centres, extents, _ = tree_vote_centres(arrays['coord'], arrays['tree_id'], cache['votes'])
        item = dict(coord=arrays['coord'], tree_id=arrays['tree_id'].astype(np.int32), votes=cache['votes'],
                    probability=cache['probability'], spread=cache['spread'], labels=cache['labels'].astype(np.int32),
                    centres=centres, extents=extents, missed=np.flatnonzero(np.isin(ids, cache['missed'])))
        self._plots[key] = item
        if len(self._plots) > self.cache_size:
            self._plots.popitem(last=False)
        return item

    def __getitem__(self, index):
        regime, row_index, draw, epoch = index
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, draw, row_index]))
        files = self.sources[regime][row_index]['files']
        plot = self.plot(regime, row_index, files[rng.integers(len(files))])
        coord, centres = plot['coord'], plot['centres']
        choice = rng.random()
        if choice < .5 and len(plot['missed']):
            anchor = centres[rng.choice(plot['missed']), :2]
        elif choice < .8 and len(centres):
            anchor = centres[rng.integers(len(centres)), :2]
        else:
            anchor = coord[rng.integers(len(coord)), :2]
        half, margin = GRID['size'] / 2., GRID['margin']
        centre = anchor + rng.uniform(-half / 2, half / 2, 2)
        reach = (half + margin) * (np.sqrt(2.) if self.augment else 1.)
        members = np.flatnonzero((np.abs(coord[:, 0] - centre[0]) <= reach) & (np.abs(coord[:, 1] - centre[1]) <= reach))
        xyz = coord[members].copy()
        votes = plot['votes'][members].copy()
        targets = centres.copy()
        if self.augment:
            angle = rng.uniform(-np.pi, np.pi)
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.asarray([[c, -s], [s, c]], np.float32)
            if rng.random() < .5:
                rotation = rotation @ np.asarray([[-1., 0.], [0., 1.]], np.float32)
            for values in (xyz, votes, targets):
                values[:, :2] = (values[:, :2] - centre) @ rotation.T + centre
            votes[:, :2] += rng.normal(0., .05, (len(votes), 2)).astype(np.float32)
            votes[:, 2] += rng.normal(0., .1, len(votes)).astype(np.float32)
            keep = rng.random(len(members)) < rng.uniform(.7, 1.)
            members, xyz, votes = members[keep], xyz[keep], votes[keep]
        origin = centre - half
        volume, ignore = rasterize_field(xyz, votes, plot['probability'][members], plot['labels'][members],
                                         plot['spread'][members], origin, known=plot['tree_id'][members] >= 0)
        heat, offset, extent, positive = field_targets(targets, plot['extents'], origin, adaptive=self.adaptive_sigma)
        return tuple(torch.from_numpy(v) for v in (volume, heat, ignore, offset, extent, positive))


class EpochSampler:
    """Holds the draws of the current epoch so one persistent DataLoader serves all epochs."""
    def __init__(self):
        self.draws = []

    def __iter__(self):
        return iter(self.draws)

    def __len__(self):
        return len(self.draws)


def training_sources(root):
    index = json.loads((root / 'stage1_ensemble/index.json').read_text())['index']
    base = root / 'stage1_ensemble'
    real = eligible_rows('train')
    helios = helios_rows('train')
    return dict(
        dense=[dict(row=r, files=[base / 'dense/train' / f"{r['dataset_id']}.npz"]) for r in real],
        thinned=[dict(row=r, files=[base / 'thinned/train' / e['file'] for e in index['thinned']['train'][r['dataset_id']]])
                 for r in real],
        helios=[dict(row=r, files=[base / 'helios/train' / f"{r['dataset_id']}.npz"]) for r in helios],
        weiser=[dict(row=r, files=[base / 'weiser/train' / f"{r['dataset_id']}.npz"]) for r in weiser_rows('train')
                if (base / 'weiser/train' / f"{r['dataset_id']}.npz").exists()]), real


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run', default='vote_field_v7')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--val-every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--mix', type=float, nargs='+', default=(.4, .4, .2),
                        help='dense, thinned, HELIOS shares; a fourth value adds the Weiser regime')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--adaptive-sigma', action='store_true',
                        help='v8: heat-target width proportional to the reach of the tree\'s votes')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is required')
    device = torch.device('cuda:0')
    root = args.output.resolve()
    protocol = json.loads((root / 'protocol_v7.json').read_text())
    frozen = json.loads((root / 'protocol.json').read_text())
    if sha256(frozen['stage1_checkpoint']) != frozen['stage1_sha256'] or sha256(root / 'protocol.json') != protocol['protocol_v1_sha256']:
        raise ValueError('Stage-1 checkpoint or protocol changed since protocol_v7 was frozen')
    run = root / 'runs' / (args.run + ('_smoke' if args.smoke else ''))
    if (run / 'DONE.json').exists():
        print((run / 'selected.json').read_text())
        return
    if run.exists() and not args.smoke:
        raise FileExistsError(f'Incomplete run protected: {run}')
    (run / 'weights').mkdir(parents=True, exist_ok=True)
    configs = sweep()
    if args.smoke:
        configs = dict(list(configs.items())[:2])
        args.epochs, args.samples, args.val_every = 1, 32, 1
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    network = VoteFieldNet(width=args.width, dropout=args.dropout).to(device)
    parameters = list(network.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * (args.samples // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=max(steps, 1), pct_start=.05)
    sources, real_rows = training_sources(root)
    configuration = dict(run=run.name, seed=args.seed, width=args.width, dropout=args.dropout, weight_decay=args.weight_decay,
                         grid=GRID, channels=CHANNEL_NAMES, trainable_parameters=sum(p.numel() for p in parameters),
                         architecture='frozen Model20, votes averaged over 4 lattice rotations -> 80x80x40 vote volume '
                                      '(0.25 m, 14 channels) -> 3-D U-Net, three down levels -> heads at 0.5 m: centre '
                                      'heat, sub-cell offset, vote extent -> NMS peaks -> selection against stage-1 '
                                      'centres -> assignment within each centre\'s reach',
                         targets='per-tree median vote (centre), 90th-percentile vote distance (extent)',
                         heat_sigma='0.6 x vote extent, clipped to 1-2.5 cells' if args.adaptive_sigma else '1.2 cells xy, 1 cell z',
                         loss='penalty-reduced focal + L1 offset + 0.5 L1 log-extent; unannotated cells ignored',
                         mix=dict(zip(('dense', 'thinned', 'helios', 'weiser'), args.mix)),
                         variants={k: sum(len(s['files']) for s in v) for k, v in sources.items()},
                         sampling='real plots source-balanced; anchors 50% stage-1-missed tree, 30% any tree, 20% any point',
                         augmentation=dict(rotation=True, flip=True, vote_jitter_m=(.05, .1), voter_keep=(.7, 1.)),
                         epochs=args.epochs, samples_per_epoch=args.samples, batch_size=args.batch_size, lr=args.lr,
                         protocol_v7_sha256=sha256(root / 'protocol_v7.json'), sweep=configs,
                         test_used_for_selection=False)
    write_json(run / 'configuration.json', configuration)
    sets = {k: v for k, v in validation_sets(root).items() if v}  # weiser only when cached
    if args.smoke:
        sets = {k: v[:1] for k, v in sets.items()}
    contexts = {name: load_context(items) for name, items in sets.items()}
    bases = {}
    for name, context in contexts.items():
        bases[name], records = stage1_summary(context)
        write_json(run / f'validation/stage1_only_{name}.json', dict(summary=bases[name], per_plot=records))
        if not args.smoke:
            for key in ('point_sb_pq', 'crown_sb_pq'):
                if abs(bases[name][key] - protocol['baselines'][name][key]) > 1e-6:
                    raise AssertionError(f'Frozen Model20 baseline not reproduced on {name}: {key}')
    dataset = FieldCrops(sources, args.seed, adaptive_sigma=args.adaptive_sigma)
    sampler = EpochSampler()
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True,
                        drop_last=True, persistent_workers=True, prefetch_factor=4)
    real_weights = balanced_weights(real_rows)
    real_weights = real_weights / real_weights.sum()
    regimes = ('dense', 'thinned', 'helios', 'weiser')[:len(args.mix)]
    if 'weiser' in regimes and not sources['weiser']:
        raise ValueError('No cached Weiser training variants (cache_stage1_ensemble.py --regimes weiser)')
    history, validations = [], []
    selected = None
    started = time.monotonic()
    scaler = torch.amp.GradScaler('cuda')

    def save(path, epoch, extra=None):
        temporary = path.with_suffix('.tmp.pt')
        torch.save(dict(network=network.state_dict(), width=args.width, dropout=args.dropout, grid=GRID, epoch=epoch,
                        model='VoteFieldNet', stage1_checkpoint=frozen['stage1_checkpoint'],
                        stage1_sha256=frozen['stage1_sha256'], stage1_config=frozen['stage1_config'],
                        symmetries=protocol['ensemble_symmetries'], configuration=configuration, **(extra or {})), temporary)
        os.replace(temporary, path)

    for epoch in range(1, args.epochs + 1):
        network.train()
        rng = np.random.default_rng(args.seed + epoch * 100003)
        draws = []
        for step in range(args.samples):
            regime = regimes[rng.choice(len(regimes), p=np.asarray(args.mix) / sum(args.mix))]
            row = (rng.integers(len(sources[regime])) if regime in ('helios', 'weiser')
                   else rng.choice(len(real_rows), p=real_weights))
            draws.append((regime, int(row), step, epoch))
        sampler.draws = draws
        totals, batches, start = Counter(), 0, time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for batch in loader:
            volume, heat, ignore, offset, extent, positive = (v.to(device, non_blocking=True) for v in batch)
            with torch.autocast('cuda', dtype=torch.float16):
                output = network(volume)
            losses = field_loss(output, heat, ignore, offset, extent, positive)
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError('Non-finite loss')
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses['loss']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 5.)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            totals.update({k: float(v.detach()) for k, v in losses.items()})
            batches += 1
        record = dict(epoch=epoch, seconds=time.monotonic() - start,
                      **{f'train_{k}': v / max(batches, 1) for k, v in totals.items()},
                      peak_vram_mb=torch.cuda.max_memory_allocated() / 1024 ** 2)
        if epoch % args.val_every == 0 or epoch == args.epochs:
            if not args.smoke:
                save(run / f'weights/epoch_{epoch:03d}.pt', epoch)
            results = {}
            for name, context in contexts.items():
                summaries, records, seconds, detection = validate(network, context, configs, device)
                write_json(run / f'validation/epoch_{epoch:03d}/{name}.json',
                           dict(summaries=summaries, per_plot=records, seconds=seconds, detection=detection,
                                stage1_only=bases[name]))
                results[name] = summaries
                truth = max(detection.get('truth', 0), 1)
                record.update({f'{name}_centre_recall_0p3': detection.get('hits_0.3', 0) / truth,
                               f'{name}_centre_precision_0p3': detection.get('hits_0.3', 0) / max(detection.get('predicted_0.3', 0), 1),
                               f'{name}_extent_log_error': detection.get('extent_abs_log_error', 0.) / max(detection.get('extent_matched', 0), 1),
                               f'{name}_detect_seconds': seconds.get('detect', 0.)})
            checks = {c: {s: gate_checks(results[s][c], bases[s], protocol['gate'][s]) for s in GATED_SETS} for c in configs}
            passed = {c: all(all(v.values()) for v in checks[c].values()) for c in configs}

            def gain(c):
                sets_ = ('dense', 'thin10') + (('weiser',) if 'weiser' in results and 'weiser' in regimes else ())
                return sum(quality(results[s][c]) - quality(bases[s]) for s in sets_)
            best = max(configs, key=lambda c: (passed[c], gain(c)))
            for c in configs:
                validations.append(dict(epoch=epoch, config=c, gate_passed=passed[c], gain=gain(c),
                                        **{f'{s}_{k}': results[s][c][k] for s in VALIDATION_SETS if s in results for k in KEYS}))
            record.update(val_config=best, val_gate_passed=passed[best], val_gain=gain(best),
                          **{f'{s}_{k}': results[s][best][k] for s in VALIDATION_SETS if s in results
                             for k in ('point_sb_pq', 'crown_sb_pq', 'small_10_tp', 'large_recall')})
            if passed[best] and not args.smoke and (selected is None or gain(best) > selected['gain']):
                selected = dict(epoch=epoch, config_name=best, config=configs[best], gain=gain(best),
                                validation={s: results[s][best] for s in VALIDATION_SETS if s in results}, checks=checks[best])
                save(run / 'weights/best.pt', epoch, dict(selected_config=configs[best], selected_config_name=best))
        save(run / 'weights/last.pt', epoch)
        history.append(record)
        fields = list(dict.fromkeys(k for r in history for k in r))
        with (run / 'training_log.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
        write_json(run / 'selected.json', dict(selected=selected, stage1_only=bases, gate=protocol['gate'],
                                               test_used_for_selection=False))
        write_workbook(run, history, validations, selected, bases)
        print(json.dumps(record), flush=True)
    if not args.smoke:
        write_json(run / 'DONE.json', dict(epochs=args.epochs, seconds=time.monotonic() - started,
                                           selected_epoch=selected and selected['epoch']))
    print(json.dumps(dict(selected=selected and {k: selected[k] for k in ('epoch', 'config_name', 'gain')}), indent=2), flush=True)


def write_workbook(run, history, validations, selected, bases):
    from openpyxl import Workbook
    book = Workbook()
    sheet = book.active
    sheet.title = 'epochs'
    for title, rows in (('epochs', history), ('validation', validations)):
        tab = sheet if title == 'epochs' else book.create_sheet(title)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        if fields:
            tab.append(fields)
            for r in rows:
                tab.append([r.get(k) for k in fields])
            tab.freeze_panes = 'A2'
    tab = book.create_sheet('model20')
    tab.append(('set', *KEYS))
    for name, base in bases.items():
        tab.append((name, *[base[k] for k in KEYS]))
    tab = book.create_sheet('selection')
    tab.append(('selected', json.dumps(selected, default=str)))
    temporary = run / 'experiments.tmp.xlsx'
    book.save(temporary)
    os.replace(temporary, run / 'experiments.xlsx')


if __name__ == '__main__':
    main()
