#!/usr/bin/env python3
"""Train the vote-space centre detector (second pass v4) on cached Model20 predictions.

Inputs are rasterised frozen Model20 votes plus stage-1 context; targets are
ground-truth tree centroids. Selection uses all native validation plots and the
gate preregistered in protocol.json. The held-out test split is never read.
"""
import argparse
import csv
import itertools
import json
import os
from collections import Counter
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
from pointcloud.vote_centers import (ASSIGN, CHANNEL_NAMES, GRID, VoteCenterNet, centre_focal_loss, centre_heatmap,
                                     detect_centres, merge_with_stage1, rasterize_votes, second_pass_labels,
                                     select_centres, stage1_centres, tree_centroids)
from scripts.cache_stage1_predictions import DEFAULT_OUTPUT, eligible_rows, sha256, write_json
from scripts.train_two_pass_refiner import (balanced_weights, evaluate_plot, gate_checks, rank, stage1_baseline,
                                            summarize, write_excel)

RAW_KEYS = ('shifted_center', 'tree_probability', 'point_probability', 'object_score', 'candidate_offset',
            'point_index', 'point_score')


class VoteCrops(Dataset):
    """Random rotated windows of train plots: vote volume, centre heatmap, ignore mask."""
    def __init__(self, rows, cache, seed, augment=True, thin_cache=None, thin_index=None, thin_probability=0.):
        if any(r['model_split'] != 'train' for r in rows):
            raise ValueError('VoteCrops accepts train rows only')
        self.rows, self.cache, self.seed, self.augment, self.epoch = rows, Path(cache), seed, augment, 0
        self.thin_cache = Path(thin_cache) if thin_cache else None
        self.thin_index, self.thin_probability = thin_index or {}, thin_probability
        self._plots = {}

    def __len__(self):
        return len(self.rows)

    def plot(self, row_index, variant=None):
        """Dense plot (variant None) or one of its density-thinned Model20 predictions."""
        key = (row_index, variant)
        if key not in self._plots:
            row = self.rows[row_index]
            arrays = load_npz(row['output'])
            coord, tree_id = arrays['coord'], arrays['tree_id']
            path = self.cache / f"{row['dataset_id']}.npz" if variant is None else self.thin_cache / variant
            with np.load(path) as archive:
                if variant is not None:
                    keep = archive['keep']
                    coord, tree_id = coord[keep], tree_id[keep]
                item = dict(coord=coord, tree_id=tree_id, votes=archive['shifted_center'].astype(np.float32),
                            probability=archive['tree_probability'].astype(np.float32),
                            labels=archive['labels'].astype(np.int64), missed=archive['missed_gt_ids'])
            ids, centres, _ = tree_centroids(item['coord'], item['tree_id'])
            item.update(centre_ids=ids, centres=centres)
            self._plots[key] = item
        return self._plots[key]

    def __getitem__(self, index):
        row_index, draw = index
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, draw, row_index]))
        variants = self.thin_index.get(self.rows[row_index]['dataset_id'], [])
        variant = None
        if variants and rng.random() < self.thin_probability:
            variant = variants[rng.integers(len(variants))]['file']
        plot = self.plot(row_index, variant)
        coord, centres = plot['coord'], plot['centres']
        choice = rng.random()
        missed = np.flatnonzero(np.isin(plot['centre_ids'], plot['missed']))
        if choice < .5 and len(missed):
            anchor = centres[rng.choice(missed), :2]
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
            votes[:, :2] += rng.normal(0., .1, (len(votes), 2)).astype(np.float32)
            votes[:, 2] += rng.normal(0., .2, len(votes)).astype(np.float32)
            keep = rng.random(len(members)) < rng.uniform(.6, 1.)
            members, xyz, votes = members[keep], xyz[keep], votes[keep]
        origin = centre - half
        volume, ignore = rasterize_votes(xyz, votes, plot['probability'][members], plot['labels'][members], origin,
                                         known=plot['tree_id'][members] >= 0)
        heat, _ = centre_heatmap(targets, origin)
        return torch.from_numpy(volume), torch.from_numpy(heat), torch.from_numpy(ignore)


def sweep():
    """Replacement (all centres from stage 2) and corrective (stage-1 centres kept where stage 2 is silent)."""
    configs = {}
    for threshold, radius in itertools.product((.3, .4, .5), (2., 3.)):
        configs[f't{threshold:g}_r{radius:g}'] = dict(threshold=threshold, consensus=False, fallback=False,
                                                    assign={**ASSIGN, 'radius': radius})
    for threshold, radius in itertools.product((.3, .4, .5, .6, .7), (2., 3.)):
        configs[f'keep_t{threshold:g}_r{radius:g}'] = dict(threshold=threshold, consensus=False, fallback=True,
                                                         assign={**ASSIGN, 'radius': radius})
    for replace, add, radius in itertools.product((.3, .4, .5), (.6, .7, .8), (2., 3.)):
        configs[f'asym_r{replace:g}_a{add:g}_r{radius:g}'] = dict(threshold=replace, add_threshold=add, consensus=False,
                                                                 fallback=True, assign={**ASSIGN, 'radius': radius})
    adaptive = {**ASSIGN, 'radius': 3., 'radius_min': 2., 'radius_base': 1.5, 'radius_per_metre': .1}
    for replace, add in itertools.product((.3, .4, .5), (.55, .6, .65, .7, .8)):
        configs[f'asym_r{replace:g}_a{add:g}_adaptive'] = dict(threshold=replace, add_threshold=add, consensus=False,
                                                              fallback=True, assign=adaptive)
    for threshold in (.4, .5):
        configs[f'keep_t{threshold:g}_adaptive'] = dict(threshold=threshold, consensus=False, fallback=True, assign=adaptive)
    for threshold in (.4, .5, .6):
        configs[f'keep_t{threshold:g}_r3_consensus'] = dict(threshold=threshold, consensus=True, fallback=True,
                                                          assign={**ASSIGN, 'radius': 3.})
        configs[f'keep_t{threshold:g}_r3_p0.3'] = dict(threshold=threshold, consensus=False, fallback=True,
                                                     assign={**ASSIGN, 'radius': 3., 'probability': .3})
    return configs


def choose_centres(centres, scores, anchors, config):
    """Apply one sweep configuration to raw detections; returns the final centre set."""
    if 'add_threshold' in config:
        return select_centres(centres, scores, anchors, config['threshold'], config['add_threshold'])[0]
    chosen = centres[scores >= config['threshold']]
    if config.get('fallback'):
        chosen, _ = merge_with_stage1(chosen, anchors)
    return chosen


def centre_match(predicted, truth, xy=1., z=3.):
    """Greedy one-to-one matches of predicted to true centres within a cylinder."""
    if not len(predicted) or not len(truth):
        return 0
    tree = cKDTree(truth[:, :2])
    used, hits = set(), 0
    for centre in predicted:
        for j in tree.query_ball_point(centre[:2], xy):
            if j not in used and abs(truth[j, 2] - centre[2]) <= z:
                used.add(j)
                hits += 1
                break
    return hits


def validate(network, rows, root, protocol, configs, context, device, flips=True):
    records = {name: [] for name in configs}
    detection = Counter()
    seconds = Counter()
    for row in rows:
        arrays, stage1, gt, ignore = context[row['dataset_id']]
        if 'raw' not in stage1:
            with np.load(root / 'stage1/val' / f"{row['dataset_id']}.npz") as archive:
                stage1['raw'] = {k: archive[k] for k in RAW_KEYS}
            stage1['centroids'] = tree_centroids(arrays['coord'], arrays['tree_id'])[1]
            stage1['anchors'] = stage1_centres(stage1['raw']['shifted_center'], stage1['labels'])
        raw = stage1['raw']
        start = time.monotonic()
        centres, scores = detect_centres(network, arrays, raw['shifted_center'], raw['tree_probability'],
                                         stage1['labels'], threshold=.1, device=device, flips=flips)
        seconds['detect'] += time.monotonic() - start
        for threshold in (.3, .5):
            keep = centres[scores >= threshold]
            detection[f'hits_{threshold:g}'] += centre_match(keep[np.argsort(-scores[scores >= threshold])], stage1['centroids'])
            detection[f'predicted_{threshold:g}'] += len(keep)
        detection['truth'] += len(stage1['centroids'])
        for name, config in configs.items():
            start = time.monotonic()
            chosen = choose_centres(centres, scores, stage1['anchors'], config)
            labels, confidence, _, _ = second_pass_labels(arrays, raw, protocol['stage1_config'], chosen,
                                                          config['assign'], config['consensus'])
            seconds['assign'] += time.monotonic() - start
            record = evaluate_plot(row, arrays, gt, ignore, labels, confidence)
            record['operations'] = {}
            records[name].append(record)
    return {name: summarize(value) for name, value in records.items()}, records, dict(seconds), dict(detection)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run', default='vote_centres_v4')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--samples', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--val-every', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--thin-probability', type=float, default=0.,
                        help='share of training crops drawn from density-thinned Model20 predictions (stage1_thinned)')
    parser.add_argument('--only', default='', help='comma-separated substrings restricting the validation sweep')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is required')
    device = torch.device('cuda:0')
    root = args.output.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    if sha256(protocol['stage1_checkpoint']) != protocol['stage1_sha256'] or sha256(protocol['manifest']) != protocol['manifest_sha256']:
        raise ValueError('Stage-1 checkpoint or manifest changed since the protocol was frozen')
    train_rows, val_rows = eligible_rows('train'), eligible_rows('val')
    run = root / 'runs' / (args.run + ('_smoke' if args.smoke else ''))
    if (run / 'DONE.json').exists():
        print((run / 'selected.json').read_text())
        return
    if run.exists() and not args.smoke:
        raise FileExistsError(f'Incomplete run protected: {run}')
    (run / 'weights').mkdir(parents=True, exist_ok=True)
    configs = sweep()
    if args.only:
        configs = {k: v for k, v in configs.items() if any(part in k for part in args.only.split(','))}
    if args.smoke:
        val_rows, configs = val_rows[:1], dict(list(configs.items())[:2])
        args.epochs, args.samples, args.val_every = 1, 16, 1
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    network = VoteCenterNet(width=args.width, dropout=args.dropout).to(device)
    parameters = list(network.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * (args.samples // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=max(steps, 1), pct_start=.1)
    configuration = dict(run=run.name, seed=args.seed, width=args.width, dropout=args.dropout,
                         weight_decay=args.weight_decay, grid=GRID, channels=CHANNEL_NAMES,
                         trainable_parameters=sum(p.numel() for p in parameters),
                         architecture='frozen Model20 votes (3-D) + stage-1 context rasterised to 80x80x40 -> '
                                      '3-D U-Net centre heatmap -> NMS peaks -> nearest-centre vote assignment',
                         loss='CenterNet penalty-reduced focal loss; cells dominated by unannotated voters ignored',
                         sampling='source-balanced real train plots; anchors 50% stage-1-missed tree, 30% any tree, 20% any point',
                         augmentation=dict(rotation=True, flip=True, vote_jitter_m=(.1, .2), voter_keep=(.6, 1.)),
                         epochs=args.epochs, samples_per_epoch=args.samples, batch_size=args.batch_size, lr=args.lr,
                         train_plots=len(train_rows), val_plots=len(val_rows),
                         inference='heatmap averaged over four axis flips of each window',
                         protocol_sha256=sha256(root / 'protocol.json'), sweep=configs,
                         thin_probability=args.thin_probability,
                         selection='dense-validation gate first; among passing candidates the best at the thinned '
                                   '(reference-scene) density: small crowns, then geometric mean of point/crown SB-PQ',
                         stage1_training_predictions='in-sample Model20 votes; the true-centre oracle is not better on '
                                                     'train than on validation, so votes are not memorised',
                         test_used_for_selection=False)
    write_json(run / 'configuration.json', configuration)
    context = {}
    base, base_records = stage1_baseline(val_rows, root / 'stage1/val', context, protocol)
    write_json(run / 'validation/stage1_only.json', dict(summary=base, per_plot=base_records))
    thin_index, thin_context, thin_base = {}, {}, None
    if args.thin_probability > 0:
        thin_index = json.loads((root / 'stage1_thinned/index.json').read_text())['index']
        thin_records = []
        for row in val_rows:
            arrays, _, gt, ignore = context[row['dataset_id']]
            with np.load(root / 'stage1_thinned/val' / thin_index['val'][row['dataset_id']][0]['file']) as archive:
                keep = archive['keep']
                size = len(arrays['coord'])
                sparse = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == size else v)
                          for k, v in arrays.items()}
                raw = dict(shifted_center=archive['shifted_center'], tree_probability=archive['tree_probability'])
                stage1 = dict(labels=archive['labels'], raw=raw, centroids=tree_centroids(sparse['coord'], sparse['tree_id'])[1],
                              anchors=stage1_centres(raw['shifted_center'], archive['labels']))
                thin_records.append(evaluate_plot(row, sparse, gt, ignore, archive['labels'].astype(np.uint32),
                                                  archive['confidence']))
            thin_context[row['dataset_id']] = (sparse, stage1, gt, ignore)
        thin_base = summarize(thin_records)
        write_json(run / 'validation/stage1_only_thinned.json', dict(summary=thin_base, per_plot=thin_records))
    thin_configs = {k: v for k, v in configs.items() if not v['consensus']}
    dataset = VoteCrops(train_rows, root / 'stage1/train', args.seed, thin_cache=root / 'stage1_thinned/train',
                        thin_index=thin_index.get('train'), thin_probability=args.thin_probability)
    weights = balanced_weights(train_rows)
    history, validations = [], []
    selected = preserving = None
    started = time.monotonic()
    scaler = torch.amp.GradScaler('cuda')

    def save(path, epoch, extra=None):
        temporary = path.with_suffix('.tmp.pt')
        torch.save(dict(network=network.state_dict(), width=args.width, dropout=args.dropout, grid=GRID, epoch=epoch,
                        stage1_checkpoint=protocol['stage1_checkpoint'], stage1_sha256=protocol['stage1_sha256'],
                        stage1_config=protocol['stage1_config'], configuration=configuration, **(extra or {})), temporary)
        os.replace(temporary, path)

    for epoch in range(1, args.epochs + 1):
        network.train()
        dataset.epoch = epoch
        rng = np.random.default_rng(args.seed + epoch * 100003)
        draws = [(int(i), step) for step, i in enumerate(rng.choice(len(train_rows), args.samples, p=weights / weights.sum()))]
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=draws, num_workers=4, pin_memory=True, drop_last=True)
        total, batches, start = 0., 0, time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for volume, heat, ignore in loader:
            volume, heat, ignore = volume.to(device), heat.to(device), ignore.to(device)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = network(volume)
            loss = centre_focal_loss(logits, heat, ignore)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss')
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 5.)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += float(loss.detach())
            batches += 1
        record = dict(epoch=epoch, seconds=time.monotonic() - start, train_loss=total / max(batches, 1),
                      peak_vram_mb=torch.cuda.max_memory_allocated() / 1024 ** 2)
        if epoch % args.val_every == 0 or epoch == args.epochs:
            summaries, records, seconds, detection = validate(network, val_rows, root, protocol, configs, context, device)
            write_json(run / f'validation/epoch_{epoch:03d}/metrics.json',
                       dict(summaries=summaries, per_plot=records, seconds=seconds, detection=detection))
            if not args.smoke:
                save(run / f'weights/epoch_{epoch:03d}.pt', epoch)
            checks = {name: gate_checks(s, base, protocol['gate']) for name, s in summaries.items()}
            for name, summary in summaries.items():
                validations.append((epoch, name, all(checks[name].values()), summary))
            thin = {}
            if thin_context:
                thin, thin_records, _, thin_detection = validate(network, val_rows, root, protocol, thin_configs,
                                                                 thin_context, device)
                write_json(run / f'validation/epoch_{epoch:03d}/metrics_thinned.json',
                           dict(summaries=thin, per_plot=thin_records, detection=thin_detection, stage1_only=thin_base))

            def order(name):      # gate first (dense), then quality at the reference scene's density
                return (rank(thin[name]) if name in thin else (0, 0.), rank(summaries[name]))
            passing = [n for n in summaries if all(checks[n].values())]
            keeping = [n for n in summaries if all(v for k, v in checks[n].items() if k != 'small_crowns')]
            best_name = max(passing or keeping or summaries, key=order)
            best = summaries[best_name]
            record.update(val_config=best_name, val_gate_passed=bool(passing), val_point_sb_pq=best['point_sb_pq'],
                          val_crown_sb_pq=best['crown_sb_pq'], val_small_10_tp=best['small_10_tp'],
                          val_large_recall=best['large_recall'], val_predicted=best['predicted_instances'],
                          centre_recall_0p3=detection.get('hits_0.3', 0) / max(detection['truth'], 1),
                          centre_precision_0p3=detection.get('hits_0.3', 0) / max(detection.get('predicted_0.3', 0), 1),
                          val_detect_seconds=seconds.get('detect', 0.))
            if thin and best_name in thin:
                record.update(thin_point_sb_pq=thin[best_name]['point_sb_pq'], thin_crown_sb_pq=thin[best_name]['crown_sb_pq'],
                              thin_small_10_tp=thin[best_name]['small_10_tp'], thin_large_recall=thin[best_name]['large_recall'])
            if passing and not args.smoke:
                name = max(passing, key=order)
                if selected is None or order(name) > tuple(tuple(v) for v in selected['order']):
                    selected = dict(epoch=epoch, config_name=name, config=configs[name], validation=summaries[name],
                                    checks=checks[name], thinned_validation=thin.get(name), order=order(name))
                    save(run / 'weights/best.pt', epoch, dict(selected_config=configs[name], selected_config_name=name))
            if keeping and not args.smoke:
                name = max(keeping, key=lambda n: rank(summaries[n]))
                if (summaries[name]['small_10_tp'] > base['small_10_tp']
                        and (preserving is None or rank(summaries[name]) > rank(preserving['validation']))):
                    preserving = dict(epoch=epoch, config_name=name, config=configs[name],
                                      validation=summaries[name], checks=checks[name])
                    save(run / 'weights/best_quality_preserving.pt', epoch, dict(selected_config=configs[name]))
        save(run / 'weights/last.pt', epoch)
        history.append(record)
        with (run / 'training_log.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in history for k in r)))
            writer.writeheader()
            writer.writerows(history)
        selection = dict(selected=selected, best_quality_preserving=preserving, stage1_only=base,
                         stage1_only_thinned=thin_base,
                         gate=protocol['gate'], test_used_for_selection=False)
        write_json(run / 'selected.json', selection)
        write_excel(run, history, validations, selection)
        print(json.dumps(record), flush=True)
    if not args.smoke:
        write_json(run / 'DONE.json', dict(epochs=args.epochs, seconds=time.monotonic() - started,
                                           selected_epoch=selected and selected['epoch'],
                                           quality_preserving_epoch=preserving and preserving['epoch']))
    print(json.dumps(dict(selected=selected and {k: selected[k] for k in ('epoch', 'config_name')},
                          best_quality_preserving=preserving and {k: preserving[k] for k in ('epoch', 'config_name')}),
                     indent=2), flush=True)


if __name__ == '__main__':
    main()
