#!/usr/bin/env python3
"""Train the stage-3 assignment network (sparse 3-D U-Net choosing a candidate centre per voxel).

Candidates come from a trained vote-field detector (cached per plot variant by
this script or by train_assignment_gbm.py). Training mixes dense, thinned and
HELIOS variants; validation covers the four validation sets and is judged with
the frozen v7 gates. The held-out test splits are never read here.
"""
import argparse
import csv
import json
import os
from collections import Counter, OrderedDict
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.assign_net import (COLUMNS, AssignNet, assign_plot, choose, pool_mask, slot_tensor, unique_rows,
                                   window_coords)
from pointcloud.candidate_assignment import K, labels_from_choice, match_candidates, slot_inputs, slot_targets
from pointcloud.centre_verifier import FEATURE_NAMES
from pointcloud.vote_ensemble import SYMMETRIES
from pointcloud.vote_field import tree_vote_centres
from scripts.cache_stage1_predictions import DEFAULT_OUTPUT, sha256, write_json
from scripts.train_assignment_gbm import cache_candidates, load_candidates
from scripts.train_two_pass_refiner import balanced_weights
from scripts.train_vote_field import (EpochSampler, GATED_SETS, VALIDATION_SETS, evaluate_field, gate_checks, load_context,
                                      load_variant, quality, stage1_summary, summarize_field, training_sources,
                                      validation_sets)
from scripts.validate_vote_field import load_network

SIZE = 20.


def subset_arrays(arrays, members):
    return dict(coord=arrays['coord'][members], tree_id=arrays['tree_id'][members], grid_coord=arrays['grid_coord'][members],
                voxel_size=arrays['voxel_size'])


class AssignCrops(Dataset):
    """Random 20 m windows: slot inputs, targets and voxel coordinates."""
    def __init__(self, sources, store, seed, max_voxels=100000, candidate_drop=.1, pool_range=(.1, .1)):
        self.sources, self.store, self.seed = sources, Path(store), seed
        self.max_voxels, self.candidate_drop, self.pool_range = max_voxels, candidate_drop, pool_range
        self._plots = OrderedDict()

    def __len__(self):
        return sum(len(v) for v in self.sources.values())

    def plot(self, regime, index, file):
        key = (regime, index, str(file))
        if key in self._plots:
            self._plots.move_to_end(key)
            return self._plots[key]
        arrays, cache = load_variant(self.sources[regime][index]['row'], file)
        candidates, features, owner = load_candidates(self.store / regime / 'train' / Path(file).name)
        ids, centres, _, _ = tree_vote_centres(arrays['coord'], arrays['tree_id'], cache['votes'])
        item = dict(arrays=dict(coord=arrays['coord'], tree_id=arrays['tree_id'].astype(np.int32),
                                grid_coord=arrays['grid_coord'], voxel_size=arrays['voxel_size']),
                    votes=cache['votes'], probability=cache['probability'], spread=cache['spread'],
                    candidates=candidates, features=features, owner=owner, centres=centres, ids=ids,
                    missed=np.flatnonzero(np.isin(ids, cache['missed'])))
        self._plots[key] = item
        if len(self._plots) > 300:
            self._plots.popitem(last=False)
        return item

    def __getitem__(self, index):
        regime, row_index, draw, epoch = index
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, draw, row_index]))
        files = self.sources[regime][row_index]['files']
        plot = self.plot(regime, row_index, files[rng.integers(len(files))])
        coord, centres = plot['arrays']['coord'], plot['centres']
        choice = rng.random()
        if choice < .5 and len(plot['missed']):
            anchor = centres[rng.choice(plot['missed']), :2]
        elif choice < .8 and len(centres):
            anchor = centres[rng.integers(len(centres)), :2]
        else:
            anchor = coord[rng.integers(len(coord)), :2]
        centre = anchor + rng.uniform(-SIZE / 4, SIZE / 4, 2)
        members = np.flatnonzero((np.abs(coord[:, 0] - centre[0]) <= SIZE / 2) & (np.abs(coord[:, 1] - centre[1]) <= SIZE / 2))
        arrays = subset_arrays(plot['arrays'], members)
        candidates, features, owner = plot['candidates'], plot['features'], plot['owner']
        keep = pool_mask(candidates, rng.uniform(*self.pool_range))
        if self.candidate_drop > 0 and rng.random() < .5:                     # simulate missed detections
            keep &= rng.random(len(owner)) >= self.candidate_drop
        if not keep.all():
            candidates = {k: v[keep] for k, v in candidates.items()}
            features = features[keep]
            owner = match_candidates(candidates, plot['ids'], plot['centres'])
        index_, slots, voxel, slot = slot_inputs(arrays, plot['votes'][members], plot['probability'][members],
                                                 plot['spread'][members], candidates)
        if not len(index_):
            return (torch.zeros((0, voxel.shape[1])), torch.zeros((0, K, slot.shape[2] + len(COLUMNS))), torch.zeros((0, K), dtype=torch.bool),
                    torch.zeros((0, 4), dtype=torch.int32), torch.zeros(0, dtype=torch.long), torch.zeros(0))
        if len(index_) > self.max_voxels:
            keep = np.sort(rng.choice(len(index_), self.max_voxels, replace=False))
            index_, slots, voxel, slot = index_[keep], slots[keep], voxel[keep], slot[keep]
        truth = arrays['tree_id'][index_]
        target = slot_targets(truth, slots, owner)
        ids, inverse, counts = np.unique(truth, return_inverse=True, return_counts=True)
        weight = np.clip(np.sqrt(400. / counts[inverse]), .5, 3.).astype(np.float32)
        weight[truth <= 0] = 1.
        grid = arrays['grid_coord'][index_].copy()
        symmetry = SYMMETRIES[rng.integers(len(SYMMETRIES))]
        grid[:, :2] = grid[:, :2] @ symmetry.T.astype(grid.dtype)
        coords = window_coords(grid)
        first = unique_rows(coords)
        tensor = slot_tensor(slot, slots, features, COLUMNS)
        return (torch.from_numpy(voxel[first]), torch.from_numpy(tensor[first]), torch.from_numpy(slots[first] >= 0),
                torch.from_numpy(coords[first]), torch.from_numpy(target[first]), torch.from_numpy(weight[first]))


def collate(batch):
    batch = [item for item in batch if len(item[0])]
    if not batch:
        return None
    for number, item in enumerate(batch):
        item[3][:, 0] = number
    return tuple(torch.cat([item[i] for item in batch]) for i in range(6)) + (len(batch),)


def validate(network, contexts, paths, pools, biases, device):
    """Sweep the candidate-pool threshold and the none bias; returns summaries and slot accuracy."""
    results, accuracy = {}, {}
    for name, context in contexts.items():
        records = {(t, b): [] for t in pools for b in biases}
        hits, count = Counter(), Counter()
        for (row, arrays, cache, gt, ignore), path in zip(context, paths[name]):
            candidates, features, _ = load_candidates(path)
            ids, centres = cache['tree_ids'], cache['centres']
            for threshold in pools:
                keep = pool_mask(candidates, threshold)
                pool = {k: v[keep] for k, v in candidates.items()}
                owner = match_candidates(pool, ids, centres)
                index, slots, probability = assign_plot(network, arrays, cache, pool, features[keep], device)
                target = slot_targets(arrays['tree_id'][index], slots, owner)
                known = target >= 0
                pick = choose(probability)
                hits[threshold] += int((np.where(pick < 0, K, pick)[known] == target[known]).sum())
                count[threshold] += int(known.sum())
                for bias in biases:
                    labels, _ = labels_from_choice(arrays, index, slots, choose(probability, bias))
                    confidence = np.zeros(len(arrays['coord']), np.float32)
                    confidence[index] = 1. - probability[:, K]
                    confidence[labels == 0] = 0.
                    records[(threshold, bias)].append(evaluate_field(row, arrays, gt, ignore, labels, confidence))
        results[name] = {f'pool{t:g}_bias{b:g}': summarize_field(r) for (t, b), r in records.items()}
        accuracy[name] = {f'pool{t:g}': hits[t] / max(count[t], 1) for t in pools}
    return results, accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--detector-run', required=True)
    parser.add_argument('--detector-weights', default='last.pt')
    parser.add_argument('--run', required=True, help='name of the assignment run')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--samples', type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--val-every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--width', type=int, default=48)
    parser.add_argument('--levels', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--max-voxels', type=int, default=100000)
    parser.add_argument('--candidate-drop', type=float, default=.1)
    parser.add_argument('--pool-range', type=float, nargs=2, default=(.1, .5),
                        help='training: detections below a threshold drawn from this range are left out of the slots')
    parser.add_argument('--thinned-per-plot', type=int, default=6)
    parser.add_argument('--mix', type=float, nargs=3, default=(.4, .4, .2))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    root = args.output.resolve()
    protocol = json.loads((root / 'protocol_v7.json').read_text())
    detector = root / 'runs' / args.detector_run
    run = root / 'runs' / (args.run + ('_smoke' if args.smoke else ''))
    if (run / 'DONE.json').exists():
        print((run / 'selected.json').read_text())
        return
    if run.exists() and not args.smoke:
        raise FileExistsError(f'Incomplete run protected: {run}')
    (run / 'weights').mkdir(parents=True, exist_ok=True)
    store = detector / 'candidates'
    payload = torch.load(detector / 'weights' / args.detector_weights, map_location='cpu', weights_only=False)
    sources, real_rows = training_sources(root)
    sets = validation_sets(root)
    if args.smoke:
        sources = {k: v[:3] for k, v in sources.items()}
        sets = {k: v[:1] for k, v in sets.items()}
        args.epochs, args.samples, args.val_every = 1, 16, 1
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    detector_network = load_network(payload)
    for regime in ('dense', 'thinned', 'helios'):          # candidates for every training variant (cached)
        for source in sources[regime]:
            if regime == 'thinned' and len(source['files']) > args.thinned_per_plot:
                source['files'] = [source['files'][i] for i in sorted(rng.choice(len(source['files']), args.thinned_per_plot, replace=False))]
            for file in source['files']:
                cache_candidates(detector_network, source['row'], file, store / regime / 'train' / Path(file).name)
    paths = {}
    for name, items in sets.items():
        paths[name] = []
        for row, file in items:
            path = store / name / 'val' / Path(file).name
            cache_candidates(detector_network, row, file, path)
            paths[name].append(path)
    del detector_network
    torch.cuda.empty_cache()
    print(f'candidates ready, {time.monotonic() - started:.0f}s', flush=True)
    torch.manual_seed(args.seed)
    network = AssignNet(width=args.width, levels=args.levels).to(device)
    parameters = list(network.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * (args.samples // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=max(steps, 1), pct_start=.05)
    biases = (0., 1.)
    pools = (.1, .2, .3, .4, .5)
    configuration = dict(run=run.name, detector_run=args.detector_run, detector_weights=args.detector_weights,
                         detector_sha256=sha256(detector / 'weights' / args.detector_weights), seed=args.seed,
                         width=args.width, levels=args.levels, k=K, trainable_parameters=sum(p.numel() for p in parameters),
                         architecture='candidate slots (K nearest centres in vote space) -> shared slot MLP -> sparse 3-D '
                                      'U-Net over voxels (0.25 m) -> shared slot scorer + none logit',
                         loss='cross-entropy over K slots + none; unannotated voxels ignored; small trees up-weighted',
                         mix=dict(zip(('dense', 'thinned', 'helios'), args.mix)), epochs=args.epochs,
                         samples_per_epoch=args.samples, batch_size=args.batch_size, lr=args.lr,
                         max_voxels=args.max_voxels, candidate_drop=args.candidate_drop, pool_range=list(args.pool_range),
                         augmentation='lattice symmetries, candidate dropout, random candidate-pool threshold', none_biases=biases, pools=pools,
                         protocol_v7_sha256=sha256(root / 'protocol_v7.json'), test_used_for_selection=False)
    write_json(run / 'configuration.json', configuration)
    contexts = {name: load_context(items) for name, items in sets.items()}
    bases = {name: stage1_summary(context)[0] for name, context in contexts.items()}
    dataset = AssignCrops(sources, store, args.seed, args.max_voxels, args.candidate_drop, tuple(args.pool_range))
    sampler = EpochSampler()
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True,
                        drop_last=True, persistent_workers=True, prefetch_factor=2, collate_fn=collate)
    real_weights = balanced_weights(real_rows) if not args.smoke else np.ones(len(sources['dense']))
    real_weights = real_weights / real_weights.sum()
    regimes = ('dense', 'thinned', 'helios')
    history, selected = [], None

    def save(path, epoch, extra=None):
        temporary = path.with_suffix('.tmp.pt')
        torch.save(dict(network=network.state_dict(), width=args.width, levels=args.levels, k=K, epoch=epoch,
                        model='AssignNet', configuration=configuration, **(extra or {})), temporary)
        os.replace(temporary, path)

    for epoch in range(1, args.epochs + 1):
        network.train()
        draw_rng = np.random.default_rng(args.seed + epoch * 100003)
        draws = []
        for step in range(args.samples):
            regime = regimes[draw_rng.choice(3, p=np.asarray(args.mix) / sum(args.mix))]
            row = (draw_rng.integers(len(sources['helios'])) if regime == 'helios'
                   else draw_rng.choice(len(sources[regime]), p=real_weights))
            draws.append((regime, int(row), step, epoch))
        sampler.draws = draws
        totals, start = Counter(), time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for batch in loader:
            if batch is None:
                continue
            voxel, slot, valid, coords, target, weight, count = batch
            voxel, slot, valid, coords, target, weight = (v.to(device, non_blocking=True) for v in (voxel, slot, valid, coords, target, weight))
            logits = network(voxel, slot, valid, coords, count)
            known = target >= 0
            if not known.any():
                continue
            loss = (F.cross_entropy(logits[known], target[known], reduction='none') * weight[known]).sum() / weight[known].sum()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                totals.update(loss=float(loss), accuracy=float((logits[known].argmax(1) == target[known]).float().mean()),
                              voxels=float(len(target)), batches=1)
        batches = max(totals['batches'], 1)
        record = dict(epoch=epoch, seconds=time.monotonic() - start, train_loss=totals['loss'] / batches,
                      train_accuracy=totals['accuracy'] / batches, voxels_per_batch=totals['voxels'] / batches,
                      peak_vram_mb=torch.cuda.max_memory_allocated() / 1024 ** 2)
        if epoch % args.val_every == 0 or epoch == args.epochs:
            results, accuracy = validate(network, contexts, paths, pools, biases, device)
            write_json(run / f'validation/epoch_{epoch:03d}.json', dict(summaries=results, slot_accuracy=accuracy, stage1_only=bases))
            if not args.smoke:
                save(run / f'weights/epoch_{epoch:03d}.pt', epoch)
            names = [f'pool{t:g}_bias{b:g}' for t in pools for b in biases]
            checks = {c: {s: gate_checks(results[s][c], bases[s], protocol['gate'][s]) for s in GATED_SETS} for c in names}
            passed = {c: all(all(v.values()) for v in checks[c].values()) for c in names}

            def gain(c):
                return sum(quality(results[s][c]) - quality(bases[s]) for s in ('dense', 'thin10'))
            best = max(names, key=lambda c: (passed[c], gain(c)))
            record.update(val_config=best, val_gate_passed=passed[best], val_gain=gain(best),
                          **{f'{s}_slot_accuracy': accuracy[s].get(best.split('_')[0], 0.) for s in results},
                          **{f'{s}_{k}': results[s][best][k] for s in results
                             for k in ('point_sb_pq', 'crown_sb_pq', 'annotated_point_sb_pq', 'small_10_tp', 'large_recall',
                                       'point_precision', 'predicted_instances')})
            if passed[best] and not args.smoke and (selected is None or gain(best) > selected['gain']):
                pool, bias = float(best.split('_')[0][4:]), float(best.split('_')[1][4:])
                selected = dict(epoch=epoch, config_name=best, pool_threshold=pool, none_bias=bias, gain=gain(best),
                                validation={s: results[s][best] for s in results}, checks=checks[best])
                save(run / 'weights/best.pt', epoch, dict(pool_threshold=pool, none_bias=bias))
        save(run / 'weights/last.pt', epoch)
        history.append(record)
        fields = list(dict.fromkeys(k for r in history for k in r))
        with (run / 'training_log.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
        write_json(run / 'selected.json', dict(selected=selected, stage1_only=bases, gate=protocol['gate'],
                                               test_used_for_selection=False))
        print(json.dumps(record), flush=True)
    if not args.smoke:
        write_json(run / 'DONE.json', dict(epochs=args.epochs, seconds=time.monotonic() - started,
                                           selected_epoch=selected and selected['epoch']))


if __name__ == '__main__':
    main()
