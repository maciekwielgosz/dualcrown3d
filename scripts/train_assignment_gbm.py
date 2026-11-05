#!/usr/bin/env python3
"""Stage 3 pilot: a tabular model that assigns voxels to candidate tree centres.

1. Run the vote-field detector on training variants and validation sets and cache
   the candidates (detections + stage-1 centres) with their set-level features.
2. Build (voxel, candidate-slot) rows from a random subset of voxels and fit a
   gradient-boosted classifier: is this candidate the one of my tree?
3. On validation, every voxel takes the slot with the highest probability (or
   none below a threshold); candidates nobody chooses disappear.

This measures how much of the 0.93 point-PQ ceiling of a learned assignment a
model without spatial context reaches. Validation plots only.

Status 2026-10-02: the candidate caching and ``load_candidates`` are used by
``train_assign_net.py``. The pilot itself was never completed: its validation
step calls the fitted scikit-learn model inside forked workers, which hung on
Athena (OpenMP after fork). Predict in the parent process before reusing it.
"""
import argparse
import json
import multiprocessing
import os
import pickle
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.candidate_assignment import K, labels_from_choice, match_candidates, pair_table, slot_inputs, slot_targets
from pointcloud.centre_verifier import FEATURE_NAMES, build_candidates, candidate_features
from pointcloud.vote_field import detect_field, stage1_centres, tree_vote_centres
from scripts.cache_stage1_predictions import DEFAULT_OUTPUT, write_json
from scripts.train_vote_field import (evaluate_field, gate_checks, load_context, load_variant, quality, stage1_summary,
                                      summarize_field, training_sources, validation_sets)
from scripts.validate_vote_field import load_network


def cache_candidates(network, row, file, target, flips=True):
    """Detector -> candidates, their features and the tree each one belongs to; cached as npz."""
    if target.exists():
        return
    arrays, cache = load_variant(row, file)
    anchors = stage1_centres(cache['votes'], cache['labels'])
    detection = detect_field(network, arrays, cache['votes'], cache['probability'], cache['labels'], cache['spread'],
                             threshold=.1, flips=flips, queries=anchors)
    candidates = build_candidates(detection, anchors)
    features = candidate_features(arrays, cache['votes'], cache['probability'], cache['spread'], cache['labels'], candidates, anchors)
    ids, centres, _, _ = tree_vote_centres(arrays['coord'], arrays['tree_id'], cache['votes'])
    owner = match_candidates(candidates, ids, centres)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f'{target.stem}.{os.getpid()}.tmp.npz')      # several jobs may fill the same cache
    np.savez_compressed(temporary, centres=candidates['centres'], scores=candidates['scores'], extents=candidates['extents'],
                        is_anchor=candidates['is_anchor'], features=features, owner=owner)
    os.replace(temporary, target)


def load_candidates(path):
    with np.load(path) as archive:
        return dict(centres=archive['centres'], scores=archive['scores'], extents=archive['extents'],
                    is_anchor=archive['is_anchor']), archive['features'], archive['owner']


def rows_for(arrays, cache, candidates, features, owner, sample=None, rng=None):
    """Pair table, validity, per-voxel index/slots/target for one plot variant."""
    index, slots, voxel, slot = slot_inputs(arrays, cache['votes'], cache['probability'], cache['spread'], candidates)
    target = slot_targets(arrays['tree_id'][index], slots, owner)
    if sample is not None and len(index) > sample:
        keep = np.sort(rng.choice(len(index), sample, replace=False))
        index, slots, voxel, slot, target = index[keep], slots[keep], voxel[keep], slot[keep], target[keep]
    table, valid = pair_table(voxel, slot)
    joined = features[np.maximum(slots, 0)].reshape(len(index) * slots.shape[1], -1)
    joined[~valid] = 0.
    return np.concatenate((table, joined), axis=1).astype(np.float32), valid, index, slots, target


def _train_rows(job):
    row, file, path, sample, seed = job
    arrays, cache = load_variant(row, file)
    candidates, features, owner = load_candidates(path)
    if not len(candidates['centres']):
        return None
    table, valid, index, slots, target = rows_for(arrays, cache, candidates, features, owner, sample, np.random.default_rng(seed))
    k = slots.shape[1]
    known = np.repeat(target >= 0, k)
    label = (np.repeat(target, k) == np.tile(np.arange(k), len(target))).astype(np.int8)
    use = valid & known
    return table[use], label[use]


_SHARED = {}


def _evaluate(i):
    row, arrays, cache, gt, ignore = _SHARED['context'][i]
    candidates, features, owner = load_candidates(_SHARED['paths'][i])
    results = {}
    if not len(candidates['centres']):
        labels = np.zeros(len(arrays['coord']), np.uint32)
        record = evaluate_field(row, arrays, gt, ignore, labels, labels.astype(np.float32))
        return {name: record for name in _SHARED['thresholds']}
    table, valid, index, slots, target = rows_for(arrays, cache, candidates, features, owner)
    k = slots.shape[1]
    probability = np.zeros(len(table), np.float32)
    if valid.any():
        probability[valid] = _SHARED['classifier'].predict_proba(table[valid])[:, 1]
    probability = probability.reshape(len(index), k)
    best = probability.argmax(1)
    top = probability[np.arange(len(index)), best]
    for name, threshold in _SHARED['thresholds'].items():
        choice = np.where(top >= threshold, best, -1)
        labels, _ = labels_from_choice(arrays, index, slots, choice)
        confidence = np.zeros(len(labels), np.float32)
        confidence[index] = top
        confidence[labels == 0] = 0.
        results[name] = evaluate_field(row, arrays, gt, ignore, labels, confidence)
    results['slot_accuracy'] = (float((best[target >= 0] == np.minimum(target[target >= 0], k - 1)).mean()) if (target >= 0).any() else 0.,
                                int((target >= 0).sum()))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run', required=True)
    parser.add_argument('--weights', default='last.pt')
    parser.add_argument('--thinned-per-plot', type=int, default=2)
    parser.add_argument('--voxels-per-variant', type=int, default=6000)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--jobs', type=int, default=12)
    parser.add_argument('--tag', default='assignment_gbm')
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.output.resolve()
    protocol = json.loads((root / 'protocol_v7.json').read_text())
    run = root / 'runs' / args.run
    payload = torch.load(run / 'weights' / args.weights, map_location='cpu', weights_only=False)
    network = load_network(payload)
    folder = run / args.tag
    store = run / 'candidates'
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    sources, _ = training_sources(root)
    jobs = []
    for regime in ('dense', 'thinned', 'helios'):
        for source in sources[regime]:
            files = source['files']
            if regime == 'thinned':
                files = [files[i] for i in rng.choice(len(files), min(args.thinned_per_plot, len(files)), replace=False)]
            for file in files:
                path = store / regime / 'train' / Path(file).name
                cache_candidates(network, source['row'], file, path)
                jobs.append((source['row'], file, path, args.voxels_per_variant, int(rng.integers(1 << 31))))
    sets = validation_sets(root)
    val_paths = {}
    for name, items in sets.items():
        val_paths[name] = []
        for row, file in items:
            path = store / name / 'val' / Path(file).name
            cache_candidates(network, row, file, path)
            val_paths[name].append(path)
    print(f'candidates cached for {len(jobs)} training variants and the validation sets, {time.monotonic() - started:.0f}s', flush=True)
    del network
    torch.cuda.empty_cache()
    with multiprocessing.get_context('fork').Pool(args.jobs) as pool:
        parts = [p for p in pool.map(_train_rows, jobs, chunksize=2) if p is not None]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    del parts
    print(f'pair rows {len(y)}, positives {int(y.sum())}, features {X.shape[1]}, {time.monotonic() - started:.0f}s', flush=True)
    classifier = HistGradientBoostingClassifier(max_iter=500, learning_rate=.08, max_leaf_nodes=63, min_samples_leaf=200,
                                                l2_regularization=1., early_stopping=True, validation_fraction=.05,
                                                random_state=args.seed)
    classifier.fit(X, y)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'assignment_gbm.pkl').open('wb') as stream:
        pickle.dump(dict(classifier=classifier, k=K, candidate_features=FEATURE_NAMES), stream)
    print(f'fitted, iterations {classifier.n_iter_}, {time.monotonic() - started:.0f}s', flush=True)
    del X, y
    thresholds = {f'p{t:g}': t for t in (.3, .4, .5, .6)}
    results, bases, accuracy = {}, {}, {}
    for name, items in sets.items():
        context = load_context(items)
        bases[name], _ = stage1_summary(context)
        _SHARED.update(context=context, paths=val_paths[name], classifier=classifier, thresholds=thresholds)
        with multiprocessing.get_context('fork').Pool(min(args.jobs, len(context))) as pool:
            per_plot = pool.map(_evaluate, range(len(context)))
        _SHARED.clear()
        results[name] = {t: summarize_field([p[t] for p in per_plot]) for t in thresholds}
        hits = sum(p['slot_accuracy'][0] * p['slot_accuracy'][1] for p in per_plot if 'slot_accuracy' in p)
        count = sum(p['slot_accuracy'][1] for p in per_plot if 'slot_accuracy' in p)
        accuracy[name] = hits / max(count, 1)
        write_json(folder / f'validation_{name}.json', dict(summaries=results[name], stage1_only=bases[name],
                                                            slot_accuracy=accuracy[name]))
        b = bases[name]
        print(f"== {name}: Model20 pPQ {b['point_sb_pq']:.4f} cPQ {b['crown_sb_pq']:.4f} annotated pPQ {b['annotated_point_sb_pq']:.4f}; "
              f"slot accuracy on annotated voxels {accuracy[name]:.3f}")
        for t, s in results[name].items():
            failed = [k for k, v in gate_checks(s, b, protocol['gate'][name]).items() if not v] if name in protocol['gate'] else []
            print(f"   {t:6s} pPQ {s['point_sb_pq']:.4f} cPQ {s['crown_sb_pq']:.4f} pP {s['point_precision']:.3f} cP {s['crown_precision']:.3f} "
                  f"small {s['small_10_tp']:3d} large {s['large_recall']:.3f} over {s['oversplit_gt']:3d} n {s['predicted_instances']:4d} "
                  f"| annotated pPQ {s['annotated_point_sb_pq']:.4f} {'PASS' if not failed else failed}", flush=True)
    gains = {t: sum(quality(results[s][t]) - quality(bases[s]) for s in ('dense', 'thin10')) for t in thresholds}
    print('gain per threshold:', {t: round(g, 4) for t, g in gains.items()})
    write_json(folder / 'summary.json', dict(gains=gains, slot_accuracy=accuracy, iterations=int(classifier.n_iter_),
                                             detector_weights=str(run / 'weights' / args.weights), test_used_for_selection=False))


if __name__ == '__main__':
    main()
