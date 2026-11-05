#!/usr/bin/env python3
"""Train the stage-2b centre verifier on top of a trained vote-field detector.

1. Runs the detector on training variants (dense, thinned, HELIOS) and on every
   validation set, builds the candidate set (detections + stage-1 centres) and its
   features, and caches them.
2. Fits a gradient-boosted classifier: is this candidate the centre of a tree?
3. Sweeps the selection threshold, suppression and assignment on validation with
   the verifier's probability as the score and applies the frozen v7 gates.

Validation plots only; the held-out test splits are never read.
"""
import argparse
import itertools
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
from sklearn.metrics import average_precision_score

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.centre_verifier import (FEATURE_NAMES, build_candidates, candidate_features, candidate_targets,
                                        combine_scores, verified_detection)
from pointcloud.vote_field import detect_field, field_labels, select_field, stage1_centres, tree_vote_centres
from scripts.cache_stage1_predictions import DEFAULT_OUTPUT, sha256, write_json
from scripts.train_vote_field import (ASSIGNMENTS, GATED_SETS, VALIDATION_SETS, evaluate_field, gate_checks, load_context,
                                      load_variant, quality, stage1_summary, summarize_field, training_sources, validation_sets)
from scripts.validate_vote_field import load_network


def plot_candidates(network, arrays, cache, flips=True):
    """Detector -> candidates, features, targets for one plot variant."""
    anchors = cache.get('anchors')
    if anchors is None:
        anchors = stage1_centres(cache['votes'], cache['labels'])
    detection = detect_field(network, arrays, cache['votes'], cache['probability'], cache['labels'], cache['spread'],
                             threshold=.1, flips=flips, queries=anchors)
    candidates = build_candidates(detection, anchors)
    features = candidate_features(arrays, cache['votes'], cache['probability'], cache['spread'], cache['labels'],
                                  candidates, anchors)
    centres = cache.get('centres')
    if centres is None:
        centres = tree_vote_centres(arrays['coord'], arrays['tree_id'], cache['votes'])[1]
    target, weight = candidate_targets(candidates, centres, arrays, cache['votes'])
    return candidates, features, target, weight


def sweep():
    configs = {}
    for rule, threshold, factor, tag in itertools.product(('mean', 'mean_heat', 'verifier'), (.4, .5, .6, .7), (0., 1.),
                                                          ('height', 'extent', 'extent1.5', 'extent1.5_norm')):
        configs[f'{rule}_t{threshold:g}_nms{factor:g}_{tag}'] = dict(
            selection=dict(mode='replace', threshold=threshold, suppress=factor), assign=ASSIGNMENTS[tag],
            verifier=dict(rule=rule))
    return configs


_SHARED = {}


def _evaluate(index):
    (row, arrays, cache, gt, ignore), candidates, probability = (_SHARED['context'][index], _SHARED['candidates'][index],
                                                                  _SHARED['probability'][index])
    results = []
    for config in _SHARED['configs'].values():
        detection = verified_detection(candidates, combine_scores(config['verifier']['rule'], probability, candidates))
        centres, extents, kinds, info = select_field(detection, np.zeros((0, 3), np.float32), config['selection'])
        labels, confidence, _ = field_labels(arrays, cache['votes'], cache['probability'], centres, extents, None, config['assign'])
        results.append(evaluate_field(row, arrays, gt, ignore, labels, confidence))
    return results


def load_verifier(payload):
    """The fitted classifier stored in a selected checkpoint, or None."""
    return pickle.loads(payload['verifier_pickle']) if payload.get('verifier_pickle') else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run', required=True)
    parser.add_argument('--weights', default='last.pt')
    parser.add_argument('--thinned-per-plot', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--jobs', type=int, default=10)
    parser.add_argument('--tag', default='verifier')
    parser.add_argument('--select', action='store_true',
                        help='freeze the best gate-passing configuration into best.pt (with the verifier inside) unless '
                             'the detector-only selection of validate_vote_field.py has a larger gain')
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.output.resolve()
    protocol = json.loads((root / 'protocol_v7.json').read_text())
    run = root / 'runs' / args.run
    weights = run / 'weights' / args.weights
    payload = torch.load(weights, map_location='cpu', weights_only=False)
    network = load_network(payload)
    folder = run / args.tag
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    # 1. training candidates
    cache_path = folder / 'train_candidates.npz'
    if cache_path.exists():
        with np.load(cache_path) as archive:
            X, y, w, regime = archive['features'], archive['target'], archive['weight'], archive['regime']
    else:
        sources, _ = training_sources(root)
        parts = []
        for code, name in enumerate(('dense', 'thinned', 'helios')):
            for number, source in enumerate(sources[name], 1):
                files = source['files']
                if name == 'thinned':
                    files = [files[i] for i in rng.choice(len(files), min(args.thinned_per_plot, len(files)), replace=False)]
                for file in files:
                    arrays, cache = load_variant(source['row'], file)
                    _, features, target, weight = plot_candidates(network, arrays, cache)
                    parts.append((features, target, weight, np.full(len(target), code, np.int8)))
                if number % 20 == 0:
                    print(f'train {name} {number}/{len(sources[name])} {time.monotonic() - started:.0f}s', flush=True)
        X, y, w, regime = (np.concatenate([p[i] for p in parts]) for i in range(4))
        np.savez_compressed(cache_path, features=X, target=y, weight=w, regime=regime)
    print(f'training candidates {len(y)}: positives {int(y.sum())}, ignored {int((w == 0).sum())}, '
          f'per regime {np.bincount(regime).tolist()}', flush=True)
    # 2. fit
    use = w > 0
    balance = np.ones(len(y), np.float32)
    for code in range(3):             # each regime carries the same total weight
        members = use & (regime == code)
        balance[members] = use.sum() / (3. * max(members.sum(), 1))
    classifier = HistGradientBoostingClassifier(max_iter=400, learning_rate=.05, max_leaf_nodes=31, min_samples_leaf=40,
                                                l2_regularization=1., early_stopping=True, validation_fraction=.15,
                                                random_state=args.seed)
    classifier.fit(X[use], y[use], sample_weight=balance[use])
    with (folder / 'verifier.pkl').open('wb') as stream:
        pickle.dump(dict(classifier=classifier, feature_names=FEATURE_NAMES, detector_weights=str(weights),
                         detector_sha256=sha256(weights)), stream)
    # 3. validation
    configs = sweep()
    results, bases, report = {}, {}, {}
    for name, items in validation_sets(root).items():
        context = load_context(items)
        bases[name], _ = stage1_summary(context)
        candidates, probabilities, targets, weights_, raw = [], [], [], [], []
        for row, arrays, cache, gt, ignore in context:
            c, features, target, weight = plot_candidates(network, arrays, cache)
            candidates.append(c)
            probabilities.append(classifier.predict_proba(features)[:, 1] if len(features) else np.zeros(0))
            targets.append(target)
            weights_.append(weight)
            raw.append(c['scores'])
        t, wv = np.concatenate(targets), np.concatenate(weights_) > 0
        report[name] = dict(candidates=int(len(t)), positives=int(t.sum()),
                            ap_detector=float(average_precision_score(t[wv], np.concatenate(raw)[wv])),
                            ap_verifier=float(average_precision_score(t[wv], np.concatenate(probabilities)[wv])))
        print(name, report[name], flush=True)
        _SHARED.update(context=context, candidates=candidates, probability=probabilities, configs=configs)
        with multiprocessing.get_context('fork').Pool(min(args.jobs, len(context))) as pool:
            per_plot = pool.map(_evaluate, range(len(context)))
        _SHARED.clear()
        records = {config: [plot[i] for plot in per_plot] for i, config in enumerate(configs)}
        results[name] = {config: summarize_field(value) for config, value in records.items()}
        write_json(folder / f'validation_{name}.json', dict(summaries=results[name], per_plot=records, stage1_only=bases[name],
                                                            ranking=report[name]))
    checks = {c: {s: gate_checks(results[s][c], bases[s], protocol['gate'][s]) for s in GATED_SETS} for c in configs}
    passing = [c for c in configs if all(all(v.values()) for v in checks[c].values())]

    def gain(c):
        return sum(quality(results[s][c]) - quality(bases[s]) for s in ('dense', 'thin10'))
    for name in VALIDATION_SETS:
        b = bases[name]
        print(f"== {name}: Model20 pPQ {b['point_sb_pq']:.4f} cPQ {b['crown_sb_pq']:.4f} annotated pPQ {b['annotated_point_sb_pq']:.4f}")
        for c in sorted(configs, key=lambda c: -quality(results[name][c]))[:6]:
            s = results[name][c]
            failed = [k for k, v in checks[c].get(name, {}).items() if not v] if name in GATED_SETS else []
            print(f"   {c:36s} pPQ {s['point_sb_pq']:.4f} cPQ {s['crown_sb_pq']:.4f} pP {s['point_precision']:.3f} cP {s['crown_precision']:.3f} "
                  f"small {s['small_10_tp']:3d} large {s['large_recall']:.3f} over {s['oversplit_gt']:3d} n {s['predicted_instances']:4d} "
                  f"| annotated pPQ {s['annotated_point_sb_pq']:.4f} {'PASS' if not failed else failed}")
    selection = None
    if passing:
        best = max(passing, key=gain)
        selection = dict(epoch=payload['epoch'], weights=args.weights, config_name=best, config=configs[best], gain=gain(best),
                         checks=checks[best], validation={s: results[s][best] for s in VALIDATION_SETS},
                         source_weights_sha256=sha256(weights))
    print('passing all gates:', [(c, round(gain(c), 4)) for c in sorted(passing, key=gain, reverse=True)[:8]])
    write_json(folder / 'selection.json', dict(selected=selection, passing=passing, ranking=report, stage1_only=bases,
                                               gate=protocol['gate'], detector_weights=str(weights),
                                               iterations=int(classifier.n_iter_), test_used_for_selection=False))
    if args.select and selection:
        detector_only = run / f'validation/full_epoch_{payload["epoch"]:03d}/selection.json'
        rival = json.loads(detector_only.read_text())['selected'] if detector_only.exists() else None
        if rival and rival['gain'] >= selection['gain']:
            print(f"detector-only configuration {rival['config_name']} has the larger gain ({rival['gain']:.4f}); best.pt unchanged")
            return
        target = run / 'weights/best.pt'
        temporary = target.with_suffix('.tmp.pt')
        torch.save({**payload, 'selected_config': configs[best], 'selected_config_name': best,
                    'verifier_pickle': pickle.dumps(classifier), 'verifier_features': FEATURE_NAMES}, temporary)
        os.replace(temporary, target)
        write_json(run / 'selected.json', dict(selected=selection, detector_only_alternative=rival, stage1_only=bases,
                                               gate=protocol['gate'], test_used_for_selection=False, best_sha256=sha256(target)))
        print(f'selected {best} with the verifier -> {target}')


if __name__ == '__main__':
    main()
