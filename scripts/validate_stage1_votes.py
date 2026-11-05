#!/usr/bin/env python3
"""Vote-quality validation of a stage-1 checkpoint, independent of any centre detector.

For every validation regime of the two-pass protocol (``dense``: the 14 native
validation plots, ``thin10``: the same plots thinned to 10 voxels/m2 with the
frozen draw, ``helios``: the 12 simulated validation plots) and for single-pass
and rotation-ensembled votes this reports

- the vote error against the true tree centroid (horizontal, vertical), pooled
  over annotated tree voxels and balanced per tree;
- the scatter of a tree's votes around their own median (what a centre detector
  sees);
- the true-centre oracle: every voxel goes to the nearest per-tree median vote
  within the tree's true extent x 1.25, scored with the protocol's point / crown
  source-balanced PQ (``oracle``), and the same with geometric centroids as
  centres (``oracle_centroid``);
- tree / non-tree accuracy of the semantic head;
- optionally the unchanged stage-1 consensus pipeline (masks + votes), which
  shows whether the mask decoder still works.

Only validation rows are read. ``--from-cache`` scores the cached Model20 votes
without a GPU, which proves the harness against the published reference values.
"""
import argparse
import json
import multiprocessing
from pathlib import Path
import sys
import time

import geopandas as gpd
import numpy as np
import shapely
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz
from pointcloud.paths import remap
from pointcloud.vote_field import ASSIGN, SCALE, field_labels, tree_vote_centres
from scripts.cache_stage1_predictions import sha256, write_json
from scripts.cache_stage1_thinned import subset
from scripts.train_vote_field import evaluate_field, summarize_field, validation_sets

TWO_PASS = PROJECT / 'outputs/dualcrown3d_two_pass_v1'
MODEL20 = PROJECT / 'outputs/dualcrown3d_joint_campaign_v1/runs/repeat_20260930/weights/best.pt'
DEFAULT_OUTPUT = PROJECT / 'outputs/dualcrown3d_stage1_scaled_v1/validation'
SETS = ('dense', 'thin10', 'helios')
ORACLE = {**ASSIGN, 'extent_scale': 1.25}
# Model20, rotation-ensembled votes, median-vote centres, true extents x 1.25 (point, crown SB-PQ).
REFERENCE = dict(dense=(.748, .683), thin10=(.732, .535), helios=(.831, .743))
ORACLE_KEYS = ('point_sb_pq', 'crown_sb_pq', 'point_precision', 'point_recall', 'crown_precision', 'crown_recall',
               'small_10_tp', 'small_10_gt', 'large_recall', 'predicted_instances')


class PointBranch(torch.nn.Module):
    """Presents the point decoder's refined offsets and semantics as the stage-1 votes."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data):
        result = dict(self.model(data))
        result['offset_m'] = result['point_offset_m']
        result['semantic_logits'] = result['point_semantic_logits']
        return result


def load_plots(name, limit=None, cached=False):
    """Validation plots of one regime: (row, arrays, gt, ignore, cache)."""
    assert name in SETS
    plots = []
    for row, path in validation_sets(TWO_PASS)[name][:limit]:
        if row['model_split'] != 'val':
            raise ValueError('Only validation rows may be read here')
        arrays = load_npz(row['output'])
        cache = {}
        path = Path(remap(path))
        if name == 'thin10' or cached:
            with np.load(path) as archive:
                if name == 'thin10':
                    arrays = subset(arrays, archive['keep'])
                if cached:
                    cache = dict(ens4=dict(votes=archive['ensemble_center'].astype(np.float32),
                                           probability=archive['ensemble_probability'].astype(np.float32)),
                                 labels=archive['labels'].astype(np.int64), confidence=archive['confidence'].astype(np.float32))
        if cached and name != 'helios':            # single-pass caches exist for the real plots only
            single = TWO_PASS / ('stage1/val' if name == 'dense' else 'stage1_thinned/val') / path.name
            with np.load(single) as archive:
                cache['single'] = dict(votes=archive['shifted_center'].astype(np.float32),
                                       probability=archive['tree_probability'].astype(np.float32))
        gt = gpd.read_file(row['gt_vector'])
        ignore = shapely.from_wkb(Path(row['ignore_vector']).read_bytes()) if row.get('ignore_vector') else None
        plots.append((row, arrays, gt, ignore, cache))
    return plots


def tree_groups(tree_id):
    use = np.flatnonzero(tree_id > 0)
    use = use[np.argsort(tree_id[use], kind='stable')]
    return np.split(use, np.flatnonzero(np.diff(tree_id[use])) + 1) if len(use) else []


def vote_errors(arrays, votes):
    """Per annotated tree voxel: horizontal / vertical distance of its vote from the true centroid.
    Per tree: medians of both and the median scaled distance of the votes from their own median."""
    coord, tree_id = arrays['coord'], arrays['tree_id']
    target = coord + arrays['instance_offset']
    difference = votes - target
    horizontal = np.linalg.norm(difference[:, :2], axis=1)
    vertical = np.abs(difference[:, 2])
    trees = []
    for members in tree_groups(tree_id):
        median = np.median(votes[members], axis=0)
        scatter = np.linalg.norm((votes[members] - median) * SCALE, axis=1)
        trees.append((len(members), np.median(horizontal[members]), np.median(vertical[members]),
                      horizontal[members].mean(), vertical[members].mean(), np.median(scatter), np.percentile(scatter, 90)))
    positive = tree_id > 0
    return dict(horizontal=horizontal[positive].astype(np.float32), vertical=vertical[positive].astype(np.float32),
                trees=np.asarray(trees, np.float32).reshape(-1, 7))


def nearest_centre_purity(arrays, votes, ids, centres):
    """Share of annotated tree voxels (above 0.5 m, trees with a centre) whose vote lies nearer to
    their own tree's median vote than to any other tree's, in the (x, y, 0.33 z) space of the second
    pass: the clustering-free measure of how separable the votes are. Pooled and tree-balanced."""
    from scipy.spatial import cKDTree
    tree_id, coord = arrays['tree_id'], arrays['coord']
    use = np.flatnonzero((tree_id > 0) & (coord[:, 2] >= .5) & np.isin(tree_id, ids))
    if not len(use) or len(centres) < 2:
        return dict(voxels=int(len(use)), correct=int(len(use)), tree_mean=1., trees=0)
    nearest = cKDTree(centres * SCALE).query(votes[use] * SCALE)[1]
    own = ids[nearest] == tree_id[use]
    _, inverse, counts = np.unique(tree_id[use], return_inverse=True, return_counts=True)
    per_tree = np.bincount(inverse, weights=own) / counts
    return dict(voxels=int(len(use)), correct=int(own.sum()), tree_mean=float(per_tree.mean()), trees=int(len(counts)))


def centroid_centres(arrays, votes, minimum_voxels=8):
    """As ``tree_vote_centres`` but with the geometric centroid as the centre of each tree."""
    coord, tree_id = arrays['coord'], arrays['tree_id']
    target = coord + arrays['instance_offset']
    centres, extents = [], []
    for members in tree_groups(np.where(coord[:, 2] >= .5, tree_id, 0)):
        if len(members) < minimum_voxels:
            continue
        centre = np.median(target[members], axis=0)
        centres.append(centre)
        extents.append(np.percentile(np.linalg.norm((votes[members] - centre) * SCALE, axis=1), 90))
    return np.asarray(centres, np.float32).reshape(-1, 3), np.asarray(extents, np.float32)


def semantic_counts(arrays, probability):
    target = arrays.get('semantic_target', np.where(arrays['tree_id'] < 0, -1, (arrays['tree_id'] > 0).astype(np.int64)))
    known = target >= 0
    predicted, truth = probability[known] >= .5, target[known] > 0
    return dict(tp=int((predicted & truth).sum()), fp=int((predicted & ~truth).sum()),
                fn=int((~predicted & truth).sum()), tn=int((~predicted & ~truth).sum()))


def score_mode(row, arrays, gt, ignore, votes, probability):
    result = dict(errors=vote_errors(arrays, votes), semantic=semantic_counts(arrays, probability))
    ids, centres, extents, _ = tree_vote_centres(arrays['coord'], arrays['tree_id'], votes)
    result['purity'] = nearest_centre_purity(arrays, votes, ids, centres)
    labels, confidence, _ = field_labels(arrays, votes, probability, centres, extents, None, ORACLE)
    result['oracle'] = evaluate_field(row, arrays, gt, ignore, labels, confidence)
    centres, extents = centroid_centres(arrays, votes)
    labels, confidence, _ = field_labels(arrays, votes, probability, centres, extents, None, ORACLE)
    result['oracle_centroid'] = evaluate_field(row, arrays, gt, ignore, labels, confidence)
    return result


_SHARED = {}


def _score(index):
    row, arrays, gt, ignore, predictions = _SHARED['plots'][index]
    result = {mode: score_mode(row, arrays, gt, ignore, value['votes'], value['probability'])
              for mode, value in predictions.items() if mode in _SHARED['modes']}
    if 'labels' in predictions:        # the unchanged stage-1 consensus labels
        result['consensus'] = evaluate_field(row, arrays, gt, ignore, predictions['labels'].astype(np.uint32),
                                             predictions['confidence'])
    return result


def quantiles(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return dict(median=None, mean=None, p90=None)
    return dict(median=float(np.median(values)), mean=float(values.mean()), p90=float(np.percentile(values, 90)))


def summarize_mode(records):
    """records: per-plot outputs of ``score_mode`` for one mode."""
    trees = np.concatenate([r['errors']['trees'] for r in records])
    counts = {k: sum(r['semantic'][k] for r in records) for k in ('tp', 'fp', 'fn', 'tn')}
    total = max(sum(counts.values()), 1)
    summary = dict(
        voxel_horizontal=quantiles(np.concatenate([r['errors']['horizontal'] for r in records])),
        voxel_vertical=quantiles(np.concatenate([r['errors']['vertical'] for r in records])),
        tree_horizontal=quantiles(trees[:, 1]), tree_vertical=quantiles(trees[:, 2]),
        tree_scatter=quantiles(trees[:, 5]), tree_extent=quantiles(trees[:, 6]), trees=int(len(trees)),
        semantic=dict(accuracy=(counts['tp'] + counts['tn']) / total,
                      tree_iou=counts['tp'] / max(counts['tp'] + counts['fp'] + counts['fn'], 1),
                      tree_recall=counts['tp'] / max(counts['tp'] + counts['fn'], 1),
                      tree_precision=counts['tp'] / max(counts['tp'] + counts['fp'], 1), **counts))
    for name in ('oracle', 'oracle_centroid'):
        full = summarize_field([r[name] for r in records])
        summary[name] = {k: full[k] for k in ORACLE_KEYS}
    purity = [r['purity'] for r in records]
    summary['nearest_centre_purity'] = dict(
        voxel=sum(p['correct'] for p in purity) / max(sum(p['voxels'] for p in purity), 1),
        tree=float(np.average([p['tree_mean'] for p in purity], weights=[max(p['trees'], 1) for p in purity])))
    return summary


def plot_record(result):
    """Compact per-plot record (the per-voxel error arrays are dropped)."""
    record = {}
    for mode, value in result.items():
        if mode == 'consensus':
            record[mode] = {k: value[k] for k in ('dataset_id', 'collection', 'tp', 'fp', 'fn', 'point_metrics')}
            continue
        trees = value['errors']['trees']
        record[mode] = dict(
            dataset_id=value['oracle']['dataset_id'], collection=value['oracle']['collection'],
            voxel_horizontal_median=float(np.median(value['errors']['horizontal'])) if len(value['errors']['horizontal']) else None,
            voxel_vertical_median=float(np.median(value['errors']['vertical'])) if len(value['errors']['vertical']) else None,
            tree_horizontal_median=float(np.median(trees[:, 1])) if len(trees) else None,
            tree_vertical_median=float(np.median(trees[:, 2])) if len(trees) else None,
            semantic=value['semantic'], purity=value['purity'],
            **{name: {k: value[name][k] for k in ('tp', 'fp', 'fn', 'point_metrics')} for name in ('oracle', 'oracle_centroid')})
    return record


def score_plots(plots, modes, jobs):
    """plots: (row, arrays, gt, ignore, predictions). Returns per-mode summaries and per-plot records."""
    _SHARED.update(plots=plots, modes=set(modes))
    if jobs > 1 and len(plots) > 1:
        with multiprocessing.get_context('fork').Pool(min(jobs, len(plots))) as pool:
            results = pool.map(_score, range(len(plots)), chunksize=1)
    else:
        results = [_score(i) for i in range(len(plots))]
    _SHARED.clear()
    summary = {mode: summarize_mode([r[mode] for r in results]) for mode in results[0] if mode != 'consensus'}
    if 'consensus' in results[0]:
        full = summarize_field([r['consensus'] for r in results])
        summary['consensus'] = {k: full[k] for k in ORACLE_KEYS}
    return summary, [plot_record(r) for r in results]


def predict_plots(model, plots, modes, consensus_config=None):
    """Run the network; returns (row, arrays, gt, ignore, predictions) per plot and the seconds spent."""
    from pointcloud.instance_output import merge_masks_with_sources
    from pointcloud.vote_ensemble import ensemble_votes
    from scripts.predict_dual_head import predict
    branch = PointBranch(model)
    output, start = [], time.monotonic()
    for row, arrays, gt, ignore, _ in plots:
        predictions = {}
        if 'single' in modes or consensus_config:
            raw = predict(model, arrays, owner_only=True, raw_object_threshold=.05)
            predictions['single'] = dict(votes=raw['shifted_center'].astype(np.float32),
                                         probability=raw['tree_probability'].astype(np.float32))
            if consensus_config:
                labels, confidence, _, _ = merge_masks_with_sources(arrays, raw, consensus_config)
                predictions.update(labels=np.asarray(labels, np.int64), confidence=np.asarray(confidence, np.float32))
            del raw
        for mode, network, symmetries in (('ens4', model, 4), ('point_single', branch, 1), ('point_ens4', branch, 4)):
            if mode in modes:
                ensemble = ensemble_votes(network, arrays, symmetries)
                predictions[mode] = dict(votes=ensemble['shifted_center'], probability=ensemble['tree_probability'])
        output.append((row, arrays, gt, ignore, predictions))
    return output, time.monotonic() - start


def evaluate_checkpoint(model, sets, modes, jobs, consensus_config=None, limit=None, loaded=None):
    """Full vote-quality validation of a model already on the GPU. ``loaded`` may hold preloaded plots."""
    summaries, per_plot, seconds = {}, {}, {}
    for name in sets:
        plots = loaded[name] if loaded else load_plots(name, limit)
        predicted, seconds[name] = predict_plots(model, plots, modes, consensus_config)
        start = time.monotonic()
        summaries[name], per_plot[name] = score_plots(predicted, modes, jobs)
        seconds[name + '_scoring'] = time.monotonic() - start
        del predicted
    return summaries, per_plot, seconds


def selection_score(summaries, mode='single'):
    """Scalar used to pick a checkpoint: mean over dense and thin10 of sqrt(point x crown oracle PQ)."""
    values = [np.sqrt(max(summaries[s][mode]['oracle']['point_sb_pq'] * summaries[s][mode]['oracle']['crown_sb_pq'], 0.))
              for s in ('dense', 'thin10') if s in summaries]
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=MODEL20)
    parser.add_argument('--name', required=True, help='folder name under --output')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--sets', nargs='+', default=list(SETS), choices=SETS)
    parser.add_argument('--modes', nargs='+', default=['single', 'ens4'],
                        choices=['single', 'ens4', 'point_single', 'point_ens4'])
    parser.add_argument('--consensus', action='store_true', help='also score the unchanged stage-1 consensus pipeline')
    parser.add_argument('--from-cache', action='store_true', help='score the cached Model20 votes; no GPU needed')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None, help='first N plots per set (smoke test)')
    args = parser.parse_args()
    torch.set_num_threads(4)
    folder = args.output.resolve() / args.name
    if (folder / 'summary.json').exists():
        raise FileExistsError(f'Validation already present: {folder}')
    folder.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((TWO_PASS / 'protocol.json').read_text())
    started = time.monotonic()
    if args.from_cache:
        checkpoint, digest = protocol['stage1_checkpoint'], protocol['stage1_sha256']
        summaries, per_plot, seconds = {}, {}, {}
        for name in args.sets:
            plots = [(row, arrays, gt, ignore, cache) for row, arrays, gt, ignore, cache in load_plots(name, args.limit, cached=True)]
            summaries[name], per_plot[name] = score_plots(plots, ('single', 'ens4'), args.jobs)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('GPU is required (or use --from-cache)')
        from scripts.train_supervision_v4 import build
        checkpoint = remap(args.checkpoint.resolve())
        digest = sha256(checkpoint)
        model, _ = build(checkpoint)
        model.eval()
        summaries, per_plot, seconds = evaluate_checkpoint(
            model, args.sets, args.modes, args.jobs, protocol['stage1_config'] if args.consensus else None, args.limit)
    for name in args.sets:
        write_json(folder / f'{name}.json', dict(summary=summaries[name], per_plot=per_plot[name]))
    reference = {}
    for name in args.sets:
        if 'ens4' in summaries[name]:
            got = summaries[name]['ens4']['oracle']
            reference[name] = dict(point_sb_pq=got['point_sb_pq'], crown_sb_pq=got['crown_sb_pq'],
                                   model20_reference=REFERENCE[name])
    write_json(folder / 'summary.json', dict(
        name=args.name, checkpoint=str(checkpoint), checkpoint_sha256=digest, from_cache=args.from_cache,
        sets=args.sets, modes=args.modes, limit=args.limit, oracle_assignment=ORACLE, summaries=summaries,
        ensemble_oracle_vs_model20_reference=reference, seconds=dict(total=time.monotonic() - started, **seconds),
        split='val only', test_used=False))
    for name in args.sets:
        for mode, value in summaries[name].items():
            if mode == 'consensus':
                print(f'{name:7s} consensus     point {value["point_sb_pq"]:.4f} crown {value["crown_sb_pq"]:.4f}')
                continue
            print(f'{name:7s} {mode:12s} oracle {value["oracle"]["point_sb_pq"]:.4f} / {value["oracle"]["crown_sb_pq"]:.4f}  '
                  f'centroid-oracle {value["oracle_centroid"]["point_sb_pq"]:.4f} / {value["oracle_centroid"]["crown_sb_pq"]:.4f}  '
                  f'tree-median err xy {value["tree_horizontal"]["median"]:.3f} z {value["tree_vertical"]["median"]:.3f}  '
                  f'voxel-median xy {value["voxel_horizontal"]["median"]:.3f} z {value["voxel_vertical"]["median"]:.3f}  '
                  f'semantic acc {value["semantic"]["accuracy"]:.4f}  purity {value["nearest_centre_purity"]["voxel"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
