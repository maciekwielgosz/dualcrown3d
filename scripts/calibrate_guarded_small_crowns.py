#!/usr/bin/env python3
"""Calibrate small-crown additions on validation; freeze once for held-out test.

Model 20 provides immutable large-tree anchors. Model 22 may add only compact,
mostly novel residual instances. Neither test labels nor the CULS test image
participate in selecting thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import geopandas as gpd
import laspy
import numpy as np
import shapely
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz, read_manifest
from pointcloud.instance_output import merge_masks, merge_masks_with_sources
from pointcloud.small_tree_guard import candidate_features, guarded_additions
from scripts.benchmark_superpoint_algorithms import load_methods, save_json, sha256
from scripts.evaluate_output20_output22_test import (OLD, NEW, PILOT, MERGE,
    wide_predict_plot, score, summary, write_cloud, write_vectors, excel)
from scripts.predict_superpoint_wide_scene import load_models
from scripts.train_superpoint_decoder_pilot import crown_records
from scripts.train_supervision_v4 import REAL

OLD_CACHE = PROJECT / 'outputs/dualcrown3d_legacy_small_tree_calibration/raw_val'
TEST_SOURCE = PROJECT / 'output_23_labeled_test_output20_vs_output22'
DEFAULT = PROJECT / 'output_24_guarded_small_crown_fusion'


def configurations():
    common = dict(min_area_m2=.25, min_voxels=8, min_height_m=1.5,
                  max_added_cover=.20)
    return {f'n{novel:g}_b{cover:g}_a{area:g}_c{confidence:g}':
            dict(**common, min_novelty=novel, max_base_cover=cover,
                 max_area_m2=area, min_confidence=confidence)
            for novel, cover, area, confidence in itertools.product(
                (.75, .90, .98), (.05, .20), (8., 15.), (.15, .30))}


def eligible(split):
    return [row for row in read_manifest(REAL, split)
            if row.get('point_eval_eligible') == 'true']


def inputs_and_signature():
    old = json.loads((OLD / 'inference_report.json').read_text())
    checkpoint = Path(old['checkpoint'])
    if sha256(checkpoint) != old['checkpoint_sha256']:
        raise ValueError('Model 20 checkpoint changed')
    payload = torch.load(NEW, map_location='cpu', weights_only=False)
    initial = Path(payload['initial_checkpoint'])
    embedding = PROJECT / 'outputs/dualcrown3d_superpoint_v1/stage2_instance_embedding/embedding.pt'
    if payload['epoch'] != 12 or payload['config']['run_name'] != 'ezsp_large_w192_q128':
        raise ValueError('Model 22 selected checkpoint changed')
    if sha256(initial) != payload['config']['initial_sha256']:
        raise ValueError('Model 22 backbone changed')
    protocol = dict(manifest_sha256=sha256(REAL), old_checkpoint_sha256=sha256(checkpoint),
                    new_checkpoint_sha256=sha256(NEW),
                    embedding_sha256=sha256(embedding), old_merge=old['mask_config'],
                    new_merge=MERGE, options=configurations(),
                    test_source=str(TEST_SOURCE))
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return old, payload, initial, embedding, protocol, signature


def cached_val_proposals(row, output, models, methods, signature):
    cache = output / 'work/val_proposals' / (row['dataset_id'] + '.npz')
    if cache.exists():
        with np.load(cache) as old:
            if str(old['signature']) != signature:
                raise ValueError(f'Stale validation proposal cache: {cache}')
            return np.asarray(old['labels']), np.asarray(old['confidence'])
    if models is None or methods is None:
        raise RuntimeError('Model 22 not loaded for missing validation cache')
    arrays = load_npz(row['output'])
    backbone, decoder, embedding = models
    csr, ezsp = methods
    seed = 20261001 + int(hashlib.sha256(row['dataset_id'].encode()).hexdigest()[:8], 16)
    raw = wide_predict_plot(arrays, backbone, decoder, embedding, csr, ezsp,
                            'cuda:0', seed=seed)
    labels, confidence, _ = merge_masks(arrays, raw, MERGE)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(cache.stem + '.tmp.npz')
    np.savez_compressed(temporary, signature=np.asarray(signature),
                        labels=labels, confidence=confidence.astype(np.float16))
    temporary.replace(cache)
    return labels, confidence


def cached_val_anchor(row, arrays, config, expected_checkpoint):
    path = OLD_CACHE / (row['dataset_id'] + '.npz')
    with np.load(path) as archive:
        if str(archive['checkpoint_sha256']) != expected_checkpoint:
            raise ValueError(f'Model 20 cache checkpoint mismatch: {path}')
        if str(archive['input_sha256']) != sha256(Path(row['output'])):
            raise ValueError(f'Model 20 cache input mismatch: {path}')
        raw = {key: archive[key] for key in archive.files}
    return merge_masks_with_sources(arrays, raw, config)[:3]


def evaluate_options(row, arrays, gt, ignore, anchor, proposal, options):
    labels20, conf20, records20 = anchor
    labels22, conf22 = proposal
    residual, features = candidate_features(arrays, labels20, labels22, conf22,
                                             records20, crown_records)
    entries = {'Model20': score(row, arrays, gt, ignore, labels20,
                                [r['geometry'] for r in records20])}
    world_xy = arrays['coord'][:, :2].astype(np.float64) + arrays['source_origin'][:2]
    records22 = crown_records({**arrays, 'world_xy': world_xy}, labels22, conf22)
    entries['Model22'] = score(row, arrays, gt, ignore, labels22,
                               [r['geometry'] for r in records22])
    for name, config in options.items():
        labels, _, records, added = guarded_additions(
            labels20, conf20, labels22, conf22, records20, residual, features, config)
        result = score(row, arrays, gt, ignore, labels,
                       [record['geometry'] for record in records])
        result['added_instances'] = len(added)
        entries[name] = result
    return entries, dict(candidates=len(features),
                         mostly_novel=sum(x['novelty'] >= .75 for x in features))


def choose(results, options):
    base = summary(results['Model20'])
    base_point = base['point']['source_balanced_pq']
    base_crown = base['crown']['source_balanced_pq']
    base_precision = base['point']['precision']
    trials = {}
    admissible = []
    for name, config in options.items():
        result = summary(results[name])
        trials[name] = dict(config=config, summary=result)
        # A new detection is useful only while preserving the main model's PQ
        # and instance precision to within a narrow validation tolerance.
        if (result['point']['source_balanced_pq'] >= base_point - .005 and
            result['crown']['source_balanced_pq'] >= base_crown - .005 and
            result['point']['precision'] >= base_precision - .01):
            admissible.append(name)
    improved = [name for name in admissible
                if trials[name]['summary']['small_crowns']['up_to_10_m2']['tp']
                > base['small_crowns']['up_to_10_m2']['tp']]
    if improved:
        selected = max(improved, key=lambda name: (
            trials[name]['summary']['small_crowns']['up_to_10_m2']['tp'],
            trials[name]['summary']['small_crowns']['up_to_4_m2']['tp'],
            trials[name]['summary']['point']['source_balanced_pq'] +
                trials[name]['summary']['crown']['source_balanced_pq']))
    else:
        selected = None
    return dict(selected=selected, base=base, raw22=summary(results['Model22']),
                trials=trials, quality_gate=dict(point_pq_min=base_point-.005,
                   crown_pq_min=base_crown-.005, point_precision_min=base_precision-.01),
                admissible=admissible)


def validation(output, old, payload, initial, embedding, protocol, signature):
    selected_path = output / 'selection.json'
    if selected_path.exists():
        selection = json.loads(selected_path.read_text())
        if selection['signature'] != signature:
            raise ValueError('Existing selection belongs to another protocol')
        return selection
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is needed for uncached model 22 validation predictions')
    torch.set_num_threads(4)
    rows = eligible('val')
    options = configurations()
    cached = all((output/'work/val_proposals'/(r['dataset_id']+'.npz')).exists() for r in rows)
    models = None if cached else load_models(payload, initial, embedding, 'cuda:0')
    if cached:
        methods = None
    else:
        _, csr, ezsp, _ = load_methods()
        methods = (csr, ezsp)
    results = {key: [] for key in ['Model20', 'Model22', *options]}
    diagnostics = []
    started = time.monotonic()
    for i, row in enumerate(rows, 1):
        arrays = load_npz(row['output'])
        anchor = cached_val_anchor(row, arrays, old['mask_config'], old['checkpoint_sha256'])
        proposal = cached_val_proposals(row, output, models, methods, signature)
        gt = gpd.read_file(row['gt_vector'])
        ignore = shapely.from_wkb(Path(row['ignore_vector']).read_bytes()) if row.get('ignore_vector') else None
        entries, info = evaluate_options(row, arrays, gt, ignore, anchor, proposal, options)
        for key, record in entries.items():
            results[key].append(record)
        diagnostics.append(dict(dataset_id=row['dataset_id'], **info))
        print(f'validation {i}/{len(rows)} {row["dataset_id"]}: '
              f'20={entries["Model20"]["predicted_instances"]}, '
              f'22={entries["Model22"]["predicted_instances"]}, '
              f'candidate={info["candidates"]}', flush=True)
    selection = choose(results, options)
    selection.update(signature=signature, protocol=protocol, split='val', plots=len(rows),
                     elapsed_seconds=time.monotonic()-started,
                     candidate_diagnostics=diagnostics,
                     baseline_small=selection['base']['small_crowns']['up_to_10_m2'])
    save_json(selected_path, selection)
    return selection


def test_anchor_and_proposal(row, arrays):
    def cloud(model):
        path = TEST_SOURCE / model / 'PointClouds' / f"trees_{row['dataset_id']}.laz"
        las = laspy.read(path)
        if len(las.points) != len(arrays['coord']):
            raise ValueError(f'Point count mismatch in {path}')
        return np.asarray(las.tree_id, np.uint32), np.asarray(las.confidence, np.float32), np.asarray(las.pred_semantic, np.float32)
    labels20, conf20, sem20 = cloud('Model20')
    labels22, conf22, sem22 = cloud('Model22')
    folder = TEST_SOURCE / 'Model20' / 'Segmentation3'
    crowns = gpd.read_file(folder / f"crowns_{row['dataset_id']}.gpkg")
    tops = gpd.read_file(folder / f"ttops_{row['dataset_id']}.gpkg")
    top_by_id = {int(r.treeID): r for r in tops.itertuples()}
    records = []
    for r in crowns.itertuples():
        top = top_by_id[int(r.treeID)]
        records.append(dict(tree_id=int(r.treeID), geometry=r.geometry,
                            height=float(top.Z), top_x=float(top.geometry.x),
                            top_y=float(top.geometry.y)))
    if set(np.unique(labels20[labels20 > 0])) != {r['tree_id'] for r in records}:
        raise ValueError(f'LAS and GPKG anchor IDs disagree: {row["dataset_id"]}')
    return (labels20, conf20, records), (labels22, conf22), np.maximum(sem20, sem22)


def test(output, selection, signature):
    if (selection['selected'] is None or
        selection['trials'][selection['selected']]['summary']['small_crowns']['up_to_10_m2']['tp']
            <= selection['base']['small_crowns']['up_to_10_m2']['tp']):
        return dict(status='No small-crown gain passed validation quality gate; test untouched')
    result_path = output / 'test_comparison.json'
    if result_path.exists():
        raise FileExistsError(f'Completed test output is protected: {result_path}')
    config = selection['trials'][selection['selected']]['config']
    before = json.loads((TEST_SOURCE / 'comparison.json').read_text())
    if (before['manifest_sha256'] != selection['protocol']['manifest_sha256'] or
        before['output20_sha256'] != selection['protocol']['old_checkpoint_sha256'] or
        before['output22_sha256'] != selection['protocol']['new_checkpoint_sha256']):
        raise ValueError('Held-out source output changed after validation selection')
    rows = eligible('test')
    records = []
    details = []
    started = time.monotonic()
    for i, row in enumerate(rows, 1):
        arrays = load_npz(row['output'])
        anchor, proposal, semantic = test_anchor_and_proposal(row, arrays)
        labels20, conf20, records20 = anchor
        labels22, conf22 = proposal
        residual, features = candidate_features(arrays, labels20, labels22, conf22,
                                                 records20, crown_records)
        labels, confidence, crowns, added = guarded_additions(
            labels20, conf20, labels22, conf22, records20, residual, features, config)
        gt = gpd.read_file(row['gt_vector'])
        ignore = shapely.from_wkb(Path(row['ignore_vector']).read_bytes()) if row.get('ignore_vector') else None
        metric = score(row, arrays, gt, ignore, labels, [r['geometry'] for r in crowns])
        metric['added_instances'] = len(added)
        folder = output / 'Model24'
        cloud = write_cloud(folder/'PointClouds'/f"trees_{row['dataset_id']}",
                            arrays, labels, confidence, semantic, gt.crs)
        vectors = write_vectors(folder/'Segmentation3', row['dataset_id'], crowns, gt.crs)
        records.append(metric)
        details.append(dict(dataset_id=row['dataset_id'], candidates=len(features),
                            accepted=len(added), cloud=cloud, vectors=vectors))
        save_json(output/'metrics/per_plot'/f"{row['dataset_id']}.json",
                  dict(signature=signature, metrics=metric, accepted=len(added),
                       cloud=cloud, vectors=vectors))
        print(f'test {i}/{len(rows)} {row["dataset_id"]}: '
              f'anchor={len(records20)}, added={len(added)}, '
              f'PQpoint={metric["point"]["tp"]} TP', flush=True)
    report = dict(signature=signature, config=config, selected_on='validation only',
                  split='held-out test', plots=len(rows), elapsed_seconds=time.monotonic()-started,
                  results={'Model20':before['results']['Model20'],
                           'Model22':before['results']['Model22'],
                           'Model24':dict(summary=summary(records), per_plot=records)},
                  details=details)
    save_json(result_path, report)
    excel(output/'test_comparison.xlsx', report['results'])
    (output/'README.md').write_text(
        '# Guarded small-crown fusion\n\n'
        'Model 20 anchors are unchanged. Model 22 can add only validation-calibrated, '
        'spatially separate small-crown instances in previously unassigned points. '
        'See selection.json for validation thresholds and test_comparison.xlsx for held-out scores. '
        'Model24/PointClouds has paired LAS/LAZ; Model24/Segmentation3 has crowns and treetops. '
        'LAS XY is in source coordinates, Z is height above ground. tree_id is the prediction; '
        'reference_tree_id is a visualization-only test label, never used to form predictions.\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--phase', choices=('all', 'val', 'test'), default='all')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    old, payload, initial, embedding, protocol, signature = inputs_and_signature()
    if args.phase in ('all', 'val'):
        selection = validation(output, old, payload, initial, embedding, protocol, signature)
        print('selected:', selection['selected'], 'admissible:', len(selection['admissible']), flush=True)
    else:
        selection = json.loads((output/'selection.json').read_text())
        if selection['signature'] != signature:
            raise ValueError('Validation selection protocol mismatch')
    if args.phase in ('all', 'test'):
        result = test(output, selection, signature)
        print(json.dumps(result.get('results', result), indent=2)[:20000], flush=True)


if __name__ == '__main__':
    main()
