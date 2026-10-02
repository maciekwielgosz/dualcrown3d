#!/usr/bin/env python3
"""Full-scene, resumable inference for the trained wide EZ-SP crop decoder.

The 20 m model is applied to the same prepared ALS voxels as output_21. Each
x-stripe is saved separately; existing stripes are verified, never overwritten.
This is an experimental transfer, not a validated whole-scene promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import geopandas as gpd
import laspy
import numpy as np
from pyproj import CRS
from rasterio.features import shapes
from rasterio.transform import Affine
from scipy.ndimage import binary_closing, binary_fill_holes
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import MultiPolygon, Point, shape
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from pointcloud.dual_head import DualHeadLitePT
from pointcloud.instance_output import _instance_members, merge_masks
from pointcloud.superpoints.decoder import group_geometry
from pointcloud.superpoints.instance_embedding import InstanceBoundaryEmbedding, edge_affinity
from pointcloud.superpoints.model import CachedInstanceModel
from pointcloud.superpoints.partition import geometric_partition, neighbor_pairs
from scripts.benchmark_superpoint_algorithms import PartitionContext, budget_match, load_methods, sha256
from scripts.evaluate_pointcloud_litept import model_input, ownership_intervals, starts_for_axis, dbh_naslund
from scripts.predict_dual_head import export_laz

PILOT = PROJECT / 'outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot'
DEFAULT_CHECKPOINT = PILOT / 'runs/ezsp_large_w192_q128/weights/best.pt'
DEFAULT_PREPARED = PROJECT / 'output_15_litept_v2_no_rectangles_pointcloud/work'
DEFAULT_OUTPUT = PROJECT / 'output_22_ezsp_wide_q128_experimental'
MERGE = dict(object_threshold=.1, mask_threshold=.5, minimum_voxels=8,
             minimum_height_m=2., minimum_area_m2=.75, merge_overlap=.6,
             merge_strategy='support_fusion_v2', fusion_min_iou=.2,
             fusion_dominance=.8, fusion_max_distance_m=1.5,
             fusion_min_probability=.5)


def load_inputs(prepared):
    with np.load(prepared / 'benchmark_pointcloud.npz') as file:
        arrays = {name: file[name] for name in ('coord', 'grid_coord', 'intensity',
                                                'source_origin', 'voxel_origin', 'voxel_size')}
    metadata = json.loads((prepared / 'preparation.json').read_text())
    return arrays, metadata


def configuration(checkpoint, prepared, maximum):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = payload['config']
    required = dict(method='ezsp', graph_width=192, graph_layers=4,
                    queries=128, memory_tokens=1536, wide_dim=192, wide_layers=2)
    if payload['epoch'] != 12 or any(config.get(k) != v for k, v in required.items()):
        raise ValueError('Checkpoint is not the selected epoch-12 wide EZ-SP q128 model')
    initial = Path(payload['initial_checkpoint'])
    embedding = PROJECT / 'outputs/dualcrown3d_superpoint_v1/stage2_instance_embedding/embedding.pt'
    if sha256(initial) != config['initial_sha256']:
        raise ValueError('Frozen encoder checkpoint differs from training')
    selected = json.loads((checkpoint.parents[1] / 'selected.json').read_text())
    if Path(selected['checkpoint']).resolve() != checkpoint.resolve() or selected['epoch'] != 12:
        raise ValueError('Checkpoint does not match pilot selection')
    if selected['validation']['config'] != dict(object_threshold=.1, mask_threshold=.5,
                                                 minimum_points=8, duplicate_iou=.6):
        raise ValueError('Unexpected selected point-mask thresholds')
    setup = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint),
                 initial_checkpoint=str(initial.resolve()), initial_sha256=sha256(initial),
                 embedding_checkpoint=str(embedding.resolve()), embedding_sha256=sha256(embedding),
                 prepared_dir=str(prepared.resolve()), preparation_sha256=sha256(prepared / 'preparation.json'),
                 max_points=maximum, window_m=20., overlap_m=8., partition='EZ-SP budget search; 0.5 m geometric target',
                 merge=MERGE, implementation_sha256=sha256(Path(__file__)),
                 status='experimental full-scene transfer; not validation-calibrated')
    signature = hashlib.sha256(json.dumps(setup, sort_keys=True).encode()).hexdigest()
    return payload, initial, embedding, setup, signature


def load_models(payload, initial, embedding_path, device):
    source = torch.load(initial, map_location='cpu', weights_only=False)
    backbone = DualHeadLitePT(**source['model_args']).to(device).eval()
    backbone.load_state_dict(source['model'], strict=True)
    decoder = CachedInstanceModel('ezsp', queries=128, memory_tokens=1536,
                                  graph_width=192, graph_layers=4,
                                  wide_dim=192, wide_layers=2).to(device).eval()
    decoder.load_state_dict(payload['model'], strict=True)
    embedding = InstanceBoundaryEmbedding().to(device).eval()
    embedding.load_state_dict(torch.load(embedding_path, map_location='cpu', weights_only=False)['model'])
    return backbone, decoder, embedding


@torch.no_grad()
def predict_window(arrays, chosen, backbone, decoder, embedding, csr, ezsp, device):
    batch = model_input(arrays['coord'][chosen], arrays['grid_coord'][chosen],
                        arrays['intensity'][chosen], device, preserve_height=True)
    feature = backbone.backbone(batch).feat
    coord = batch['coord'].cpu().numpy()
    edges = neighbor_pairs(coord, radius_m=.75, k=12)
    if len(edges):
        feature_np = feature.float()
        latent = embedding(feature_np)
        weight = edge_affinity(latent, torch.from_numpy(edges.astype(np.int64)).to(device)).cpu().numpy().astype(np.float32)
        target = len(coord) / len(np.unique(geometric_partition(coord, .5)))
        context = PartitionContext(coord, latent.cpu().numpy().astype(np.float32),
                                   edges, weight, csr)
        groups, _, _ = budget_match(lambda reg: context.run_ez(reg, ezsp), target,
                                    lower=1e-5, upper=20., steps=8)
        del context
    else:
        groups = np.arange(len(coord), dtype=np.int32)
    groups, centers, group_edges = group_geometry(coord, groups)
    batch.update(feature=feature,
                 groups=torch.from_numpy(groups).to(device),
                 centers=torch.from_numpy(centers).to(device),
                 group_edges=torch.from_numpy(group_edges).to(device))
    result = decoder(batch)
    masks = result['instance_masks']
    return (masks['mask_logits'].sigmoid().cpu().numpy(),
            masks['object_logits'].sigmoid().cpu().numpy(),
            result['point_semantic_logits'].softmax(1)[:, 1].cpu().numpy())


def save_stripe(path, signature, owners, semantics, offsets, members, probabilities, scores, windows):
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, signature=np.asarray(signature),
        owned_index=np.concatenate(owners) if owners else np.empty(0, np.int32),
        owned_semantic=np.concatenate(semantics) if semantics else np.empty(0, np.float32),
        candidate_offset=np.asarray(offsets, np.int64),
        point_index=np.concatenate(members) if members else np.empty(0, np.int32),
        point_score=np.concatenate(probabilities) if probabilities else np.empty(0, np.float16),
        object_score=np.asarray(scores, np.float32), windows=np.int32(windows))
    os.replace(temporary, path)


def predict_stripes(arrays, output, payload, initial, embedding_path, signature,
                    max_points, device, max_stripes=0):
    xyz = arrays['coord']
    xs = starts_for_axis(float(xyz[:, 0].min()), float(xyz[:, 0].max()), 20., 8.)
    ys = starts_for_axis(float(xyz[:, 1].min()), float(xyz[:, 1].max()), 20., 8.)
    xo = ownership_intervals(xs, 20., float(xyz[:, 0].min()),
                             float(np.nextafter(xyz[:, 0].max(), np.float32(np.inf))))
    yo = ownership_intervals(ys, 20., float(xyz[:, 1].min()),
                             float(np.nextafter(xyz[:, 1].max(), np.float32(np.inf))))
    stripe_dir = output / 'work/stripes'
    stripe_dir.mkdir(parents=True, exist_ok=True)
    index = cKDTree(xyz[:, :2])
    backbone, decoder, embedding = load_models(payload, initial, embedding_path, device)
    _, csr, ezsp, _ = load_methods()
    start = time.monotonic()
    processed = 0
    for xi, x in enumerate(xs):
        path = stripe_dir / f'stripe_{xi:03d}.npz'
        rng = np.random.default_rng(20261001 + xi)
        if path.exists():
            with np.load(path) as old:
                if str(old['signature']) != signature:
                    raise ValueError(f'Existing stripe from another run: {path}')
            continue
        if max_stripes and processed >= max_stripes:
            break
        owners, semantics, members, probabilities, scores = [], [], [], [], []
        offsets = [0]
        windows = 0
        for yi, y in enumerate(ys):
            context = np.asarray(sorted(index.query_ball_point([x + 10., y + 10.], 10.0001, p=np.inf)), dtype=np.int64)
            if not len(context):
                continue
            x_end = float(xyz[:, 0].max()) if xi == len(xs) - 1 else x + 20.
            y_end = float(xyz[:, 1].max()) if yi == len(ys) - 1 else y + 20.
            context = context[(xyz[context, 0] >= x) & (xyz[context, 0] <= x_end)
                              & (xyz[context, 1] >= y) & (xyz[context, 1] <= y_end)]
            a, b = xo[xi]
            c, d = yo[yi]
            owner = context[(xyz[context, 0] >= a) & (xyz[context, 0] < b)
                            & (xyz[context, 1] >= c) & (xyz[context, 1] < d)]
            if not len(owner):
                continue
            for owned in np.array_split(owner, max(1, math.ceil(len(owner) / max_points))):
                extra = np.setdiff1d(context, owned, assume_unique=True)
                capacity = max_points - len(owned)
                if len(extra) > capacity:
                    extra = rng.choice(extra, capacity, replace=False)
                chosen = np.sort(np.concatenate((owned, extra)))
                own_local = np.flatnonzero(np.isin(chosen, owned))
                masks, object_score, semantic = predict_window(
                    arrays, chosen, backbone, decoder, embedding, csr, ezsp, device)
                owners.append(owned.astype(np.int32))
                semantics.append(semantic[own_local].astype(np.float32))
                for q in np.flatnonzero(object_score >= .1):
                    retained = np.flatnonzero(masks[q] >= .2)
                    if len(retained) < 4:
                        continue
                    center = np.average(xyz[chosen[retained], :2], axis=0,
                                        weights=masks[q, retained])
                    if not (a <= center[0] < b and c <= center[1] < d):
                        continue
                    members.append(chosen[retained].astype(np.int32))
                    probabilities.append(masks[q, retained].astype(np.float16))
                    scores.append(float(object_score[q]))
                    offsets.append(offsets[-1] + len(retained))
                windows += 1
        save_stripe(path, signature, owners, semantics, offsets, members,
                    probabilities, scores, windows)
        processed += 1
        print(f'stripe {xi + 1}/{len(xs)}: {windows} windows, {len(scores)} proposals, '
              f'{time.monotonic() - start:.1f}s elapsed', flush=True)
    return len(xs)


def collect_stripes(output, signature, n_stripes, n_points):
    offsets, members, probabilities, objects = [0], [], [], []
    point_probability = np.zeros(n_points, np.float32)
    visited = np.zeros(n_points, np.uint8)
    windows = 0
    for xi in range(n_stripes):
        path = output / 'work/stripes' / f'stripe_{xi:03d}.npz'
        if not path.exists():
            raise RuntimeError(f'Incomplete prediction; missing {path}')
        with np.load(path) as file:
            if str(file['signature']) != signature:
                raise ValueError(f'Stripe signature mismatch: {path}')
            owned = file['owned_index']
            if np.any(visited[owned]):
                raise AssertionError(f'Repeated owner indices in {path}')
            visited[owned] = 1
            point_probability[owned] = file['owned_semantic']
            local = file['candidate_offset']
            offsets.extend((local[1:] + offsets[-1]).tolist())
            members.append(file['point_index'])
            probabilities.append(file['point_score'])
            objects.append(file['object_score'])
            windows += int(file['windows'])
    if not visited.all():
        raise AssertionError(f'Ownership left {int((visited == 0).sum())} voxels uncovered')
    return dict(candidate_offset=np.asarray(offsets, np.int64),
                point_index=np.concatenate(members),
                point_score=np.concatenate(probabilities),
                object_score=np.concatenate(objects),
                point_probability=point_probability), windows


def support_polygon(xy, resolution=.5):
    origin = np.floor(xy.min(0) / resolution) * resolution
    grid = np.floor((xy - origin) / resolution).astype(np.int32)
    occupancy = np.zeros((int(grid[:, 1].max()) + 5, int(grid[:, 0].max()) + 5), bool)
    occupancy[grid[:, 1] + 2, grid[:, 0] + 2] = True
    filled = binary_fill_holes(binary_closing(occupancy, structure=np.ones((3, 3))) | occupancy)
    transform = Affine(resolution, 0, origin[0] - 2 * resolution,
                       0, resolution, origin[1] - 2 * resolution)
    pieces = [shape(geom) for geom, value in shapes(filled.astype(np.uint8),
                                                    mask=filled, transform=transform) if value]
    geometry = shapely.make_valid(shapely.union_all(pieces))
    polygons = [part for part in shapely.get_parts(geometry) if part.geom_type == 'Polygon']
    return MultiPolygon(polygons)


def build_instances(arrays, metadata, labels, confidence, original_instances):
    xy = arrays['coord'][:, :2].astype(np.float64) + arrays['source_origin'][:2]
    core = []
    for old, (item, indices) in enumerate(zip(original_instances, _instance_members(labels)), 1):
        if not any(tile['bounds'][0] <= item['top_x'] < tile['bounds'][2] and
                   tile['bounds'][1] <= item['top_y'] < tile['bounds'][3]
                   for tile in metadata['tiles']):
            labels[indices] = 0
            confidence[indices] = 0
            continue
        geometry = support_polygon(xy[indices])
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(f'Invalid reconstructed crown {old}')
        top = indices[np.argmax(arrays['coord'][indices, 2])]
        number = len(core) + 1
        labels[indices] = number
        core.append(dict(tree_id=number, geometry=geometry, points=len(indices),
                         confidence=float(confidence[indices].mean()),
                         height=float(arrays['coord'][top, 2]),
                         top_x=float(xy[top, 0]), top_y=float(xy[top, 1])))
    return core


def write_vectors(instances, metadata, folder):
    folder.mkdir(parents=True, exist_ok=True)
    report = []
    for tile in metadata['tiles']:
        left, bottom, right, top = tile['bounds']
        items = [item for item in instances if left <= item['top_x'] < right and
                 bottom <= item['top_y'] < top]
        crowns = gpd.GeoDataFrame(dict(treeID=[item['tree_id'] for item in items],
                                          area_m2=[item['geometry'].area for item in items]),
                                     geometry=[item['geometry'] for item in items],
                                     crs=metadata['chm_crs'])
        tops = gpd.GeoDataFrame(dict(treeID=[item['tree_id'] for item in items],
                                       Z=[item['height'] for item in items],
                                       dbh=[round(dbh_naslund(item['height']), 2) for item in items]),
                                  geometry=[Point(item['top_x'], item['top_y'], item['height']) for item in items],
                                  crs=metadata['chm_crs'])
        crowns.to_file(folder / f"crowns_{tile['tile_id']}.gpkg", driver='GPKG', index=False)
        tops.to_file(folder / f"ttops_{tile['tile_id']}.gpkg", driver='GPKG', index=False)
        if not crowns.is_valid.all() or crowns.is_empty.any() or len(crowns) != len(tops):
            raise AssertionError(f'Invalid vectors: {tile["tile_id"]}')
        report.append(dict(tile_id=tile['tile_id'], crowns=len(crowns), treetops=len(tops)))
    return report


def export_scene(arrays, metadata, output, signature, n_stripes, setup):
    started = time.monotonic()
    raw, windows = collect_stripes(output, signature, n_stripes, len(arrays['coord']))
    print(f'collected {windows} windows and {len(raw["object_score"])} proposals; merging', flush=True)
    labels, confidence, instances = merge_masks(arrays, raw, MERGE)
    core = build_instances(arrays, metadata, labels, confidence, instances)
    if not core:
        raise RuntimeError('No instances after merge; export was not written')
    point_vectors = write_vectors(core, metadata, output / 'PointHead/Segmentation3')
    benchmark_vectors = write_vectors(core, metadata, output / 'Segmentation3')
    clouds = export_laz(output, arrays, metadata, labels, np.zeros_like(labels), confidence,
                        (raw['point_probability'] >= .3).astype(np.uint8),
                        (labels > 0).astype(np.uint8))
    point_ids = set(np.unique(labels[labels > 0]).tolist())
    polygon_ids = set()
    for tile in metadata['tiles']:
        stem = tile['tile_id']
        crown = gpd.read_file(output / 'Segmentation3' / f'crowns_{stem}.gpkg')
        tops = gpd.read_file(output / 'Segmentation3' / f'ttops_{stem}.gpkg')
        if set(crown.treeID) != set(tops.treeID) or not crown.is_valid.all():
            raise AssertionError(f'Export ID or geometry mismatch: {stem}')
        polygon_ids.update(int(value) for value in crown.treeID)
    if polygon_ids != point_ids:
        raise AssertionError('Polygon and point instance IDs differ')
    report = dict(**setup, signature=signature, windows=windows,
                  candidates=len(raw['object_score']), instances=len(core),
                  export_seconds=time.monotonic() - started,
                  point_vectors=point_vectors, benchmark_vectors=benchmark_vectors,
                  clouds=clouds, exact_instance_id_match=True,
                  warning='Thresholds selected on 20m validation crops; whole-scene quality and timing are not directly comparable to output_21')
    (output / 'inference_report.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'README.md').write_text(
        '# EZ-SP wide q128 experimental inference\n\n'
        'This is the epoch-12 wide EZ-SP crop checkpoint transferred to the same four ALS tiles as output_21. '
        'PointClouds has original points and XYZ; tree_id=0 is unassigned. '
        'Segmentation3 and PointHead/Segmentation3 contain identical crown/treetop IDs. '
        'Crown polygons are filled 0.5 m raster supports reconstructed from point masks; '
        'this model has no independent full-crown raster head. '
        'The 20 m crop thresholds and global window merger have not been validated on whole scenes. '
        'See inference_report.json for checkpoint hashes, parameters, and counts.\n')
    print(json.dumps(dict(output=str(output), windows=windows, instances=len(core),
                          clouds=len(clouds), export_seconds=report['export_seconds']), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--prepared-dir', type=Path, default=DEFAULT_PREPARED)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--max-points', type=int, default=12000)
    parser.add_argument('--stage', choices=('predict', 'export', 'all'), default='all')
    parser.add_argument('--max-stripes', type=int, default=0,
                        help='Prediction smoke test only; positive value processes this many new x-stripes')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.max_points < 128 or (args.max_stripes and args.stage != 'predict'):
        parser.error('Use at least 128 points and --max-stripes only with --stage predict')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    torch.set_num_threads(4)
    output = args.output_dir.resolve()
    if (output / 'inference_report.json').exists():
        raise FileExistsError(f'Completed inference output is protected: {output}')
    checkpoint = args.checkpoint.resolve()
    prepared = args.prepared_dir.resolve()
    payload, initial, embedding, setup, signature = configuration(checkpoint, prepared, args.max_points)
    output.mkdir(parents=True, exist_ok=True)
    run_file = output / 'work/run.json'
    run_file.parent.mkdir(exist_ok=True)
    if run_file.exists():
        if json.loads(run_file.read_text())['signature'] != signature:
            raise ValueError(f'Existing output has a different run signature: {run_file}')
    else:
        run_file.write_text(json.dumps(dict(**setup, signature=signature), indent=2) + '\n')
    arrays, metadata = load_inputs(prepared)
    xs = starts_for_axis(float(arrays['coord'][:, 0].min()),
                         float(arrays['coord'][:, 0].max()), 20., 8.)
    if args.stage in ('predict', 'all'):
        predict_stripes(arrays, output, payload, initial, embedding, signature,
                        args.max_points, args.device, args.max_stripes)
    if args.stage in ('export', 'all'):
        export_scene(arrays, metadata, output, signature, len(xs), setup)


if __name__ == '__main__':
    main()
