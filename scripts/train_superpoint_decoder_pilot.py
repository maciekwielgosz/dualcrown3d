#!/usr/bin/env python3
"""Train/evaluate revisable point-instance decoders on matched cached ALS crops.

This evaluates actual predictions, never oracle majority labels. Frozen LitePT
features are reused; this is a crop pilot, not a full-scene deployment benchmark.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sys
import time

import geopandas as gpd
import laspy
import numpy as np
from openpyxl import Workbook
from pyproj import CRS
from rasterio.features import shapes
from rasterio.transform import Affine
from scipy.ndimage import binary_closing, binary_fill_holes
import shapely
from shapely.geometry import Point, shape
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.dual_head import dense_losses
from pointcloud.instance_output import point_instance_metrics
from pointcloud.superpoints.decoder import assign_masks
from pointcloud.superpoints.model import CachedInstanceModel
from scripts.benchmark_superpoint_algorithms import save_json, sha256
from scripts.evaluate_combined_full_crowns import aggregate, metrics

DEFAULT = PROJECT / 'outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot'


@lru_cache(maxsize=128)
def load_crop(path):
    with np.load(path) as file:
        return {key: file[key] for key in file.files}


def batch_for(arrays, method):
    batch = {key: torch.from_numpy(arrays[key]).cuda() for key in
             ('coord', 'feat', 'tree_id', 'semantic_target', 'instance_offset')}
    batch['feature'] = torch.from_numpy(arrays['feature'].astype(np.float32)).cuda()
    if method in ('fixed', 'ezsp'):
        for key, source in [('groups', 'groups'), ('centers', 'centers'), ('group_edges', 'edges')]:
            batch[key] = torch.from_numpy(arrays[f'{method}_{source}']).cuda()
    return batch


def crown_records(arrays, labels, confidence):
    """Filled raster footprints per predicted ID, preserving disconnected parts.

    Each ID uses the same point support as the LAZ; there is no convex hull
    across separated components. A 0.5m raster closing fills sampling holes.
    """
    xy = arrays['world_xy']
    resolution = .5
    records = []
    for identifier in np.unique(labels[labels > 0]):
        indices = np.flatnonzero(labels == identifier)
        coordinates = xy[indices]
        origin = np.floor(coordinates.min(0) / resolution) * resolution
        grid = np.floor((coordinates - origin) / resolution).astype(int)
        occupancy = np.zeros((grid[:, 1].max() + 5, grid[:, 0].max() + 5), bool)
        occupancy[grid[:, 1] + 2, grid[:, 0] + 2] = True
        filled = binary_fill_holes(binary_closing(occupancy, structure=np.ones((3, 3))) | occupancy)
        transform = Affine(resolution, 0, origin[0] - 2 * resolution,
                           0, resolution, origin[1] - 2 * resolution)
        pieces = [shape(geom) for geom, value in shapes(filled.astype(np.uint8), mask=filled, transform=transform) if value]
        geometry = shapely.union_all(pieces)
        if geometry.is_empty:
            continue
        top = indices[np.argmax(arrays['coord'][indices, 2])]
        records.append(dict(tree_id=int(identifier), geometry=geometry, points=len(indices),
                            confidence=float(confidence[indices].mean()),
                            height=float(arrays['coord'][top, 2]), top_x=float(xy[top, 0]), top_y=float(xy[top, 1])))
    return records


def point_diagnostics(truth, labels):
    result = point_instance_metrics(truth, labels)
    unique, count = np.unique(truth[truth > 0], return_counts=True)
    sparse = unique[count <= 100]
    known = truth >= 0
    hits = 0
    for identifier in sparse:
        target = truth == identifier
        choices, intersection = np.unique(labels[target], return_counts=True)
        for choice, overlap in zip(choices, intersection):
            if choice > 0 and overlap / ((target | ((labels == choice) & known)).sum()) >= .5:
                hits += 1
                break
    result.update(sparse_tree_tp=hits, sparse_tree_count=len(sparse),
                  annotated_tree_points=int((truth > 0).sum()),
                  assigned_tree_points=int(((truth > 0) & (labels > 0)).sum()))
    return result


def configurations():
    return {f'o{obj:g}_m{mask:g}': dict(object_threshold=obj, mask_threshold=mask,
                                      minimum_points=8, duplicate_iou=.6)
            for obj in (.1, .25, .4) for mask in (.4, .5, .6)}


@torch.no_grad()
def evaluate(model, entries, configs):
    model.eval()
    rows = {key: [] for key in configs}
    for entry in entries:
        arrays = load_crop(entry['cache'])
        data = batch_for(arrays, model.method)
        torch.cuda.synchronize()
        started = time.monotonic()
        output = model(data)['instance_masks']
        probability = output['mask_logits'].sigmoid().cpu().numpy()
        objects = output['object_logits'].sigmoid().cpu().numpy()
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        gt = [shapely.from_wkb(bytes.fromhex(value)) for value in entry['gt_wkb']]
        ignore = shapely.from_wkb(bytes.fromhex(entry['ignore_wkb']))
        for name, config in configs.items():
            start = time.monotonic()
            labels, confidence, _ = assign_masks(probability, objects, **config)
            polygons = crown_records(arrays, labels, confidence)
            record = {key: entry[key] for key in ('dataset_id', 'source_dataset', 'collection')}
            record.update(point=point_diagnostics(arrays['tree_id'], labels),
                          crown=metrics(gt, [r['geometry'] for r in polygons], ignore),
                          decoder_seconds=seconds, assignment_export_geometry_seconds=time.monotonic() - start,
                          predicted_instances=len(polygons))
            if model.method in ('fixed', 'ezsp'):
                groups = arrays[f'{model.method}_groups']
                pairs = np.unique(np.column_stack((groups, labels)), axis=0)
                mixed = np.bincount(pairs[:, 0], minlength=int(groups.max()) + 1) > 1
                record['groups_split_by_final_prediction'] = int(mixed.sum())
            rows[name].append(record)
        del data, output
    results = {}
    for name, per_plot in rows.items():
        point = aggregate([{**row, **row['point']} for row in per_plot])
        crown = aggregate([{**row, **row['crown']} for row in per_plot])
        sparse_count = sum(row['point']['sparse_tree_count'] for row in per_plot)
        total = sum(row['point']['annotated_tree_points'] for row in per_plot)
        results[name] = dict(config=configs[name], point=point, crown=crown, per_plot=per_plot,
            sparse_tree_recall_le100_voxels=(sum(row['point']['sparse_tree_tp'] for row in per_plot) / sparse_count if sparse_count else None),
            annotated_tree_point_coverage=sum(row['point']['assigned_tree_points'] for row in per_plot) / max(total, 1),
            score=math.sqrt(point['source_balanced_pq'] * crown['source_balanced_pq']))
    return results


def write_excel(root):
    sheets = dict(comparison=[], configurations=[], epochs=[], validation_grid=[], per_plot=[], artifacts=[])
    for run in sorted((root / 'runs').glob('*')):
        config_file = run / 'configuration.json'
        if not config_file.exists():
            continue
        config = json.loads(config_file.read_text())
        sheets['configurations'].append(config)
        history = run / 'history.json'
        if history.exists():
            sheets['epochs'].extend(dict(method=run.name, **row) for row in json.loads(history.read_text())['epochs'])
        selected = run / 'selected.json'
        if selected.exists():
            selection = json.loads(selected.read_text())
            value = selection['validation']
            sheets['comparison'].append(dict(method=run.name, epoch=selection['epoch'],
                point_pq=value['point']['source_balanced_pq'], point_f1=value['point']['source_balanced_f1'],
                crown_pq=value['crown']['source_balanced_pq'], crown_f1=value['crown']['source_balanced_f1'],
                sparse_tree_recall=value['sparse_tree_recall_le100_voxels'], coverage=value['annotated_tree_point_coverage'],
                checkpoint=selection['checkpoint'], score=value['score']))
            sheets['per_plot'].extend(dict(method=run.name, **row) for row in value['per_plot'])
            sheets['artifacts'].append(dict(method=run.name, checkpoint=selection['checkpoint'],
                                            exports=str(run / 'validation_exports')))
        for path in sorted((run / 'validation').glob('epoch_*.json')):
            epoch = int(path.stem.split('_')[-1])
            for name, value in json.loads(path.read_text()).items():
                sheets['validation_grid'].append(dict(method=run.name, epoch=epoch, thresholds=name,
                    score=value['score'], point_pq=value['point']['source_balanced_pq'],
                    crown_pq=value['crown']['source_balanced_pq'], point_f1=value['point']['source_balanced_f1']))
    book = Workbook()
    book.remove(book.active)
    for name, rows in sheets.items():
        sheet = book.create_sheet(name)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        if fields:
            sheet.append(fields)
            for row in rows:
                sheet.append([json.dumps(row.get(key)) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in fields])
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
    tmp = root / 'experiments.tmp.xlsx'
    book.save(tmp)
    os.replace(tmp, root / 'experiments.xlsx')
    save_json(root / 'comparison.json', dict(comparison=sheets['comparison']))


@torch.no_grad()
def export_predictions(model, entries, selection, folder):
    folder.mkdir(parents=True)
    model.eval()
    artifacts = []
    for entry in entries:
        arrays = load_crop(entry['cache'])
        output = model(batch_for(arrays, model.method))['instance_masks']
        labels, confidence, _ = assign_masks(output['mask_logits'].sigmoid().cpu().numpy(),
            output['object_logits'].sigmoid().cpu().numpy(), **selection['validation']['config'])
        records = crown_records(arrays, labels, confidence)
        key = entry['dataset_id']
        point_dir, polygon_dir = folder / 'PointClouds', folder / 'Segmentation3'
        point_dir.mkdir(exist_ok=True)
        polygon_dir.mkdir(exist_ok=True)
        header = laspy.LasHeader(point_format=3, version='1.2')
        header.offsets = np.array([*arrays['world_xy'].min(0), 0.])
        header.scales = np.array([.001, .001, .001])
        if entry['crs']:
            header.add_crs(CRS.from_user_input(entry['crs']))
        for name, dtype in [('tree_id', np.uint32), ('reference_tree_id', np.int32),
                            ('height_agl', np.float32), ('confidence', np.float32)]:
            header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
        cloud = laspy.LasData(header)
        cloud.x, cloud.y, cloud.z = arrays['world_xy'][:, 0], arrays['world_xy'][:, 1], arrays['coord'][:, 2]
        cloud.tree_id, cloud.reference_tree_id = labels.astype(np.uint32), arrays['tree_id'].astype(np.int32)
        cloud.height_agl, cloud.confidence = arrays['coord'][:, 2], confidence
        colors = np.random.default_rng(20261001).integers(6000, 65000, size=(int(labels.max()) + 1, 3), dtype=np.uint16)
        colors[0] = 16000
        cloud.red, cloud.green, cloud.blue = colors[labels].T
        cloud.write(point_dir / f'trees_{key}.laz')
        if records:
            frame = gpd.GeoDataFrame(records, geometry='geometry', crs=entry['crs'])
            frame.to_file(polygon_dir / f'crowns_{key}.gpkg', driver='GPKG')
            tops = frame.copy()
            tops.geometry = [Point(row['top_x'], row['top_y']) for row in records]
            tops.to_file(polygon_dir / f'ttops_{key}.gpkg', driver='GPKG')
        artifacts.append(dict(dataset_id=key, points=len(labels), instances=len(records),
                              laz=str(point_dir / f'trees_{key}.laz'), vectors_written=bool(records)))
    save_json(folder / 'export_manifest.json', dict(artifacts=artifacts,
        coordinates='XY in source CRS; Z is normalized height AGL, not absolute elevation',
        sampling='20m validation crops, downsampled input voxels (max12000), not full source clouds',
        colors='categorical prediction tree IDs; gray=unassigned; reference_tree_id is validation truth',
        polygon_method='0.5m occupancy, 3x3 closing and hole fill, all components retained',
        checkpoint=selection['checkpoint']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT)
    p.add_argument('--method', choices=('control', 'fixed', 'ezsp', 'retained'), required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--eval-every', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261001)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--run-name', type=str)
    p.add_argument('--queries', type=int, default=96)
    p.add_argument('--memory-tokens', type=int, default=1024)
    p.add_argument('--graph-width', type=int, default=96)
    p.add_argument('--graph-layers', type=int, default=3)
    p.add_argument('--wide-dim', type=int, default=0)
    p.add_argument('--wide-layers', type=int, default=2)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root = args.output.resolve()
    prepared = json.loads((root / 'prepared.json').read_text())
    train = [r for r in prepared['entries'] if r['split'] == 'train']
    val = [r for r in prepared['entries'] if r['split'] == 'val']
    if {r['group_id'] for r in train} & {r['group_id'] for r in val}:
        raise ValueError('Train/validation group leakage')
    model = CachedInstanceModel(args.method, queries=args.queries, memory_tokens=args.memory_tokens,
                                graph_width=args.graph_width, graph_layers=args.graph_layers,
                                wide_dim=args.wide_dim, wide_layers=args.wide_layers).cuda()
    model.initialize(torch.load(prepared['initial_checkpoint'], map_location='cpu', weights_only=False)['model'])
    model.decoder.epoch = 100
    if args.smoke:
        model.train()
        batch = batch_for(load_crop(train[0]['cache']), args.method)
        losses = dense_losses(model(batch), batch)
        losses['loss'].backward()
        if not all(torch.isfinite(v).all() for v in losses.values()):
            raise FloatingPointError('Nonfinite smoke loss')
        print(json.dumps(dict(method=args.method, loss=float(losses['loss'].detach()),
                              gpu=torch.cuda.get_device_name(), peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2)), flush=True)
        return
    run_name = args.run_name or args.method
    if not run_name.replace('_', '').isalnum():
        raise ValueError('Run name must contain only letters, digits and underscores')
    run = root / 'runs' / run_name
    if run.exists():
        raise FileExistsError(f'Protected training run: {run}')
    (run / 'weights').mkdir(parents=True)
    (run / 'validation').mkdir()
    config = dict(method=args.method, run_name=run_name, seed=args.seed,
        epochs=args.epochs if args.method != 'retained' else 0,
        train_crops=len(train), val_crops=len(val), initial_checkpoint=prepared['initial_checkpoint'],
        initial_sha256=prepared['initial_sha256'], architecture=('frozen LitePT72 -> ' +
            (f'{args.graph_layers} graph layers width{args.graph_width} -> ' if args.method in ('fixed', 'ezsp') else '') +
            f'128D point masks, {args.queries} queries, 3 masked transformer layers, {args.memory_tokens} memory tokens' +
            (f', {args.wide_layers} extra decoder layers width{args.wide_dim}' if args.wide_dim else '')),
        lr_decoder=1e-4, lr_graph=3e-4, weight_decay=.02,
        graph_width=args.graph_width if args.method in ('fixed', 'ezsp') else 0,
        wide_dim=args.wide_dim, wide_layers=args.wide_layers,
        queries=args.queries, memory_tokens=args.memory_tokens,
        graph_layers=args.graph_layers if args.method in ('fixed', 'ezsp') else 0,
        initialization_exposure='retained encoder previously exposed to HELIOS',
        selection='source-balanced geometric mean point/crown PQ on validation; 9 equal threshold options per variant',
        test_used=False, crop_m=20., gpu=torch.cuda.get_device_name(),
        point_refinement='original per-point features retained; masks not constrained to partition IDs',
        training_augmentation='none: frozen deterministic feature crop pilot',
        code_sha256={name: sha256(PROJECT / name) for name in (
            'pointcloud/superpoints/decoder.py', 'pointcloud/superpoints/model.py',
            'pointcloud/superpoints/wide_decoder.py',
            'scripts/train_superpoint_decoder_pilot.py')})
    save_json(run / 'configuration.json', config)
    params = [dict(params=[p for name, p in model.decoder.named_parameters() if not name.startswith('graph.')], lr=1e-4)]
    if hasattr(model.decoder, 'graph'):
        params.append(dict(params=model.decoder.graph.parameters(), lr=3e-4))
    optimizer = torch.optim.AdamW(params, weight_decay=.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=5e-6)
    history = []
    best = -1.
    selected = None
    epochs = 0 if args.method == 'retained' else args.epochs
    for epoch in range(epochs + 1):
        record = dict(epoch=epoch)
        if epoch:
            model.train()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            values = []
            for index in np.random.default_rng(args.seed + epoch).permutation(len(train)):
                batch = batch_for(load_crop(train[index]['cache']), args.method)
                optimizer.zero_grad(set_to_none=True)
                losses = dense_losses(model(batch), batch)
                if not torch.isfinite(losses['loss']):
                    raise FloatingPointError(f'Nonfinite loss epoch={epoch} plot={index}')
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 2., error_if_nonfinite=True)
                optimizer.step()
                values.append({key: float(value.detach()) for key, value in losses.items()})
            scheduler.step()
            torch.cuda.synchronize()
            record.update(seconds=time.monotonic() - started, peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2,
                          **{key: float(np.mean([row[key] for row in values])) for key in values[0]})
        if epoch == 0 or epoch % args.eval_every == 0 or epoch == epochs:
            validation = evaluate(model, val, configurations())
            save_json(run / 'validation' / f'epoch_{epoch:03d}.json', validation)
            name = max(validation, key=lambda key: validation[key]['score'])
            value = validation[name]
            record.update(validation_score=value['score'], point_pq=value['point']['source_balanced_pq'],
                          crown_pq=value['crown']['source_balanced_pq'], thresholds=name)
            if value['score'] > best:
                best = value['score']
                checkpoint = run / 'weights/best.pt'
                torch.save(dict(model=model.state_dict(), config=config, epoch=epoch,
                                initial_checkpoint=prepared['initial_checkpoint']), checkpoint)
                selected = dict(epoch=epoch, validation=value, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
                save_json(run / 'selected.json', selected)
        history.append(record)
        save_json(run / 'history.json', dict(epochs=history))
        torch.save(dict(model=model.state_dict(), config=config, epoch=epoch,
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), run / 'weights/last.pt')
        write_excel(root)
        print(json.dumps(dict(method=args.method, **record)), flush=True)
    model.load_state_dict(torch.load(selected['checkpoint'], map_location='cuda', weights_only=False)['model'])
    export_predictions(model, val, selected, run / 'validation_exports')
    save_json(run / 'DONE.json', dict(selected_epoch=selected['epoch'], score=best, completed=True))
    write_excel(root)


if __name__ == '__main__':
    main()
