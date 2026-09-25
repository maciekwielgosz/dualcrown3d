#!/usr/bin/env python3
"""Single-backbone dual-head inference, GIS outputs and full-resolution LAZ."""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import torch
from pyproj import CRS
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Point

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.dual_head import DualHeadLitePT
from pointcloud.instance_output import merge_masks_with_sources
from scripts.evaluate_pointcloud_litept import starts_for_axis, ownership_intervals, model_input, cluster_candidates, filter_candidates, dbh_naslund


def predict(model, arrays, max_points=40000, owner_only=True):
    start = time.monotonic()
    xyz = arrays['coord']
    size, overlap = 20., 8.
    xs = starts_for_axis(float(xyz[:, 0].min()), float(xyz[:, 0].max()), size, overlap)
    ys = starts_for_axis(float(xyz[:, 1].min()), float(xyz[:, 1].max()), size, overlap)
    # Half-open ownership at the upper edge must survive float32 conversion.
    xo = ownership_intervals(xs, size, float(xyz[:, 0].min()), float(np.nextafter(xyz[:, 0].max(), np.float32(np.inf))))
    yo = ownership_intervals(ys, size, float(xyz[:, 1].min()), float(np.nextafter(xyz[:, 1].max(), np.float32(np.inf))))
    index = cKDTree(xyz[:, :2])
    old_prob = np.zeros(len(xyz), np.float32)
    old_center = np.zeros_like(xyz)
    point_prob = np.zeros(len(xyz), np.float32)
    visited = np.zeros(len(xyz), np.uint8)
    offsets, members, probabilities, scores = [0], [], [], []
    device = next(model.parameters()).device
    rng = np.random.default_rng(20260925)
    windows = 0
    model.eval()
    with torch.no_grad():
        for xi, x in enumerate(xs):
            for yi, y in enumerate(ys):
                context = np.asarray(sorted(index.query_ball_point([x+10., y+10.], 10.0001, p=np.inf)), dtype=np.int64)
                if not len(context):
                    continue
                context = context[(xyz[context, 0] >= x) & (xyz[context, 0] <= x+size)
                                  & (xyz[context, 1] >= y) & (xyz[context, 1] <= y+size)]
                a, b = xo[xi]
                c, d = yo[yi]
                owner = context[(xyz[context, 0] >= a) & (xyz[context, 0] < b)
                                & (xyz[context, 1] >= c) & (xyz[context, 1] < d)]
                if not len(owner):
                    continue
                # Cover every owned voxel even when a dense window exceeds capacity.
                for owned in np.array_split(owner, max(1, math.ceil(len(owner)/max_points))):
                    extra = np.setdiff1d(context, owned, assume_unique=True)
                    capacity = max_points - len(owned)
                    if len(extra) > capacity:
                        extra = rng.choice(extra, capacity, replace=False)
                    chosen = np.sort(np.concatenate((owned, extra)))
                    own_local = np.flatnonzero(np.isin(chosen, owned))
                    batch = model_input(xyz[chosen], arrays['grid_coord'][chosen], arrays['intensity'][chosen], device, preserve_height=True)
                    result = model(batch)
                    old_prob[owned] = result['semantic_logits'].softmax(1)[own_local, 1].cpu().numpy()
                    old_center[owned] = xyz[owned] + result['offset_m'][own_local].cpu().numpy()
                    point_prob[owned] = result['point_semantic_logits'].softmax(1)[own_local, 1].cpu().numpy()
                    visited[owned] += 1
                    masks = result['instance_masks']['mask_logits'].sigmoid().cpu().numpy()
                    quality = result['instance_masks']['object_logits'].sigmoid().cpu().numpy()
                    for q in np.flatnonzero(quality >= .05):
                        retained = np.flatnonzero(masks[q] >= .2)
                        if len(retained) < 4:
                            continue
                        center = np.average(xyz[chosen[retained], :2], axis=0, weights=masks[q, retained])
                        if owner_only and not (a <= center[0] < b and c <= center[1] < d):
                            continue
                        members.append(chosen[retained].astype(np.int32))
                        probabilities.append(masks[q, retained].astype(np.float16))
                        scores.append(float(quality[q]))
                        offsets.append(offsets[-1] + len(retained))
                    windows += 1
                if windows % 200 == 0:
                    print(f'{windows} windows; {np.count_nonzero(visited):,}/{len(xyz):,} owned voxels; {time.monotonic()-start:.1f}s', flush=True)
    if not np.all(visited == 1):
        raise AssertionError(f'Ownership failed: missing={int((visited==0).sum())}, repeated={int((visited>1).sum())}')
    return dict(tree_probability=old_prob, shifted_center=old_center, point_probability=point_prob,
                object_score=np.asarray(scores, np.float32), candidate_offset=np.asarray(offsets, np.int64),
                point_index=np.concatenate(members) if members else np.empty(0, np.int32),
                point_score=np.concatenate(probabilities) if probabilities else np.empty(0, np.float16),
                seconds=np.float64(time.monotonic()-start), windows=np.int64(windows))


def legacy_instances(arrays, raw, metadata, config):
    xy = arrays['coord'][:, :2] + arrays['source_origin'][:2]
    labels = np.zeros(len(xy), np.uint32)
    instances = []
    mapping = []
    for tile in metadata['tiles']:
        left, bottom, right, top = tile['bounds']
        take = np.flatnonzero((xy[:, 0] >= left-5) & (xy[:, 0] < right+5)
                             & (xy[:, 1] >= bottom-5) & (xy[:, 1] < top+5))
        subset = dict(coord=arrays['coord'][take], tree_probability=raw['tree_probability'][take],
                      shifted_center=raw['shifted_center'][take], source_origin=arrays['source_origin'],
                      voxel_size=arrays['voxel_size'], plot_max_z=arrays['coord'][take, 2].max())
        candidates = filter_candidates(cluster_candidates(subset, config, return_members=True), subset, config)
        selected = [p for p in candidates if p['height'] >= 2 and left <= p['top_x'] < right and bottom <= p['top_y'] < top]
        selected.sort(key=lambda p: (-p['top_y'], p['top_x']))
        for local, item in enumerate(selected, 1):
            identifier = len(instances) + 1
            member = take[item.pop('point_indices')]
            unassigned = labels[member] == 0
            labels[member[unassigned]] = identifier
            item.update(tree_id=identifier, local_tree_id=local, tile_id=tile['tile_id'])
            instances.append(item)
            mapping.append(dict(legacy_tree_id=identifier, tile_id=tile['tile_id'], gpkg_treeID=local))
    return labels, instances, mapping


def write_vectors(instances, metadata, folder, legacy=False):
    folder.mkdir(parents=True, exist_ok=True)
    report = []
    for tile in metadata['tiles']:
        left, bottom, right, top = tile['bounds']
        items = [p for p in instances if (p.get('tile_id') == tile['tile_id'] if legacy else
                 left <= p['top_x'] < right and bottom <= p['top_y'] < top)]
        identifiers = [p['local_tree_id'] if legacy else p['tree_id'] for p in items]
        crs = metadata['chm_crs']
        crowns = gpd.GeoDataFrame(dict(treeID=identifiers, area_m2=[p['geometry'].area for p in items]),
                                 geometry=[MultiPolygon([p['geometry']]) for p in items], crs=crs)
        tops = gpd.GeoDataFrame(dict(treeID=identifiers, Z=[p['height'] for p in items],
                                    dbh=[round(dbh_naslund(p['height']), 2) for p in items]),
                               geometry=[Point(p['top_x'], p['top_y'], p['height']) for p in items], crs=crs)
        crowns.to_file(folder / f"crowns_{tile['tile_id']}.gpkg", driver='GPKG', index=False)
        tops.to_file(folder / f"ttops_{tile['tile_id']}.gpkg", driver='GPKG', index=False)
        if not crowns.is_valid.all() or crowns.is_empty.any() or len(crowns) != len(tops):
            raise AssertionError('Invalid crown/treetop output')
        report.append(dict(tile_id=tile['tile_id'], crowns=len(crowns), treetops=len(tops)))
    return report


def color_ids(ids):
    value = np.asarray(ids, np.uint64) * np.uint64(2654435761)
    rgb = np.column_stack([48 + ((value >> shift) & 207) for shift in (0, 8, 16)]).astype(np.uint16) * 257
    rgb[ids == 0] = 18000
    return rgb


def export_laz(output, arrays, metadata, new_ids, old_ids, confidence, semantic, assignment_source=None):
    """Use the preparation grid's exact inverse, retaining original XYZ and LAS fields."""
    clouds = []
    left, bottom, right, top = metadata['processing_bounds']
    for source in metadata['als_files']:
        cloud = laspy.read(source)
        mask = ((cloud.x >= left) & (cloud.x < right) & (cloud.y >= bottom) & (cloud.y < top))
        cloud = laspy.LasData(cloud.header.copy(), points=cloud.points[mask].copy())
        clouds.append(cloud)
    xyz = np.concatenate([np.column_stack((c.x, c.y, c.z)) for c in clouds])
    classification = np.concatenate([np.asarray(c.classification) for c in clouds])
    resolution = metadata['ground_grid_size_m']
    width, height = math.ceil((right-left)/resolution), math.ceil((top-bottom)/resolution)
    cc = np.clip(np.floor((xyz[:, 0]-left)/resolution).astype(np.int64), 0, width-1)
    rr = np.clip(np.floor((xyz[:, 1]-bottom)/resolution).astype(np.int64), 0, height-1)
    cell = rr * width + cc
    ground = classification == 2
    sums = np.bincount(cell[ground], weights=xyz[ground, 2], minlength=height*width)
    counts = np.bincount(cell[ground], minlength=height*width)
    terrain = (sums/np.maximum(counts, 1)).reshape(height, width)
    nearest = distance_transform_edt(counts.reshape(height, width) == 0, return_distances=False, return_indices=True)
    terrain = terrain[tuple(nearest)]
    normalized = xyz.copy()
    normalized[:, 2] -= terrain[rr, cc]
    grid = np.floor((normalized-arrays['voxel_origin']) / float(arrays['voxel_size'])).astype(np.int64)
    source_grid = arrays['grid_coord'].astype(np.int64)
    extent = source_grid.max(0) + 1
    encode = lambda g: g[:, 0] + extent[0]*g[:, 1] + extent[0]*extent[1]*g[:, 2]
    keys = encode(source_grid)
    order = np.argsort(keys)
    wanted = encode(grid)
    positions = np.searchsorted(keys[order], wanted)
    good = positions < len(order)
    good[good] &= keys[order[positions[good]]] == wanted[good]
    if not good.all():
        raise AssertionError(f'Exact voxel inverse failed for {(~good).sum()} source points')
    inverse = order[positions]
    ids, legacy = new_ids[inverse].copy(), old_ids[inverse].copy()
    scores, classes = confidence[inverse].copy(), semantic[inverse].copy()
    sources = ((new_ids > 0).astype(np.uint8) if assignment_source is None else assignment_source)[inverse].copy()
    classes[ids > 0] = 1
    ids[ground] = 0
    legacy[ground] = 0
    scores[ground] = 0
    classes[ground] = 0
    sources[ids == 0] = 0
    status = np.where(ids > 0, 1, np.where(classes == 1, 2, 0)).astype(np.uint8)
    folder = output / 'PointClouds'
    folder.mkdir(parents=True, exist_ok=True)
    reports = []
    cursor = 0
    for source_number, cloud in enumerate(clouds):
        sl = slice(cursor, cursor+len(cloud.points))
        cursor += len(cloud.points)
        if 'red' not in set(cloud.point_format.dimension_names):
            formats = {0: 2, 1: 3, 4: 5, 6: 7, 9: 10}
            cloud = laspy.convert(cloud, point_format_id=formats[cloud.header.point_format.id])
        cloud.header.add_crs(CRS(metadata['chm_crs']))
        for name, dtype in [('tree_id', np.uint32), ('legacy_tree_id', np.uint32),
                            ('tree_confidence', np.float32), ('pred_semantic', np.uint8), ('height_agl', np.float32),
                            ('assignment_source', np.uint8), ('segmentation_status', np.uint8)]:
            if name in set(cloud.point_format.dimension_names):
                raise ValueError(f'Source already contains predicted field {name}')
            cloud.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
        cloud.tree_id, cloud.legacy_tree_id = ids[sl], legacy[sl]
        cloud.tree_confidence, cloud.pred_semantic = scores[sl], classes[sl]
        cloud.height_agl = normalized[sl, 2].astype(np.float32)
        cloud.assignment_source, cloud.segmentation_status = sources[sl], status[sl]
        colors = color_ids(ids[sl])
        cloud.red, cloud.green, cloud.blue = colors.T
        for tile in metadata['tiles']:
            x0, y0, x1, y1 = tile['bounds']
            take = (cloud.x >= x0) & (cloud.x < x1) & (cloud.y >= y0) & (cloud.y < y1)
            if not take.any():
                continue
            part = laspy.LasData(cloud.header.copy(), points=cloud.points[take].copy())
            suffix = f'_source{source_number}' if len(clouds) > 1 else ''
            path = folder / f"trees_{tile['tile_id']}{suffix}.laz"
            part.write(path)
            check = laspy.read(path)
            for dimension in ('X', 'Y', 'Z', 'tree_id', 'legacy_tree_id', 'classification', 'intensity',
                              'assignment_source', 'segmentation_status'):
                if not np.array_equal(np.asarray(check[dimension]), np.asarray(part[dimension])):
                    raise AssertionError(f'LAZ round-trip changed {dimension}')
            if check.header.parse_crs().to_epsg() != 2180:
                raise AssertionError('Unexpected LAZ CRS')
            reports.append(dict(file=str(path), points=len(part.points), labelled_points=int((part.tree_id > 0).sum()),
                                trees=len(np.unique(part.tree_id[part.tree_id > 0])), exact_voxel_mapping=True,
                                assignment_source={str(k): int((part.assignment_source == k).sum()) for k in range(6)},
                                unassigned_predicted_tree_points=int((part.segmentation_status == 2).sum())))
            print(f'LAZ: {path.name}: {len(part.points):,} original points', flush=True)
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=PROJECT / 'outputs/dual_head_satv2_litept_v3/weights/best.pt')
    p.add_argument('--selection', type=Path, default=PROJECT / 'configs/dual_head_complete_consensus.json')
    p.add_argument('--prepared-dir', type=Path, default=PROJECT / 'output_15_litept_v2_no_rectangles_pointcloud/work')
    p.add_argument('--output-dir', type=Path, default=PROJECT / 'output_19_dual_head_complete_consensus')
    p.add_argument('--stage', choices=('predict', 'export', 'all'), default='all')
    p.add_argument('--raw-predictions', type=Path,
                   help='Reuse an existing prediction NPZ; requires --stage export')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.raw_predictions is not None and args.stage != 'export':
        p.error('--raw-predictions requires --stage export')
    torch.set_num_threads(4)
    output = args.output_dir.resolve()
    work = output / 'work'
    work.mkdir(parents=True, exist_ok=True)
    with np.load(args.prepared_dir / 'benchmark_pointcloud.npz') as f:
        arrays = {k: f[k] for k in f.files}
    metadata = json.loads((args.prepared_dir / 'preparation.json').read_text())
    selection = json.loads(args.selection.read_text())
    if selection.get('acceptance_passed') is False:
        raise ValueError('Refusing a selection that failed validation acceptance')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != selection['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the validation-frozen selection')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    raw_path = args.raw_predictions.resolve() if args.raw_predictions else work / 'dual_predictions.npz'
    if args.stage in ('predict', 'all'):
        if raw_path.exists():
            raise FileExistsError(raw_path)
        if not torch.cuda.is_available() and args.device.startswith('cuda'):
            raise RuntimeError('CUDA required')
        model = DualHeadLitePT().to(args.device)
        model.load_state_dict(checkpoint['model'], strict=True)
        raw = predict(model, arrays, owner_only=selection['config'].get('owner_only', True))
        raw['checkpoint_sha256'] = np.asarray(selection['checkpoint_sha256'])
        np.savez_compressed(raw_path, **raw)
    if args.stage in ('export', 'all'):
        export_start = time.monotonic()
        if (output / 'inference_report.json').exists():
            raise FileExistsError('Completed outputs already exist')
        with np.load(raw_path) as f:
            raw = {k: f[k] for k in f.files}
        if str(raw['checkpoint_sha256']) != selection['checkpoint_sha256']:
            raise ValueError('Raw predictions use a different checkpoint')
        if len(raw['tree_probability']) != len(arrays['coord']):
            raise ValueError('Raw predictions and prepared point cloud have different sizes')
        new_ids, confidence, new_instances, assignment_source = merge_masks_with_sources(arrays, raw, selection['config'])
        # Apply the same core-tile ownership contract as the GIS outputs.
        core_instances = [p for p in new_instances if any(
            t['bounds'][0] <= p['top_x'] < t['bounds'][2]
            and t['bounds'][1] <= p['top_y'] < t['bounds'][3] for t in metadata['tiles'])]
        renumber = np.zeros(len(new_instances)+1, np.uint32)
        for i, item in enumerate(core_instances, 1):
            renumber[item['tree_id']] = i
            item['tree_id'] = i
        new_ids = renumber[new_ids]
        confidence[new_ids == 0] = 0.
        assignment_source[new_ids == 0] = 0
        new_instances = core_instances
        old_ids, old_instances, mapping = legacy_instances(arrays, raw, metadata, checkpoint['config']['legacy_cluster_config'])
        legacy_report = write_vectors(old_instances, metadata, output / 'Segmentation3', legacy=True)
        point_report = write_vectors(new_instances, metadata, output / 'PointHead/Segmentation3')
        with (output / 'legacy_tree_id_map.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['legacy_tree_id', 'tile_id', 'gpkg_treeID'])
            writer.writeheader()
            writer.writerows(mapping)
        clouds = export_laz(output, arrays, metadata, new_ids, old_ids, confidence,
                            (raw['point_probability'] >= .3).astype(np.uint8), assignment_source)
        report = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=selection['checkpoint_sha256'],
                      selection=str(args.selection.resolve()), raw_predictions=str(raw_path.resolve()),
                      reused_raw_predictions=args.raw_predictions is not None,
                      export_seconds=time.monotonic()-export_start,
                      mask_config=selection['config'], inference_seconds=float(raw['seconds']), windows=int(raw['windows']),
                      legacy_trees=len(old_instances), mask_trees=len(new_instances),
                      legacy_vectors=legacy_report, point_vectors=point_report, clouds=clouds,
                      original_xyz_preserved=True, source_classification_preserved=True,
                      point_ids_global_across_tiles=True, voxel_size=.25, window_size=20., overlap=8.)
        (output / 'inference_report.json').write_text(json.dumps(report, indent=2) + '\n')
        (output / 'README.md').write_text(
            '# Two-head LitePT / SATv2-inspired point masks\n\n'
            'Open `PointClouds/trees_*.laz` in CloudCompare and accept the proposed Global Shift. '
            'Choose Properties > Colors > RGB for per-tree colours. If displaying `tree_id` as a scalar field, '
            'restore its complete DISPLAYED range on each cloud; saturation alone does not restore hidden points. '
            '`tree_id = 0` means no assigned tree. Coordinates are original elevations, not normalized heights; '
            '`height_agl` provides the separate normalized height.\n\n'
            '- `tree_id`: final mask/consensus instance ID, globally unique across tiles.\n'
            '- `legacy_tree_id`: old branch instance ID; see `legacy_tree_id_map.csv` for per-tile polygon IDs.\n'
            '- `tree_confidence`: originating-head score (mask probability times quality, or semantic probability); '
            'interpret with assignment_source and the selected configuration. These scores are uncalibrated, '
            'not comparable accuracy estimates.\n'
            '- `pred_semantic`: 0 = background, 1 = tree; source `classification` is preserved.\n\n'
            '- `assignment_source`: 0 = unassigned, 1 = mask/support-fusion anchor, 2 = cross-head completion, '
            '3 = vote-head instance (anchor or added), 4 = local centre-consistent recovery, 5 = added mask-head instance.\n'
            '- `segmentation_status`: 0 = predicted background/ground, 1 = assigned instance, 2 = predicted tree without an instance. '
            'This is a model diagnostic, not ground truth.\n\n'
            '`Segmentation3` contains the legacy crowns/treetops. `PointHead/Segmentation3` contains '
            'filled polygons and treetops derived from the new point masks; their `treeID` matches LAZ `tree_id`. '
            'Both use EPSG:2180. The legacy vectors are unchanged; the final consensus uses evidence from both branches.\n\n'
            f'Mask postprocessing: `{selection["config"].get("merge_strategy", "legacy")}`. '
            'Support fusion, when selected, matches overlapping proposals to fixed tree anchors, '
            'recovers nearby unassigned points and rebuilds polygons from the final support. '
            'Dual consensus also matches centre-vote instances, can recover missing trees, and optionally performs '
            'one-pass spatial/centre-consistent completion. It does not force every point into a tree.\n\n'
            'This is a compact adaptation of https://arxiv.org/abs/2606.08206 with LitePT-S, '
            '96 ISA queries and 3 masked cross-attention layers. It is not a reproduction of the paper\'s metrics. '
            'See `inference_report.json` and the training run\'s `point_comparison.json`/`experiments.xlsx`.\n',
            encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
