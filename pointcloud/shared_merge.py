"""Pairwise scene stitching for shared point and crown queries.

Candidate overlap is measured against individual instances. Competing instances
may keep separate support; only the disputed voxels choose their best score.
"""
from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, label as components
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import MultiPoint, Polygon, shape
from shapely.ops import unary_union


def _candidate(raw, index, arrays, config):
    if raw['object_score'][index] < config['object_threshold']:
        return None
    if raw['candidate_quality'][index] < config['quality_threshold']:
        return None
    a, b = raw['candidate_offset'][index:index+2]
    score = np.asarray(raw['point_score'][a:b], np.float32)
    keep = score >= config['mask_threshold']
    ids = np.asarray(raw['point_index'][a:b][keep], np.int32)
    score = score[keep]
    if len(ids) < config['minimum_voxels']:
        return None
    xyz = arrays['coord'][ids]
    if xyz[:, 2].max() < config.get('minimum_height_m', 2.):
        return None
    hull = MultiPoint(xyz[:, :2]).convex_hull.buffer(float(arrays['voxel_size'])/2.)
    if hull.area < config.get('minimum_area_m2', .75):
        return None
    ca, cb = raw['crown_offset'][index:index+2]
    crown_score = np.asarray(raw['crown_score'][ca:cb], np.float32)
    crown_keep = crown_score >= config['crown_threshold']
    return dict(indices=ids, scores=score, center=np.median(xyz[:, :2], axis=0),
                cells=np.asarray(raw['crown_cell'][ca:cb][crown_keep], np.int64),
                cell_scores=crown_score[crown_keep],
                object_score=float(raw['object_score'][index]),
                quality_score=float(raw['candidate_quality'][index]),
                weight=float(raw['object_score'][index] * raw['candidate_quality'][index]),
                ranking=float(raw['object_score'][index] * raw['candidate_quality'][index] * score.mean()))


def _intersection(a, b):
    return len(np.intersect1d(a, b, assume_unique=True))


def _geometry_from_cells(cells, raw):
    if not len(cells):
        return None
    width = int(raw['crown_grid_width'])
    resolution = float(raw['crown_grid_size'])
    xy = np.column_stack((cells % width, cells // width)).astype(np.int64)
    xy -= xy.min(0)
    x0, y0 = int((cells % width).min()), int((cells // width).min())
    occupancy = np.zeros((int(xy[:, 1].max())+1, int(xy[:, 0].max())+1), np.uint8)
    occupancy[xy[:, 1], xy[:, 0]] = 1
    padded = np.pad(occupancy, 1)
    closed = binary_closing(padded, structure=np.ones((3, 3)), border_value=0)
    closed = binary_fill_holes(closed)
    labels, count = components(closed)
    if not count:
        return None
    largest = int(np.argmax(np.bincount(labels[labels > 0])))
    mask = (labels[1:-1, 1:-1] == largest).astype(np.uint8)
    if not mask.any():
        return None
    origin = raw['crown_grid_origin']
    transform = Affine(resolution, 0, float(origin[0]) + x0*resolution,
                       0, resolution, float(origin[1]) + y0*resolution)
    polygons = [shape(geom) for geom, value in shapes(mask, mask=mask.astype(bool), transform=transform) if value]
    return unary_union(polygons) if polygons else None


def merge_shared_queries(arrays, raw, config):
    """Return global IDs, point confidence, filled crown records and provenance."""
    required = {'candidate_quality', 'crown_offset', 'crown_cell', 'crown_score',
                'crown_grid_origin', 'crown_grid_width', 'crown_grid_size'}
    if missing := required.difference(raw):
        raise ValueError(f'Shared-query predictions missing: {sorted(missing)}')
    if not 0 < config['duplicate_iou'] < 1 or not 0 < config['minimum_unique_fraction'] <= 1:
        raise ValueError('Invalid duplicate or unique support threshold')
    key = (config['mask_threshold'], config['crown_threshold'], config['minimum_voxels'],
           config.get('minimum_height_m', 2.), config.get('minimum_area_m2', .75))
    cached = raw.get('_shared_candidate_cache')
    if cached is None or cached[0] != key:
        loose = {**config, 'object_threshold': 0., 'quality_threshold': 0.}
        base = [_candidate(raw, i, arrays, loose) for i in range(len(raw['object_score']))]
        base = [candidate for candidate in base if candidate is not None]
        raw['_shared_candidate_cache'] = (key, base)
    else:
        base = cached[1]
    candidates = [candidate for candidate in base
                  if candidate['object_score'] >= config['object_threshold']
                  and candidate['quality_score'] >= config['quality_threshold']]
    candidates.sort(key=lambda c: -c['ranking'])
    groups = []
    spatial = defaultdict(list)
    min_points = config['minimum_voxels']
    cell_m = config.get('spatial_index_m', 2.)
    for candidate in candidates:
        center = candidate['center']
        cell = np.floor(center/cell_m).astype(int)
        nearby = set()
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nearby.update(spatial[(int(cell[0]+dx), int(cell[1]+dy))])
        duplicate = None
        for gid in nearby:
            group = groups[gid]
            if np.linalg.norm(center-group['center']) > config.get('duplicate_distance_m', 2.):
                continue
            overlap = _intersection(candidate['indices'], group['support'])
            union = len(candidate['indices'])+len(group['support'])-overlap
            if overlap/max(union, 1) >= config['duplicate_iou'] or (
                    overlap/max(min(len(candidate['indices']),len(group['support'])),1)
                    >= config.get('duplicate_containment', .8)):
                duplicate = gid
                break
        if duplicate is not None:
            group = groups[duplicate]
            group['proposals'].append(candidate)
            group['support'] = np.union1d(group['support'], candidate['indices'])
            continue
        nearby_support = np.concatenate([groups[gid]['support'] for gid in nearby]) if nearby else np.empty(0,np.int32)
        unique = np.setdiff1d(candidate['indices'], nearby_support, assume_unique=False)
        if len(unique) < min_points or len(unique)/len(candidate['indices']) < config['minimum_unique_fraction']:
            continue
        if MultiPoint(arrays['coord'][unique, :2]).convex_hull.buffer(
                float(arrays['voxel_size'])/2.).area < config.get('minimum_area_m2', .75):
            continue
        gid = len(groups)
        groups.append(dict(center=center, support=candidate['indices'],
                           proposals=[candidate]))
        spatial[(int(cell[0]),int(cell[1]))].append(gid)

    def assign(active):
        ids = np.zeros(len(arrays['coord']), np.uint32)
        confidence = np.zeros(len(ids), np.float32)
        for gid in active:
            for proposal in groups[gid]['proposals']:
                member = proposal['indices']
                offers = proposal['weight']*proposal['scores']
                better = offers > confidence[member]
                ids[member[better]] = gid+1
                confidence[member[better]] = offers[better]
        return ids, confidence

    active = list(range(len(groups)))
    for _ in range(2):
        ids, confidence = assign(active)
        valid = []
        for gid in active:
            member = np.flatnonzero(ids == gid+1)
            if len(member) < min_points or arrays['coord'][member, 2].max() < config.get('minimum_height_m', 2.):
                continue
            if MultiPoint(arrays['coord'][member, :2]).convex_hull.buffer(
                    float(arrays['voxel_size'])/2.).area >= config.get('minimum_area_m2', .75):
                valid.append(gid)
        if len(valid) == len(active):
            break
        active = valid
    ids, confidence = assign(active)
    dense = np.zeros(len(groups)+1, np.uint32)
    dense[np.asarray(active,dtype=np.int64)+1] = np.arange(1,len(active)+1,dtype=np.uint32)
    ids = dense[ids]
    confidence[ids == 0] = 0.
    records = []
    origin = arrays['source_origin'][:2]
    for gid in active:
        instance_id = int(dense[gid+1])
        member = np.flatnonzero(ids == instance_id)
        if not len(member):
            continue
        xyz = arrays['coord'][member]
        top = int(np.argmax(xyz[:, 2]))
        hull = MultiPoint(xyz[:, :2]).convex_hull.buffer(float(arrays['voxel_size'])/2.)
        crown_cells = np.concatenate([p['cells'] for p in groups[gid]['proposals']])
        head = _geometry_from_cells(np.unique(crown_cells),raw)
        mode = config.get('crown_output', 'head_union_support')
        if mode == 'support_hull' or head is None:
            geometry = hull
        elif mode == 'head':
            geometry = head
        elif mode == 'head_union_support':
            geometry = head.union(hull)
        else:
            raise ValueError(f'Unknown crown_output: {mode}')
        if geometry.geom_type == 'MultiPolygon':
            geometry = max(geometry.geoms,key=lambda g:g.area)
        geometry = Polygon(geometry.exterior)
        # Raw raster cell coordinates are scene-relative, just like point coordinates.
        from shapely.affinity import translate
        geometry = translate(geometry, xoff=float(origin[0]), yoff=float(origin[1]))
        records.append(dict(tree_id=instance_id, geometry=geometry,
                            confidence=float(confidence[member].mean()), points=len(member),
                            height=float(xyz[top,2]), top_x=float(xyz[top,0]+origin[0]),
                            top_y=float(xyz[top,1]+origin[1])))
    return ids, confidence, records, (ids>0).astype(np.uint8)
