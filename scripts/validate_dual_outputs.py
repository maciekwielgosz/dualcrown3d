#!/usr/bin/env python3
"""Check labelled LAZ against original XYZ/attributes and both GeoPackage branches."""
import argparse
import json
from pathlib import Path
import geopandas as gpd
import laspy
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=PROJECT / 'output_17_dual_head_support_fusion')
    p.add_argument('--preparation', type=Path, default=PROJECT / 'output_15_litept_v2_no_rectangles_pointcloud/work/preparation.json')
    args = p.parse_args()
    output = args.output_dir
    metadata = json.loads(args.preparation.read_text())
    vectors, point_ids, legacy_total = [], set(), 0
    for branch in ('Segmentation3', 'PointHead/Segmentation3'):
        for tile in metadata['tiles']:
            crowns = gpd.read_file(output / branch / f"crowns_{tile['tile_id']}.gpkg")
            tops = gpd.read_file(output / branch / f"ttops_{tile['tile_id']}.gpkg")
            assert crowns.crs.to_epsg() == tops.crs.to_epsg() == 2180
            assert crowns.is_valid.all() and not crowns.is_empty.any()
            assert set(crowns.treeID) == set(tops.treeID)
            assert crowns.treeID.is_unique and tops.treeID.is_unique
            assert np.allclose(crowns.area_m2, crowns.area)
            assert all(len(p.interiors) == 0 for g in crowns.geometry for p in g.geoms)
            assert tops.has_z.all()
            if branch.startswith('PointHead'):
                assert not point_ids.intersection(crowns.treeID)
                point_ids.update(crowns.treeID)
            else:
                legacy_total += len(crowns)
            vectors.append(dict(branch=branch, tile=tile['tile_id'], crowns=len(crowns), passed=True))
    clouds = []
    # Stream the source once per source file, retaining only the four target extents.
    for source_number, source_path in enumerate(metadata['als_files']):
        originals = {tile['tile_id']: [] for tile in metadata['tiles']}
        with laspy.open(source_path) as reader:
            for chunk in reader.chunk_iterator(2_000_000):
                x, y = np.asarray(chunk.x), np.asarray(chunk.y)
                for tile in metadata['tiles']:
                    left, bottom, right, top = tile['bounds']
                    inside = (x >= left) & (x < right) & (y >= bottom) & (y < top)
                    if inside.any():
                        originals[tile['tile_id']].append(np.column_stack((x[inside], y[inside], np.asarray(chunk.z)[inside],
                              np.asarray(chunk.intensity)[inside], np.asarray(chunk.classification)[inside])))
        for tile in metadata['tiles']:
            parts = originals[tile['tile_id']]
            if not parts:
                continue
            expected = np.concatenate(parts)
            suffix = f'_source{source_number}' if len(metadata['als_files']) > 1 else ''
            path = output / 'PointClouds' / f"trees_{tile['tile_id']}{suffix}.laz"
            cloud = laspy.read(path)
            actual = np.column_stack((cloud.x, cloud.y, cloud.z, cloud.intensity, cloud.classification))
            assert np.array_equal(expected, actual), f'Changed coordinates/source fields: {path}'
            assert cloud.header.parse_crs().to_epsg() == 2180
            ids = np.asarray(cloud.tree_id)
            old_ids = np.asarray(cloud.legacy_tree_id)
            assert set(np.unique(ids)) - {0} <= point_ids, 'Point IDs missing from polygon outputs'
            assert old_ids.max(initial=0) <= legacy_total
            assert np.isfinite(cloud.tree_confidence).all()
            assert ((cloud.tree_confidence >= 0) & (cloud.tree_confidence <= 1)).all()
            assert np.isfinite(cloud.height_agl).all()
            assert (ids[np.asarray(cloud.classification) == 2] == 0).all()
            assert (np.asarray(cloud.pred_semantic)[ids > 0] == 1).all()
            clouds.append(dict(file=path.name, points=len(ids), original_xyz_and_fields_equal=True,
                               labelled_points=int((ids > 0).sum()), ids_match_polygons=True))
            print(f'Validated {path.name}: {len(ids):,} original points', flush=True)
    report = dict(passed=True, vectors=vectors, clouds=clouds, total_original_points=sum(c['points'] for c in clouds))
    (output / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
