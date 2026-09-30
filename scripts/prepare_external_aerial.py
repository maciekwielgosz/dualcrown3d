#!/usr/bin/env python3
"""Prepare verified external aerial tree-instance data in the project format.

New sources are training-only. Existing validation and test rows are copied from
the frozen no-rectangles v2 manifest so comparisons remain paired and honest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from shapely.geometry import MultiPoint, Polygon


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
EXTERNAL = WORKSPACE / "external_lidar_datasets"
BASE = WORKSPACE / "combined_als_crowns_no_rectangles_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepared_record(folder: Path) -> dict | None:
    """Return metadata only when every required prepared artifact exists."""
    metadata_path = folder / "metadata.json"
    required = (
        folder / "points_0p25m.npz",
        folder / "points.laz",
        folder / "crowns_full.gpkg",
    )
    if not metadata_path.exists() or not all(path.exists() for path in required):
        return None
    metadata = json.loads(metadata_path.read_text())
    chm = Path(metadata.get("chm", ""))
    if not chm.is_file():
        return None
    return metadata


def voxelize(
    xyz: np.ndarray,
    intensity: np.ndarray,
    tree_id: np.ndarray,
    classification: np.ndarray,
    voxel_size: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep one point per voxel, preferring supervised tree points."""
    voxel_origin = np.floor(xyz.min(0) / voxel_size) * voxel_size
    grid = np.floor((xyz - voxel_origin) / voxel_size).astype(np.int32)
    priority = (tree_id > 0).astype(np.int8)
    shifted = grid.astype(np.int64) - grid.min(0).astype(np.int64)
    extent = shifted.max(0) + 1
    keys = shifted[:, 0] + extent[0] * (shifted[:, 1] + extent[1] * shifted[:, 2])
    # Sort by voxel first and supervision second, so a labelled tree point wins
    # a collision without the expensive np.unique(..., axis=0) path.
    order = np.lexsort((-priority, keys))
    _, first = np.unique(keys[order], return_index=True)
    selected = np.sort(order[first])
    return (
        xyz[selected],
        intensity[selected],
        tree_id[selected],
        classification[selected],
        voxel_origin,
    )


def normalize_height(
    xyz: np.ndarray, ground: np.ndarray, cell_size: float = 0.5
) -> tuple[np.ndarray, str]:
    result = xyz.copy()
    if ground.any() and xyz[:, 2].min() <= 1.0 and np.median(xyz[ground, 2]) <= 1.0:
        result[:, 2] -= float(np.median(xyz[ground, 2]))
        return result, "already_normalized_ground_median"
    left, bottom = np.floor(xyz[:, :2].min(0) / cell_size) * cell_size
    right, top = np.ceil(xyz[:, :2].max(0) / cell_size) * cell_size
    width = max(1, int(round((right - left) / cell_size)))
    height = max(1, int(round((top - bottom) / cell_size)))
    cc = np.clip(np.floor((xyz[:, 0] - left) / cell_size).astype(np.int64), 0, width - 1)
    rr = np.clip(np.floor((top - xyz[:, 1]) / cell_size).astype(np.int64), 0, height - 1)
    cells = rr * width + cc
    if ground.any():
        sums = np.bincount(cells[ground], weights=xyz[ground, 2], minlength=height * width)
        counts = np.bincount(cells[ground], minlength=height * width)
        surface = (sums / np.maximum(counts, 1)).reshape(height, width)
        nearest = distance_transform_edt(
            counts.reshape(height, width) == 0,
            return_distances=False,
            return_indices=True,
        )
        terrain = surface[tuple(nearest)]
        method = "source_ground_nearest_0p5m"
    else:
        low = np.full(height * width, np.inf)
        np.minimum.at(low, cells, xyz[:, 2])
        low = low.reshape(height, width)
        nearest = distance_transform_edt(
            ~np.isfinite(low), return_distances=False, return_indices=True
        )
        terrain = low[tuple(nearest)]
        method = "estimated_lower_envelope_0p5m"
    result[:, 2] -= terrain[rr, cc]
    return result, method


def exclude_boundary_instances(
    xy: np.ndarray, ids: np.ndarray, bounds: tuple[float, float, float, float], margin: float
) -> tuple[np.ndarray, list[int]]:
    left, bottom, right, top = bounds
    excluded = []
    output = ids.copy()
    for identifier in np.unique(ids[ids > 0]):
        points = xy[ids == identifier]
        if (
            points[:, 0].min() <= left + margin
            or points[:, 0].max() >= right - margin
            or points[:, 1].min() <= bottom + margin
            or points[:, 1].max() >= top - margin
        ):
            output[ids == identifier] = -1
            excluded.append(int(identifier))
    return output, excluded


def crown_records(xyz: np.ndarray, ids: np.ndarray, origin: np.ndarray) -> list[dict]:
    records = []
    for identifier in np.unique(ids[ids > 0]):
        points = xyz[ids == identifier]
        if len(points) < 4:
            continue
        world_xy = points[:, :2].astype(np.float64) + origin[:2]
        geometry = MultiPoint(world_xy).convex_hull.buffer(0.125)
        if geometry.is_empty or geometry.area < 0.25:
            continue
        geometry = Polygon(geometry.exterior)
        records.append(
            {
                "treeID": int(identifier),
                "area_m2": float(geometry.area),
                "points": int(len(points)),
                "geometry": geometry,
            }
        )
    return records


def write_standard_laz(
    path: Path,
    xyz: np.ndarray,
    intensity: np.ndarray,
    ids: np.ndarray,
    classification: np.ndarray,
    crs: str,
) -> None:
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.asarray([0.001, 0.001, 0.001])
    header.offsets = np.floor(xyz.min(0))
    if crs:
        header.add_crs(CRS.from_user_input(crs))
    header.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    cloud.intensity = np.rint(intensity * 65535).astype(np.uint16)
    cloud.classification = classification
    cloud.treeID = ids.astype(np.int32)
    cloud.write(path)


def prepare_arrays(
    *,
    dataset_id: str,
    collection: str,
    source_dataset: str,
    source_path: Path,
    folder: Path,
    xyz: np.ndarray,
    intensity: np.ndarray,
    ids: np.ndarray,
    classification: np.ndarray,
    ground_class: int,
    boundary_margin: float,
    annotation_method: str,
    licence: str,
    doi: str,
    spatial_crs: str = "",
    crs_wkt: str = "",
) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.int64)
    classification = np.asarray(classification, dtype=np.uint8)
    finite = np.isfinite(xyz).all(1) & np.isfinite(intensity)
    xyz, intensity, ids, classification = (
        xyz[finite], intensity[finite], ids[finite], classification[finite]
    )
    raw_points = len(xyz)
    bounds = (
        float(xyz[:, 0].min()),
        float(xyz[:, 1].min()),
        float(xyz[:, 0].max()),
        float(xyz[:, 1].max()),
    )
    ids, boundary_ids = exclude_boundary_instances(xyz[:, :2], ids, bounds, boundary_margin)
    ignored_points_removed = int(np.count_nonzero(ids < 0))
    retained = ids >= 0
    xyz, intensity, ids, classification = (
        xyz[retained], intensity[retained], ids[retained], classification[retained]
    )
    xyz, height_method = normalize_height(xyz, classification == ground_class)
    xyz[:, 2] = np.maximum(xyz[:, 2], -2.0)
    xyz, intensity, ids, classification, voxel_origin = voxelize(
        xyz, intensity, ids, classification
    )
    left, bottom = np.floor(xyz[:, :2].min(0) * 2) / 2
    origin = np.asarray([left, bottom, 0.0], dtype=np.float64)
    coord = (xyz - origin).astype(np.float32)
    grid = np.floor((xyz - voxel_origin) / 0.25).astype(np.int32)
    lo, hi = np.percentile(intensity, [1, 99])
    intensity = np.clip((intensity - lo) / max(float(hi - lo), 1.0), 0, 1).astype(np.float32)

    records = crown_records(coord, ids, origin)
    valid_ids = np.asarray([record["treeID"] for record in records], dtype=np.int64)
    retained = ~((ids > 0) & ~np.isin(ids, valid_ids))
    ignored_points_removed += int(np.count_nonzero(~retained))
    xyz, intensity, ids, classification, coord, grid = (
        xyz[retained],
        intensity[retained],
        ids[retained],
        classification[retained],
        coord[retained],
        grid[retained],
    )
    positive = ids > 0
    offsets = np.zeros_like(coord)
    if positive.any():
        unique, inverse, counts = np.unique(ids[positive], return_inverse=True, return_counts=True)
        sums = np.zeros((len(unique), 3), dtype=np.float64)
        np.add.at(sums, inverse, coord[positive])
        centers = sums / counts[:, None]
        offsets[positive] = centers[inverse] - coord[positive]

    npz = folder / "points_0p25m.npz"
    np.savez_compressed(
        npz,
        coord=coord,
        grid_coord=grid,
        intensity=intensity,
        tree_id=ids.astype(np.int32),
        instance_offset=offsets,
        source_origin=origin,
        voxel_origin=voxel_origin,
        voxel_size=np.float32(0.25),
        raw_points=np.int64(raw_points),
    )
    standard_laz = folder / "points.laz"
    write_standard_laz(
        standard_laz,
        xyz,
        intensity,
        ids,
        classification,
        crs_wkt or spatial_crs,
    )

    crowns = gpd.GeoDataFrame(
        records,
        columns=["treeID", "area_m2", "points", "geometry"],
        geometry="geometry",
        crs=crs_wkt or spatial_crs or None,
    )
    crowns.to_file(folder / "crowns_full.gpkg", layer="crowns_gt", driver="GPKG")

    right, top = np.ceil(xyz[:, :2].max(0) * 2) / 2
    width, height = max(1, int(round((right - left) * 2))), max(1, int(round((top - bottom) * 2)))
    cc = np.clip(np.floor((xyz[:, 0] - left) * 2).astype(np.int64), 0, width - 1)
    rr = np.clip(np.floor((top - xyz[:, 1]) * 2).astype(np.int64), 0, height - 1)
    chm = np.full(height * width, -9999.0, dtype=np.float32)
    np.maximum.at(chm, rr * width + cc, np.maximum(xyz[:, 2], 0))
    chm_path = folder / f"chm_{dataset_id}.tif"
    with rasterio.open(
        chm_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        transform=from_origin(left, top, 0.5, 0.5),
        nodata=-9999,
        compress="deflate",
        crs=crs_wkt or spatial_crs or None,
    ) as dst:
        dst.write(chm.reshape(height, width), 1)
        dst.update_tags(height_normalization=height_method, uses_tree_id="false")

    metadata = {
        "dataset_id": dataset_id,
        "source_dataset": source_dataset,
        "collection": collection,
        "official_split": "train",
        "original_las": str(source_path.resolve()),
        "annotation_method": annotation_method,
        "spatial_crs": spatial_crs,
        "bounds": json.dumps(list(bounds)),
        "crs_wkt": crs_wkt,
        "source_sha256": sha256(source_path),
        "group_id": hashlib.sha256(dataset_id.encode()).hexdigest()[:16],
        "model_split": "train",
        "source_las": str(standard_laz.resolve()),
        "output": str(npz.resolve()),
        "gt_vector": str((folder / "crowns_full.gpkg").resolve()),
        "chm": str(chm_path.resolve()),
        "raw_points": raw_points,
        "voxels": len(coord),
        "instances": len(records),
        "tree_voxels": int(positive.sum()),
        "voxel_size": 0.25,
        "height_normalization": height_method,
        "gt_method": "filled_convex_hull_of_complete_instance_points",
        "copy_verified": True,
        "grid_version": 2,
        "repaired_grid_collisions": 0,
        "coordinates_sha256": "",
        "supervision_changed": False,
        "npz_sha256": sha256(npz),
        "standard_laz_sha256": sha256(standard_laz),
        "licence": licence,
        "doi": doi,
        "boundary_instances_ignored": len(boundary_ids),
        "ignored_points_removed": ignored_points_removed,
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def prepare_dales(
    source_root: Path, output: Path, limit: int | None = None, overwrite: bool = False
) -> list[dict]:
    rows = []
    paths = sorted((source_root / "DALES-2" / "train").glob("*.laz"))
    for path in paths[:limit]:
        target = output / "external_train" / f"dales2__{path.stem}"
        done = prepared_record(target)
        if done is not None and not overwrite:
            rows.append(done)
            continue
        cloud = laspy.read(path)
        xyz = np.column_stack((cloud.x, cloud.y, cloud.z))
        semantic = np.asarray(cloud.classification)
        ids = np.where(semantic == 5, np.asarray(cloud.instance), 0)
        rows.append(
            prepare_arrays(
                dataset_id=f"dales2__{path.stem}",
                collection="DALES2",
                source_dataset="external_aerial",
                source_path=path,
                folder=target,
                xyz=xyz,
                intensity=np.asarray(cloud.intensity),
                ids=ids,
                classification=semantic,
                ground_class=0,
                boundary_margin=0.75,
                annotation_method="human_in_the_loop_point_instance",
                licence="MIT",
                doi="DALES 2, CVPRW 2026",
            )
        )
        print(f"Prepared {path.name}: {rows[-1]['instances']} complete trees", flush=True)
    return rows


def read_pcd_ascii(path: Path, chunk_size: int = 1_000_000):
    header_lines = 0
    fields = None
    with path.open("rb") as stream:
        for raw in stream:
            header_lines += 1
            line = raw.decode("ascii").strip()
            if line.startswith("FIELDS "):
                fields = line.split()[1:]
            if line == "DATA ascii":
                break
    if not fields:
        raise ValueError(f"Missing PCD fields in {path}")
    yield from pd.read_csv(
        path,
        sep=r"\s+",
        names=fields,
        skiprows=header_lines,
        chunksize=chunk_size,
        dtype=np.float32,
    )


def prepare_synthetic(
    source_root: Path, output: Path, limit: int | None = None, overwrite: bool = False
) -> list[dict]:
    rows = []
    paths = sorted((source_root / "SyntheticForest" / "raw").glob("*.pcd"))
    for path in paths[:limit]:
        dataset_id = f"synthetic_forest__{path.stem.lower()}"
        target = output / "external_train" / dataset_id
        done = prepared_record(target)
        if done is not None and not overwrite:
            rows.append(done)
            continue
        pieces = []
        for number, frame in enumerate(read_pcd_ascii(path), 1):
            xyz = frame[["x", "y", "z"]].to_numpy(np.float32)
            semantic = frame["classification"].to_numpy(np.int16)
            ids = np.where(np.isin(semantic, [2, 3]), frame["instance"].to_numpy(np.int64), 0)
            intensity = frame["Intensity"].to_numpy(np.float32)
            mapped = np.where(semantic == 1, 2, np.where(np.isin(semantic, [2, 3]), 5, 1)).astype(np.uint8)
            xyz, intensity, ids, mapped, _ = voxelize(xyz, intensity, ids, mapped)
            pieces.append((xyz, intensity, ids, mapped))
            print(f"{path.name}: parsed chunk {number}", flush=True)
        xyz = np.concatenate([p[0] for p in pieces])
        intensity = np.concatenate([p[1] for p in pieces])
        ids = np.concatenate([p[2] for p in pieces])
        semantic = np.concatenate([p[3] for p in pieces])
        rows.append(
            prepare_arrays(
                dataset_id=dataset_id,
                collection="SYNTHETIC_FOREST",
                source_dataset="external_aerial",
                source_path=path,
                folder=target,
                xyz=xyz,
                intensity=intensity,
                ids=ids,
                classification=semantic,
                ground_class=2,
                boundary_margin=0.0,
                annotation_method="physics_simulation_exact_instance",
                licence="CC-BY-4.0",
                doi="10.5281/zenodo.17568131",
            )
        )
        print(f"Prepared {path.name}: {rows[-1]['instances']} complete trees", flush=True)
    return rows


def prepare_lapalma(
    source_root: Path, output: Path, limit: int | None = None, overwrite: bool = False
) -> list[dict]:
    path = source_root / "LaPalma" / "original" / "Cloud_segmented_limoneros.las"
    if not path.exists() or limit == 0:
        return []
    target = output / "external_train" / "lapalma__limoneros_manual"
    done = prepared_record(target)
    if done is not None and not overwrite:
        return [done]
    cloud = laspy.read(path)
    crs = cloud.header.parse_crs()
    ids = np.asarray(cloud.Tree_ID_1, dtype=np.int64)
    row = prepare_arrays(
        dataset_id="lapalma__limoneros_manual",
        collection="LAPALMA",
        source_dataset="external_aerial",
        source_path=path,
        folder=target,
        xyz=np.column_stack((cloud.x, cloud.y, cloud.z)),
        intensity=np.asarray(cloud.intensity),
        ids=ids,
        classification=np.asarray(cloud.classification),
        ground_class=2,
        boundary_margin=0.25,
        annotation_method="manual_point_instance",
        licence="CC-BY-4.0",
        doi="10.5281/zenodo.14051046",
        spatial_crs=crs.to_authority()[0] + ":" + crs.to_authority()[1] if crs and crs.to_authority() else "",
        crs_wkt=crs.to_wkt() if crs else "",
    )
    print(f"Prepared {path.name}: {row['instances']} complete trees", flush=True)
    return [row]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=EXTERNAL)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "combined_als_crowns_augmented_v3")
    parser.add_argument(
        "--sources", choices=("all", "dales", "synthetic", "lapalma"), default="all"
    )
    parser.add_argument("--limit", type=int, help="Prepare at most this many files per selected source")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild already prepared external files")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    external_rows = []
    if args.sources in ("all", "dales"):
        external_rows.extend(prepare_dales(args.source_root, args.output, args.limit, args.overwrite))
    if args.sources in ("all", "synthetic"):
        external_rows.extend(prepare_synthetic(args.source_root, args.output, args.limit, args.overwrite))
    # La Palma is already present in ideas_als/LA_PALMA. Keep the importer for
    # reproducibility, but never add the duplicate through the default `all` run.
    if args.sources == "lapalma":
        external_rows.extend(prepare_lapalma(args.source_root, args.output, args.limit, args.overwrite))
    base_rows = read_csv(args.base / "manifest.csv")
    rows = base_rows + external_rows
    write_csv(args.output / "manifest.csv", rows)
    for split in ("train", "val", "test"):
        write_csv(args.output / f"manifest_{split}.csv", [r for r in rows if r["model_split"] == split])
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_manifest": str((args.base / "manifest.csv").resolve()),
        "base_manifest_sha256": sha256(args.base / "manifest.csv"),
        "new_sources_are_training_only": True,
        "validation_and_test_unchanged": True,
        "external_plots": len(external_rows),
        "external_instances": sum(int(r["instances"]) for r in external_rows),
        "external_voxels": sum(int(r["voxels"]) for r in external_rows),
        "manifest_sha256": sha256(args.output / "manifest.csv"),
    }
    (args.output / "READY.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
