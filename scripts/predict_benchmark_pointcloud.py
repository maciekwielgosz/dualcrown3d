#!/usr/bin/env python3
"""Run the direct point-cloud LitePT model on the four benchmark tiles."""

from __future__ import annotations

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
import rasterio
import torch
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from shapely.geometry import MultiPolygon, Point


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.evaluate_pointcloud_litept import (  # noqa: E402
    cluster_instances,
    dbh_naslund,
    load_model,
    predict_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "predict", "export", "validate", "all"))
    parser.add_argument("--chm-dir", type=Path, default=WORKSPACE_DIR / "run_r" / "data_input")
    parser.add_argument(
        "--als-dir", type=Path, default=WORKSPACE_DIR / "input_data_ALS" / "dane_ALS"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_DIR / "outputs" / "training" / "litept_tree_instance" / "weights" / "best.pt",
    )
    parser.add_argument(
        "--cluster-config",
        type=Path,
        default=PROJECT_DIR / "outputs" / "evaluation" / "litept_tree_instance_combined" / "selected_cluster_config.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "output_14_litept_pointcloud"
    )
    parser.add_argument("--voxel-size", type=float, default=0.25)
    parser.add_argument("--ground-grid-size", type=float, default=0.5)
    parser.add_argument("--context-buffer", type=float, default=8.0)
    parser.add_argument("--tile-size", type=float, default=20.0)
    parser.add_argument("--overlap", type=float, default=8.0)
    parser.add_argument("--max-points", type=int, default=40_000)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def benchmark_extent(chm_dir: Path):
    descriptions = []
    for path in sorted(chm_dir.resolve().glob("chm_*.tif")):
        with rasterio.open(path) as dataset:
            descriptions.append(
                {
                    "path": str(path),
                    "tile_id": path.stem.removeprefix("chm_"),
                    "bounds": list(dataset.bounds),
                    "crs": str(dataset.crs),
                }
            )
    if not descriptions:
        raise FileNotFoundError(f"No chm_*.tif in {chm_dir}")
    bounds = [
        min(item["bounds"][0] for item in descriptions),
        min(item["bounds"][1] for item in descriptions),
        max(item["bounds"][2] for item in descriptions),
        max(item["bounds"][3] for item in descriptions),
    ]
    return bounds, descriptions


def prepare(args: argparse.Namespace) -> None:
    started = time.monotonic()
    output = args.output_dir.resolve()
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    bounds, tiles = benchmark_extent(args.chm_dir)
    left, bottom, right, top = bounds
    processing_bounds = [left-args.context_buffer, bottom-args.context_buffer,
                         right+args.context_buffer, top+args.context_buffer]
    process_left, process_bottom, process_right, process_top = processing_bounds
    sources = []
    chunks = []
    intensities = []
    classifications = []
    raw_points = selected_points = 0
    crs_values = set()
    for path in sorted(args.als_dir.resolve().glob("*.la[sz]")):
        with laspy.open(path) as reader:
            minimum, maximum = reader.header.mins, reader.header.maxs
            intersects = maximum[0] > process_left and minimum[0] < process_right and maximum[1] > process_bottom and minimum[1] < process_top
            if not intersects:
                continue
            sources.append(str(path))
            # Some supplied LAZ files contain an empty WKT VLR.  Laspy can
            # still read every point, but pyproj rejects that metadata value.
            # The output CRS is taken from the benchmark CHM tiles below.
            try:
                parsed_crs = reader.header.parse_crs()
            except Exception as error:
                print(f"Warning: ignoring invalid CRS metadata in {path}: {error}", flush=True)
                parsed_crs = None
            if parsed_crs is not None:
                crs_values.add(str(parsed_crs))
            for points in reader.chunk_iterator(2_000_000):
                raw_points += len(points)
                x = np.asarray(points.x, dtype=np.float64)
                y = np.asarray(points.y, dtype=np.float64)
                z = np.asarray(points.z, dtype=np.float64)
                intensity = np.asarray(points.intensity, dtype=np.float32)
                classification = np.asarray(points.classification, dtype=np.uint8)
                mask = (
                    np.isfinite(x)
                    & np.isfinite(y)
                    & np.isfinite(z)
                    & np.isfinite(intensity)
                    & (x >= process_left)
                    & (x < process_right)
                    & (y >= process_bottom)
                    & (y < process_top)
                )
                if np.any(mask):
                    chunks.append(np.column_stack((x[mask], y[mask], z[mask])))
                    intensities.append(intensity[mask])
                    classifications.append(classification[mask])
                    selected_points += int(np.count_nonzero(mask))
    if not chunks:
        raise FileNotFoundError("No ALS points intersect the benchmark CHM extent")
    xyz = np.concatenate(chunks)
    intensity = np.concatenate(intensities)
    classification = np.concatenate(classifications)
    # Match training preprocessing: estimate ground only from LAS class 2 on
    # a 0.5 m grid, fill empty cells from the nearest ground cell, and feed
    # normalized height rather than elevation above one global minimum.
    width = max(1, int(math.ceil((process_right-process_left)/args.ground_grid_size)))
    height = max(1, int(math.ceil((process_top-process_bottom)/args.ground_grid_size)))
    cc = np.clip(np.floor((xyz[:, 0]-process_left)/args.ground_grid_size).astype(np.int64), 0, width-1)
    rr = np.clip(np.floor((xyz[:, 1]-process_bottom)/args.ground_grid_size).astype(np.int64), 0, height-1)
    cells = rr*width+cc
    ground = classification == 2
    if not np.any(ground):
        raise RuntimeError("No LAS class-2 ground points in the processing extent")
    sums = np.bincount(cells[ground], weights=xyz[ground, 2], minlength=height*width)
    counts = np.bincount(cells[ground], minlength=height*width)
    terrain = (sums/np.maximum(counts, 1)).reshape(height, width)
    nearest = distance_transform_edt(counts.reshape(height, width) == 0,
                                     return_distances=False, return_indices=True)
    terrain = terrain[tuple(nearest)]
    normalized_height = xyz[:, 2]-terrain[rr, cc]
    normalized_xyz = np.column_stack((xyz[:, :2], normalized_height))
    voxel_origin = normalized_xyz.min(axis=0)
    grid = np.floor((normalized_xyz - voxel_origin) / args.voxel_size).astype(np.int32)
    extent = grid.max(axis=0).astype(np.int64) + 1
    key = grid[:, 0].astype(np.int64)
    key += extent[0] * grid[:, 1].astype(np.int64)
    key += extent[0] * extent[1] * grid[:, 2].astype(np.int64)
    _, index = np.unique(key, return_index=True)
    index.sort()
    normalized_xyz, intensity, grid = normalized_xyz[index], intensity[index], grid[index]
    low, high = np.percentile(intensity, (1, 99))
    high = max(float(high), float(low) + 1.0)
    intensity = np.clip((intensity - low) / (high - low), 0, 1).astype(np.float32)
    source_origin = np.asarray([process_left, process_bottom, 0.0], dtype=np.float64)
    np.savez_compressed(
        work / "benchmark_pointcloud.npz",
        coord=(normalized_xyz - source_origin).astype(np.float32),
        grid_coord=grid,
        intensity=intensity,
        tree_id=np.zeros(len(normalized_xyz), dtype=np.int32),
        instance_offset=np.zeros((len(normalized_xyz), 3), dtype=np.float32),
        source_origin=source_origin,
        voxel_origin=voxel_origin,
        voxel_size=np.float32(args.voxel_size),
    )
    metadata = {
        "bounds": bounds,
        "processing_bounds": processing_bounds,
        "tiles": tiles,
        "chm_crs": tiles[0]["crs"],
        "als_crs": sorted(crs_values),
        "als_files": sources,
        "raw_points_read": raw_points,
        "points_in_extent": selected_points,
        "voxels": len(normalized_xyz),
        "voxel_size_m": args.voxel_size,
        "height_normalization": "LAS classification 2 mean on 0.5 m grid + nearest ground cell",
        "ground_grid_size_m": args.ground_grid_size,
        "ground_points": int(np.count_nonzero(ground)),
        "normalized_height_percentiles": np.percentile(normalized_height, [0, 1, 50, 99, 100]).tolist(),
        "intensity_percentiles": [float(low), float(high)],
        "seconds": time.monotonic() - started,
    }
    (work / "preparation.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def predict(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    with np.load(args.output_dir.resolve() / "work" / "benchmark_pointcloud.npz") as data:
        arrays = {key: data[key] for key in data.files}
    args.preserve_height = True
    raw = predict_plot(model, arrays, args, device, seed=20260924)
    np.savez_compressed(args.output_dir.resolve() / "work" / "raw_predictions.npz", **raw)
    print(
        f"Predicted {int(raw['predicted_voxels']):,}/{int(raw['source_voxels']):,} "
        f"voxels in {float(raw['seconds']):.1f}s across {int(raw['tiles'])} windows"
    )


def pointcloud_heights(raw: dict[str, np.ndarray], instances: list[dict]) -> list[float]:
    """Estimate tree heights from the point cloud, without sampling the CHM.

    The LAS elevations are absolute.  For each predicted top we estimate local
    ground as the second percentile of all returns within 5 m.  Ground and low
    vegetation returns make this robust while keeping the model input strictly
    point-cloud based.
    """
    coord = raw["coord"]
    origin = raw["source_origin"]
    xy_world = coord[:, :2] + origin[:2]
    tree = cKDTree(xy_world)
    heights = []
    for item in instances:
        indices = tree.query_ball_point((item["top_x"], item["top_y"]), r=5.0)
        if indices:
            ground = float(np.percentile(coord[np.asarray(indices), 2], 2.0))
        else:
            ground = float(np.min(coord[:, 2]))
        heights.append(max(float(item["height"]) - ground, 0.0))
    return heights


def export(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    metadata = json.loads((output / "work" / "preparation.json").read_text(encoding="utf-8"))
    config_payload = json.loads(args.cluster_config.resolve().read_text(encoding="utf-8"))
    selected = config_payload.get("config") or config_payload.get("model", {}).get("config")
    if selected is None:
        raise KeyError("Cluster configuration must contain config or model.config")
    selected = dict(selected)
    # FOR-instance reference crowns contain only dominant trees. The benchmark
    # output should retain all trees at least 2 m tall, so disable that relative
    # height filter and apply the CHM threshold below.
    selected["height_ratio"] = 0.0
    with np.load(output / "work" / "raw_predictions.npz") as data:
        raw = {key: data[key] for key in data.files}
    started = time.monotonic()
    origin = raw["source_origin"]
    world_xy = raw["coord"][:, :2] + origin[:2]
    instances_by_tile: dict[str, list[dict]] = {}
    all_instances: list[dict] = []
    for tile in metadata["tiles"]:
        left, bottom, right, top = tile["bounds"]
        context = (
            (world_xy[:, 0] >= left - 5.0)
            & (world_xy[:, 0] < right + 5.0)
            & (world_xy[:, 1] >= bottom - 5.0)
            & (world_xy[:, 1] < top + 5.0)
        )
        subset = {
            key: value[context]
            for key, value in raw.items()
            if key in ("coord", "tree_probability", "shifted_center")
        }
        subset.update(
            {
                "source_origin": origin,
                "voxel_size": raw["voxel_size"],
                "plot_max_z": np.float32(subset["coord"][:, 2].max()),
            }
        )
        candidates = cluster_instances(subset, selected)
        chosen = [
            item
            for item in candidates
            if left <= item["top_x"] < right and bottom <= item["top_y"] < top
        ]
        instances_by_tile[tile["tile_id"]] = chosen
        all_instances.extend(chosen)

    # Prepared Z already is height above point-cloud-derived terrain.
    for item in all_instances:
        item["height"] = max(float(item["height"]), 0.0)
    instances_by_tile = {
        tile_id: [item for item in instances if item["height"] >= 2.0]
        for tile_id, instances in instances_by_tile.items()
    }
    segmentation = output / "Segmentation3"
    segmentation.mkdir(parents=True, exist_ok=True)
    summaries = []
    total = 0
    for tile in metadata["tiles"]:
        tile_id = tile["tile_id"]
        chosen = instances_by_tile[tile_id]
        chosen.sort(key=lambda item: (-item["top_y"], item["top_x"]))
        crown_rows, crown_geometry, top_rows, top_geometry = [], [], [], []
        for tree_id, item in enumerate(chosen, start=1):
            crown_rows.append({"treeID": tree_id, "area_m2": float(item["geometry"].area)})
            crown_geometry.append(MultiPolygon([item["geometry"]]))
            top_rows.append(
                {"treeID": tree_id, "Z": item["height"], "dbh": round(dbh_naslund(item["height"]), 2)}
            )
            top_geometry.append(Point(item["top_x"], item["top_y"], item["height"]))
        crowns = gpd.GeoDataFrame(crown_rows, columns=["treeID", "area_m2"], geometry=crown_geometry, crs=metadata["chm_crs"])
        tops = gpd.GeoDataFrame(top_rows, columns=["treeID", "Z", "dbh"], geometry=top_geometry, crs=metadata["chm_crs"])
        crowns.to_file(segmentation / f"crowns_{tile_id}.gpkg", driver="GPKG", engine="pyogrio", index=False)
        tops.to_file(segmentation / f"ttops_{tile_id}.gpkg", driver="GPKG", engine="pyogrio", index=False)
        summaries.append({"tile_id": tile_id, "crowns": len(crowns), "treetops": len(tops)})
        total += len(crowns)
        print(f"{tile_id}: {len(crowns)} crowns")
    with (output / "prediction_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    configuration = {
        "name": output.name,
        "model": "LitePT-S v2 direct point-cloud tree instance segmentation, trained without IDTREES",
        "model_input": "XYZ + normalized intensity only",
        "height_source": metadata["height_normalization"],
        "weights": str(args.weights.resolve()),
        "weights_sha256": hashlib.sha256(args.weights.resolve().read_bytes()).hexdigest(),
        "cluster_config": selected,
        "preparation": metadata,
        "inference": {
            "seconds": float(raw["seconds"]),
            "tiles": int(raw["tiles"]),
            "source_voxels": int(raw["source_voxels"]),
            "predicted_voxels": int(raw["predicted_voxels"]),
            "window_size_m": args.tile_size,
            "window_overlap_m": args.overlap,
            "max_points_per_window": args.max_points,
        },
        "clustering_seconds": time.monotonic() - started,
        "total_trees": total,
        "tiles": summaries,
        "output_schema": {
            "crowns": ["treeID", "area_m2", "geometry: MultiPolygon"],
            "ttops": ["treeID", "Z", "dbh", "geometry: Point Z"],
        },
    }
    (output / "benchmark_configuration.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# LitePT-S v2 inference on run_r LiDAR extent\n\n"
        "Direct point-cloud inference for the four CHM extents in `run_r/data_input`. "
        "The model input is XYZ height-normalized from LAS class-2 ground plus normalized intensity; "
        "CHM is used only to define tile bounds and output CRS. The checkpoint was trained without IDTREES.\n\n"
        "Open `Segmentation3/crowns_*.gpkg` in QGIS to inspect filled crown outlines and "
        "`ttops_*.gpkg` for matching 3-D treetops. `preview_crowns.png` is a quick-look image, "
        "and `validation.json` contains the GIS integrity checks. See "
        "`benchmark_configuration.json` for full provenance.\n",
        encoding="utf-8",
    )
    print(f"Exported {total:,} trees to {segmentation}")


def validate(args: argparse.Namespace) -> None:
    """Check GIS deliverables and create a quick-look image for QGIS review."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = args.output_dir.resolve()
    segmentation = output / "Segmentation3"
    metadata = json.loads((output / "work" / "preparation.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    rows = []
    datasets = {}
    for tile in metadata["tiles"]:
        tile_id = tile["tile_id"]
        crown_path = segmentation / f"crowns_{tile_id}.gpkg"
        top_path = segmentation / f"ttops_{tile_id}.gpkg"
        if not crown_path.exists() or not top_path.exists():
            errors.append(f"{tile_id}: missing crown or treetop GeoPackage")
            continue
        crowns = gpd.read_file(crown_path, engine="pyogrio")
        tops = gpd.read_file(top_path, engine="pyogrio")
        datasets[tile_id] = (crowns, tops)
        crown_ids = set(crowns["treeID"].astype(int))
        top_ids = set(tops["treeID"].astype(int))
        invalid = int((~crowns.geometry.is_valid).sum())
        empty = int(crowns.geometry.is_empty.sum())
        holes = sum(
            len(part.interiors)
            for geometry in crowns.geometry
            for part in (geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry])
        )
        multipart = sum(
            len(geometry.geoms) > 1
            for geometry in crowns.geometry
            if geometry.geom_type == "MultiPolygon"
        )
        area_error = (
            float(np.max(np.abs(crowns.geometry.area - crowns["area_m2"]))) if len(crowns) else 0.0
        )
        left, bottom, right, top = tile["bounds"]
        inside = (
            (tops.geometry.x >= left)
            & (tops.geometry.x < right)
            & (tops.geometry.y >= bottom)
            & (tops.geometry.y < top)
        )
        epsg_crowns = crowns.crs.to_epsg() if crowns.crs else None
        epsg_tops = tops.crs.to_epsg() if tops.crs else None
        if len(crowns) != len(tops) or crown_ids != top_ids:
            errors.append(f"{tile_id}: crown/treetop IDs do not match")
        if invalid or empty:
            errors.append(f"{tile_id}: {invalid} invalid and {empty} empty crowns")
        if holes:
            errors.append(f"{tile_id}: {holes} polygon holes")
        if not bool(inside.all()):
            errors.append(f"{tile_id}: {int((~inside).sum())} treetops outside tile")
        if epsg_crowns != 2180 or epsg_tops != 2180:
            errors.append(f"{tile_id}: unexpected CRS {epsg_crowns}/{epsg_tops}")
        if area_error > 1e-6:
            errors.append(f"{tile_id}: area attribute mismatch {area_error}")
        rows.append(
            {
                "tile_id": tile_id,
                "crowns": len(crowns),
                "treetops": len(tops),
                "invalid_crowns": invalid,
                "empty_crowns": empty,
                "polygon_holes": holes,
                "multipart_crowns": int(multipart),
                "max_area_error_m2": area_error,
                "epsg": epsg_crowns,
                "treetops_inside_tile": int(inside.sum()),
            }
        )

    validation = {
        "passed": not errors and len(rows) == len(metadata["tiles"]),
        "errors": errors,
        "tiles": rows,
        "expected_tile_count": len(metadata["tiles"]),
        "validated_tile_count": len(rows),
    }
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if datasets:
        figure, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)
        for axis, tile in zip(axes.ravel(), metadata["tiles"]):
            tile_id = tile["tile_id"]
            with rasterio.open(tile["path"]) as source:
                chm = source.read(1, masked=True)
                bounds = source.bounds
            axis.imshow(
                chm,
                extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                origin="upper",
                cmap="gray",
                vmin=0,
                vmax=40,
            )
            if tile_id in datasets:
                crowns, tops = datasets[tile_id]
                crowns.geometry.boundary.plot(ax=axis, color="#ff2a2a", linewidth=0.35)
                tops.plot(ax=axis, color="#00ffff", markersize=0.35)
            axis.set_title(f"{tile_id} — {len(datasets.get(tile_id, ([], []))[0])} crowns")
            axis.set_aspect("equal")
        figure.suptitle("LitePT-S v2: crown outlines (red) and treetops (cyan) on CHM")
        figure.savefig(output / "preview_crowns.png", dpi=180)
        plt.close(figure)
    print(f"Validation {'passed' if validation['passed'] else 'failed'}: {output / 'validation.json'}")
    if errors:
        raise RuntimeError("; ".join(errors))


def main() -> int:
    args = parse_args()
    if args.stage in ("prepare", "all"):
        prepare(args)
    if args.stage in ("predict", "all"):
        predict(args)
    if args.stage in ("export", "all"):
        export(args)
    if args.stage in ("validate", "all"):
        validate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
