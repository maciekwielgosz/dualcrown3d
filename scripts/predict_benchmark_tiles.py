#!/usr/bin/env python3
"""Run a trained YOLO model on the four CHM benchmark tiles.

The command is split into three stages because the existing ``treescan``
environment owns the geospatial/LAZ dependencies, while ``.venv`` owns
PyTorch and Ultralytics:

1. ``prepare``: physical or CHM-only inputs -> overlapping PNGs.
2. ``infer``: YOLO prediction on those PNGs.
3. ``export``: global de-duplication and benchmark-compatible GeoPackages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_INPUT = WORKSPACE_DIR / "run_r" / "data_input"
DEFAULT_ALS = WORKSPACE_DIR / "input_data_ALS" / "dane_ALS"
DEFAULT_DTM = WORKSPACE_DIR / "CHM_DTM_DSM_Morphometry" / "DTM"
DEFAULT_WEIGHTS = PROJECT_DIR / "outputs" / "training" / "yolo11s_physical" / "weights" / "best.pt"
DEFAULT_OUTPUT = PROJECT_DIR / "output_09_yolo11s_physical"
MODEL_CONFIDENCE = 0.41
PIXEL_SIZE_M = 0.5
PATCH_SIZE_PX = 64
PATCH_OVERLAP_PX = 24
MIN_CANOPY_HEIGHT_M = 2.0
CHM_CLIP_MAX_M = 45.0
POINT_CHUNK_SIZE = 2_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "export", "all"))
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--als-dir", type=Path, default=DEFAULT_ALS)
    parser.add_argument("--dtm-dir", type=Path, default=DEFAULT_DTM)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--input-mode",
        choices=("physical", "chm_only"),
        default="physical",
        help="physical: CHM+density+intensity; chm_only: normalized CHM repeated three times",
    )
    parser.add_argument("--confidence", type=float, default=MODEL_CONFIDENCE)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE_PX)
    parser.add_argument("--overlap", type=int, default=PATCH_OVERLAP_PX)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def configure_geo_environment() -> None:
    prefix = Path(sys.prefix)
    for variable, relative in (
        ("PROJ_DATA", "share/proj"),
        ("GDAL_DATA", "share/gdal"),
        ("GDAL_DRIVER_PATH", "lib/gdalplugins"),
    ):
        candidate = prefix / relative
        if candidate.is_dir():
            os.environ[variable] = str(candidate)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tile_id(path: Path) -> str:
    return path.stem.removeprefix("chm_")


def starts_for_axis(length: int, patch_size: int, overlap: int) -> list[int]:
    if patch_size <= overlap:
        raise ValueError("patch-size must be larger than overlap")
    if length <= patch_size:
        return [0]
    stride = patch_size - overlap
    starts = list(range(0, length - patch_size + 1, stride))
    final = length - patch_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def ownership_intervals(starts: list[int], patch_size: int, length: int) -> list[tuple[float, float]]:
    centers = [start + patch_size / 2 for start in starts]
    intervals = []
    for index in range(len(starts)):
        lower = 0.0 if index == 0 else (centers[index - 1] + centers[index]) / 2
        upper = float(length) if index == len(starts) - 1 else (centers[index] + centers[index + 1]) / 2
        intervals.append((lower, upper))
    return intervals


def normalize_patch(
    chm: np.ndarray,
    density: np.ndarray,
    intensity: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, dict]:
    chm_u8 = np.rint(np.clip(chm, 0, CHM_CLIP_MAX_M) / CHM_CLIP_MAX_M * 255).astype(np.uint8)
    positive_density = density[(density > 0) & valid]
    density_limit = float(np.percentile(positive_density, 99.5)) if positive_density.size else 1.0
    density_limit = max(density_limit, 1.0)
    density_u8 = np.rint(
        np.clip(np.log1p(density), 0, np.log1p(density_limit))
        / np.log1p(density_limit)
        * 255
    ).astype(np.uint8)
    populated = (density > 0) & valid
    values = intensity[populated]
    if values.size:
        intensity_low = float(np.percentile(values, 1.0))
        intensity_high = float(np.percentile(values, 99.0))
    else:
        intensity_low, intensity_high = 0.0, 1.0
    if intensity_high <= intensity_low:
        intensity_high = intensity_low + 1.0
    intensity_u8 = np.rint(
        np.clip((intensity - intensity_low) / (intensity_high - intensity_low), 0, 1) * 255
    ).astype(np.uint8)
    image = np.stack((chm_u8, density_u8, intensity_u8), axis=-1)
    image[~valid] = 0
    return image, {
        "density_clip_count": density_limit,
        "intensity_clip_low": intensity_low,
        "intensity_clip_high": intensity_high,
    }


def prepare(args: argparse.Namespace) -> None:
    configure_geo_environment()
    import laspy
    import rasterio
    from affine import Affine

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    work_dir = output_dir / "work"
    patch_dir = work_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    chm_paths = sorted(input_dir.glob("chm_*.tif"))
    if not chm_paths:
        raise FileNotFoundError(f"No chm_*.tif files in {input_dir}")

    descriptions = []
    for path in chm_paths:
        with rasterio.open(path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"Missing CRS: {path}")
            if not np.allclose(dataset.res, (PIXEL_SIZE_M, PIXEL_SIZE_M)):
                raise ValueError(f"Unexpected resolution in {path}: {dataset.res}")
            descriptions.append(
                {
                    "path": path,
                    "id": tile_id(path),
                    "bounds": dataset.bounds,
                    "crs": dataset.crs,
                    "transform": dataset.transform,
                    "shape": dataset.shape,
                    "nodata": dataset.nodata,
                }
            )
    crs = descriptions[0]["crs"]
    if any(item["crs"] != crs for item in descriptions):
        raise ValueError("CHM rasters do not share one CRS")
    left = min(item["bounds"].left for item in descriptions)
    bottom = min(item["bounds"].bottom for item in descriptions)
    right = max(item["bounds"].right for item in descriptions)
    top = max(item["bounds"].top for item in descriptions)
    width = int(round((right - left) / PIXEL_SIZE_M))
    height = int(round((top - bottom) / PIXEL_SIZE_M))
    transform = Affine(PIXEL_SIZE_M, 0, left, 0, -PIXEL_SIZE_M, top)
    chm = np.full((height, width), np.nan, dtype=np.float32)
    dtm = np.full((height, width), np.nan, dtype=np.float32) if args.input_mode == "physical" else None
    valid = np.zeros((height, width), dtype=bool)

    tile_metadata = []
    for item in descriptions:
        path = item["path"]
        with rasterio.open(path) as dataset:
            values = dataset.read(1)
            tile_valid = np.isfinite(values)
            if dataset.nodata is not None:
                tile_valid &= values != dataset.nodata
        row0 = int(round((top - item["bounds"].top) / PIXEL_SIZE_M))
        col0 = int(round((item["bounds"].left - left) / PIXEL_SIZE_M))
        rows = slice(row0, row0 + values.shape[0])
        cols = slice(col0, col0 + values.shape[1])
        destination = chm[rows, cols]
        destination[tile_valid] = values[tile_valid]
        valid[rows, cols] |= tile_valid

        dtm_path = None
        if args.input_mode == "physical":
            dtm_path = args.dtm_dir.resolve() / f"dtm_{item['id']}.tif"
            if not dtm_path.is_file():
                raise FileNotFoundError(dtm_path)
            with rasterio.open(dtm_path) as dataset:
                terrain = dataset.read(1)
                if dataset.shape != item["shape"] or dataset.transform != item["transform"]:
                    raise ValueError(f"DTM grid does not match CHM: {dtm_path}")
            terrain_valid = np.isfinite(terrain) & tile_valid
            dtm_destination = dtm[rows, cols]
            dtm_destination[terrain_valid] = terrain[terrain_valid]

        x_token, y_token = (int(value) for value in item["id"].split("_"))
        tile_metadata.append(
            {
                "tile_id": item["id"],
                "chm": str(path),
                "dtm": str(dtm_path) if dtm_path is not None else None,
                "core_bounds": [x_token, y_token, x_token + 500, y_token + 500],
                "raster_bounds": list(item["bounds"]),
            }
        )

    density = np.zeros((height, width), dtype=np.uint32)
    intensity_sum = np.zeros((height, width), dtype=np.float64)
    intensity_count = np.zeros((height, width), dtype=np.uint32)
    laz_paths = []
    points_read = 0
    canopy_points = 0
    started = time.monotonic()
    if args.input_mode == "physical":
        for path in sorted(args.als_dir.resolve().glob("*.laz")):
            with laspy.open(path) as reader:
                minimum, maximum = reader.header.mins, reader.header.maxs
                intersects = maximum[0] > left and minimum[0] < right and maximum[1] > bottom and minimum[1] < top
                if intersects:
                    laz_paths.append(path)
        if not laz_paths:
            raise FileNotFoundError("No LAZ file intersects the benchmark CHM extent")
        flat_dtm = dtm.ravel()
        flat_valid = valid.ravel()
        for laz_path in laz_paths:
            print(f"Rasterizing {laz_path.name}", flush=True)
            with laspy.open(laz_path) as reader:
                dimensions = set(reader.header.point_format.dimension_names)
                if not {"intensity", "classification"}.issubset(dimensions):
                    raise ValueError(f"Required LAS dimensions missing in {laz_path}")
                for points in reader.chunk_iterator(POINT_CHUNK_SIZE):
                    points_read += len(points)
                    x = np.asarray(points.x, dtype=np.float64)
                    y = np.asarray(points.y, dtype=np.float64)
                    z = np.asarray(points.z, dtype=np.float64)
                    intensity_values = np.asarray(points.intensity, dtype=np.float64)
                    classification = np.asarray(points.classification, dtype=np.uint8)
                    selected = (
                        np.isfinite(x)
                        & np.isfinite(y)
                        & np.isfinite(z)
                        & np.isfinite(intensity_values)
                        & ~np.isin(classification, (2, 3))
                        & (x >= left)
                        & (x < right)
                        & (y >= bottom)
                        & (y < top)
                    )
                    if not np.any(selected):
                        continue
                    x, y, z, intensity_values = (
                        x[selected], y[selected], z[selected], intensity_values[selected]
                    )
                    columns = np.floor((x - left) / PIXEL_SIZE_M).astype(np.int64)
                    rows = np.floor((top - y) / PIXEL_SIZE_M).astype(np.int64)
                    inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
                    rows, columns, z, intensity_values = (
                        rows[inside], columns[inside], z[inside], intensity_values[inside]
                    )
                    cells = rows * width + columns
                    point_heights = z - flat_dtm[cells]
                    canopy = (
                        flat_valid[cells]
                        & np.isfinite(point_heights)
                        & (point_heights >= MIN_CANOPY_HEIGHT_M)
                    )
                    if not np.any(canopy):
                        continue
                    rows, columns, intensity_values = (
                        rows[canopy], columns[canopy], intensity_values[canopy]
                    )
                    canopy_points += len(rows)
                    np.add.at(density, (rows, columns), 1)
                    np.add.at(intensity_sum, (rows, columns), intensity_values)
                    np.add.at(intensity_count, (rows, columns), 1)
    mean_intensity = np.zeros((height, width), dtype=np.float32)
    populated = intensity_count > 0
    mean_intensity[populated] = (intensity_sum[populated] / intensity_count[populated]).astype(np.float32)

    np.savez_compressed(work_dir / "mosaic.npz", chm=chm, valid=valid.astype(np.uint8))
    row_starts = starts_for_axis(height, args.patch_size, args.overlap)
    col_starts = starts_for_axis(width, args.patch_size, args.overlap)
    row_owners = ownership_intervals(row_starts, args.patch_size, height)
    col_owners = ownership_intervals(col_starts, args.patch_size, width)
    patch_rows = []
    for row_index, row0 in enumerate(row_starts):
        for col_index, col0 in enumerate(col_starts):
            window = np.s_[row0 : row0 + args.patch_size, col0 : col0 + args.patch_size]
            patch_valid = valid[window]
            if not np.any(patch_valid):
                continue
            patch_chm = np.nan_to_num(chm[window], nan=0.0)
            if args.input_mode == "chm_only":
                chm_u8 = np.rint(
                    np.clip(patch_chm, 0, CHM_CLIP_MAX_M) / CHM_CLIP_MAX_M * 255
                ).astype(np.uint8)
                image = np.repeat(chm_u8[..., None], 3, axis=-1)
                image[~patch_valid] = 0
                normalization = {"density_clip_count": None, "intensity_clip_low": None, "intensity_clip_high": None}
            else:
                image, normalization = normalize_patch(
                    patch_chm,
                    density[window],
                    mean_intensity[window],
                    patch_valid,
                )
            patch_name = f"patch_r{row0:04d}_c{col0:04d}.png"
            patch_path = patch_dir / patch_name
            Image.fromarray(image).save(patch_path)
            patch_rows.append(
                {
                    "path": str(patch_path),
                    "row": row0,
                    "column": col0,
                    "owner_row_min": row_owners[row_index][0],
                    "owner_row_max": row_owners[row_index][1],
                    "owner_column_min": col_owners[col_index][0],
                    "owner_column_max": col_owners[col_index][1],
                    "valid_fraction": float(np.mean(patch_valid)),
                    **normalization,
                }
            )
    metadata = {
        "version": 2,
        "input_mode": args.input_mode,
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "crs": str(crs),
        "transform": list(transform)[:6],
        "width": width,
        "height": height,
        "bounds": [left, bottom, right, top],
        "pixel_size_m": PIXEL_SIZE_M,
        "channel_order": (
            ["CHM", "CHM", "CHM"]
            if args.input_mode == "chm_only"
            else ["CHM", "canopy_point_density", "mean_canopy_intensity"]
        ),
        "input_channels_use_tree_id": False,
        "minimum_canopy_height_m": MIN_CANOPY_HEIGHT_M,
        "chm_clip_max_m": CHM_CLIP_MAX_M,
        "density_percentile": 99.5,
        "intensity_percentiles": [1.0, 99.0],
        "patch_size_px": args.patch_size,
        "patch_size_m": args.patch_size * PIXEL_SIZE_M,
        "patch_overlap_px": args.overlap,
        "patch_overlap_m": args.overlap * PIXEL_SIZE_M,
        "patches": len(patch_rows),
        "points_read": points_read,
        "canopy_points_used": canopy_points,
        "laz_files": [str(path) for path in laz_paths],
        "tiles": tile_metadata,
        "elapsed_seconds": time.monotonic() - started,
    }
    (work_dir / "patch_manifest.json").write_text(
        json.dumps({"metadata": metadata, "patches": patch_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    detail = (
        "from CHM only"
        if args.input_mode == "chm_only"
        else f"from {canopy_points:,} canopy points"
    )
    print(f"Prepared {len(patch_rows)} patches {detail} in {metadata['elapsed_seconds']:.1f}s")


def polygon_centroid(coordinates: np.ndarray) -> tuple[float, float]:
    if len(coordinates) < 3:
        return float(np.mean(coordinates[:, 0])), float(np.mean(coordinates[:, 1]))
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area6 = 3.0 * np.sum(cross)
    if abs(area6) < 1e-9:
        return float(np.mean(x)), float(np.mean(y))
    return (
        float(np.sum((x + np.roll(x, -1)) * cross) / area6),
        float(np.sum((y + np.roll(y, -1)) * cross) / area6),
    )


def infer(args: argparse.Namespace) -> None:
    import torch
    from ultralytics import YOLO

    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "work" / "patch_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    patch_rows = payload["patches"]
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    model = YOLO(str(weights))
    predictions = []
    started = time.monotonic()
    chunk_size = max(args.batch * 8, args.batch)
    for offset in range(0, len(patch_rows), chunk_size):
        chunk = patch_rows[offset : offset + chunk_size]
        results = model.predict(
            source=[row["path"] for row in chunk],
            imgsz=320,
            conf=args.confidence,
            iou=0.7,
            max_det=300,
            batch=args.batch,
            device=device,
            retina_masks=True,
            verbose=False,
        )
        for patch, result in zip(chunk, results, strict=True):
            if result.masks is None or result.boxes is None:
                continue
            confidences = result.boxes.conf.detach().cpu().numpy()
            for confidence, local_polygon in zip(confidences, result.masks.xy, strict=True):
                local_polygon = np.asarray(local_polygon, dtype=np.float64)
                if len(local_polygon) < 3:
                    continue
                global_polygon = local_polygon + np.array([patch["column"], patch["row"]])
                centroid_x, centroid_y = polygon_centroid(global_polygon)
                if not (
                    patch["owner_column_min"] <= centroid_x < patch["owner_column_max"]
                    and patch["owner_row_min"] <= centroid_y < patch["owner_row_max"]
                ):
                    continue
                predictions.append(
                    {
                        "confidence": float(confidence),
                        "polygon_pixel_xy": global_polygon.tolist(),
                        "centroid_pixel_xy": [centroid_x, centroid_y],
                        "source_patch": Path(patch["path"]).name,
                    }
                )
        print(f"Predicted {min(offset + len(chunk), len(patch_rows))}/{len(patch_rows)} patches", flush=True)
    inference = {
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "confidence": args.confidence,
        "image_size": 320,
        "nms_iou_within_patch": 0.7,
        "device": str(device),
        "torch": torch.__version__,
        "patches": len(patch_rows),
        "accepted_patch_predictions": len(predictions),
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "work" / "raw_predictions.json").write_text(
        json.dumps({"inference": inference, "predictions": predictions}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Accepted {len(predictions):,} patch predictions in {inference['elapsed_seconds']:.1f}s")


def dbh_naslund(height: float, a: float = 10.0, b: float = 0.6) -> float:
    hm = max(height - 1.3, 0.0)
    coefficient_a = -hm * b
    coefficient_c = -hm * a
    return (-coefficient_a + math.sqrt(coefficient_a**2 - 4.0 * coefficient_c)) / 2.0


def export(args: argparse.Namespace) -> None:
    configure_geo_environment()
    import geopandas as gpd
    import rasterio
    from affine import Affine
    from shapely import contains_xy
    from shapely.geometry import MultiPolygon, Point, Polygon
    from shapely.strtree import STRtree

    output_dir = args.output_dir.resolve()
    work_dir = output_dir / "work"
    manifest = json.loads((work_dir / "patch_manifest.json").read_text(encoding="utf-8"))
    raw = json.loads((work_dir / "raw_predictions.json").read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    transform = Affine(*metadata["transform"])
    inverse = ~transform
    mosaic = np.load(work_dir / "mosaic.npz")
    chm = mosaic["chm"]
    valid = mosaic["valid"].astype(bool)
    height, width = chm.shape

    polygons = []
    confidences = []
    pixel_polygons = []
    for item in raw["predictions"]:
        pixel_polygon = np.asarray(item["polygon_pixel_xy"], dtype=np.float64)
        world_coordinates = [transform * (float(x), float(y)) for x, y in pixel_polygon]
        polygon = Polygon(world_coordinates)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area < 0.25:
            continue
        if isinstance(polygon, MultiPolygon):
            polygon = max(polygon.geoms, key=lambda geometry: geometry.area)
        polygons.append(polygon)
        confidences.append(float(item["confidence"]))
        pixel_polygons.append(pixel_polygon)

    # Resolve rare cross-window duplicates left after ownership filtering.
    tree = STRtree(polygons)
    order = sorted(range(len(polygons)), key=lambda index: confidences[index], reverse=True)
    suppressed = np.zeros(len(polygons), dtype=bool)
    kept = []
    for index in order:
        if suppressed[index]:
            continue
        kept.append(index)
        geometry = polygons[index]
        for candidate in tree.query(geometry, predicate="intersects"):
            candidate = int(candidate)
            if candidate == index or suppressed[candidate] or confidences[candidate] > confidences[index]:
                continue
            intersection = geometry.intersection(polygons[candidate]).area
            union = geometry.area + polygons[candidate].area - intersection
            if union > 0 and intersection / union >= 0.5:
                suppressed[candidate] = True

    # Locate the CHM maximum of every crown before assigning it to a 500 m
    # benchmark tile. The historical workflow assigns a crown by its treetop,
    # not by its polygon centroid, so this also keeps every output treetop in
    # the core bounds encoded by its file name.
    crown_tops = {}
    for index in kept:
        polygon = polygons[index]
        min_col, max_row = inverse * (polygon.bounds[0], polygon.bounds[1])
        max_col, min_row = inverse * (polygon.bounds[2], polygon.bounds[3])
        row0 = max(0, int(math.floor(min_row)) - 1)
        row1 = min(height, int(math.ceil(max_row)) + 2)
        col0 = max(0, int(math.floor(min_col)) - 1)
        col1 = min(width, int(math.ceil(max_col)) + 2)
        row_grid, col_grid = np.mgrid[row0:row1, col0:col1]
        x_grid = transform.c + (col_grid + 0.5) * transform.a
        y_grid = transform.f + (row_grid + 0.5) * transform.e
        inside = contains_xy(polygon, x_grid, y_grid) & valid[row0:row1, col0:col1]
        values = chm[row0:row1, col0:col1]
        inside &= np.isfinite(values)
        if np.any(inside):
            scores = np.where(inside, values, -np.inf)
            local_row, local_col = np.unravel_index(int(np.argmax(scores)), scores.shape)
            top_row, top_col = row0 + local_row, col0 + local_col
            tree_height = float(chm[top_row, top_col])
            x_top, y_top = transform * (top_col + 0.5, top_row + 0.5)
        else:
            centroid = polygon.centroid
            top_col, top_row = inverse * (centroid.x, centroid.y)
            top_col = min(max(int(top_col), 0), width - 1)
            top_row = min(max(int(top_row), 0), height - 1)
            tree_height = float(chm[top_row, top_col]) if np.isfinite(chm[top_row, top_col]) else 0.0
            x_top, y_top = centroid.x, centroid.y
        if tree_height >= MIN_CANOPY_HEIGHT_M:
            crown_tops[index] = (tree_height, float(x_top), float(y_top))

    segmentation_dir = output_dir / "Segmentation3"
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    total = 0
    for tile in metadata["tiles"]:
        tile_name = tile["tile_id"]
        core_left, core_bottom, core_right, core_top = tile["core_bounds"]
        selected = []
        for index, (_, x_top, y_top) in crown_tops.items():
            if core_left <= x_top < core_right and core_bottom <= y_top < core_top:
                selected.append(index)
        selected.sort(key=lambda index: (-crown_tops[index][2], crown_tops[index][1]))
        crown_rows = []
        crown_geometries = []
        treetop_rows = []
        treetop_geometries = []
        for tree_id_value, index in enumerate(selected, start=1):
            polygon = polygons[index]
            tree_height, x_top, y_top = crown_tops[index]
            crown_rows.append(
                {
                    "treeID": tree_id_value,
                    "area_m2": float(polygon.area),
                }
            )
            crown_geometries.append(MultiPolygon([polygon]))
            treetop_rows.append(
                {
                    "treeID": tree_id_value,
                    "Z": tree_height,
                    "dbh": round(dbh_naslund(tree_height), 2),
                }
            )
            treetop_geometries.append(Point(float(x_top), float(y_top), tree_height))

        crowns = gpd.GeoDataFrame(crown_rows, geometry=crown_geometries, crs=metadata["crs"])
        treetops = gpd.GeoDataFrame(treetop_rows, geometry=treetop_geometries, crs=metadata["crs"])
        crown_path = segmentation_dir / f"crowns_{tile_name}.gpkg"
        treetop_path = segmentation_dir / f"ttops_{tile_name}.gpkg"
        crowns.to_file(crown_path, layer=crown_path.stem, driver="GPKG", engine="pyogrio", index=False)
        treetops.to_file(treetop_path, layer=treetop_path.stem, driver="GPKG", engine="pyogrio", index=False)
        summary_rows.append(
            {
                "tile_id": tile_name,
                "crowns": len(crowns),
                "treetops": len(treetops),
                "mean_confidence": float(np.mean([confidences[index] for index in selected])) if selected else 0.0,
                "min_confidence": float(np.min([confidences[index] for index in selected])) if selected else 0.0,
                "max_confidence": float(np.max([confidences[index] for index in selected])) if selected else 0.0,
            }
        )
        total += len(crowns)
        print(f"{tile_name}: {len(crowns)} crowns and treetops")

    with (output_dir / "prediction_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    configuration = {
        "name": output_dir.name,
        "description": (
            "YOLO11s-seg CHM-only input: normalized CHM repeated in three technical channels"
            if metadata["input_mode"] == "chm_only"
            else "YOLO11s-seg physical input: CHM + ALS density + mean ALS intensity"
        ),
        "input_directory": metadata["input_directory"],
        "output_directory": str(output_dir),
        "model": raw["inference"],
        "preprocessing": metadata,
        "global_duplicate_iou": 0.5,
        "output_schema": {
            "crowns": ["treeID", "area_m2", "geometry: MultiPolygon"],
            "ttops": ["treeID", "Z", "dbh", "geometry: Point Z"],
        },
        "tiles": summary_rows,
        "total_trees": total,
        "comparability_note": (
            "File names, layers, CRS and attribute schemas match segmentatiion_benchmark. "
            + (
                "The model input uses CHM only; three identical channels are a technical adapter for pretrained RGB weights."
                if metadata["input_mode"] == "chm_only"
                else "The DL method additionally uses ALS density and intensity and therefore is not a CHM-only ablation."
            )
        ),
    }
    (output_dir / "benchmark_configuration.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    benchmark_summary = {
        "variant": output_dir.name,
        **{f"tile_{row['tile_id']}": row["crowns"] for row in summary_rows},
        "total_trees": total,
        "tiles": len(summary_rows),
        "crowns_files": len(summary_rows),
        "ttops_files": len(summary_rows),
        "crs": metadata["crs"],
        "invalid_geometries": 0,
        "empty_geometries": 0,
    }
    summary_path = output_dir / "benchmark_outputs_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(benchmark_summary))
        writer.writeheader()
        writer.writerow(benchmark_summary)

    historical_summary = WORKSPACE_DIR / "segmentatiion_benchmark" / "benchmark_outputs_summary.csv"
    if historical_summary.is_file():
        with historical_summary.open("r", encoding="utf-8", newline="") as stream:
            comparison_rows = list(csv.DictReader(stream))
        comparison_rows.append({key: str(value) for key, value in benchmark_summary.items()})
        comparison_path = output_dir / "comparison_with_segmentatiion_benchmark.csv"
        with comparison_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(benchmark_summary))
            writer.writeheader()
            writer.writerows(comparison_rows)

    if metadata["input_mode"] == "chm_only":
        title = "YOLO11s CHM-only: predykcja kafli benchmarkowych"
        model_name = "YOLO11s-seg CHM-only"
        input_description = (
            "Wejście modelu obejmuje wyłącznie CHM znormalizowany w zakresie 0–45 m "
            "i powielony do trzech identycznych kanałów technicznych. Nie użyto ALS ani DTM."
        )
        comparison_note = (
            "Jest to właściwa ablacja CHM-only i można ją bezpośrednio porównywać z metodami "
            "benchmarku korzystającymi wyłącznie z CHM."
        )
    else:
        title = "YOLO11s physical: predykcja kafli benchmarkowych"
        model_name = "YOLO11s-seg physical"
        input_description = (
            "Wejście modelu obejmuje CHM oraz gęstość i średnią intensywność punktów z pliku "
            f"`{Path(metadata['laz_files'][0]).name}`. ALS pokrywa cały analizowany obszar."
        )
        comparison_note = (
            "Metoda DL wykorzystuje dwa dodatkowe kanały ALS, zatem nie jest ablacją CHM-only."
        )

    readme = f"""# {title}

Wynik modelu `{model_name}` dla czterech CHM z `{metadata['input_directory']}`.
{input_description}

Wynik zawiera {total} drzew. Folder `Segmentation3` ma taki sam układ nazw,
warstw, CRS i atrybutów jak warianty w `segmentatiion_benchmark`:

- `crowns_<tile>.gpkg`: `treeID`, `area_m2`, MultiPolygon,
- `ttops_<tile>.gpkg`: `treeID`, `Z`, `dbh`, Point Z.

Model wykonuje inferencję w oknach {metadata['patch_size_m']:g} m z zakładką
{metadata['patch_overlap_m']:g} m. Próg pewności {raw['inference']['confidence']:.2f}
został wcześniej dobrany wyłącznie na walidacji FOR-instance. Obowiązuje ten sam
minimalny próg wysokości 2 m co w benchmarku.

To porównanie formatu i liczby detekcji; dla tych kafli nie ma referencyjnych
koron GT, więc nie można policzyć F1 ani PQ. {comparison_note}
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Exported {total:,} total trees to {segmentation_dir}")


def main() -> int:
    args = parse_args()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "infer":
        infer(args)
    elif args.stage == "export":
        export(args)
    else:
        raise SystemExit(
            "Stage 'all' cannot cross the two isolated Python environments. "
            "Run prepare/export with treescan and infer with .venv."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
