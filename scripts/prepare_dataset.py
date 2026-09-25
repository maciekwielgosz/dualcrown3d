#!/usr/bin/env python3
"""Build leakage-free ALS images and YOLO instance-segmentation labels.

Input channels never use treeID.  Existing georeferenced topmost crown vectors
are used only as labels and retain their source treeID in a sidecar manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = Path(sys.prefix)
for variable, relative in (
    ("PROJ_DATA", "share/proj"),
    ("GDAL_DATA", "share/gdal"),
    ("GDAL_DRIVER_PATH", "lib/gdalplugins"),
):
    candidate = ENV_PREFIX / relative
    if candidate.is_dir():
        os.environ[variable] = str(candidate)

import geopandas as gpd
import laspy
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt, median_filter, minimum_filter
from shapely.geometry import MultiPolygon, Polygon


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "default.json"
DEFAULT_LAS_DIR = WORKSPACE_DIR / "FOR-instance"
DEFAULT_REUSED_DIR = PROJECT_DIR / "reused" / "for_instance_chm_gt_0p5m"
DEFAULT_ARTIFACT_DIR = PROJECT_DIR / "artifacts" / "physical_3band_0p5m"
DEFAULT_DATASET_DIR = PROJECT_DIR / "dataset"

GROUND_CLASS = 2
OUTSIDE_CLASS = 3
NODATA_FLOAT = -9999.0
CHUNK_SIZE = 2_000_000
TOPMOST_LAYER = "crowns_gt_topmost"


@dataclass(frozen=True)
class RasterGrid:
    width: int
    height: int
    transform: Affine
    crs: object

    @property
    def cells(self) -> int:
        return self.width * self.height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--las-dir", type=Path, default=DEFAULT_LAS_DIR)
    parser.add_argument("--reused-dir", type=Path, default=DEFAULT_REUSED_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "dataset_id",
        "collection",
        "split",
        "source_las",
        "chm_file",
        "topmost_gt_vector",
    }
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"Manifest lacks columns: {sorted(missing)}")
    return rows


def final_split(row: dict[str, str], validation_ids: set[str]) -> str:
    if row["split"] == "test":
        return "test"
    return "val" if row["dataset_id"] in validation_ids else "train"


def grid_from_reference(path: Path) -> RasterGrid:
    with rasterio.open(path) as dataset:
        return RasterGrid(
            width=dataset.width,
            height=dataset.height,
            transform=dataset.transform,
            crs=dataset.crs,
        )


def point_cells(
    x: np.ndarray, y: np.ndarray, grid: RasterGrid
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inverse = ~grid.transform
    columns = np.floor(inverse.a * x + inverse.b * y + inverse.c).astype(np.int64)
    rows = np.floor(inverse.d * x + inverse.e * y + inverse.f).astype(np.int64)
    inside = (
        (rows >= 0)
        & (rows < grid.height)
        & (columns >= 0)
        & (columns < grid.width)
    )
    return rows[inside], columns[inside], inside


def build_dtm_and_coverage(las_path: Path, grid: RasterGrid) -> tuple[np.ndarray, np.ndarray, str]:
    ground_sum = np.zeros(grid.cells, dtype=np.float64)
    ground_count = np.zeros(grid.cells, dtype=np.uint32)
    coverage_count = np.zeros(grid.cells, dtype=np.uint32)
    surface_min = np.full(grid.cells, np.inf, dtype=np.float64)

    with laspy.open(las_path) as reader:
        for points in reader.chunk_iterator(CHUNK_SIZE):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            classification = np.asarray(points.classification, dtype=np.uint8)
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            if not np.any(finite):
                continue
            x, y, z, classification = (
                x[finite],
                y[finite],
                z[finite],
                classification[finite],
            )
            rows, columns, inside = point_cells(x, y, grid)
            if not np.any(inside):
                continue
            z = z[inside]
            classification = classification[inside]
            cells = rows * grid.width + columns
            valid_area = classification != OUTSIDE_CLASS
            np.add.at(coverage_count, cells[valid_area], 1)
            ground = classification == GROUND_CLASS
            np.add.at(ground_sum, cells[ground], z[ground])
            np.add.at(ground_count, cells[ground], 1)
            nonground_valid = valid_area & (classification != GROUND_CLASS)
            np.minimum.at(surface_min, cells[nonground_valid], z[nonground_valid])

    known = ground_count > 0
    if np.any(known):
        dtm = np.full(grid.cells, np.nan, dtype=np.float64)
        dtm[known] = ground_sum[known] / ground_count[known]
        dtm = dtm.reshape((grid.height, grid.width))
        nearest = distance_transform_edt(
            ~known.reshape((grid.height, grid.width)),
            return_distances=False,
            return_indices=True,
        )
        dtm = dtm[tuple(nearest)]
        method = "class2-nearest"
    else:
        known = np.isfinite(surface_min)
        if not np.any(known):
            raise ValueError(f"No usable terrain or vegetation points in {las_path}")
        surface = surface_min.reshape((grid.height, grid.width))
        nearest = distance_transform_edt(
            ~known.reshape((grid.height, grid.width)),
            return_distances=False,
            return_indices=True,
        )
        surface = surface[tuple(nearest)]
        dtm = median_filter(minimum_filter(surface, size=5), size=9)
        method = "cell-min-lower-envelope"
    return dtm, coverage_count.reshape((grid.height, grid.width)) > 0, method


def rasterize_features(
    las_path: Path,
    grid: RasterGrid,
    dtm: np.ndarray,
    minimum_canopy_height_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chm = np.full(grid.cells, -np.inf, dtype=np.float64)
    density = np.zeros(grid.cells, dtype=np.uint32)
    intensity_sum = np.zeros(grid.cells, dtype=np.float64)
    intensity_count = np.zeros(grid.cells, dtype=np.uint32)
    flat_dtm = dtm.ravel()

    with laspy.open(las_path) as reader:
        dimensions = set(reader.header.point_format.dimension_names)
        if "intensity" not in dimensions:
            raise ValueError(f"LAS has no intensity dimension: {las_path}")
        for points in reader.chunk_iterator(CHUNK_SIZE):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            intensity = np.asarray(points.intensity, dtype=np.float64)
            classification = np.asarray(points.classification, dtype=np.uint8)
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(intensity)
            selected = finite & ~np.isin(classification, (GROUND_CLASS, OUTSIDE_CLASS))
            if not np.any(selected):
                continue
            x, y, z, intensity = (
                x[selected],
                y[selected],
                z[selected],
                intensity[selected],
            )
            rows, columns, inside = point_cells(x, y, grid)
            if not np.any(inside):
                continue
            z, intensity = z[inside], intensity[inside]
            cells = rows * grid.width + columns
            heights = z - flat_dtm[cells]
            valid_height = np.isfinite(heights) & (heights >= 0)
            cells, heights, intensity = (
                cells[valid_height],
                heights[valid_height],
                intensity[valid_height],
            )
            np.maximum.at(chm, cells, heights)
            canopy = heights >= minimum_canopy_height_m
            np.add.at(density, cells[canopy], 1)
            np.add.at(intensity_sum, cells[canopy], intensity[canopy])
            np.add.at(intensity_count, cells[canopy], 1)

    chm[~np.isfinite(chm)] = 0.0
    mean_intensity = np.zeros(grid.cells, dtype=np.float64)
    populated = intensity_count > 0
    mean_intensity[populated] = intensity_sum[populated] / intensity_count[populated]
    shape = (grid.height, grid.width)
    return chm.reshape(shape), density.reshape(shape), mean_intensity.reshape(shape)


def normalize_uint8(
    chm: np.ndarray,
    density: np.ndarray,
    intensity: np.ndarray,
    valid: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, dict]:
    chm_limit = float(config["chm_clip_max_m"])
    chm_u8 = np.rint(np.clip(chm, 0, chm_limit) / chm_limit * 255).astype(np.uint8)

    positive_density = density[(density > 0) & valid]
    density_limit = (
        float(np.percentile(positive_density, config["density_percentile"]))
        if positive_density.size
        else 1.0
    )
    density_limit = max(density_limit, 1.0)
    density_u8 = np.rint(
        np.clip(np.log1p(density), 0, np.log1p(density_limit))
        / np.log1p(density_limit)
        * 255
    ).astype(np.uint8)

    populated = (density > 0) & valid
    values = intensity[populated]
    if values.size:
        intensity_low = float(np.percentile(values, config["intensity_low_percentile"]))
        intensity_high = float(np.percentile(values, config["intensity_high_percentile"]))
    else:
        intensity_low, intensity_high = 0.0, 1.0
    if intensity_high <= intensity_low:
        intensity_high = intensity_low + 1.0
    intensity_u8 = np.rint(
        np.clip((intensity - intensity_low) / (intensity_high - intensity_low), 0, 1)
        * 255
    ).astype(np.uint8)

    image = np.stack((chm_u8, density_u8, intensity_u8), axis=-1)
    image[~valid] = 0
    normalization = {
        "channel_order": ["chm", "density", "mean_intensity"],
        "chm_clip_max_m": chm_limit,
        "density_clip_count": density_limit,
        "density_transform": "log1p",
        "intensity_clip_low": intensity_low,
        "intensity_clip_high": intensity_high,
    }
    return image, normalization


def polygon_parts(geometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [part for part in geometry.geoms if not part.is_empty]
    return [part for part in getattr(geometry, "geoms", []) if isinstance(part, Polygon)]


def yolo_polygon(
    polygon: Polygon, grid: RasterGrid, simplify_m: float = 0.20
) -> list[float] | None:
    candidate = polygon.simplify(simplify_m, preserve_topology=True)
    if candidate.is_empty or not isinstance(candidate, Polygon):
        candidate = polygon
    coordinates = list(candidate.exterior.coords)
    if len(coordinates) > 1 and coordinates[0] == coordinates[-1]:
        coordinates = coordinates[:-1]
    if len(coordinates) < 3:
        return None
    inverse = ~grid.transform
    values: list[float] = []
    for x, y in coordinates:
        column, row = inverse * (x, y)
        values.extend(
            (
                min(max(column / grid.width, 0.0), 1.0),
                min(max(row / grid.height, 0.0), 1.0),
            )
        )
    if len(values) < 6:
        return None
    return values


def write_labels(
    vector_path: Path,
    grid: RasterGrid,
    label_path: Path,
    dataset_id: str,
    split: str,
) -> tuple[list[dict], dict]:
    crowns = gpd.read_file(vector_path, layer=TOPMOST_LAYER)
    if grid.crs and crowns.crs and crowns.crs != grid.crs:
        crowns = crowns.to_crs(grid.crs)
    lines: list[str] = []
    instances: list[dict] = []
    total_area = 0.0
    retained_area = 0.0
    multipart_count = 0
    for _, crown in crowns.sort_values("treeID").iterrows():
        parts = polygon_parts(crown.geometry)
        if not parts:
            continue
        total_area += float(sum(part.area for part in parts))
        if len(parts) > 1:
            multipart_count += 1
        part = max(parts, key=lambda item: item.area)
        values = yolo_polygon(part, grid)
        if values is None:
            continue
        retained_area += float(part.area)
        line = "0 " + " ".join(f"{value:.8f}" for value in values)
        lines.append(line)
        instances.append(
            {
                "dataset_id": dataset_id,
                "split": split,
                "label_line": len(lines) - 1,
                "source_tree_id": int(crown["treeID"]),
                "source_area_m2": float(crown.geometry.area),
                "label_area_m2": float(part.area),
            }
        )
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    stats = {
        "source_instances": int(len(crowns)),
        "written_instances": len(lines),
        "multipart_instances": multipart_count,
        "area_retained_fraction": retained_area / total_area if total_area else 1.0,
    }
    return instances, stats


def write_raw_raster(
    path: Path,
    grid: RasterGrid,
    valid: np.ndarray,
    chm: np.ndarray,
    density: np.ndarray,
    intensity: np.ndarray,
    tags: dict[str, str],
) -> None:
    stack = np.stack((chm, density.astype(np.float64), intensity)).astype(np.float32)
    stack[:, ~valid] = NODATA_FLOAT
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 3,
        "dtype": "float32",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": NODATA_FLOAT,
        "compress": "deflate",
        "predictor": 3,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(stack)
        destination.set_band_description(1, "CHM metres")
        destination.set_band_description(2, "canopy point count")
        destination.set_band_description(3, "mean canopy intensity")
        destination.update_tags(**tags)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = load_json(args.config.resolve())
    las_dir = args.las_dir.resolve()
    reused_dir = args.reused_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    manifest_path = reused_dir / "for_instance_file_manifest.csv"
    rows = read_manifest(manifest_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    validation_ids = set(config["validation_dataset_ids"])
    available_ids = {row["dataset_id"] for row in rows}
    unknown_validation = validation_ids - available_ids
    if unknown_validation and args.limit <= 0:
        raise ValueError(f"Unknown validation IDs: {sorted(unknown_validation)}")

    dataset_rows: list[dict] = []
    instance_rows: list[dict] = []
    preparation_rows: list[dict] = []
    started_all = time.monotonic()

    for index, row in enumerate(rows, start=1):
        dataset_id = row["dataset_id"]
        split = final_split(row, validation_ids)
        las_path = las_dir / row["source_las"]
        reference_chm = reused_dir / row["chm_file"]
        vector_path = reused_dir / row["topmost_gt_vector"]
        raw_path = artifact_dir / f"{dataset_id}.tif"
        image_path = dataset_dir / "images" / split / f"{dataset_id}.png"
        label_path = dataset_dir / "labels" / split / f"{dataset_id}.txt"
        norm_path = artifact_dir / f"{dataset_id}.normalization.json"
        expected = (raw_path, image_path, label_path, norm_path)
        print(f"[{index}/{len(rows)}] {dataset_id} -> {split}", flush=True)
        plot_started = time.monotonic()

        if all(path.is_file() for path in expected) and not args.overwrite:
            with rasterio.open(raw_path) as raw:
                grid = RasterGrid(raw.width, raw.height, raw.transform, raw.crs)
            instances, label_stats = write_labels(
                vector_path, grid, label_path, dataset_id, split
            )
            instance_rows.extend(instances)
            normalization = load_json(norm_path)
            status = "reused"
        else:
            if not las_path.is_file():
                raise FileNotFoundError(las_path)
            grid = grid_from_reference(reference_chm)
            dtm, valid, dtm_method = build_dtm_and_coverage(las_path, grid)
            chm, density, intensity = rasterize_features(
                las_path,
                grid,
                dtm,
                float(config["minimum_canopy_height_m"]),
            )
            image, normalization = normalize_uint8(
                chm, density, intensity, valid, config
            )
            write_raw_raster(
                raw_path,
                grid,
                valid,
                chm,
                density,
                intensity,
                {
                    "DATASET": "FOR-instance",
                    "DATASET_ID": dataset_id,
                    "COLLECTION": row["collection"],
                    "OFFICIAL_SPLIT": row["split"],
                    "MODEL_SPLIT": split,
                    "SOURCE_LAS": row["source_las"],
                    "DTM_METHOD": dtm_method,
                    "TREE_ID_USED_FOR_INPUT": "false",
                    "EXCLUDED_CLASSES": "2,3",
                },
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(image_path)
            norm_path.write_text(
                json.dumps(normalization, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            instances, label_stats = write_labels(
                vector_path, grid, label_path, dataset_id, split
            )
            instance_rows.extend(instances)
            status = "created"

        elapsed = time.monotonic() - plot_started
        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "collection": row["collection"],
                "official_split": row["split"],
                "model_split": split,
                "source_las": str(las_path),
                "raw_raster": str(raw_path),
                "gt_raster": str(reused_dir / row["topmost_gt_raster"]),
                "gt_vector": str(vector_path),
                "image": str(image_path),
                "label": str(label_path),
                "width": grid.width,
                "height": grid.height,
                "crs": str(grid.crs),
                "transform": ",".join(str(value) for value in grid.transform[:6]),
            }
        )
        preparation_rows.append(
            {
                "dataset_id": dataset_id,
                "split": split,
                "status": status,
                "elapsed_seconds": round(elapsed, 6),
                **label_stats,
                **normalization,
            }
        )

    manifests_dir = PROJECT_DIR / "manifests"
    reports_dir = PROJECT_DIR / "reports"
    write_csv(
        manifests_dir / "dataset_manifest.csv",
        dataset_rows,
        list(dataset_rows[0]),
    )
    write_csv(
        manifests_dir / "instance_tree_id_map.csv",
        instance_rows,
        [
            "dataset_id",
            "split",
            "label_line",
            "source_tree_id",
            "source_area_m2",
            "label_area_m2",
        ],
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    split_counts = Counter(row["model_split"] for row in dataset_rows)
    split_instances = Counter(row["split"] for row in instance_rows)
    summary = {
        "plots": len(dataset_rows),
        "plots_by_split": dict(sorted(split_counts.items())),
        "instances_by_split": dict(sorted(split_instances.items())),
        "elapsed_seconds": time.monotonic() - started_all,
        "input_channels_use_tree_id": False,
        "input_channel_order": ["CHM", "density", "mean_intensity"],
        "pixel_size_m": config["pixel_size_m"],
        "preparation": preparation_rows,
    }
    (reports_dir / "dataset_preparation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    dataset_yaml = (
        f"path: {dataset_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: tree\n"
    )
    (dataset_dir / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("plots", "plots_by_split", "instances_by_split", "elapsed_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
