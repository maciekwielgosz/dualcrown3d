#!/usr/bin/env python3
"""Build a CHM-only YOLO dataset from ideas_als/dev and FOR-instance.

The input image is derived from XYZ and Classification only.  ``treeID`` is
used exclusively to create 2-D instance labels.  The official ideas_als test
split is never included in training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import laspy
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt, median_filter, minimum_filter
from scipy.spatial import ConvexHull, QhullError


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_IDEAS_DIR = WORKSPACE_DIR / "ideas_als"
DEFAULT_FOR_DATASET = PROJECT_DIR / "dataset_chm_only"
DEFAULT_FOR_MANIFEST = PROJECT_DIR / "manifests" / "dataset_manifest_chm_only.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "dataset_chm_only_ideas_combined"
DEFAULT_ARTIFACTS = PROJECT_DIR / "artifacts" / "ideas_chm_only_0p5m"
PIXEL_SIZE_M = 0.5
CHM_CLIP_MAX_M = 45.0
GROUND_CLASS = 2
CHUNK_SIZE = 2_000_000
MIN_CROWN_AREA_M2 = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ideas-dir", type=Path, default=DEFAULT_IDEAS_DIR)
    parser.add_argument("--for-dataset", type=Path, default=DEFAULT_FOR_DATASET)
    parser.add_argument("--for-manifest", type=Path, default=DEFAULT_FOR_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def grid_from_header(header) -> tuple[int, int, Affine, object]:
    minimum = header.mins
    maximum = header.maxs
    left = math.floor(float(minimum[0]) / PIXEL_SIZE_M) * PIXEL_SIZE_M
    bottom = math.floor(float(minimum[1]) / PIXEL_SIZE_M) * PIXEL_SIZE_M
    right = math.ceil(float(maximum[0]) / PIXEL_SIZE_M) * PIXEL_SIZE_M
    top = math.ceil(float(maximum[1]) / PIXEL_SIZE_M) * PIXEL_SIZE_M
    width = max(1, int(round((right - left) / PIXEL_SIZE_M)))
    height = max(1, int(round((top - bottom) / PIXEL_SIZE_M)))
    transform = Affine(PIXEL_SIZE_M, 0.0, left, 0.0, -PIXEL_SIZE_M, top)
    return width, height, transform, header.parse_crs()


def point_cells(
    x: np.ndarray,
    y: np.ndarray,
    width: int,
    height: int,
    transform: Affine,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = np.floor((x - transform.c) / transform.a).astype(np.int64)
    rows = np.floor((transform.f - y) / -transform.e).astype(np.int64)
    inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    return rows[inside], columns[inside], inside


def build_terrain_and_coverage(
    las_path: Path,
    width: int,
    height: int,
    transform: Affine,
) -> tuple[np.ndarray, np.ndarray, str]:
    cells_count = width * height
    ground_sum = np.zeros(cells_count, dtype=np.float64)
    ground_count = np.zeros(cells_count, dtype=np.uint32)
    coverage_count = np.zeros(cells_count, dtype=np.uint32)
    surface_min = np.full(cells_count, np.inf, dtype=np.float64)
    with laspy.open(las_path) as reader:
        for points in reader.chunk_iterator(CHUNK_SIZE):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            classification = np.asarray(points.classification, dtype=np.uint8)
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            if not np.any(finite):
                continue
            x, y, z, classification = x[finite], y[finite], z[finite], classification[finite]
            rows, columns, inside = point_cells(x, y, width, height, transform)
            z, classification = z[inside], classification[inside]
            cells = rows * width + columns
            np.add.at(coverage_count, cells, 1)
            np.minimum.at(surface_min, cells, z)
            ground = classification == GROUND_CLASS
            np.add.at(ground_sum, cells[ground], z[ground])
            np.add.at(ground_count, cells[ground], 1)
    known_ground = ground_count > 0
    if np.any(known_ground):
        terrain = np.full(cells_count, np.nan, dtype=np.float64)
        terrain[known_ground] = ground_sum[known_ground] / ground_count[known_ground]
        terrain = terrain.reshape(height, width)
        indices = distance_transform_edt(
            ~known_ground.reshape(height, width), return_distances=False, return_indices=True
        )
        terrain = terrain[tuple(indices)]
        method = "class2-nearest"
    else:
        known_surface = np.isfinite(surface_min)
        if not np.any(known_surface):
            raise ValueError(f"No finite points in {las_path}")
        surface = surface_min.reshape(height, width)
        indices = distance_transform_edt(
            ~known_surface.reshape(height, width), return_distances=False, return_indices=True
        )
        surface = surface[tuple(indices)]
        terrain = median_filter(minimum_filter(surface, size=5), size=9)
        method = "cell-min-lower-envelope"
    return terrain, coverage_count.reshape(height, width) > 0, method


def rasterize_chm_and_tree_cells(
    las_path: Path,
    width: int,
    height: int,
    transform: Affine,
    terrain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cells_count = width * height
    chm = np.full(cells_count, -np.inf, dtype=np.float64)
    pairs_by_chunk: list[np.ndarray] = []
    terrain_flat = terrain.ravel()
    with laspy.open(las_path) as reader:
        dimensions = set(reader.header.point_format.dimension_names)
        if "treeID" not in dimensions:
            raise ValueError(f"Missing treeID: {las_path}")
        for points in reader.chunk_iterator(CHUNK_SIZE):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            classification = np.asarray(points.classification, dtype=np.uint8)
            tree_ids = np.asarray(points.treeID, dtype=np.int64)
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            selected = finite & (classification != GROUND_CLASS)
            if not np.any(selected):
                continue
            x, y, z, tree_ids = x[selected], y[selected], z[selected], tree_ids[selected]
            rows, columns, inside = point_cells(x, y, width, height, transform)
            z, tree_ids = z[inside], tree_ids[inside]
            cells = rows * width + columns
            heights = z - terrain_flat[cells]
            valid_height = np.isfinite(heights) & (heights >= 0.0)
            cells, heights, tree_ids = cells[valid_height], heights[valid_height], tree_ids[valid_height]
            np.maximum.at(chm, cells, heights)
            labelled = tree_ids > 0
            if np.any(labelled):
                encoded = tree_ids[labelled] * cells_count + cells[labelled]
                pairs_by_chunk.append(np.unique(encoded))
    chm[~np.isfinite(chm)] = 0.0
    encoded_pairs = np.unique(np.concatenate(pairs_by_chunk)) if pairs_by_chunk else np.empty(0, dtype=np.int64)
    return chm.reshape(height, width), encoded_pairs


def hull_label(
    cells: np.ndarray,
    width: int,
    height: int,
) -> tuple[list[float], float] | None:
    rows = cells // width
    columns = cells % width
    corners = np.concatenate(
        (
            np.column_stack((columns, rows)),
            np.column_stack((columns + 1, rows)),
            np.column_stack((columns + 1, rows + 1)),
            np.column_stack((columns, rows + 1)),
        ),
        axis=0,
    ).astype(np.float64)
    corners = np.unique(corners, axis=0)
    if len(corners) < 3:
        return None
    try:
        hull = ConvexHull(corners)
    except QhullError:
        return None
    polygon = corners[hull.vertices]
    x, y = polygon[:, 0], polygon[:, 1]
    area_px = 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
    area_m2 = area_px * PIXEL_SIZE_M**2
    if area_m2 < MIN_CROWN_AREA_M2:
        return None
    values = np.column_stack((x / width, y / height)).ravel().tolist()
    return values, area_m2


def process_plot(job: dict) -> dict:
    las_path = Path(job["las_path"])
    image_path = Path(job["image_path"])
    label_path = Path(job["label_path"])
    raster_path = Path(job["raster_path"])
    if (
        not job["overwrite"]
        and image_path.is_file()
        and label_path.is_file()
        and raster_path.is_file()
    ):
        labels = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line]
        return {**job, "status": "reused", "instances": len(labels), "elapsed_seconds": 0.0}

    started = time.monotonic()
    with laspy.open(las_path) as reader:
        width, height, transform, crs = grid_from_header(reader.header)
    terrain, valid, terrain_method = build_terrain_and_coverage(
        las_path, width, height, transform
    )
    chm, encoded_pairs = rasterize_chm_and_tree_cells(
        las_path, width, height, transform, terrain
    )
    image = np.rint(np.clip(chm, 0, CHM_CLIP_MAX_M) / CHM_CLIP_MAX_M * 255).astype(np.uint8)
    image[~valid] = 0
    image_rgb = np.repeat(image[..., None], 3, axis=-1)

    cells_count = width * height
    tree_ids = encoded_pairs // cells_count
    cells = encoded_pairs % cells_count
    lines: list[str] = []
    instance_rows: list[dict] = []
    for tree_id in np.unique(tree_ids):
        tree_cells = cells[tree_ids == tree_id]
        result = hull_label(tree_cells, width, height)
        if result is None:
            continue
        values, area_m2 = result
        lines.append("0 " + " ".join(f"{value:.8f}" for value in values))
        instance_rows.append(
            {
                "dataset_id": job["dataset_id"],
                "source_tree_id": int(tree_id),
                "label_line": len(lines) - 1,
                "occupied_cells": len(tree_cells),
                "convex_hull_area_m2": area_m2,
            }
        )

    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(image_path)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "crs": crs,
        "nodata": -9999.0,
        "compress": "deflate",
        "predictor": 3,
    }
    with rasterio.open(raster_path, "w", **profile) as destination:
        values = chm.astype(np.float32)
        values[~valid] = -9999.0
        destination.write(values, 1)
        destination.set_band_description(1, "CHM metres")
        destination.update_tags(
            DATASET="ideas_als",
            DATASET_ID=job["dataset_id"],
            SOURCE_LAS=str(las_path),
            DTM_METHOD=terrain_method,
            TREE_ID_USED_FOR_INPUT="false",
            LABEL_GEOMETRY="convex hull of occupied treeID cells",
        )
    return {
        **job,
        "status": "created",
        "instances": len(lines),
        "width": width,
        "height": height,
        "crs": str(crs),
        "terrain_method": terrain_method,
        "instance_rows": instance_rows,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> int:
    args = parse_args()
    ideas_dir = args.ideas_dir.resolve()
    output_dir = args.output_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    split_rows = read_csv(ideas_dir / "data_split_metadata.csv")
    source_metadata = {
        row["path"]: row for row in read_csv(ideas_dir / "source_metadata.csv")
    }
    dev_rows = [row for row in split_rows if row["split"] == "dev"]
    if args.limit > 0:
        dev_rows = dev_rows[: args.limit]

    jobs = []
    for row in dev_rows:
        dataset_id = f"ideas__{row['folder'].lower()}__{Path(row['path']).stem}"
        jobs.append(
            {
                "dataset_id": dataset_id,
                "collection": row["folder"],
                "annotation_method": source_metadata[row["path"]]["annotation_method"],
                "source_split": row["split"],
                "las_path": str(ideas_dir / row["path"]),
                "image_path": str(output_dir / "images" / "train" / f"{dataset_id}.png"),
                "label_path": str(output_dir / "labels" / "train" / f"{dataset_id}.txt"),
                "raster_path": str(artifact_dir / f"{dataset_id}.tif"),
                "overwrite": args.overwrite,
            }
        )

    results = []
    if args.workers == 1:
        iterator = ((process_plot(job), job) for job in jobs)
        for index, (result, _) in enumerate(iterator, start=1):
            results.append(result)
            print(
                f"[{index}/{len(jobs)}] {result['dataset_id']}: "
                f"{result['instances']} instances ({result['status']})",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_map = {executor.submit(process_plot, job): job for job in jobs}
            for index, future in enumerate(as_completed(future_map), start=1):
                result = future.result()
                results.append(result)
                print(
                    f"[{index}/{len(jobs)}] {result['dataset_id']}: "
                    f"{result['instances']} instances ({result['status']})",
                    flush=True,
                )
    results.sort(key=lambda row: row["dataset_id"])

    manifest_rows: list[dict] = []
    instance_rows: list[dict] = []
    for result in results:
        manifest_rows.append(
            {
                "dataset_id": result["dataset_id"],
                "source_dataset": "ideas_als",
                "collection": result["collection"],
                "source_split": result["source_split"],
                "model_split": "train",
                "annotation_method": result["annotation_method"],
                "source_las": result["las_path"],
                "image": result["image_path"],
                "label": result["label_path"],
                "raw_raster": result["raster_path"],
                "gt_raster": "",
                "instances": result["instances"],
            }
        )
        instance_rows.extend(result.get("instance_rows", []))

    for row in read_csv(args.for_manifest.resolve()):
        split = row["model_split"]
        dataset_id = f"for__{row['dataset_id']}"
        image_source = Path(row["image"])
        label_source = Path(row["label"])
        image_destination = output_dir / "images" / split / f"{dataset_id}.png"
        label_destination = output_dir / "labels" / split / f"{dataset_id}.txt"
        link_or_copy(image_source, image_destination)
        link_or_copy(label_source, label_destination)
        instances = sum(1 for line in label_source.read_text(encoding="utf-8").splitlines() if line)
        manifest_rows.append(
            {
                "dataset_id": dataset_id,
                "source_dataset": "FOR-instance",
                "collection": row["collection"],
                "source_split": row["official_split"],
                "model_split": split,
                "annotation_method": "topmost_crown_reference",
                "source_las": row["source_las"],
                "image": str(image_destination),
                "label": str(label_destination),
                "raw_raster": row["raw_raster"],
                "gt_raster": row.get("gt_raster", ""),
                "instances": instances,
            }
        )

    yaml = (
        f"path: {output_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: tree\n"
    )
    (output_dir / "dataset.yaml").write_text(yaml, encoding="utf-8")
    write_csv(
        PROJECT_DIR / "manifests" / "dataset_manifest_chm_only_ideas_combined.csv",
        manifest_rows,
        list(manifest_rows[0]),
    )
    if instance_rows:
        write_csv(
            PROJECT_DIR / "manifests" / "ideas_instance_tree_id_map.csv",
            instance_rows,
            list(instance_rows[0]),
        )
    plot_counts = Counter(row["model_split"] for row in manifest_rows)
    instance_counts = Counter()
    for row in manifest_rows:
        instance_counts[row["model_split"]] += int(row["instances"])
    summary = {
        "dataset": "ideas_als/dev + FOR-instance",
        "ideas_official_test_used": False,
        "input_channels_use_tree_id": False,
        "input_encoding": "normalized CHM repeated identically into RGB",
        "label_encoding_ideas": "convex hull of 0.5 m cells occupied by treeID points",
        "pixel_size_m": PIXEL_SIZE_M,
        "chm_clip_max_m": CHM_CLIP_MAX_M,
        "plots_by_split": dict(sorted(plot_counts.items())),
        "instances_by_split": dict(sorted(instance_counts.items())),
        "ideas_collections": dict(sorted(Counter(row["collection"] for row in results).items())),
        "ideas_annotation_methods": dict(
            sorted(Counter(row["annotation_method"] for row in results).items())
        ),
    }
    reports_dir = PROJECT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "ideas_chm_only_dataset.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
