#!/usr/bin/env python3
"""Prepare direct point-cloud training data from ideas_als and FOR-instance.

The network input is built from XYZ and intensity. ``treeID`` is retained only
as the instance-supervision target. The official ideas_als test partition is
never used for training or validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import laspy
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_IDEAS = WORKSPACE_DIR / "ideas_als"
DEFAULT_FOR = WORKSPACE_DIR / "FOR-instance"
DEFAULT_FOR_MANIFEST = (
    PROJECT_DIR / "reused" / "for_instance_chm_gt_0p5m" / "for_instance_file_manifest.csv"
)
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "default.json"
DEFAULT_OUTPUT = PROJECT_DIR / "pointcloud_dataset_0p25m"
DEFAULT_MANIFEST = PROJECT_DIR / "manifests" / "pointcloud_dataset_0p25m.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ideas-dir", type=Path, default=DEFAULT_IDEAS)
    parser.add_argument("--for-dir", type=Path, default=DEFAULT_FOR)
    parser.add_argument("--for-manifest", type=Path, default=DEFAULT_FOR_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--voxel-size", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def make_jobs(args: argparse.Namespace) -> list[dict]:
    ideas_dir = args.ideas_dir.resolve()
    for_dir = args.for_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_metadata = {
        row["path"]: row for row in read_csv(ideas_dir / "source_metadata.csv")
    }
    jobs: list[dict] = []
    for row in read_csv(ideas_dir / "data_split_metadata.csv"):
        if row["split"] != "dev":
            continue
        dataset_id = f"ideas__{row['folder'].lower()}__{Path(row['path']).stem}"
        jobs.append(
            {
                "dataset_id": dataset_id,
                "source_dataset": "ideas_als",
                "collection": row["folder"],
                "official_split": row["split"],
                "model_split": "train",
                "annotation_method": source_metadata[row["path"]]["annotation_method"],
                "source_las": str(ideas_dir / row["path"]),
                "output": str(output_dir / "train" / f"{dataset_id}.npz"),
                "voxel_size": args.voxel_size,
                "overwrite": args.overwrite,
            }
        )

    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    validation = set(config["validation_dataset_ids"])
    for row in read_csv(args.for_manifest.resolve()):
        if row["split"] == "test":
            model_split = "test"
        elif row["dataset_id"] in validation:
            model_split = "val"
        else:
            model_split = "train"
        jobs.append(
            {
                "dataset_id": f"for__{row['dataset_id']}",
                "source_dataset": "FOR-instance",
                "collection": row["collection"],
                "official_split": row["split"],
                "model_split": model_split,
                "annotation_method": "point_native",
                "source_las": str(for_dir / row["source_las"]),
                "output": str(output_dir / model_split / f"for__{row['dataset_id']}.npz"),
                "gt_raster": str(
                    args.for_manifest.resolve().parent / row["topmost_gt_raster"]
                ),
                "gt_vector": str(
                    args.for_manifest.resolve().parent / row["topmost_gt_vector"]
                ),
                "voxel_size": args.voxel_size,
                "overwrite": args.overwrite,
            }
        )
    jobs.sort(key=lambda item: item["dataset_id"])
    return jobs[: args.limit] if args.limit > 0 else jobs


def voxel_indices(xyz: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    origin = np.min(xyz, axis=0)
    grid = np.floor((xyz - origin) / voxel_size).astype(np.int32)
    extent = grid.max(axis=0).astype(np.int64) + 1
    key = grid[:, 0].astype(np.int64)
    key += extent[0] * grid[:, 1].astype(np.int64)
    key += extent[0] * extent[1] * grid[:, 2].astype(np.int64)
    _, index = np.unique(key, return_index=True)
    index.sort()
    return index, origin


def process(job: dict) -> dict:
    started = time.monotonic()
    source = Path(job["source_las"])
    output = Path(job["output"])
    if output.is_file() and not job["overwrite"]:
        with np.load(output) as data:
            return {
                **job,
                "status": "reused",
                "raw_points": int(data["raw_points"]),
                "voxels": int(len(data["coord"])),
                "tree_voxels": int(np.count_nonzero(data["tree_id"] > 0)),
                "instances": int(len(np.unique(data["tree_id"][data["tree_id"] > 0]))),
                "elapsed_seconds": time.monotonic() - started,
            }

    las = laspy.read(source)
    dimensions = set(las.point_format.dimension_names)
    if "treeID" not in dimensions:
        raise ValueError(f"Missing treeID dimension: {source}")
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    intensity = np.asarray(las.intensity, dtype=np.float32)
    tree_id = np.asarray(las.treeID, dtype=np.int64)
    finite = np.all(np.isfinite(xyz), axis=1) & np.isfinite(intensity)
    xyz, intensity, tree_id = xyz[finite], intensity[finite], tree_id[finite]
    raw_points = len(xyz)
    if raw_points == 0:
        raise ValueError(f"No finite points: {source}")

    selected, source_origin = voxel_indices(xyz, float(job["voxel_size"]))
    xyz = xyz[selected]
    intensity = intensity[selected]
    tree_id = tree_id[selected]
    coord = (xyz - source_origin).astype(np.float32)
    grid_coord = np.floor(coord / float(job["voxel_size"])).astype(np.int32)

    low, high = np.percentile(intensity, (1.0, 99.0))
    if high <= low:
        high = low + 1.0
    intensity = np.clip((intensity - low) / (high - low), 0.0, 1.0).astype(np.float32)

    instance_offset = np.zeros_like(coord, dtype=np.float32)
    positive_ids = np.unique(tree_id[tree_id > 0])
    for value in positive_ids:
        mask = tree_id == value
        center = np.mean(coord[mask], axis=0, dtype=np.float64).astype(np.float32)
        instance_offset[mask] = center - coord[mask]

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        coord=coord,
        grid_coord=grid_coord,
        intensity=intensity,
        tree_id=tree_id.astype(np.int32),
        instance_offset=instance_offset,
        source_origin=source_origin.astype(np.float64),
        voxel_size=np.float32(job["voxel_size"]),
        raw_points=np.int64(raw_points),
    )
    return {
        **job,
        "status": "created",
        "raw_points": raw_points,
        "voxels": len(coord),
        "tree_voxels": int(np.count_nonzero(tree_id > 0)),
        "instances": int(len(positive_ids)),
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> int:
    args = parse_args()
    if not 0 < args.voxel_size <= 1:
        raise ValueError("voxel-size must be in (0, 1]")
    jobs = make_jobs(args)
    results: list[dict] = []
    if args.workers <= 1:
        for index, job in enumerate(jobs, start=1):
            result = process(job)
            results.append(result)
            print(
                f"[{index}/{len(jobs)}] {result['dataset_id']}: "
                f"{result['voxels']:,} voxels, {result['instances']} trees",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process, job): job for job in jobs}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                print(
                    f"[{index}/{len(jobs)}] {result['dataset_id']}: "
                    f"{result['voxels']:,} voxels, {result['instances']} trees",
                    flush=True,
                )
    results.sort(key=lambda item: item["dataset_id"])
    fields = [
        "dataset_id",
        "source_dataset",
        "collection",
        "official_split",
        "model_split",
        "annotation_method",
        "source_las",
        "output",
        "gt_raster",
        "gt_vector",
        "raw_points",
        "voxels",
        "tree_voxels",
        "instances",
        "voxel_size",
    ]
    args.manifest.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.resolve().open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    by_split: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        subset = [row for row in results if row["model_split"] == split]
        by_split[split] = {
            "plots": len(subset),
            "raw_points": sum(int(row["raw_points"]) for row in subset),
            "voxels": sum(int(row["voxels"]) for row in subset),
            "instances": sum(int(row["instances"]) for row in subset),
        }
    report = {
        "voxel_size_m": args.voxel_size,
        "features": ["centered_x", "centered_y", "height", "normalized_intensity"],
        "target": "per-voxel tree/non-tree and 3D offset to instance centroid",
        "tree_id_used_as_input": False,
        "ideas_official_test_used": False,
        "splits": by_split,
        "manifest": str(args.manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
    }
    report_path = PROJECT_DIR / "reports" / "pointcloud_dataset_0p25m.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
