#!/usr/bin/env python3
"""Prepare optional high-quality close-range tree instances.

These data are deliberately kept separate from the default aerial split. They
can be mixed in only for an explicit sensor-domain augmentation experiment.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import laspy
import numpy as np
from plyfile import PlyData


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
EXTERNAL = WORKSPACE / "external_lidar_datasets"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_external_aerial import prepared_record, prepare_arrays, sha256, write_csv  # noqa: E402


def prepare_treescan_one(job: tuple[Path, Path]) -> tuple[str, dict]:
    path, target = job
    cloud = laspy.read(path)
    source_ids = np.asarray(cloud.treeID, dtype=np.int64)
    complete = np.asarray(cloud.completelyInside) > 0
    ids = np.where((source_ids > 0) & complete, source_ids, 0)
    ids[(source_ids > 0) & ~complete] = -1
    row = prepare_arrays(
        dataset_id=f"treescanpl10k__{path.stem.lower()}",
        collection="TREESCANPL10K",
        source_dataset="external_tls",
        source_path=path,
        folder=target,
        xyz=np.column_stack((cloud.x, cloud.y, cloud.z)),
        intensity=np.asarray(cloud.intensity),
        ids=ids,
        classification=np.zeros(len(cloud.points), dtype=np.uint8),
        ground_class=255,
        boundary_margin=-1.0,
        annotation_method="automated_segmentation_expert_corrected_and_quality_controlled",
        licence="CC-BY-4.0",
        doi="10.5281/zenodo.19127709",
    )
    return path.name, row


def prepare_treescan(
    source_root: Path, output: Path, overwrite: bool, workers: int
) -> list[dict]:
    paths = sorted((source_root / "TreeScanPL10k" / "original").glob("*.laz"))
    rows_by_path: dict[Path, dict] = {}
    jobs = []
    for path in paths:
        target = output / "treescanpl10k" / path.stem
        done = prepared_record(target)
        if done is not None and not overwrite:
            rows_by_path[path] = done
        else:
            jobs.append((path, target))
    if workers == 1:
        results = map(prepare_treescan_one, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(prepare_treescan_one, jobs)
    try:
        for (path, _), (name, row) in zip(jobs, results):
            rows_by_path[path] = row
            print(f"Prepared {name}: {row['instances']} complete trees", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    return [rows_by_path[path] for path in paths]


def read_avocado(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    record = np.dtype(
        [
            ("x", "<f8"),
            ("y", "<f8"),
            ("z_ned", "<f8"),
            ("matter", "<u4"),
            ("tree", "<u4"),
            ("height", "<f8"),
        ]
    )
    trailing = path.stat().st_size % record.itemsize
    if trailing:
        print(f"Warning: ignoring {trailing} trailing bytes in {path}", flush=True)
    data = np.fromfile(path, dtype=record, count=path.stat().st_size // record.itemsize)
    xyz = np.column_stack((data["x"], data["y"], data["height"]))
    source_ids = data["tree"].astype(np.int64)
    ids = np.where(np.isin(source_ids, [1, 2, 3]), source_ids, 0)
    ids[source_ids == 4] = -1
    classification = np.where(source_ids == 0, 2, 5).astype(np.uint8)
    return xyz, np.zeros(len(data), dtype=np.float32), ids, classification


def prepare_avocado(source_root: Path, output: Path, overwrite: bool) -> list[dict]:
    rows = []
    root = source_root / "AvocadoTrees" / "original"
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".bin")
    for path in paths:
        relative = path.relative_to(root).with_suffix("")
        slug = "__".join(relative.parts).replace(" ", "_").lower()
        target = output / "avocado_manual" / slug
        done = prepared_record(target)
        if done is not None and not overwrite:
            rows.append(done)
            continue
        xyz, intensity, ids, classification = read_avocado(path)
        rows.append(
            prepare_arrays(
                dataset_id=f"avocado_manual__{slug}",
                collection="AVOCADO_MANUAL",
                source_dataset="external_close_range",
                source_path=path,
                folder=target,
                xyz=xyz,
                intensity=intensity,
                ids=ids,
                classification=classification,
                ground_class=2,
                boundary_margin=-1.0,
                annotation_method="manual_point_instance",
                licence="CC-BY-4.0",
                doi="10.17632/h49fpprg6c.1",
            )
        )
        print(f"Prepared {path.name}: {rows[-1]['instances']} complete trees", flush=True)
    return rows


def prepare_tls_benchmark(source_root: Path, output: Path, overwrite: bool) -> list[dict]:
    rows = []
    root = source_root / "TLSBenchmark" / "original"
    paths = sorted(root.glob("*/*_train.ply"))
    for path in paths:
        site = path.stem.removesuffix("_train")
        target = output / "tls_benchmark_manual" / site
        done = prepared_record(target)
        if done is not None and not overwrite:
            rows.append(done)
            continue
        vertex = PlyData.read(path, mmap=True)["vertex"].data
        semantic = np.asarray(vertex["semantic"], dtype=np.int32)
        source_ids = np.asarray(vertex["instance"], dtype=np.int64)
        ids = np.where((semantic == 1) & (source_ids > 0), source_ids, 0)
        rows.append(
            prepare_arrays(
                dataset_id=f"tls_benchmark__{site}",
                collection=f"TLS_BENCHMARK_{site.upper()}",
                source_dataset="external_tls",
                source_path=path,
                folder=target,
                xyz=np.column_stack((vertex["x"], vertex["y"], vertex["z"])),
                intensity=np.zeros(len(vertex), dtype=np.float32),
                ids=ids,
                classification=semantic.astype(np.uint8),
                ground_class=0,
                boundary_margin=0.25,
                annotation_method="manual_point_instance_second_operator_quality_control",
                licence="CC-BY-4.0",
                doi="10.5281/zenodo.16875688",
            )
        )
        print(f"Prepared {path.name}: {rows[-1]['instances']} complete trees", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=EXTERNAL)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "external_tls_tree_instances_v1")
    parser.add_argument(
        "--sources", choices=("all", "treescan", "avocado", "tlsbenchmark"), default="all"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Parallel TreeScan plots")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.sources in ("all", "treescan"):
        rows.extend(prepare_treescan(args.source_root, args.output, args.overwrite, args.workers))
    if args.sources in ("all", "avocado"):
        rows.extend(prepare_avocado(args.source_root, args.output, args.overwrite))
    if args.sources in ("all", "tlsbenchmark"):
        rows.extend(prepare_tls_benchmark(args.source_root, args.output, args.overwrite))
    write_csv(args.output / "manifest.csv", rows)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sensor_domain": "TLS_MLS_close_range_optional",
        "not_in_default_aerial_split": True,
        "plots": len(rows),
        "instances": sum(int(row["instances"]) for row in rows),
        "voxels": sum(int(row["voxels"]) for row in rows),
        "manifest_sha256": sha256(args.output / "manifest.csv") if rows else "",
    }
    (args.output / "READY.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
