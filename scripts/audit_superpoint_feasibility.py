#!/usr/bin/env python3
"""Stage-1 fixed-partition and GT-assisted superpoint oracle diagnostics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz
from pointcloud.superpoints.partition import (
    geometric_partition, neighbor_pairs, partition_diagnostics,
)

INVENTORY = PROJECT / "outputs/dualcrown3d_superpoint_v1/stage0/inventory.csv"
OUTPUT = PROJECT / "outputs/dualcrown3d_superpoint_v1/stage1"


def representative_rows(inventory: Path) -> list[dict]:
    with inventory.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (PROJECT.parent / "combined_als_crowns_supervision_v4/manifest.csv").open(newline="") as stream:
        eligible = {row["dataset_id"] for row in csv.DictReader(stream)
                    if (row["model_split"] == "train" and row["train_eligible"] == "true")
                    or (row["model_split"] == "val" and row["point_eval_eligible"] == "true")}
    groups = defaultdict(list)
    for row in rows:
        if (row["kind"] != "native_als" or row["split"] not in ("train", "val")
                or row["dataset_id"] not in eligible):
            continue
        groups[(row["split"], row["source_dataset"], row["collection"])].append(row)
    selected = []
    for key, candidates in sorted(groups.items()):
        # Median voxel count is deterministic and avoids choosing easy extremes.
        ordered = sorted(candidates, key=lambda row: (int(row["voxels"]), row["dataset_id"]))
        selected.append(ordered[len(ordered) // 2])
    return selected


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cells", type=float, nargs="+", default=[.5, 1., 1.5])
    parser.add_argument("--neighbor-radius", type=float, default=1.)
    args = parser.parse_args()
    if len(set(args.cells)) != len(args.cells) or any(value <= 0 for value in args.cells):
        raise ValueError("Distinct positive cell sizes required")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Completed stage-1 output protected: {output}")
    rows = representative_rows(args.inventory)
    if not rows:
        raise ValueError("No representative native ALS rows")
    output.mkdir(parents=True)
    results = []
    for index, row in enumerate(rows, 1):
        arrays = load_npz(row["npz"])
        coord, truth = arrays["coord"], arrays["tree_id"]
        started = time.monotonic()
        edges = neighbor_pairs(coord, args.neighbor_radius)
        graph_seconds = time.monotonic() - started
        for cell in args.cells:
            started = time.monotonic()
            groups = geometric_partition(coord, cell)
            diag = partition_diagnostics(coord, truth, groups, edges)
            record = dict(dataset_id=row["dataset_id"], split=row["split"],
                          source_dataset=row["source_dataset"],
                          collection=row["collection"], cell_m=cell,
                          neighbor_radius_m=args.neighbor_radius,
                          neighbor_edges=len(edges), graph_seconds=graph_seconds,
                          partition_seconds=time.monotonic() - started,
                          **{key: value for key, value in diag.items() if key != "oracle"},
                          **{f"oracle_{key}": value for key, value in diag["oracle"].items()})
            results.append(record)
            write_csv(output / "per_plot_progress.csv", results)
            print(f"{index}/{len(rows)} {row['dataset_id']} cell={cell}: "
                  f"compression={diag['compression']:.2f} "
                  f"oracle PQ={diag['oracle']['pq']:.3f} "
                  f"boundary={diag['boundary_recall']}", flush=True)
    write_csv(output / "per_plot.csv", results)
    summary = []
    for cell in args.cells:
        current = [r for r in results if r["cell_m"] == cell]
        for split in ("train", "val", "all"):
            subset = [r for r in current if split == "all" or r["split"] == split]
            measured = [r["boundary_recall"] for r in subset if r["boundary_recall"] is not None]
            summary.append(dict(cell_m=cell, split=split, plots=len(subset),
                mean_compression=float(np.mean([r["compression"] for r in subset])),
                mean_oracle_pq=float(np.mean([r["oracle_pq"] for r in subset])),
                mean_oracle_f1=float(np.mean([r["oracle_f1"] for r in subset])),
                mean_boundary_recall=float(np.mean(measured)) if measured else None,
                mean_known_point_purity=float(np.mean([r["known_point_purity"] for r in subset])),
                mean_mixed_group_fraction=float(np.mean([r["mixed_group_fraction"] for r in subset])),
                disconnected_trees=sum(r["graph_disconnected_trees"] for r in subset),
                oracle_trees=sum(r["gt_trees"] for r in subset)))
    write_csv(output / "summary.csv", summary)
    (output / "summary.json").write_text(json.dumps(dict(
        method="3-D geometric cells; GT majority vote per cell; same-ID grouping is a diagnostic oracle only",
        selection="median-size eligible native ALS plot per split/source collection",
        selected_plots=[r["dataset_id"] for r in rows],
        ignores_unknown_labels=True, test_used=False,
        local_graph=dict(radius_m=args.neighbor_radius, k=8),
        summary=summary), indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
