#!/usr/bin/env python3
"""Stage-1b candidate-graph connectivity on the frozen feasibility plots."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz
from pointcloud.superpoints.partition import geometric_partition, centroid_graph_connectivity

SOURCE = PROJECT / "outputs/dualcrown3d_superpoint_v1/stage1/summary.json"
INVENTORY = PROJECT / "outputs/dualcrown3d_superpoint_v1/stage0/inventory.csv"
OUTPUT = PROJECT / "outputs/dualcrown3d_superpoint_v1/stage1_connectivity"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Completed connectivity audit protected: {OUTPUT}")
    selected = set(json.loads(SOURCE.read_text())["selected_plots"])
    with INVENTORY.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["dataset_id"] in selected and row["split"] == "val"]
    if len(rows) != 5:
        raise ValueError(f"Expected five representative validation plots, got {len(rows)}")
    OUTPUT.mkdir(parents=True)
    findings = []
    for row in rows:
        data = load_npz(row["npz"])
        group = geometric_partition(data["coord"], .5)
        for radius, neighbors in ((4., 32), (8., 64)):
            start = time.monotonic()
            result = centroid_graph_connectivity(data["coord"], data["tree_id"],
                group, radius_m=radius, k=neighbors)
            record = dict(dataset_id=row["dataset_id"], collection=row["collection"],
                          cell_m=.5, graph_radius_m=radius, graph_k=neighbors,
                          seconds=time.monotonic() - start, **result)
            findings.append(record)
            print(json.dumps(record), flush=True)
    with (OUTPUT / "per_plot.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(findings[0]))
        writer.writeheader()
        writer.writerows(findings)
    (OUTPUT / "summary.json").write_text(json.dumps(findings, indent=2) + "\n")


if __name__ == "__main__":
    main()
