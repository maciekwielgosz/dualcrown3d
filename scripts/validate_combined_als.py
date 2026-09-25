#!/usr/bin/env python3
"""Validate copied data and crown geometry without changing any input."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "combined_als_crowns_v1")
    root = parser.parse_args().root.resolve()
    with (root / "manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    issues, missing_point_ids = [], []
    total = Counter()
    for i, row in enumerate(rows):
        gt = gpd.read_file(row["gt_vector"])
        if not gt.geometry.is_valid.all() or gt.geometry.is_empty.any() or (gt.geometry.area <= 0).any():
            issues.append(dict(dataset_id=row["dataset_id"], issue="invalid_crown_geometry"))
        if not gt.treeID.is_unique:
            issues.append(dict(dataset_id=row["dataset_id"], issue="duplicate_tree_ID"))
        with np.load(row["output"]) as data:
            if not np.isfinite(data["coord"]).all() or not np.isfinite(data["intensity"]).all():
                issues.append(dict(dataset_id=row["dataset_id"], issue="nonfinite_model_input"))
            grid = data["grid_coord"].astype(np.int64)
            extent = grid.max(0)+1
            keys = grid[:,0]+extent[0]*grid[:,1]+extent[0]*extent[1]*grid[:,2]
            if len(np.unique(keys)) != len(keys):
                issues.append(dict(dataset_id=row["dataset_id"], issue="duplicate_sparse_grid"))
            missing = set(gt.treeID.astype(int))-set(np.unique(data["tree_id"]))
            if missing:
                missing_point_ids.append(dict(dataset_id=row["dataset_id"], ids=sorted(map(int, missing))))
        with rasterio.open(row["chm"]) as chm:
            if chm.tags().get("uses_tree_id") != "false":
                issues.append(dict(dataset_id=row["dataset_id"], issue="missing_label_independence_marker"))
        total[row["model_split"]+"_crowns"] += len(gt)
        if (i+1) % 100 == 0:
            print(f"Validated {i+1}/{len(rows)}", flush=True)
    result = dict(plots=len(rows), crown_counts=dict(total), issues=issues, crowns_without_voxel_labels=missing_point_ids,
                  note="Small crowns can lose all support under 0.25 m voxelization; retained in full-polygon evaluation, not silently dropped.")
    (root / "validation_report.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
