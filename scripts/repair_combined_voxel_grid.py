#!/usr/bin/env python3
"""Repair grid indices only; assert coordinates and supervision do not change.

The original sampling origin must also be the integer-grid origin. Re-centering
on the min of the already selected points can merge neighboring voxels. Cache
replacements are atomic, so concurrent training readers see a complete NPZ.
"""
import csv
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import laspy
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt, median_filter, minimum_filter

from build_combined_als import ROOT, sha256, voxel_indices, write_csv

DATASET = ROOT / "combined_als_crowns_v1"


def repair(row):
    cache = Path(row["output"])
    metadata = cache.parent / "metadata.json"
    meta = json.loads(metadata.read_text())
    if meta.get("grid_version") == 2:
        return meta
    with np.load(cache) as data:
        arrays = {k: data[k] for k in data.files}
    original_coord_hash = __import__("hashlib").sha256(arrays["coord"].tobytes()).hexdigest()
    las = laspy.read(row["source_las"])
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    classification = np.asarray(las.classification)
    finite = np.isfinite(xyz).all(1)
    xyz, classification = xyz[finite], classification[finite]
    with rasterio.open(row["chm"]) as source:
        shape, transform = source.shape, source.transform
    h, w = shape
    rr, cc = rasterio.transform.rowcol(transform, xyz[:, 0], xyz[:, 1])
    rr, cc = np.clip(rr, 0, h-1), np.clip(cc, 0, w-1)
    cells = rr*w+cc
    ground = classification == 2
    if xyz[:, 2].min() <= 2.:
        dtm = np.zeros(shape)
    elif ground.any():
        sums = np.bincount(cells[ground], weights=xyz[ground, 2], minlength=h*w)
        counts = np.bincount(cells[ground], minlength=h*w)
        dtm = (sums/np.maximum(counts, 1)).reshape(shape)
        nearest = distance_transform_edt(counts.reshape(shape) == 0, return_distances=False, return_indices=True)
        dtm = dtm[tuple(nearest)]
    else:
        low = np.full(h*w, np.inf)
        np.minimum.at(low, cells, xyz[:, 2])
        low = low.reshape(shape)
        nearest = distance_transform_edt(~np.isfinite(low), return_distances=False, return_indices=True)
        dtm = median_filter(minimum_filter(low[tuple(nearest)], size=5), size=9)
    xyz[:, 2] -= dtm[rr, cc]
    selected, origin = voxel_indices(xyz, .25)
    coord = (xyz[selected]-arrays["source_origin"]).astype(np.float32)
    if not np.array_equal(coord, arrays["coord"]):
        raise RuntimeError(f"Coordinates changed; refusing repair: {row['dataset_id']}")
    grid = np.floor((xyz[selected]-origin)/.25).astype(np.int32)
    extent = grid.max(0).astype(np.int64)+1
    key = grid[:,0]+extent[0]*grid[:,1]+extent[0]*extent[1]*grid[:,2]
    if len(np.unique(key)) != len(key):
        raise RuntimeError("Rebuilt grid still contains duplicates")
    old = arrays["grid_coord"].astype(np.int64)
    extent = old.max(0)+1
    old_key = old[:,0]+extent[0]*old[:,1]+extent[0]*extent[1]*old[:,2]
    collisions = len(old_key)-len(np.unique(old_key))
    recovery = cache.parent / "grid_coord_v1_recovery.npz"
    if not recovery.exists():
        np.savez_compressed(recovery, grid_coord=arrays["grid_coord"])
    arrays["grid_coord"], arrays["voxel_origin"] = grid, origin
    tmp = cache.with_name("points_0p25m.gridfix.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, cache)
    meta.update(grid_version=2, repaired_grid_collisions=collisions, coordinates_sha256=original_coord_hash,
                supervision_changed=False, npz_sha256=sha256(cache))
    tmp_meta = metadata.with_suffix(".tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2)+"\n")
    os.replace(tmp_meta, metadata)
    return meta


def main():
    with (DATASET / "manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    results = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for i, result in enumerate(executor.map(repair, rows), 1):
            results.append(result)
            if i % 50 == 0:
                print(f"Repaired {i}/{len(rows)}", flush=True)
    for name in ("manifest.csv", "manifest_train.csv", "manifest_val.csv", "manifest_test.csv", "READY.json"):
        backup = DATASET / (name+".grid_v1")
        if not backup.exists():
            shutil.copy2(DATASET / name, backup)
    write_csv(DATASET / "manifest.csv", results)
    for split in ("train", "val", "test"):
        write_csv(DATASET / f"manifest_{split}.csv", [r for r in results if r["model_split"] == split])
    summary = dict(plots=len(results), grid_version=2, repaired_collisions=sum(r["repaired_grid_collisions"] for r in results),
                   coordinates_and_supervision_unchanged=True, split_unchanged=True,
                   previous_full_inference_metrics_require_reevaluation=True,
                   manifest_sha256=sha256(DATASET / "manifest.csv"))
    (DATASET / "grid_repair_report.json").write_text(json.dumps(summary, indent=2)+"\n")
    (DATASET / "READY.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
