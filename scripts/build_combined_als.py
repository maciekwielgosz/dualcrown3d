#!/usr/bin/env python3
"""Copy labelled ALS, freeze spatial groups, normalize heights and prepare full crowns.

Original datasets are never changed. Original split tags are retained as metadata.
Bounds overlapping (or within 2 m) in a known common CRS are indivisible. Local-frame
plots cannot be checked geographically; this limitation is recorded explicitly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt, median_filter, minimum_filter
from shapely.geometry import box

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
RUN_R_CODE = ROOT / "run_r" / "code"
REFERENCE_CODE = PROJECT / "reused" / "reference_scripts"
sys.path.insert(0, str(RUN_R_CODE if RUN_R_CODE.is_dir() else REFERENCE_CODE))
import for_instance_to_chm_gt as converter
from prepare_pointcloud_dataset import voxel_indices


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory():
    metadata = {r["path"]: r for r in read_csv(ROOT / "ideas_als/source_metadata.csv")}
    rows, missing = [], []
    for dataset in ("FOR-instance", "ideas_als"):
        for item in read_csv(ROOT / dataset / "data_split_metadata.csv"):
            path = ROOT / dataset / item["path"]
            if not path.exists():
                missing.append({"source_dataset": dataset, "path": item["path"], "reason": "missing_on_disk"})
                continue
            with laspy.open(path) as reader:
                header = reader.header
                crs = header.parse_crs()
                # These two missing VLRs have documented projected coordinates.
                spatial_crs = crs.to_epsg() if crs else {"RMIT": 28355, "TUWIEN": 32633}.get(item["folder"])
                bounds = [*header.mins[:2], *header.maxs[:2]]
            prefix = "for" if dataset == "FOR-instance" else "ideas"
            rows.append(dict(dataset_id=f"{prefix}__{item['folder']}__{path.stem}",
                             source_dataset=dataset, collection=item["folder"],
                             official_split=item["split"], original_las=str(path),
                             annotation_method=metadata[item["path"]]["annotation_method"] if dataset == "ideas_als" else "point_native",
                             spatial_crs=spatial_crs, bounds=list(map(float, bounds)),
                             crs_wkt=crs.to_wkt() if crs else "", source_sha256=sha256(path)))
    return rows, missing


def assign_splits(rows, seed, policy="combined"):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        parent[find(j)] = find(i)
    links = []
    for i, row in enumerate(rows):
        for j in range(i):
            other = rows[j]
            same_file = row["source_sha256"] == other["source_sha256"]
            close = (row["spatial_crs"] is not None and row["spatial_crs"] == other["spatial_crs"]
                     and box(*row["bounds"]).distance(box(*other["bounds"])) <= 2.0)
            if same_file or close:
                union(i, j)
                links.append(dict(a=row["dataset_id"], b=other["dataset_id"], reason="identical_file" if same_file else "bounds_distance_le_2m"))
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    dev = defaultdict(list)
    for group in groups.values():
        gid = hashlib.sha256("|".join(sorted(r["dataset_id"] for r in group)).encode()).hexdigest()[:16]
        split = "test" if policy == "official" and any(r["official_split"] == "test" for r in group) else "train"
        for row in group:
            row.update(group_id=gid, model_split=split)
        if split == "train":
            dev[tuple(sorted({r["source_dataset"] + ":" + r["collection"] for r in group}))].append(group)
    for stratum, candidates in sorted(dev.items()):
        # Never split a plot, including collections represented by one plot.
        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda g: hashlib.sha256(f"{seed}:{g[0]['group_id']}".encode()).hexdigest())
        if policy == "official":
            selections = [("val", candidates[:max(1, round(.2 * len(candidates)))])]
        else:
            ntest = max(1, round(.15 * len(candidates)))
            nval = max(1, round(.15 * len(candidates))) if len(candidates) >= 3 else 0
            selections = [("test", candidates[:ntest]), ("val", candidates[ntest:ntest+nval])]
        for split, selected in selections:
            for group in selected:
                for row in group:
                    row["model_split"] = split
    assert all(len({r["model_split"] for r in g}) == 1 for g in groups.values())
    return links


def full_crowns(row, xy, ids, transform, shape):
    source = Path(row["original_las"])
    polygon_source = None
    if row["collection"] == "ECODSE":
        name = source.stem.removeprefix("plot_").removesuffix("_annotated")
        polygon_source = ROOT / f"ideas_als/_build/vectors/ecodse_{name}.gpkg"
    elif row["collection"] == "IDTREES":
        tag, site = re.match(r"plot_(train|test)_([A-Z]+)_", source.stem).groups()
        polygon_source = ROOT / f"ideas_als/_build/vectors/idtrees_{tag}_{site}.gpkg"
    crs = row["crs_wkt"] or (f"EPSG:{row['spatial_crs']}" if row["spatial_crs"] else None)
    if polygon_source:
        frame = gpd.read_file(polygon_source, bbox=tuple(row["bounds"]))
        frame = frame[frame.treeID.isin(np.unique(ids[ids > 0]))].copy()
        footprint = box(*row["bounds"])
        frame.geometry = frame.geometry.intersection(footprint)
        frame = frame[~frame.geometry.is_empty & (frame.geometry.area >= 0.25)].copy()
        frame = frame[["treeID", "geometry"]]
        method = "official_crown_polygon_clipped_to_plot"
    else:
        # Preserve the established full (overlapping) crown projection protocol,
        # not the visible/topmost target used by the earlier YOLO experiments.
        rr, cc = rasterio.transform.rowcol(transform, xy[:, 0], xy[:, 1])
        rr, cc = np.clip(rr, 0, shape[0]-1), np.clip(cc, 0, shape[1]-1)
        records = []
        for identifier in np.unique(ids[ids > 0]):
            mask = np.zeros(shape, dtype=bool)
            selected = ids == identifier
            mask[rr[selected], cc[selected]] = True
            # Pad so closing cannot create artificial foreground at plot edges.
            clean = converter.clean_crown_mask(np.pad(mask, 2))[2:-2, 2:-2]
            geom = converter.mask_geometry(clean, transform)
            if not geom.is_empty and geom.area >= 0.25:
                records.append(dict(treeID=int(identifier), geometry=geom))
        frame = gpd.GeoDataFrame(records, columns=["treeID", "geometry"], geometry="geometry", crs=crs)
        method = "full_labelled_point_projection_0p5m_closing_fill_holes_largest_component"
    frame["evaluation_eligible"] = 1
    frame["gt_method"] = method
    return frame, method, polygon_source


def prepare(row):
    row = dict(row)
    folder = Path(row["output"]).parent
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / "metadata.json"
    if done.exists():
        cached = json.loads(done.read_text())
        if cached["source_sha256"] != row["source_sha256"]:
            raise RuntimeError(f"Source changed: {row['dataset_id']}")
        return cached
    original, copy = Path(row["original_las"]), Path(row["source_las"])
    if copy.exists() and sha256(copy) != row["source_sha256"]:
        raise RuntimeError(f"Unexpected existing copy: {copy}")
    if not copy.exists():
        shutil.copy2(original, copy)
    if sha256(copy) != row["source_sha256"]:
        raise RuntimeError(f"Copy verification failed: {copy}")
    las = laspy.read(copy)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    ids = np.asarray(las.treeID).astype(np.int64)
    intensity = np.asarray(las.intensity).astype(np.float32)
    classification = np.asarray(las.classification)
    finite = np.isfinite(xyz).all(axis=1)
    xyz, ids, intensity, classification = xyz[finite], ids[finite], intensity[finite], classification[finite]
    raw_points = len(xyz)
    left, bottom = np.floor(xyz[:, :2].min(axis=0) * 2) / 2
    right, top = np.ceil(xyz[:, :2].max(axis=0) * 2) / 2
    width, height = max(1, int(round((right-left)*2))), max(1, int(round((top-bottom)*2)))
    shape = (height, width)
    transform = from_origin(left, top, 0.5, 0.5)
    rr, cc = rasterio.transform.rowcol(transform, xyz[:, 0], xyz[:, 1])
    rr, cc = np.clip(rr, 0, height-1), np.clip(cc, 0, width-1)
    cells = rr * width + cc
    # Normalization reads XYZ and terrain class only, never instance IDs.
    ground = classification == 2
    if xyz[:, 2].min() <= 2.0:
        dtm, method = np.zeros(shape), "already_height_normalized"
    elif ground.any():
        sums = np.bincount(cells[ground], weights=xyz[ground, 2], minlength=height*width)
        counts = np.bincount(cells[ground], minlength=height*width)
        dtm = (sums / np.maximum(counts, 1)).reshape(shape)
        nearest = distance_transform_edt(counts.reshape(shape) == 0, return_distances=False, return_indices=True)
        dtm, method = dtm[tuple(nearest)], "ground_class2_nearest"
    else:
        low = np.full(height*width, np.inf)
        np.minimum.at(low, cells, xyz[:, 2])
        low = low.reshape(shape)
        nearest = distance_transform_edt(~np.isfinite(low), return_distances=False, return_indices=True)
        dtm = median_filter(minimum_filter(low[tuple(nearest)], size=5), size=9)
        method = "estimated_lower_envelope"
    normalized = xyz[:, 2] - dtm[rr, cc]
    chm = np.full(height*width, -9999., dtype=np.float32)
    np.maximum.at(chm, cells, np.maximum(normalized, 0))
    crs = row["crs_wkt"] or (f"EPSG:{row['spatial_crs']}" if row["spatial_crs"] else None)
    with rasterio.open(row["chm"], "w", driver="GTiff", width=width, height=height, count=1, dtype="float32", transform=transform, crs=crs, nodata=-9999, compress="deflate") as dst:
        dst.write(chm.reshape(shape), 1)
        dst.update_tags(height_normalization=method, uses_tree_id="false")
    frame, gt_method, polygon_source = full_crowns(row, xyz[:, :2], ids, transform, shape)
    frame.to_file(row["gt_vector"], layer="crowns_gt", driver="GPKG")
    if polygon_source:
        shutil.copy2(polygon_source, folder / "original_crown_annotations.gpkg")
    # Ground and annotation-outside points must not become crown positives.
    ids[np.isin(classification, [2, 3])] = 0
    eligible = set(frame.treeID.astype(int))
    ids[~np.isin(ids, list(eligible))] = 0
    origin = np.array([left, bottom, 0.], dtype=np.float64)
    xyz[:, 2] = normalized
    selected, voxel_origin = voxel_indices(xyz, 0.25)
    coord = (xyz[selected] - origin).astype(np.float32)
    ids, intensity = ids[selected], intensity[selected]
    lo, hi = np.percentile(intensity, [1, 99])
    intensity = np.clip((intensity-lo)/max(float(hi-lo), 1.), 0, 1)
    offsets = np.zeros_like(coord)
    for identifier in np.unique(ids[ids > 0]):
        selected_tree = ids == identifier
        offsets[selected_tree] = coord[selected_tree].mean(axis=0) - coord[selected_tree]
    np.savez_compressed(row["output"], coord=coord, grid_coord=np.floor((xyz[selected]-voxel_origin)/.25).astype(np.int32), intensity=intensity, tree_id=ids.astype(np.int32), instance_offset=offsets, source_origin=origin, voxel_origin=voxel_origin, voxel_size=np.float32(.25), raw_points=np.int64(raw_points))
    row.update(raw_points=raw_points, voxels=len(coord), instances=len(frame), tree_voxels=int((ids>0).sum()), voxel_size=.25, height_normalization=method, gt_method=gt_method, copy_verified=True,
               grid_version=2, npz_sha256=sha256(row["output"]))
    done.write_text(json.dumps(row, indent=2) + "\n")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "combined_als_crowns_v1")
    parser.add_argument("--split-policy", choices=("combined", "official"), default="combined")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for dataset in ("ideas_als", "FOR-instance"):
        source_root = ROOT / dataset
        sources = list(source_root.glob("*.csv")) + list(source_root.glob("readMe.txt")) + list(source_root.glob("*/tree_data_*.csv"))
        for source in sources:
            destination = args.output / "source_metadata" / dataset / source.relative_to(source_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha256(destination) != sha256(source):
                raise RuntimeError(f"Metadata source changed: {source}")
            if not destination.exists():
                shutil.copy2(source, destination)
    if args.metadata_only:
        return
    frozen = args.output / "split_inventory.json"
    if frozen.exists():
        rows = json.loads(frozen.read_text())
    else:
        rows, missing = inventory()
        links = assign_splits(rows, args.seed, args.split_policy)
        for row in rows:
            folder = args.output.resolve() / row["model_split"] / row["dataset_id"]
            row.update(source_las=str(folder / "points.las"), output=str(folder / "points_0p25m.npz"),
                       gt_vector=str(folder / "crowns_full.gpkg"), chm=str(folder / f"chm_{row['dataset_id']}.tif"))
        frozen.write_text(json.dumps(rows, indent=2) + "\n")
        write_csv(args.output / "missing_sources.csv", missing)
        write_csv(args.output / "spatial_links.csv", links)
        summary = dict(seed=args.seed, counts=dict(Counter(r["model_split"] for r in rows)),
                       by_source=dict(Counter(f"{r['model_split']}/{r['source_dataset']}/{r['collection']}" for r in rows)),
                       missing=len(missing), spatial_links=len(links),
                       split_policy=args.split_policy, target_group_ratios="70/15/15" if args.split_policy == "combined" else "official test + 80/20 dev",
                       official_test_preserved=args.split_policy == "official", local_frames_not_geographically_verifiable=True,
                       target="full crowns; overlapping polygons, not topmost visible fragments",
                       test_status="historical official tests previously inspected; not a pristine independent test",
                       initialization="new training must not load former mixed-data checkpoints",
                       selection="source-balanced full-crown PQ@0.50 on val; test not for tuning")
        (args.output / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(Counter(r["model_split"] for r in rows), flush=True)
    if args.audit_only:
        return
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, result in enumerate(executor.map(prepare, rows), 1):
            results.append(result)
            print(f"[{i}/{len(rows)}] {result['model_split']} {result['dataset_id']} {result['instances']} crowns", flush=True)
    write_csv(args.output / "manifest.csv", results)
    for split in ("train", "val", "test"):
        write_csv(args.output / f"manifest_{split}.csv", [r for r in results if r["model_split"] == split])
    (args.output / "READY.json").write_text(json.dumps(dict(plots=len(results), manifest_sha256=sha256(args.output / "manifest.csv")), indent=2)+"\n")


if __name__ == "__main__":
    main()
