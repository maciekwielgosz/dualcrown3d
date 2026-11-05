#!/usr/bin/env python3
"""Weiser 2022 paired ALS (real and simulated) in the DualCrown3D dataset format.

Real sparse scans of the twelve Weiser plots with the dataset's hand-checked per-tree segments
as instance labels, and 30 m tiles simulated from the Weiser plot-TLS scenes (trunk_reconstruction,
``simulate_weiser_tiles.py``). Both are the forest type (tall mixed stands) in which Model20
splits 40 % of the large crowns and swallows 44 % of the 10-20 m trees
(``trunk_reconstruction/tools/segmentation_errors.py``).

Labels: ``tree_id`` > 0 = a reference tree's segment; -1 = vegetation of trees without a
reference segment (unknown, never background); 0 = ground (``height_agl`` < 0.3 m or class 2).
Crown polygons for the evaluation are convex hulls of each reference tree's points above
max(2 m, 0.35 h). Heights come from the scans' ``height_agl``.

Splits follow ``trunk_reconstruction/configs/real_holdout.json``: held-out plots -> ``test``
(never trained on, never used for selection), BR02 and BR08 -> ``val``, the other five -> ``train``.
Real variants: native density plus 14 and 10 pts/m2 (three thinning seeds each) for training,
14 pts/m2 seed 0 for val and test. Simulated tiles: ``--sim-variants`` per window (default 2).

    .venv-gpu/bin/python scripts/prepare_weiser_dualcrown.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.prepare_treescan_helios_dualcrown import make_chm, sha256, voxelise, write_csv  # noqa: E402

TRUNK = PROJECT.parent / "trunk_reconstruction"
VAL_PLOTS = {"BR02", "BR08"}
GROUND_HAG = 0.3


def split_of(plot: str, held: set[str]) -> str:
    return "test" if plot in held else "val" if plot in VAL_PLOTS else "train"


def crown_polygons(xyz: np.ndarray, hag: np.ndarray, tree_id: np.ndarray, ref: pd.DataFrame, plot: str,
                   split: str) -> gpd.GeoDataFrame:
    rows = []
    ref = ref.set_index("tree_id")
    for tid in np.unique(tree_id[tree_id > 0]):
        m = tree_id == tid
        h = hag[m].max()
        top = m & (hag >= max(2.0, 0.35 * h))
        pts = xyz[top if top.sum() >= 3 else m, :2]
        if len(pts) < 3:
            continue
        rows.append({"plot_id": plot, "tree_id": int(tid), "global_tree_id": f"{plot}_{int(tid)}",
                     "species_code": str(ref.species.get(tid, "")) if "species" in ref else "", "complete": True,
                     "split": split, "geometry": MultiPoint(pts).convex_hull.buffer(0.25)})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:25832")


def convert(xyz: np.ndarray, hag: np.ndarray, tree_id: np.ndarray, intensity: np.ndarray, ground: np.ndarray,
            target: Path, dataset_id: str, split: str, plot: str, ref: pd.DataFrame, source: str,
            voxel_size: float, extra: dict) -> dict:
    tree_id = tree_id.astype(np.int64).copy()
    tree_id[(tree_id <= 0) & ~ground & (hag >= GROUND_HAG)] = -1  # unlabelled vegetation: unknown
    tree_id[ground | (hag < GROUND_HAG)] = 0
    coords = np.column_stack((xyz[:, 0], xyz[:, 1], hag)).astype(np.float64)
    finite = np.isfinite(coords).all(1)
    coords, intensity, tree_id = coords[finite], intensity[finite].astype(np.float32), tree_id[finite]
    raw_points = len(coords)
    vox, inten, tid, voxel_origin = voxelise(coords, intensity, tree_id, voxel_size, True)
    if inten.max() > 0:  # same intensity scaling as the HELIOS conversion; simulated tiles have none (0)
        lo, hi = np.percentile(inten, (1, 99))
        inten = np.clip((inten - lo) / max(float(hi - lo), 1.0), 0, 1).astype(np.float32)
    xy_origin = np.floor(vox[:, :2].min(0) * 2) / 2
    source_origin = np.asarray([xy_origin[0], xy_origin[1], 0.0], dtype=np.float64)
    coord = (vox - source_origin).astype(np.float32)
    grid = np.floor((vox - voxel_origin) / voxel_size).astype(np.int32)
    positive = tid > 0
    offsets = np.zeros((len(coord), 3), np.float32)
    if positive.any():
        unique, inverse, counts = np.unique(tid[positive], return_inverse=True, return_counts=True)
        sums = np.zeros((len(unique), 3))
        np.add.at(sums, inverse, coord[positive])
        offsets[positive] = sums[inverse] / counts[inverse, None] - coord[positive]
    target.mkdir(parents=True, exist_ok=True)
    npz_path = target / "points_0p25m.npz"
    np.savez_compressed(npz_path, coord=coord, grid_coord=grid, intensity=inten, tree_id=tid.astype(np.int32),
                        instance_offset=offsets, source_origin=source_origin, voxel_origin=voxel_origin,
                        voxel_size=np.float32(voxel_size), raw_points=np.int64(raw_points))
    crowns = crown_polygons(coords, coords[:, 2], tree_id, ref, plot, split)
    crowns_path = target / "crowns_full.gpkg"
    crowns.to_file(crowns_path, driver="GPKG")
    chm_path = target / f"chm_{dataset_id}.tif"
    make_chm(chm_path, coords)
    row = {"dataset_id": dataset_id, "source_dataset": source, "collection": "WEISER_2022", "official_split": split,
           "original_las": extra.pop("original"), "annotation_method": extra.pop("annotation"),
           "spatial_crs": "EPSG:25832", "bounds": json.dumps([float(v) for v in (*coords[:, :2].min(0), *coords[:, :2].max(0))]),
           "crs_wkt": "", "source_sha256": extra.pop("source_sha256", ""), "group_id": hashlib.sha256(plot.encode()).hexdigest()[:16],
           "model_split": split, "parent_plot": plot, "source_las": extra.pop("source_las", ""), "output": str(npz_path.resolve()),
           "gt_vector": str(crowns_path.resolve()), "chm": str(chm_path.resolve()), "raw_points": raw_points,
           "voxels": len(coord), "instances": int(len(crowns)), "tree_voxels": int(positive.sum()),
           "unknown_voxels": int((tid == -1).sum()), "voxel_size": voxel_size,
           "height_normalization": "provided_height_agl", "gt_method": "reference_segment_convex_hull",
           "copy_verified": True, "grid_version": 2, "repaired_grid_collisions": 0, "supervision_changed": False,
           "voxel_representative_label_independent": False, "npz_sha256": sha256(npz_path), **extra}
    (target / "metadata.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def real_rows(output: Path, held: set[str], voxel_size: float) -> list[dict]:
    man = pd.read_csv(TRUNK / "data/paired/weiser/manifest.csv")
    rows = []
    for r in man.itertuples():
        split = split_of(r.plot, held)
        if split != "train" and not (r.density == "d14" and r.seed == 0):
            continue
        path = TRUNK / r.path
        name = path.stem
        dataset_id = f"weiser__{name}"
        target = output / split / dataset_id
        if (target / "metadata.json").exists():
            rows.append(json.loads((target / "metadata.json").read_text()))
            continue
        las = laspy.read(path)
        ref = pd.read_csv(path.parent / "reference.csv")
        row = convert(np.c_[las.x, las.y, las.z], np.asarray(las.height_agl, np.float64), np.asarray(las.tree_id),
                      np.asarray(las.intensity, np.float32), np.asarray(las.classification) == 2, target, dataset_id,
                      split, r.plot, ref, "WEISER_2022_ALS", voxel_size,
                      dict(original=str(path), annotation="reference_segments_pulse_thinned", source_sha256=sha256(path),
                           density=r.density, seed=int(r.seed), pts_m2=float(r.pts_m2)))
        rows.append(row)
        print(f"{dataset_id}: {row['voxels']} voxels, {row['instances']} trees, {split}", flush=True)
    return rows


def sim_rows(output: Path, held: set[str], voxel_size: float, variants: int) -> list[dict]:
    man = pd.read_csv(TRUNK / "data/sim/weiser_v2/manifest.csv")
    man = man[man.variant < variants]
    rows = []
    for r in man.itertuples():
        plot = r.plot.split("_")[1]
        if plot in held:
            continue
        split = split_of(plot, held)
        path = TRUNK / r.path
        dataset_id = f"weiser_sim__{r.plot}_v{r.variant}"
        target = output / split / dataset_id
        if (target / "metadata.json").exists():
            rows.append(json.loads((target / "metadata.json").read_text()))
            continue
        d = np.load(path)
        xyz = d["xyz"].astype(np.float64)
        # window-local x, y: shift into the plot frame so tiles of one plot do not overlap in the grid
        ref = pd.read_csv(TRUNK / "data/paired/weiser" / plot / "reference.csv")
        row = convert(xyz, d["hag"].astype(np.float64), d["tree"], np.zeros(len(xyz), np.float32), d["ground"],
                      target, dataset_id, split, plot, ref, "WEISER_2022_SIM_V2", voxel_size,
                      dict(original=str(path), annotation="simulator_v2_weiser_tls_scene_reference_ids",
                           window=r.plot, variant=int(r.variant), density=float(r.density)))
        rows.append(row)
        if len(rows) % 50 == 0:
            print(f"{len(rows)} simulated tiles", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=PROJECT.parent / "dualcrown3d_weiser_v1")
    ap.add_argument("--voxel-size", type=float, default=0.25)
    ap.add_argument("--sim-variants", type=int, default=2)
    ap.add_argument("--no-sim", action="store_true")
    a = ap.parse_args()
    held = set(json.loads((TRUNK / "configs/real_holdout.json").read_text())["weiser"])
    a.output.mkdir(parents=True, exist_ok=True)
    rows = real_rows(a.output, held, a.voxel_size)
    if not a.no_sim:
        rows += sim_rows(a.output, held, a.voxel_size, a.sim_variants)
    write_csv(a.output / "manifest.csv", rows)
    for split in ("train", "val", "test"):
        write_csv(a.output / f"manifest_{split}.csv", [r for r in rows if r["model_split"] == split])
    summary = pd.DataFrame(rows).groupby(["source_dataset", "model_split"]).agg(plots=("dataset_id", "size"),
                                                                                trees=("instances", "sum"))
    print(summary.to_string())
    (a.output / "READY.json").write_text(json.dumps({"rows": len(rows), "holdout_test_plots": sorted(held),
                                                     "val_plots": sorted(VAL_PLOTS)}, indent=2))


if __name__ == "__main__":
    main()
