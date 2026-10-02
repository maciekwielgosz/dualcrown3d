#!/usr/bin/env python3
"""Stage-0 inventory and split/label contract checks for superpoint work."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = WORKSPACE / "DL_model_version"
INPUTS = {
    "native_als": WORKSPACE / "combined_als_crowns_supervision_v4/manifest.csv",
    "helios": WORKSPACE / "dualcrown3d_treescan_helios_v1/manifest.csv",
    "helios_flights": WORKSPACE / "TreeScanPL10k_HELIOS_ALS_flight_augmentation_v2/manifest.csv",
}
OUTPUT = PROJECT / "outputs/dualcrown3d_superpoint_v1/stage0"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def parent_key(kind: str, row: dict) -> str:
    if kind == "native_als":
        return row["group_id"]
    return row.get("parent_plot") or row["dataset_id"].removeprefix("treescan_helios__")


def audit() -> tuple[list[dict], dict]:
    items = []
    summaries = {}
    parent_splits = defaultdict(set)
    missing = []
    for kind, manifest in INPUTS.items():
        rows = read_rows(manifest)
        counts = Counter(row["model_split"] for row in rows)
        summaries[kind] = dict(manifest=str(manifest), sha256=digest(manifest),
                               records=len(rows), splits=dict(counts))
        for row in rows:
            source = Path(row["output"])
            reference = Path(row["gt_vector"])
            if not source.is_file() or not reference.is_file():
                missing.append(row["dataset_id"])
                continue
            with np.load(source) as data:
                coord = data["coord"]
                ids = data["tree_id"]
                semantic = data["semantic_target"] if "semantic_target" in data else None
                origin = data["source_origin"].tolist()
                finite = bool(np.isfinite(coord).all())
                label_counts = dict(negative=int(np.sum(ids < 0)),
                                    background=int(np.sum(ids == 0)),
                                    tree=int(np.sum(ids > 0)))
                semantic_values = (np.unique(semantic).astype(int).tolist()
                                   if semantic is not None else [])
                if semantic is not None and not set(semantic_values) <= {-1, 0, 1}:
                    raise ValueError(f"Invalid semantic targets: {source}")
                if kind == "native_als" and semantic is None:
                    raise ValueError(f"Missing corrected tri-state labels: {source}")
            if not finite or len(coord) != len(ids):
                raise ValueError(f"Invalid coordinate/label array: {source}")
            bounds = json.loads(row["bounds"])
            area = max((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]), 1.)
            parent = parent_key(kind, row)
            parent_splits[("real" if kind == "native_als" else "treescan", parent)].add(
                row["model_split"])
            items.append(dict(kind=kind, dataset_id=row["dataset_id"],
                              source_dataset=row["source_dataset"],
                              collection=row["collection"], split=row["model_split"],
                              acquisition_id=row.get("flight_variant") or row["dataset_id"],
                              parent_group=parent, source_las=row["source_las"],
                              npz=str(source), gt_vector=str(reference),
                              annotation_method=row["annotation_method"],
                              annotation_protocol=row.get("annotation_protocol", ""),
                              crs=row["spatial_crs"], source_origin=origin,
                              height_definition=row["height_normalization"],
                              voxel_size_m=float(row["voxel_size"]),
                              raw_points=int(row["raw_points"]), voxels=len(coord),
                              raw_density_m2=int(row["raw_points"]) / area,
                              voxel_density_m2=len(coord) / area,
                              tree_instances=int(len(np.unique(ids[ids > 0]))),
                              reference_instances=int(row["instances"]),
                              unknown_voxels=label_counts["negative"],
                              background_voxels=label_counts["background"],
                              tree_voxels=label_counts["tree"],
                              semantic_values=semantic_values,
                              npz_sha256=row["npz_sha256"],
                              source_sha256=row["source_sha256"],
                              ignore_vector=row.get("ignore_vector", "")))
    split_conflicts = [{"parent": key, "splits": sorted(values)}
                       for key, values in parent_splits.items() if len(values) > 1]
    summary = dict(inputs=summaries, checked_records=len(items),
                   missing_records=missing, parent_split_conflicts=split_conflicts,
                   modality_pair_status=(
                       "Synthetic ALS and source TLS share a modeled TreeScan parent and exact scene IDs; "
                       "registered, temporally matched real TLS/ALS pairs with shared tree IDs are not established"),
                   label_contract="native v4: tree_id -1 unknown, 0 known background, >0 instance",
                   heldout_note="historically exposed real ALS test is a regression set, not a new holdout")
    if missing or split_conflicts:
        raise ValueError(f"Data contract failed: {summary}")
    return items, summary


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Completed stage-0 output protected: {OUTPUT}")
    items, summary = audit()
    OUTPUT.mkdir(parents=True)
    with (OUTPUT / "inventory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, list) else value
                          for key, value in item.items()} for item in items)
    (OUTPUT / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
