#!/usr/bin/env python3
"""Create a controlled CHM-only YOLO dataset by repeating CHM into RGB."""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

ENV_PREFIX = Path(sys.prefix)
for variable, relative in (("PROJ_DATA", "share/proj"), ("GDAL_DATA", "share/gdal")):
    candidate = ENV_PREFIX / relative
    if candidate.is_dir():
        os.environ[variable] = str(candidate)

import numpy as np
import rasterio
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DATASET = PROJECT_DIR / "dataset"
TARGET_DATASET = PROJECT_DIR / "dataset_chm_only"
SOURCE_MANIFEST = PROJECT_DIR / "manifests" / "dataset_manifest.csv"
TARGET_MANIFEST = PROJECT_DIR / "manifests" / "dataset_manifest_chm_only.csv"
RAW_DIR = PROJECT_DIR / "artifacts" / "physical_3band_0p5m"
CHM_CLIP_MAX_M = 45.0


def main() -> int:
    with SOURCE_MANIFEST.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    updated = []
    for index, row in enumerate(rows, start=1):
        dataset_id = row["dataset_id"]
        split = row["model_split"]
        raw_path = RAW_DIR / f"{dataset_id}.tif"
        with rasterio.open(raw_path) as dataset:
            chm = dataset.read(1).astype(np.float64)
            valid = np.isfinite(chm)
            if dataset.nodata is not None:
                valid &= chm != dataset.nodata
        normalized = np.rint(
            np.clip(np.where(valid, chm, 0.0), 0, CHM_CLIP_MAX_M)
            / CHM_CLIP_MAX_M
            * 255
        ).astype(np.uint8)
        image = np.repeat(normalized[..., None], 3, axis=2)
        image[~valid] = 0

        image_path = TARGET_DATASET / "images" / split / f"{dataset_id}.png"
        label_path = TARGET_DATASET / "labels" / split / f"{dataset_id}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(image_path)
        shutil.copy2(SOURCE_DATASET / "labels" / split / f"{dataset_id}.txt", label_path)
        updated.append({**row, "image": str(image_path), "label": str(label_path)})
        print(f"[{index}/{len(rows)}] {dataset_id} -> {split}")

    (TARGET_DATASET / "dataset.yaml").write_text(
        f"path: {TARGET_DATASET}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: tree\n",
        encoding="utf-8",
    )
    TARGET_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with TARGET_MANIFEST.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(updated[0]))
        writer.writeheader()
        writer.writerows(updated)
    method = {
        "experiment": "CHM-only ablation",
        "source_feature": "leakage-free CHM derived without treeID",
        "encoding": "normalized CHM repeated identically into R, G and B",
        "chm_clip_max_m": CHM_CLIP_MAX_M,
        "input_channels_use_tree_id": False,
        "split_reused_from": str(SOURCE_MANIFEST),
        "plots_by_split": {
            split: sum(row["model_split"] == split for row in rows)
            for split in ("train", "val", "test")
        },
    }
    (PROJECT_DIR / "reports" / "chm_only_method.json").write_text(
        json.dumps(method, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Created CHM-only dataset with {len(rows)} plots in {TARGET_DATASET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
