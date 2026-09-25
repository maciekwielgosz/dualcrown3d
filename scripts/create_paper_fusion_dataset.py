#!/usr/bin/env python3
"""Create the paper-inspired colored point-map representation from raw bands.

The paper does not publish exact height bins or color maps.  This explicit,
deterministic approximation colorizes density and intensity after mixing each
with normalized canopy height, then averages the two RGB maps.
"""

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
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib
import numpy as np
import rasterio
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DATASET = PROJECT_DIR / "dataset"
TARGET_DATASET = PROJECT_DIR / "dataset_paper_fusion"
MANIFEST = PROJECT_DIR / "manifests" / "dataset_manifest.csv"
RAW_DIR = PROJECT_DIR / "artifacts" / "physical_3band_0p5m"


def main() -> int:
    with MANIFEST.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    updated = []
    for index, row in enumerate(rows, start=1):
        dataset_id = row["dataset_id"]
        split = row["model_split"]
        raw_path = RAW_DIR / f"{dataset_id}.tif"
        normalization = json.loads(
            (RAW_DIR / f"{dataset_id}.normalization.json").read_text(encoding="utf-8")
        )
        with rasterio.open(raw_path) as dataset:
            raw = dataset.read().astype(np.float64)
            valid = raw[0] != dataset.nodata
        chm, density, intensity = raw
        height_normalized = np.clip(chm / normalization["chm_clip_max_m"], 0, 1)
        density_normalized = np.clip(
            np.log1p(np.maximum(density, 0))
            / np.log1p(max(normalization["density_clip_count"], 1.0)),
            0,
            1,
        )
        intensity_range = normalization["intensity_clip_high"] - normalization["intensity_clip_low"]
        intensity_normalized = np.clip(
            (intensity - normalization["intensity_clip_low"]) / max(intensity_range, 1.0),
            0,
            1,
        )
        density_value = 0.65 * density_normalized + 0.35 * height_normalized
        intensity_value = 0.65 * intensity_normalized + 0.35 * height_normalized
        density_rgb = matplotlib.colormaps["turbo"](density_value)[..., :3]
        intensity_rgb = matplotlib.colormaps["viridis"](intensity_value)[..., :3]
        fused = np.rint((density_rgb + intensity_rgb) * 0.5 * 255).astype(np.uint8)
        canopy = valid & (chm >= 2.0) & (density > 0)
        fused[~canopy] = 0

        image_path = TARGET_DATASET / "images" / split / f"{dataset_id}.png"
        label_path = TARGET_DATASET / "labels" / split / f"{dataset_id}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(fused).save(image_path)
        shutil.copy2(SOURCE_DATASET / "labels" / split / f"{dataset_id}.txt", label_path)
        updated.append({**row, "image": str(image_path), "label": str(label_path)})
        print(f"[{index}/{len(rows)}] {dataset_id}")

    (TARGET_DATASET / "dataset.yaml").write_text(
        f"path: {TARGET_DATASET}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: tree\n",
        encoding="utf-8",
    )
    output_manifest = PROJECT_DIR / "manifests" / "dataset_manifest_paper_fusion.csv"
    with output_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(updated[0]))
        writer.writeheader()
        writer.writerows(updated)
    method = {
        "status": "paper-inspired approximation; exact authors' code unavailable",
        "density_colormap": "turbo",
        "intensity_colormap": "viridis",
        "feature_formula": "color_value = 0.65 * normalized_feature + 0.35 * normalized_CHM",
        "rgb_fusion": "arithmetic mean of density RGB and intensity RGB",
        "minimum_canopy_height_m": 2.0,
        "input_channels_use_tree_id": False,
    }
    reports = PROJECT_DIR / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "paper_fusion_method.json").write_text(
        json.dumps(method, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
