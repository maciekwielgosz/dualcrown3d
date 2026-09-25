#!/usr/bin/env python3
"""Compare crown-union area with CHM canopy area for benchmark outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
TILES = ("764000_197000", "764000_197500", "764500_197000", "764500_197500")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chm-dir", type=Path, default=WORKSPACE_DIR / "run_r" / "data_input")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "output_10_yolo11s_chm_only" / "canopy_coverage_comparison.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = {
        "watershed_baseline": WORKSPACE_DIR
        / "segmentatiion_benchmark"
        / "output_00_baseline"
        / "Segmentation3",
        "yolo11s_physical": PROJECT_DIR / "output_09_yolo11s_physical" / "Segmentation3",
        "yolo11s_chm_only": PROJECT_DIR / "output_10_yolo11s_chm_only" / "Segmentation3",
        "yolo11s_chm_only_ideas_finetuned": PROJECT_DIR
        / "output_11_yolo11s_chm_only_ideas_finetuned"
        / "Segmentation3",
        "yolo11s_chm_only_ideas_recall": PROJECT_DIR
        / "output_12_yolo11s_chm_only_ideas_recall"
        / "Segmentation3",
        "yolo11s_chm_only_ideas_combined_recall": PROJECT_DIR
        / "output_13_yolo11s_chm_only_ideas_combined_recall"
        / "Segmentation3",
    }
    rows = []
    for variant, directory in variants.items():
        if not directory.is_dir():
            continue
        for tile in TILES:
            with rasterio.open(args.chm_dir / f"chm_{tile}.tif") as dataset:
                chm = dataset.read(1)
                valid = np.isfinite(chm)
                if dataset.nodata is not None:
                    valid &= chm != dataset.nodata
                canopy_area = float(
                    np.count_nonzero(valid & (chm >= 2.0))
                    * abs(dataset.transform.a * dataset.transform.e)
                )
            crowns = gpd.read_file(directory / f"crowns_{tile}.gpkg")
            crown_union_area = float(crowns.geometry.union_all().area) if len(crowns) else 0.0
            rows.append(
                {
                    "variant": variant,
                    "tile_id": tile,
                    "crowns": len(crowns),
                    "crown_union_area_m2": crown_union_area,
                    "chm_canopy_area_ge_2m_m2": canopy_area,
                    "crown_union_to_canopy_ratio": crown_union_area / canopy_area if canopy_area else 0.0,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
