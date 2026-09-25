#!/usr/bin/env python3
"""Export YOLO pixel polygons and tree_id rasters to georeferenced files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ENV_PREFIX = Path(sys.prefix)
for variable, relative in (
    ("PROJ_DATA", "share/proj"),
    ("GDAL_DATA", "share/gdal"),
    ("GDAL_DRIVER_PATH", "lib/gdalplugins"),
):
    candidate = ENV_PREFIX / relative
    if candidate.is_dir():
        os.environ[variable] = str(candidate)

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine
from shapely.geometry import Polygon


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=PROJECT_DIR / "outputs/evaluation/yolo11s_physical",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prediction_dir = args.evaluation_dir.resolve() / "predictions"
    output_dir = args.evaluation_dir.resolve() / "geospatial"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(prediction_dir.glob("predictions_*.json"))
    if not files:
        raise SystemExit(f"No prediction JSON files in {prediction_dir}")
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset_id = payload["dataset_id"]
        transform = Affine(*payload["transform"])
        gpkg_path = output_dir / f"crowns_{dataset_id}.gpkg"
        tif_path = output_dir / f"tree_id_{dataset_id}.tif"
        if not args.overwrite and gpkg_path.exists() and tif_path.exists():
            continue
        rows = []
        geometries = []
        for item in payload["objects"]:
            coordinates = [transform * (float(x), float(y)) for x, y in item["polygon_xy"]]
            polygon = Polygon(coordinates)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty:
                continue
            rows.append(
                {
                    "tree_id": int(item["tree_id"]),
                    "confidence": float(item["confidence"]),
                    "pixel_area": int(item["pixel_area"]),
                    "dataset_id": dataset_id,
                    "collection": payload["collection"],
                }
            )
            geometries.append(polygon)
        crowns = gpd.GeoDataFrame(rows, geometry=geometries, crs=payload["crs"])
        crowns.to_file(gpkg_path, layer="crowns", driver="GPKG", engine="pyogrio", index=False)

        labels = np.asarray(Image.open(prediction_dir / f"tree_id_{dataset_id}.png")).astype(np.uint16)
        profile = {
            "driver": "GTiff",
            "width": int(payload["width"]),
            "height": int(payload["height"]),
            "count": 1,
            "dtype": "uint16",
            "crs": payload["crs"],
            "transform": transform,
            "nodata": int(payload["nodata"]),
            "compress": "deflate",
            "predictor": 2,
        }
        with rasterio.open(tif_path, "w", **profile) as destination:
            destination.write(labels, 1)
            destination.set_band_description(1, "topmost predicted tree_id")
            destination.update_tags(
                DATASET_ID=dataset_id,
                TREE_ID_TYPE="prediction-sequential",
                OVERLAP_RULE="highest-confidence instance wins",
                VECTOR_OUTPUT="overlapping instance polygons retained in GeoPackage",
            )
        print(f"exported {dataset_id}: {len(crowns)} crowns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
