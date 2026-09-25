#!/usr/bin/env python3
"""Predict, calibrate and evaluate LitePT tree instances on FOR-instance."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pointcloud.data import load_npz  # noqa: E402
from pointcloud.model import LitePTTreeInstance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "calibrate", "evaluate", "all"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_DIR / "manifests" / "pointcloud_dataset_0p25m.csv",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_DIR / "outputs" / "training" / "litept_tree_instance" / "weights" / "best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "evaluation" / "litept_tree_instance",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tile-size", type=float, default=20.0)
    parser.add_argument("--overlap", type=float, default=8.0)
    parser.add_argument("--max-points", type=int, default=70_000)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--probability", type=float)
    parser.add_argument("--vote-smoothing", type=float)
    parser.add_argument("--peak-separation", type=float)
    parser.add_argument("--peak-threshold-fraction", type=float)
    parser.add_argument("--assignment-radius", type=float)
    parser.add_argument("--min-voxels", type=int)
    parser.add_argument("--height-ratio", type=float)
    return parser.parse_args()


def read_manifest(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["model_split"] == split]


def starts_for_axis(minimum: float, maximum: float, size: float, overlap: float) -> list[float]:
    length = maximum - minimum
    if length <= size:
        return [minimum]
    stride = size - overlap
    starts = list(np.arange(minimum, maximum - size + 1e-6, stride, dtype=float))
    final = maximum - size
    if not starts or abs(starts[-1] - final) > 1e-6:
        starts.append(final)
    return starts


def ownership_intervals(starts: list[float], size: float, minimum: float, maximum: float):
    centers = [value + size / 2 for value in starts]
    return [
        (
            minimum if index == 0 else (centers[index - 1] + centers[index]) / 2,
            maximum if index == len(starts) - 1 else (centers[index] + centers[index + 1]) / 2,
        )
        for index in range(len(starts))
    ]


def model_input(
    coord: np.ndarray,
    grid_coord: np.ndarray,
    intensity: np.ndarray,
    device: torch.device,
    preserve_height: bool = False,
):
    local = coord.astype(np.float32, copy=True)
    local[:, :2] -= np.mean(local[:, :2], axis=0, keepdims=True)
    if not preserve_height:
        local[:, 2] -= np.min(local[:, 2])
    grid = grid_coord.astype(np.int32, copy=True)
    grid -= grid.min(axis=0, keepdims=True)
    extent = grid.max(axis=0).astype(np.int64) + 1
    keys = grid[:, 0].astype(np.int64) + extent[0] * grid[:, 1] + extent[0] * extent[1] * grid[:, 2]
    if len(np.unique(keys)) != len(keys):
        raise ValueError("Duplicate sparse-grid coordinates: repair voxelization before inference")
    features = np.column_stack((local, intensity)).astype(np.float32)
    return {
        "coord": torch.from_numpy(local).to(device),
        "grid_coord": torch.from_numpy(grid).to(device),
        "feat": torch.from_numpy(features).to(device),
        "offset": torch.tensor([len(local)], dtype=torch.long, device=device),
    }


def predict_plot(
    model: LitePTTreeInstance,
    arrays: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, np.ndarray | float | int]:
    coord = arrays["coord"]
    intensity = arrays["intensity"]
    voxel_size = float(arrays["voxel_size"])
    x_starts = starts_for_axis(float(coord[:, 0].min()), float(coord[:, 0].max()), args.tile_size, args.overlap)
    y_starts = starts_for_axis(float(coord[:, 1].min()), float(coord[:, 1].max()), args.tile_size, args.overlap)
    x_owners = ownership_intervals(x_starts, args.tile_size, float(coord[:, 0].min()), float(coord[:, 0].max()) + 1e-5)
    y_owners = ownership_intervals(y_starts, args.tile_size, float(coord[:, 1].min()), float(coord[:, 1].max()) + 1e-5)
    rng = np.random.default_rng(seed)
    coordinates: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    shifted_centers: list[np.ndarray] = []
    gt_tree_ids: list[np.ndarray] = []
    started = time.monotonic()
    tiles = 0
    model.eval()
    with torch.no_grad():
        for xi, x0 in enumerate(x_starts):
            for yi, y0 in enumerate(y_starts):
                context = np.flatnonzero(
                    (coord[:, 0] >= x0)
                    & (coord[:, 0] <= x0 + args.tile_size)
                    & (coord[:, 1] >= y0)
                    & (coord[:, 1] <= y0 + args.tile_size)
                )
                if len(context) == 0:
                    continue
                ox0, ox1 = x_owners[xi]
                oy0, oy1 = y_owners[yi]
                owner = context[
                    (coord[context, 0] >= ox0)
                    & (coord[context, 0] < ox1)
                    & (coord[context, 1] >= oy0)
                    & (coord[context, 1] < oy1)
                ]
                if len(owner) == 0:
                    continue
                if len(owner) >= args.max_points:
                    chosen = np.sort(rng.choice(owner, args.max_points, replace=False))
                else:
                    other = np.setdiff1d(context, owner, assume_unique=True)
                    capacity = args.max_points - len(owner)
                    if len(other) > capacity:
                        other = rng.choice(other, capacity, replace=False)
                    chosen = np.sort(np.concatenate((owner, other)))
                owner_lookup = np.isin(chosen, owner, assume_unique=False)
                batch = model_input(
                    coord[chosen], arrays["grid_coord"][chosen], intensity[chosen], device,
                    preserve_height=getattr(args, "preserve_height", False),
                )
                # FP32 avoids an unsupported spconv FP16 inference-kernel
                # combination on the RTX PRO 500 Blackwell GPU.
                prediction = model(batch)
                probability = prediction["semantic_logits"].softmax(dim=1)[:, 1]
                offset = prediction["offset_m"]
                probability = probability.float().cpu().numpy()[owner_lookup]
                offset = offset.float().cpu().numpy()[owner_lookup]
                owned_indices = chosen[owner_lookup]
                coordinates.append(coord[owned_indices])
                probabilities.append(probability)
                shifted_centers.append(coord[owned_indices] + offset)
                gt_tree_ids.append(arrays["tree_id"][owned_indices])
                tiles += 1
    return {
        "coord": np.concatenate(coordinates).astype(np.float32),
        "tree_probability": np.concatenate(probabilities).astype(np.float32),
        "shifted_center": np.concatenate(shifted_centers).astype(np.float32),
        "gt_tree_id": np.concatenate(gt_tree_ids).astype(np.int32),
        "source_origin": arrays["source_origin"],
        "voxel_size": np.float32(voxel_size),
        "plot_max_z": np.float32(coord[:, 2].max()),
        "source_voxels": np.int64(len(coord)),
        "predicted_voxels": np.int64(sum(len(item) for item in coordinates)),
        "tiles": np.int64(tiles),
        "seconds": np.float32(time.monotonic() - started),
    }


def predict_split(args: argparse.Namespace, split: str, model, device) -> None:
    raw_dir = args.output_dir.resolve() / "raw" / split
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest.resolve(), split)
    summaries = []
    for index, row in enumerate(rows, start=1):
        arrays = load_npz(row["output"])
        prediction = predict_plot(model, arrays, args, device, seed=20260924 + index)
        np.savez_compressed(raw_dir / f"{row['dataset_id']}.npz", **prediction)
        summary = {
            "dataset_id": row["dataset_id"],
            "split": split,
            "source_voxels": int(prediction["source_voxels"]),
            "predicted_voxels": int(prediction["predicted_voxels"]),
            "tiles": int(prediction["tiles"]),
            "seconds": float(prediction["seconds"]),
        }
        summaries.append(summary)
        print(f"[{index}/{len(rows)}] {row['dataset_id']}: {summary}", flush=True)
    with (args.output_dir.resolve() / f"prediction_speed_{split}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def polygon_from_points(xy: np.ndarray, radius: float) -> Polygon | None:
    if len(xy) == 0:
        return None
    geometry = MultiPoint(xy).convex_hull
    if not isinstance(geometry, Polygon):
        geometry = geometry.buffer(radius, cap_style="square")
    else:
        geometry = geometry.buffer(radius, join_style="mitre")
    if geometry.is_empty or not geometry.is_valid:
        geometry = geometry.buffer(0)
    return geometry if isinstance(geometry, Polygon) and not geometry.is_empty else None


def cluster_candidates(
    raw: dict[str, np.ndarray], config: dict, return_members: bool = False
) -> list[dict]:
    probability = raw["tree_probability"]
    selected = np.flatnonzero(probability >= config["probability"])
    if len(selected) == 0:
        return []
    coord = raw["coord"]
    votes = raw["shifted_center"][selected, :2]
    resolution = 0.25
    lower = coord[:, :2].min(axis=0) - 2.0
    upper = coord[:, :2].max(axis=0) + 2.0
    inside = np.all((votes >= lower) & (votes <= upper), axis=1)
    selected = selected[inside]
    votes = votes[inside]
    if len(selected) == 0:
        return []

    shape = np.ceil((upper - lower) / resolution).astype(int) + 1
    grid = np.floor((votes - lower) / resolution).astype(int)
    density = np.zeros(shape, dtype=np.float32)
    np.add.at(density, (grid[:, 0], grid[:, 1]), 1.0)
    density = gaussian_filter(
        density, sigma=config["vote_smoothing"] / resolution
    )
    window = 2 * int(math.ceil(config["peak_separation"] / resolution)) + 1
    maxima = maximum_filter(density, size=window, mode="constant")
    peak_grid = np.argwhere(
        (density == maxima)
        & (density >= config["peak_threshold_fraction"] * float(density.max()))
    )
    if len(peak_grid) == 0:
        return []
    peak_xy = lower + (peak_grid + 0.5) * resolution
    distances, labels = cKDTree(peak_xy).query(
        votes, distance_upper_bound=config["assignment_radius"]
    )
    labels[~np.isfinite(distances)] = len(peak_xy)
    origin = raw["source_origin"]
    voxel_size = float(raw["voxel_size"])
    instances = []
    for label in range(len(peak_xy)):
        members = selected[labels == label]
        if len(members) == 0:
            continue
        points = raw["coord"][members]
        maximum = float(points[:, 2].max())
        polygon = polygon_from_points(points[:, :2] + origin[:2], voxel_size / 2)
        if polygon is None or polygon.area < 0.75:
            continue
        top_index = int(np.argmax(points[:, 2]))
        instances.append(
            {
                "geometry": polygon,
                "confidence": float(np.mean(probability[members])),
                "points": int(len(members)),
                "height": maximum,
                "top_x": float(points[top_index, 0] + origin[0]),
                "top_y": float(points[top_index, 1] + origin[1]),
                **({"point_indices": members} if return_members else {}),
            }
        )
    return sorted(instances, key=lambda item: item["confidence"], reverse=True)


def filter_candidates(candidates: list[dict], raw: dict[str, np.ndarray], config: dict) -> list[dict]:
    minimum_height = config["height_ratio"] * float(raw["plot_max_z"])
    return [
        item
        for item in candidates
        if item["points"] >= config["min_voxels"] and item["height"] >= minimum_height
    ]


def cluster_instances(raw: dict[str, np.ndarray], config: dict) -> list[dict]:
    candidates = cluster_candidates(raw, config)
    return filter_candidates(candidates, raw, config)


def match_raster(gt: np.ndarray, transform, instances: list[dict], threshold: float = 0.5) -> dict:
    valid = gt >= 0
    gt_ids = np.unique(gt[gt > 0])
    gt_masks = [gt == value for value in gt_ids]
    pred_masks = [
        rasterize(
            [(item["geometry"], 1)],
            out_shape=gt.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
        ).astype(bool)
        & valid
        for item in instances
    ]
    ious = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float64)
    for gi, gt_mask in enumerate(gt_masks):
        for pi, pred_mask in enumerate(pred_masks):
            intersection = np.count_nonzero(gt_mask & pred_mask)
            union = np.count_nonzero(gt_mask | pred_mask)
            ious[gi, pi] = intersection / union if union else 0.0
    matched = []
    if ious.size:
        gt_index, pred_index = linear_sum_assignment(-ious)
        matched = [
            float(ious[g, p])
            for g, p in zip(gt_index, pred_index, strict=True)
            if ious[g, p] >= threshold
        ]
    tp = len(matched)
    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp
    return {
        "gt_count": len(gt_masks),
        "prediction_count": len(pred_masks),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sum_iou": float(sum(matched)),
    }


def aggregate(metrics: list[dict]) -> dict:
    totals = defaultdict(float)
    for row in metrics:
        for key, value in row.items():
            if isinstance(value, (int, float, np.number)):
                totals[key] += value
    tp, fp, fn = int(totals["tp"]), int(totals["fp"]), int(totals["fn"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    sq = totals["sum_iou"] / tp if tp else 0.0
    return {
        "plots": len(metrics),
        "gt_count": int(totals["gt_count"]),
        "prediction_count": int(totals["prediction_count"]),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "segmentation_quality": sq,
        "panoptic_quality": sq * f1,
    }


def score_config(args: argparse.Namespace, split: str, config: dict, export: bool = False):
    rows = read_manifest(args.manifest.resolve(), split)
    per_plot = []
    exports = []
    for row in rows:
        with np.load(args.output_dir.resolve() / "raw" / split / f"{row['dataset_id']}.npz") as data:
            raw = {key: data[key] for key in data.files}
        instances = cluster_instances(raw, config)
        with rasterio.open(row["gt_raster"]) as dataset:
            gt = dataset.read(1)
            metrics = match_raster(gt, dataset.transform, instances)
            crs = dataset.crs
        per_plot.append({"dataset_id": row["dataset_id"], **metrics})
        if export:
            exports.append((row, instances, crs))
    return aggregate(per_plot), per_plot, exports


def calibrate(args: argparse.Namespace) -> dict:
    plot_data = []
    for row in read_manifest(args.manifest.resolve(), "val"):
        with np.load(args.output_dir.resolve() / "raw" / "val" / f"{row['dataset_id']}.npz") as data:
            raw = {key: data[key] for key in data.files}
        with rasterio.open(row["gt_raster"]) as dataset:
            gt = dataset.read(1)
            transform = dataset.transform
        plot_data.append((row, raw, gt, transform))

    candidate_metrics: dict[tuple, list[dict]] = defaultdict(list)
    for probability, smoothing, separation, peak_threshold_fraction, assignment_radius in itertools.product(
        (0.35, 0.45, 0.55), (0.25, 0.5, 0.75), (1.0, 1.5), (0.1, 0.2, 0.3), (1.0, 1.5)
    ):
        for _, raw, gt, transform in plot_data:
            base_config = {
                "probability": probability,
                "vote_smoothing": smoothing,
                "peak_separation": separation,
                "peak_threshold_fraction": peak_threshold_fraction,
                "assignment_radius": assignment_radius,
            }
            base = cluster_candidates(raw, base_config)
            for min_voxels, height_ratio in itertools.product((24,), (0.25, 0.4)):
                config = {
                    **base_config,
                    "min_voxels": min_voxels,
                    "height_ratio": height_ratio,
                }
                instances = filter_candidates(base, raw, config)
                key = (
                    probability,
                    smoothing,
                    separation,
                    peak_threshold_fraction,
                    assignment_radius,
                    min_voxels,
                    height_ratio,
                )
                candidate_metrics[key].append(match_raster(gt, transform, instances))

    candidates = []
    for key, per_plot in candidate_metrics.items():
        (
            probability,
            smoothing,
            separation,
            peak_threshold_fraction,
            assignment_radius,
            min_voxels,
            height_ratio,
        ) = key
        config = {
            "probability": probability,
            "vote_smoothing": smoothing,
            "peak_separation": separation,
            "peak_threshold_fraction": peak_threshold_fraction,
            "assignment_radius": assignment_radius,
            "min_voxels": min_voxels,
            "height_ratio": height_ratio,
        }
        metrics = aggregate(per_plot)
        candidates.append({**config, **metrics})
        print(config, f"F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f}")
    selected = max(candidates, key=lambda row: (row["f1"], row["panoptic_quality"], row["precision"]))
    path = args.output_dir.resolve() / "cluster_calibration.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    selection = {
        key: selected[key]
        for key in (
            "probability",
            "vote_smoothing",
            "peak_separation",
            "peak_threshold_fraction",
            "assignment_radius",
            "min_voxels",
            "height_ratio",
        )
    }
    (args.output_dir.resolve() / "selected_cluster_config.json").write_text(
        json.dumps({"config": selection, "validation": selected}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Selected", json.dumps(selected, indent=2, sort_keys=True))
    return selection


def dbh_naslund(height: float, a: float = 10.0, b: float = 0.6) -> float:
    hm = max(height - 1.3, 0.0)
    coefficient_a = -hm * b
    coefficient_c = -hm * a
    return (-coefficient_a + math.sqrt(coefficient_a**2 - 4.0 * coefficient_c)) / 2.0


def export_predictions(output_dir: Path, exports) -> None:
    geospatial = output_dir / "geospatial"
    geospatial.mkdir(parents=True, exist_ok=True)
    for row, instances, crs in exports:
        crown_rows = []
        crown_geometry = []
        top_rows = []
        top_geometry = []
        for tree_id, item in enumerate(instances, start=1):
            crown_rows.append(
                {"treeID": tree_id, "area_m2": item["geometry"].area, "confidence": item["confidence"]}
            )
            crown_geometry.append(MultiPolygon([item["geometry"]]))
            top_rows.append(
                {"treeID": tree_id, "Z": item["height"], "dbh": round(dbh_naslund(item["height"]), 2)}
            )
            top_geometry.append(Point(item["top_x"], item["top_y"], item["height"]))
        crowns = gpd.GeoDataFrame(crown_rows, geometry=crown_geometry, crs=crs)
        tops = gpd.GeoDataFrame(top_rows, geometry=top_geometry, crs=crs)
        dataset_id = row["dataset_id"].removeprefix("for__")
        crowns.to_file(geospatial / f"crowns_{dataset_id}.gpkg", driver="GPKG", engine="pyogrio", index=False)
        tops.to_file(geospatial / f"ttops_{dataset_id}.gpkg", driver="GPKG", engine="pyogrio", index=False)


def evaluate(args: argparse.Namespace) -> None:
    selected_path = args.output_dir.resolve() / "selected_cluster_config.json"
    config = json.loads(selected_path.read_text(encoding="utf-8"))["config"]
    overrides = {
        "probability": args.probability,
        "vote_smoothing": args.vote_smoothing,
        "peak_separation": args.peak_separation,
        "peak_threshold_fraction": args.peak_threshold_fraction,
        "assignment_radius": args.assignment_radius,
        "min_voxels": args.min_voxels,
        "height_ratio": args.height_ratio,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    metrics, per_plot, exports = score_config(args, "test", config, export=True)
    export_predictions(args.output_dir.resolve(), exports)
    with (args.output_dir.resolve() / "test_per_plot.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_plot[0]))
        writer.writeheader()
        writer.writerows(per_plot)
    speed_rows = list(csv.DictReader((args.output_dir.resolve() / "prediction_speed_test.csv").open()))
    report = {
        "protocol": {
            "input": "direct XYZ + intensity point cloud",
            "voxel_size_m": 0.25,
            "instance_match": "Hungarian one-to-one at 2D crown-mask IoU >= 0.5",
            "cluster_config_selected_on": "FOR-instance validation only",
            "test_used_for_selection": False,
            "cluster_config": config,
        },
        "test": metrics,
        "speed": {
            "plots": len(speed_rows),
            "seconds": sum(float(row["seconds"]) for row in speed_rows),
            "source_voxels": sum(int(row["source_voxels"]) for row in speed_rows),
            "predicted_voxels": sum(int(row["predicted_voxels"]) for row in speed_rows),
        },
    }
    report["speed"]["source_voxels_per_second"] = report["speed"]["source_voxels"] / report["speed"]["seconds"]
    (args.output_dir.resolve() / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def load_model(args: argparse.Namespace, device: torch.device):
    checkpoint = torch.load(args.weights.resolve(), map_location=device, weights_only=False)
    model = LitePTTreeInstance(patch_size=args.patch_size).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.stage in ("predict", "all"):
        model = load_model(args, device)
        predict_split(args, "val", model, device)
        predict_split(args, "test", model, device)
    if args.stage in ("calibrate", "all"):
        calibrate(args)
    if args.stage in ("evaluate", "all"):
        evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
