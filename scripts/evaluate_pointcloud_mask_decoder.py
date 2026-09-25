#!/usr/bin/env python3
"""Predict, calibrate and evaluate the LitePT query-mask decoder."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from shapely import STRtree


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pointcloud.data import load_npz  # noqa: E402
from pointcloud.mask_decoder import LitePTMaskDecoder  # noqa: E402
from scripts.evaluate_pointcloud_litept import (  # noqa: E402
    aggregate,
    export_predictions,
    match_raster,
    model_input,
    ownership_intervals,
    polygon_from_points,
    starts_for_axis,
)


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
        default=PROJECT_DIR
        / "outputs"
        / "training"
        / "litept_mask_decoder"
        / "weights"
        / "best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "evaluation" / "litept_mask_decoder",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tile-size", type=float, default=20.0)
    parser.add_argument("--overlap", type=float, default=8.0)
    parser.add_argument("--max-points", type=int, default=100_000)
    parser.add_argument("--raw-object-threshold", type=float, default=0.05)
    parser.add_argument("--raw-mask-threshold", type=float, default=0.1)
    parser.add_argument("--object-threshold", type=float)
    parser.add_argument("--mask-threshold", type=float)
    parser.add_argument("--minimum-voxels", type=int)
    parser.add_argument("--height-ratio", type=float)
    parser.add_argument("--nms-iou", type=float)
    return parser.parse_args()


def read_manifest(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["model_split"] == split]


def load_model(args: argparse.Namespace, device: torch.device) -> LitePTMaskDecoder:
    checkpoint = torch.load(args.weights.resolve(), map_location=device, weights_only=False)
    saved = checkpoint.get("args", {})
    model = LitePTMaskDecoder(
        patch_size=int(saved.get("patch_size", 256)),
        queries=int(saved.get("queries", 64)),
        hidden_dim=int(saved.get("hidden_dim", 128)),
        decoder_layers=int(saved.get("decoder_layers", 3)),
        attention_heads=int(saved.get("attention_heads", 4)),
        memory_tokens=int(saved.get("memory_tokens", 2048)),
        spatial_prior=bool(saved.get("spatial_prior", False)),
        masked_attention=bool(saved.get("masked_attention", False)),
        auxiliary_losses=bool(saved.get("auxiliary_losses", False)),
        semantic_head=bool(saved.get("semantic_head", False)),
        backbone_type=saved.get("backbone_type", "litept"),
        anchor_mode=saved.get("anchor_mode", "fps"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def predict_plot(
    model: LitePTMaskDecoder,
    arrays: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, np.ndarray]:
    coord = arrays["coord"]
    intensity = arrays["intensity"]
    x_starts = starts_for_axis(
        float(coord[:, 0].min()), float(coord[:, 0].max()), args.tile_size, args.overlap
    )
    y_starts = starts_for_axis(
        float(coord[:, 1].min()), float(coord[:, 1].max()), args.tile_size, args.overlap
    )
    x_owners = ownership_intervals(
        x_starts, args.tile_size, float(coord[:, 0].min()), float(coord[:, 0].max()) + 1e-5
    )
    y_owners = ownership_intervals(
        y_starts, args.tile_size, float(coord[:, 1].min()), float(coord[:, 1].max()) + 1e-5
    )
    rng = np.random.default_rng(seed)
    object_scores: list[float] = []
    candidate_offsets = [0]
    point_indices: list[np.ndarray] = []
    point_scores: list[np.ndarray] = []
    candidate_owner: list[bool] = []
    tiles = 0
    started = time.monotonic()
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
                batch = model_input(
                    coord[chosen], arrays["grid_coord"][chosen], intensity[chosen], device,
                    preserve_height=getattr(args, "preserve_height", False),
                )
                prediction = model(batch)
                objects = prediction["object_logits"].sigmoid().float().cpu().numpy()
                masks = prediction["mask_logits"].sigmoid().float().cpu().numpy()
                for query_index in np.flatnonzero(objects >= args.raw_object_threshold):
                    probabilities = masks[query_index]
                    retained = np.flatnonzero(probabilities >= args.raw_mask_threshold)
                    if len(retained) < 4:
                        continue
                    weights = probabilities[retained]
                    center = np.average(coord[chosen[retained], :2], axis=0, weights=weights)
                    owned = ox0 <= center[0] < ox1 and oy0 <= center[1] < oy1
                    if not owned and not getattr(args, 'retain_all_context_masks', False):
                        continue
                    indices = chosen[retained].astype(np.int32)
                    object_scores.append(float(objects[query_index]))
                    candidate_owner.append(bool(owned))
                    point_indices.append(indices)
                    point_scores.append(probabilities[retained].astype(np.float16))
                    candidate_offsets.append(candidate_offsets[-1] + len(indices))
                tiles += 1
    return {
        "object_score": np.asarray(object_scores, dtype=np.float32),
        "candidate_owner": np.asarray(candidate_owner, dtype=bool),
        "candidate_offset": np.asarray(candidate_offsets, dtype=np.int64),
        "point_index": np.concatenate(point_indices) if point_indices else np.zeros(0, np.int32),
        "point_score": np.concatenate(point_scores) if point_scores else np.zeros(0, np.float16),
        "tiles": np.int64(tiles),
        "seconds": np.float32(time.monotonic() - started),
        "source_voxels": np.int64(len(coord)),
    }


def predict_split(args: argparse.Namespace, split: str, model, device) -> None:
    raw_dir = args.output_dir.resolve() / "raw" / split
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest.resolve(), split)
    summaries = []
    for number, row in enumerate(rows, start=1):
        arrays = load_npz(row["output"])
        raw = predict_plot(model, arrays, args, device, 20260924 + number)
        np.savez_compressed(raw_dir / f"{row['dataset_id']}.npz", **raw)
        summary = {
            "dataset_id": row["dataset_id"],
            "split": split,
            "source_voxels": int(raw["source_voxels"]),
            "candidates": len(raw["object_score"]),
            "tiles": int(raw["tiles"]),
            "seconds": float(raw["seconds"]),
        }
        summaries.append(summary)
        print(f"[{number}/{len(rows)}] {summary}", flush=True)
    with (args.output_dir.resolve() / f"prediction_speed_{split}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def polygon_iou(first, second) -> float:
    intersection = first.intersection(second).area
    if intersection <= 0:
        return 0.0
    union = first.area + second.area - intersection
    return intersection / union if union else 0.0


def candidate_geometries(
    arrays: dict[str, np.ndarray], raw: dict[str, np.ndarray], config: dict
) -> list[dict]:
    coord = arrays["coord"]
    origin = arrays["source_origin"]
    radius = float(arrays["voxel_size"]) / 2
    plot_height = float(coord[:, 2].max())
    instances = []
    offsets = raw["candidate_offset"]
    for candidate, object_score in enumerate(raw["object_score"]):
        start, end = int(offsets[candidate]), int(offsets[candidate + 1])
        scores = raw["point_score"][start:end]
        selected = scores >= config["mask_threshold"]
        if int(np.count_nonzero(selected)) < config["minimum_voxels"]:
            continue
        indices = raw["point_index"][start:end][selected]
        points = coord[indices]
        height = float(points[:, 2].max())
        if height < config["height_ratio"] * plot_height:
            continue
        geometry = polygon_from_points(points[:, :2] + origin[:2], radius)
        if geometry is None or geometry.area < 0.75:
            continue
        top = int(np.argmax(points[:, 2]))
        instances.append(
            {
                "geometry": geometry,
                "object_score": float(object_score),
                "confidence": float(object_score * np.mean(scores[selected])),
                "points": len(indices),
                "height": height,
                "top_x": float(points[top, 0] + origin[0]),
                "top_y": float(points[top, 1] + origin[1]),
            }
        )
    return instances


def filter_instances(
    instances: list[dict],
    object_threshold: float,
    nms_iou: float,
    nms_center_distance: float = 0.0,
) -> list[dict]:
    kept = []
    ranked = sorted(
        (item for item in instances if item["object_score"] >= object_threshold),
        key=lambda value: value["confidence"],
        reverse=True,
    )
    if nms_center_distance <= 0 and ranked:
        # Exact same greedy NMS, but only compare geometries whose bounds can
        # intersect. Dense tile grids otherwise spend minutes on disjoint pairs.
        index = STRtree([item["geometry"] for item in ranked])
        accepted = set()
        for i, item in enumerate(ranked):
            neighbours = index.query(item["geometry"])
            if all(polygon_iou(item["geometry"], ranked[int(j)]["geometry"]) < nms_iou
                   for j in neighbours if int(j) in accepted):
                accepted.add(i)
                kept.append(item)
        return kept
    for item in ranked:
        if all(
            polygon_iou(item["geometry"], other["geometry"]) < nms_iou
            and (
                nms_center_distance <= 0
                or item["geometry"].centroid.distance(other["geometry"].centroid)
                >= nms_center_distance
            )
            for other in kept
        ):
            kept.append(item)
    return kept


def candidates_to_instances(
    arrays: dict[str, np.ndarray], raw: dict[str, np.ndarray], config: dict
) -> list[dict]:
    instances = candidate_geometries(arrays, raw, config)
    return filter_instances(
        instances,
        config["object_threshold"],
        config["nms_iou"],
        config.get("nms_center_distance", 0.0),
    )


def load_plot_data(args: argparse.Namespace, split: str):
    plots = []
    for row in read_manifest(args.manifest.resolve(), split):
        arrays = load_npz(row["output"])
        with np.load(args.output_dir.resolve() / "raw" / split / f"{row['dataset_id']}.npz") as data:
            raw = {key: data[key] for key in data.files}
        with rasterio.open(row["gt_raster"]) as dataset:
            gt = dataset.read(1)
            transform = dataset.transform
            crs = dataset.crs
        plots.append((row, arrays, raw, gt, transform, crs))
    return plots


def score_plots(plots, config: dict, export: bool = False):
    per_plot = []
    exports = []
    for row, arrays, raw, gt, transform, crs in plots:
        instances = candidates_to_instances(arrays, raw, config)
        metrics = match_raster(gt, transform, instances)
        per_plot.append({"dataset_id": row["dataset_id"], **metrics})
        if export:
            exports.append((row, instances, crs))
    return aggregate(per_plot), per_plot, exports


def calibrate(args: argparse.Namespace) -> dict:
    plots = load_plot_data(args, "val")
    candidates = []
    for shape_values in itertools.product((0.4, 0.5, 0.6, 0.7, 0.8), (12,), (0.25,)):
        shape_config = dict(
            zip(
                ("mask_threshold", "minimum_voxels", "height_ratio"),
                shape_values,
                strict=True,
            )
        )
        prepared = [
            (row, candidate_geometries(arrays, raw, shape_config), gt, transform)
            for row, arrays, raw, gt, transform, _ in plots
        ]
        for object_threshold, nms_iou, nms_center_distance in itertools.product(
            (0.3, 0.4, 0.5, 0.6, 0.7), (0.5,), (0.0, 0.75, 1.25, 1.75)
        ):
            config = {
                "object_threshold": object_threshold,
                **shape_config,
                "nms_iou": nms_iou,
                "nms_center_distance": nms_center_distance,
            }
            per_plot = []
            for row, instances, gt, transform in prepared:
                filtered = filter_instances(
                    instances, object_threshold, nms_iou, nms_center_distance
                )
                per_plot.append(
                    {"dataset_id": row["dataset_id"], **match_raster(gt, transform, filtered)}
                )
            metrics = aggregate(per_plot)
            candidates.append({**config, **metrics})
            print(
                config,
                f"F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f}",
                flush=True,
            )
    selected = max(
        candidates, key=lambda row: (row["f1"], row["panoptic_quality"], row["precision"])
    )
    with (args.output_dir.resolve() / "mask_calibration.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    keys = (
        "object_threshold",
        "mask_threshold",
        "minimum_voxels",
        "height_ratio",
        "nms_iou",
        "nms_center_distance",
    )
    selection = {key: selected[key] for key in keys}
    (args.output_dir.resolve() / "selected_mask_config.json").write_text(
        json.dumps({"config": selection, "validation": selected}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Selected", json.dumps(selected, indent=2, sort_keys=True))
    return selection


def evaluate(args: argparse.Namespace) -> None:
    selected = json.loads(
        (args.output_dir.resolve() / "selected_mask_config.json").read_text(encoding="utf-8")
    )["config"]
    overrides = {
        "object_threshold": args.object_threshold,
        "mask_threshold": args.mask_threshold,
        "minimum_voxels": args.minimum_voxels,
        "height_ratio": args.height_ratio,
        "nms_iou": args.nms_iou,
    }
    selected.update({key: value for key, value in overrides.items() if value is not None})
    metrics, per_plot, exports = score_plots(load_plot_data(args, "test"), selected, export=True)
    export_predictions(args.output_dir.resolve(), exports)
    with (args.output_dir.resolve() / "test_per_plot.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_plot[0]))
        writer.writeheader()
        writer.writerows(per_plot)
    speed = list(
        csv.DictReader((args.output_dir.resolve() / "prediction_speed_test.csv").open())
    )
    seconds = sum(float(row["seconds"]) for row in speed)
    voxels = sum(int(row["source_voxels"]) for row in speed)
    report = {
        "protocol": {
            "input": "direct XYZ + intensity point cloud",
            "decoder": "64-query lightweight mask transformer",
            "matching": "Hungarian mask assignment during training",
            "instance_match": "Hungarian one-to-one at 2D crown-mask IoU >= 0.5",
            "config_selected_on": "FOR-instance validation only",
            "test_used_for_selection": False,
            "config": selected,
        },
        "test": metrics,
        "speed": {
            "plots": len(speed),
            "seconds": seconds,
            "source_voxels": voxels,
            "source_voxels_per_second": voxels / seconds,
        },
    }
    (args.output_dir.resolve() / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


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
