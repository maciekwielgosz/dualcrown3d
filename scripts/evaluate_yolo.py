#!/usr/bin/env python3
"""Calibrate confidence on validation data and evaluate a frozen YOLO model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class Prediction:
    confidence: float
    polygon: np.ndarray
    mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "manifests/dataset_manifest.csv")
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "dataset/dataset.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/evaluation/yolo11s_physical")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def rasterize_polygon(polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if polygon.shape[0] < 3:
        return mask.astype(bool)
    points = np.rint(polygon).astype(np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, shape[0] - 1)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def extract_predictions(result, shape: tuple[int, int]) -> list[Prediction]:
    if result.masks is None or result.boxes is None:
        return []
    confidences = result.boxes.conf.detach().cpu().numpy()
    predictions: list[Prediction] = []
    for confidence, polygon in zip(confidences, result.masks.xy, strict=True):
        polygon = np.asarray(polygon, dtype=np.float64)
        mask = rasterize_polygon(polygon, shape)
        if np.any(mask):
            predictions.append(Prediction(float(confidence), polygon, mask))
    return predictions


def load_gt(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.int64, copy=False)


def predict_rows(model: YOLO, rows: list[dict[str, str]], args: argparse.Namespace):
    sources = [row["image"] for row in rows]
    started = time.perf_counter()
    results = model.predict(
        source=sources,
        imgsz=args.imgsz,
        conf=0.001,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        batch=args.batch,
        retina_masks=True,
        verbose=False,
    )
    wall_seconds = time.perf_counter() - started
    output = []
    speed = defaultdict(float)
    for row, result in zip(rows, results, strict=True):
        gt = load_gt(Path(row["gt_raster"]))
        predictions = extract_predictions(result, gt.shape)
        output.append((row, gt, predictions))
        for key, value in result.speed.items():
            speed[key] += float(value)
    count = max(len(results), 1)
    return output, wall_seconds, {key: value / count for key, value in speed.items()}


def match_plot(gt: np.ndarray, predictions: list[Prediction], confidence: float, iou_threshold: float) -> dict:
    valid = gt >= 0
    gt_ids = np.unique(gt[gt > 0])
    gt_masks = [(gt == tree_id) for tree_id in gt_ids]
    selected: list[tuple[Prediction, np.ndarray]] = []
    for prediction in predictions:
        if prediction.confidence < confidence:
            continue
        mask = prediction.mask & valid
        if np.any(mask):
            selected.append((prediction, mask))

    ious = np.zeros((len(gt_masks), len(selected)), dtype=np.float64)
    intersections = np.zeros_like(ious)
    for gt_index, gt_mask in enumerate(gt_masks):
        for prediction_index, (_, prediction_mask) in enumerate(selected):
            intersection = np.count_nonzero(gt_mask & prediction_mask)
            union = np.count_nonzero(gt_mask | prediction_mask)
            intersections[gt_index, prediction_index] = intersection
            ious[gt_index, prediction_index] = intersection / union if union else 0.0

    matched: list[tuple[int, int, float]] = []
    if ious.size:
        gt_indices, prediction_indices = linear_sum_assignment(-ious)
        matched = [
            (int(g), int(p), float(ious[g, p]))
            for g, p in zip(gt_indices, prediction_indices, strict=True)
            if ious[g, p] >= iou_threshold
        ]
    tp = len(matched)
    fp = len(selected) - tp
    fn = len(gt_masks) - tp
    denominator = tp + 0.5 * fp + 0.5 * fn
    sum_iou = sum(item[2] for item in matched)
    dices = []
    for gt_index, prediction_index, _ in matched:
        a = np.count_nonzero(gt_masks[gt_index])
        b = np.count_nonzero(selected[prediction_index][1])
        intersection = intersections[gt_index, prediction_index]
        dices.append(2 * intersection / (a + b) if a + b else 0.0)

    gt_foreground = gt > 0
    prediction_foreground = np.zeros(gt.shape, dtype=bool)
    for _, mask in selected:
        prediction_foreground |= mask
    area_intersection = int(np.count_nonzero(gt_foreground & prediction_foreground))
    gt_area = int(np.count_nonzero(gt_foreground))
    prediction_area = int(np.count_nonzero(prediction_foreground))
    area_union = gt_area + prediction_area - area_intersection

    return {
        "gt_count": len(gt_masks),
        "prediction_count": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sum_iou": sum_iou,
        "sum_dice": float(sum(dices)),
        "area_intersection_px": area_intersection,
        "gt_area_px": gt_area,
        "prediction_area_px": prediction_area,
        "area_union_px": area_union,
        "recognition_denominator": denominator,
    }


def aggregate(rows: list[dict]) -> dict:
    totals = defaultdict(float)
    count_errors = []
    for row in rows:
        for key in (
            "gt_count", "prediction_count", "tp", "fp", "fn", "sum_iou",
            "sum_dice", "area_intersection_px", "gt_area_px",
            "prediction_area_px", "area_union_px", "recognition_denominator",
        ):
            totals[key] += row[key]
        count_errors.append(row["prediction_count"] - row["gt_count"])
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rq = tp / totals["recognition_denominator"] if totals["recognition_denominator"] else 0.0
    sq = totals["sum_iou"] / tp if tp else 0.0
    errors = np.asarray(count_errors, dtype=np.float64)
    return {
        "plots": len(rows),
        "gt_count": int(totals["gt_count"]),
        "prediction_count": int(totals["prediction_count"]),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "recognition_quality": rq,
        "segmentation_quality": sq,
        "panoptic_quality": totals["sum_iou"] / totals["recognition_denominator"] if totals["recognition_denominator"] else 0.0,
        "mean_iou_matched": sq,
        "mean_dice_matched": totals["sum_dice"] / tp if tp else 0.0,
        "area_precision": totals["area_intersection_px"] / totals["prediction_area_px"] if totals["prediction_area_px"] else 0.0,
        "area_recall": totals["area_intersection_px"] / totals["gt_area_px"] if totals["gt_area_px"] else 0.0,
        "area_iou": totals["area_intersection_px"] / totals["area_union_px"] if totals["area_union_px"] else 0.0,
        "tree_count_mae": float(np.mean(np.abs(errors))) if errors.size else 0.0,
        "tree_count_rmse": float(np.sqrt(np.mean(errors**2))) if errors.size else 0.0,
        "tree_count_bias": float(np.mean(errors)) if errors.size else 0.0,
    }


def evaluate_set(predicted_rows, confidence: float, iou_threshold: float):
    per_plot = []
    for row, gt, predictions in predicted_rows:
        metrics = match_plot(gt, predictions, confidence, iou_threshold)
        per_plot.append({"dataset_id": row["dataset_id"], "collection": row["collection"], **metrics})
    overall = aggregate(per_plot)
    by_collection = {}
    collections = sorted({row["collection"] for row in per_plot})
    for collection in collections:
        by_collection[collection] = aggregate([row for row in per_plot if row["collection"] == collection])
    return overall, by_collection, per_plot


def calibrate(predicted_rows) -> tuple[float, list[dict]]:
    candidates = np.round(np.arange(0.01, 0.81, 0.01), 2)
    table = []
    for confidence in candidates:
        overall, _, _ = evaluate_set(predicted_rows, float(confidence), 0.5)
        table.append({"confidence": float(confidence), **overall})
    best = max(table, key=lambda row: (row["f1"], row["panoptic_quality"], row["confidence"]))
    return float(best["confidence"]), table


def save_predictions(predicted_rows, confidence: float, output_dir: Path) -> None:
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for row, gt, predictions in predicted_rows:
        valid = gt >= 0
        selected = [prediction for prediction in predictions if prediction.confidence >= confidence]
        selected.sort(key=lambda item: item.confidence, reverse=True)
        labels = np.zeros(gt.shape, dtype=np.uint16)
        labels[~valid] = np.iinfo(np.uint16).max
        objects = []
        for tree_id, prediction in enumerate(selected, start=1):
            mask = prediction.mask & valid
            labels[mask & (labels == 0)] = tree_id
            objects.append(
                {
                    "tree_id": tree_id,
                    "confidence": prediction.confidence,
                    "polygon_xy": prediction.polygon.tolist(),
                    "pixel_area": int(np.count_nonzero(mask)),
                }
            )
        Image.fromarray(labels).save(prediction_dir / f"tree_id_{row['dataset_id']}.png")
        payload = {
            "dataset_id": row["dataset_id"],
            "collection": row["collection"],
            "crs": row["crs"],
            "transform": [float(value) for value in row["transform"].split(",")],
            "width": int(row["width"]),
            "height": int(row["height"]),
            "nodata": 65535,
            "objects": objects,
        }
        (prediction_dir / f"predictions_{row['dataset_id']}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device is None:
        args.device = 0 if torch.cuda.is_available() else "cpu"
    rows = read_manifest(args.manifest.resolve())
    validation_rows = [row for row in rows if row["model_split"] == "val"]
    test_rows = [row for row in rows if row["model_split"] == "test"]
    model = YOLO(str(args.weights.resolve()))

    validation_predictions, validation_wall, validation_speed = predict_rows(model, validation_rows, args)
    confidence, calibration = calibrate(validation_predictions)
    write_csv(args.output_dir / "confidence_calibration.csv", calibration)
    validation_overall, validation_by_collection, validation_per_plot = evaluate_set(
        validation_predictions, confidence, 0.5
    )

    test_predictions, test_wall, test_speed = predict_rows(model, test_rows, args)
    thresholds = {}
    primary_per_plot = []
    for threshold in (0.25, 0.5, 0.75):
        overall, by_collection, per_plot = evaluate_set(test_predictions, confidence, threshold)
        thresholds[str(threshold)] = {"overall": overall, "by_collection": by_collection}
        if threshold == 0.5:
            primary_per_plot = per_plot
    save_predictions(test_predictions, confidence, args.output_dir)
    write_csv(args.output_dir / "test_per_plot_iou50.csv", primary_per_plot)

    ultralytics_metrics = model.val(
        data=str(args.data.resolve()),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(args.output_dir),
        name="ultralytics_test",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    builtin = {key: float(value) for key, value in ultralytics_metrics.results_dict.items()}
    report = {
        "weights": str(args.weights.resolve()),
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(args.device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "protocol": {
            "confidence_selected_on": "validation",
            "selected_confidence": confidence,
            "matching": "Hungarian one-to-one",
            "test_used_for_selection": False,
            "nms_iou": args.iou,
            "image_size": args.imgsz,
        },
        "validation": {
            "wall_seconds": validation_wall,
            "mean_speed_ms_per_image": validation_speed,
            "overall_iou50": validation_overall,
            "by_collection_iou50": validation_by_collection,
            "per_plot_iou50": validation_per_plot,
        },
        "test": {
            "wall_seconds": test_wall,
            "images_per_second_end_to_end": len(test_rows) / test_wall if test_wall else None,
            "mean_speed_ms_per_image": test_speed,
            "inference_fps": 1000.0 / test_speed["inference"] if test_speed.get("inference") else None,
            "custom_instance_metrics": thresholds,
            "ultralytics_coco_metrics": builtin,
        },
    }
    (args.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selected_confidence": confidence, "test_iou50": thresholds["0.5"]["overall"], "coco": builtin}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
