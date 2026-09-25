#!/usr/bin/env python3
"""Full-polygon evaluation shared by classical segmentation and point networks."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
from functools import lru_cache
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import rasterio
import shapely
import torch
from scipy.optimize import linear_sum_assignment
from shapely.geometry import MultiPoint, Point, Polygon

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "run_r/code"))
from pointcloud.data import load_npz, read_manifest
from pointcloud.experiment_log import ROOT, record
from scripts.evaluate_pointcloud_mask_decoder import load_model, predict_plot, filter_instances
from scripts.evaluate_pointcloud_litept import predict_plot as predict_offsets, cluster_instances
import pcopw_chunks_500m_Segmentacja as classical

PROTOCOL_VERSION = "full_crowns_v3_class3_ignore_grid2"


@lru_cache(maxsize=1024)
def annotation_ignore(source_las, chm_path):
    """Explicitly unannotated class-3 canopy, never inferred from GT crown union.

    Ignore only 0.5 m cells containing outside points and no non-ground inside
    points. Ground returns do not establish crown annotation coverage. Keep
    boundary/mixed cells scored, conservatively. This mask is evaluation-only.
    """
    import laspy
    from rasterio.features import shapes
    from shapely.geometry import shape
    key = hashlib.sha256(str(source_las).encode()).hexdigest()
    path = ROOT / "annotation_ignore_v1" / f"{key}.wkb"
    if path.exists():
        return shapely.from_wkb(path.read_bytes())
    with rasterio.open(chm_path) as raster:
        transform, height, width = raster.transform, raster.height, raster.width
    outside = np.zeros(height*width, dtype=bool)
    inside = np.zeros_like(outside)
    with laspy.open(source_las) as reader:
        for points in reader.chunk_iterator(1_000_000):
            classes = np.asarray(points.classification)
            rr, cc = rasterio.transform.rowcol(transform, np.asarray(points.x), np.asarray(points.y))
            rr, cc = np.asarray(rr), np.asarray(cc)
            valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            cells = rr[valid]*width + cc[valid]
            classes = classes[valid]
            outside[cells[classes == 3]] = True
            inside[cells[(classes != 2) & (classes != 3)]] = True
    mask = (outside & ~inside).reshape(height, width).astype(np.uint8)
    geometry = shapely.union_all([shape(g) for g, value in shapes(mask, mask=mask.astype(bool), transform=transform) if value])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(shapely.to_wkb(geometry))
    return geometry


def plot_metrics(row, gt, predictions):
    return metrics(gt, predictions, annotation_ignore(row["source_las"], row["chm"]))


def metrics(gt, predictions, ignore=None):
    gt = list(gt)
    predictions = list(predictions)
    overlap = np.zeros((len(gt), len(predictions)), dtype=np.float64)
    if predictions:
        index = shapely.STRtree(predictions)
        for i, geom in enumerate(gt):
            for j in index.query(geom, predicate="intersects"):
                inter = geom.intersection(predictions[j]).area
                overlap[i, j] = inter / max(geom.area + predictions[j].area - inter, 1e-9)
    # Cardinality-first assignment, with sufficient bonus for arbitrary matrices.
    valid = overlap >= .5
    score = valid * (min(overlap.shape, default=0) + 1. + overlap)
    a, b = linear_sum_assignment(-score)
    good = overlap[a, b] >= .5
    tp = int(good.sum())
    total_iou = float(overlap[a[good], b[good]].sum())
    ignored = 0
    if ignore is not None and not ignore.is_empty:
        matched = set(b[good].tolist())
        for j, geom in enumerate(predictions):
            # Do not forgive duplicate detections of a scored reference crown.
            if j not in matched and not np.any(valid[:, j]) and geom.intersection(ignore).area / max(geom.area, 1e-9) >= .5:
                ignored += 1
    return dict(tp=tp, fp=len(predictions)-tp-ignored, fn=len(gt)-tp, iou_sum=total_iou, ignored_predictions=ignored)


def totals(rows):
    values = {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn", "iou_sum")}
    values["ignored_predictions"] = sum(row.get("ignored_predictions", 0) for row in rows)
    tp, fp, fn = values["tp"], values["fp"], values["fn"]
    denominator = tp + .5*(fp+fn)
    values.update(precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
                  f1=tp/max(denominator, 1), pq=values["iou_sum"]/max(denominator, 1),
                  sq=values["iou_sum"]/max(tp, 1))
    return values


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_dataset"]+":"+row["collection"]].append(row)
    result = totals(rows)
    result["by_source"] = {k: totals(v) for k, v in grouped.items()}
    result["source_balanced_pq"] = float(np.mean([v["pq"] for v in result["by_source"].values()]))
    result["source_balanced_f1"] = float(np.mean([v["f1"] for v in result["by_source"].values()]))
    return result


def polygon_candidates(arrays, raw, threshold, shape_mode="concave"):
    result = []
    xyz = arrays["coord"]
    offsets = raw["candidate_offset"]
    for i, object_score in enumerate(raw["object_score"]):
        start, end = offsets[i:i+2]
        probabilities = raw["point_score"][start:end]
        ids = raw["point_index"][start:end][probabilities >= threshold]
        if len(ids) < 8:
            continue
        points = xyz[ids]
        if points[:, 2].max() < 2:
            continue
        xy = points[:, :2].astype(np.float64) + arrays["source_origin"][:2]
        # Connected, hole-free crown; no top-height cut discarding lower branches.
        unique_xy = np.unique(np.round(xy*4)/4, axis=0)
        geom = MultiPoint(unique_xy)
        geom = (shapely.concave_hull(geom, ratio=.25, allow_holes=False) if shape_mode == "concave" else geom.convex_hull).buffer(.125)
        if geom.is_empty or geom.area < .25:
            continue
        if geom.geom_type == "Polygon":
            geom = Polygon(geom.exterior)
        top = int(points[:, 2].argmax())
        result.append(dict(geometry=geom, object_score=float(object_score), confidence=float(object_score*np.mean(probabilities[probabilities>=threshold])),
                           height=float(points[top, 2]), top_x=float(xy[top, 0]), top_y=float(xy[top, 1])))
    return result


def export(folder, row, instances, crs):
    folder.mkdir(parents=True, exist_ok=True)
    crowns = gpd.GeoDataFrame(dict(treeID=np.arange(1, len(instances)+1), confidence=[r["confidence"] for r in instances], area_m2=[r["geometry"].area for r in instances]), geometry=[r["geometry"] for r in instances], crs=crs)
    heights = np.asarray([r["height"] for r in instances], dtype=float)
    tops = gpd.GeoDataFrame(dict(treeID=np.arange(1, len(instances)+1), Z=heights, dbh=np.round(classical.dbh_naslund(heights), 2)), geometry=[Point(r["top_x"], r["top_y"], r["height"]) for r in instances], crs=crs)
    crowns.to_file(folder / f"crowns_{row['dataset_id']}.gpkg", driver="GPKG")
    tops.to_file(folder / f"ttops_{row['dataset_id']}.gpkg", driver="GPKG")


def evaluate_model(model, manifest, output_dir, max_points=60000, split="val", config=None, export_outputs=False, tile_size=20.):
    if split == "test" and config is None:
        raise ValueError("Test requires a config frozen on validation")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(Path(manifest), split)
    args = SimpleNamespace(tile_size=tile_size, overlap=tile_size*.4, max_points=max_points, raw_object_threshold=.05, raw_mask_threshold=.2, preserve_height=True)
    device = next(model.parameters()).device
    model.eval()
    if hasattr(model.backbone, "shuffle_orders"):
        model.backbone.shuffle_orders = False
    original_aux = model.auxiliary_losses
    model.auxiliary_losses = False
    started = time.monotonic()
    plots = []
    inference_seconds = 0.
    configs = [config] if config else [dict(object_threshold=o, mask_threshold=m, nms_iou=.5, shape_mode="concave") for o, m in itertools.product((.2, .4, .6), (.4, .6))]
    for number, row in enumerate(rows):
        arrays = load_npz(row["output"])
        raw = predict_plot(model, arrays, args, device, 20260925+number)
        gt = gpd.read_file(row["gt_vector"])
        inference_seconds += float(raw["seconds"])
        candidates = {m: polygon_candidates(arrays, raw, m) for m in {c["mask_threshold"] for c in configs}}
        plots.append((row, gt, candidates))
        if (number+1) % 10 == 0 or number+1 == len(rows):
            print(f"full-crown {split}: predicted {number+1}/{len(rows)} plots ({time.monotonic()-started:.1f}s)", flush=True)
    results = []
    for cfg in configs:
        per_plot = []
        for row, gt, candidates in plots:
            pred = filter_instances(candidates[cfg["mask_threshold"]], cfg["object_threshold"], cfg["nms_iou"])
            per_plot.append({"dataset_id": row["dataset_id"], "collection": row["collection"], "source_dataset": row["source_dataset"], **plot_metrics(row, gt.geometry, [r["geometry"] for r in pred])})
        results.append(dict(config=cfg, metrics=aggregate(per_plot), per_plot=per_plot))
        print(f"full-crown {split}: {cfg} PQ={results[-1]['metrics']['source_balanced_pq']:.4f}", flush=True)
    best = max(results, key=lambda r: (r["metrics"]["source_balanced_pq"], r["metrics"]["f1"]))
    if export_outputs:
        cfg = best["config"]
        for row, gt, candidates in plots:
            export(output_dir / "Segmentation3", row, filter_instances(candidates[cfg["mask_threshold"]], cfg["object_threshold"], cfg["nms_iou"]), gt.crs)
    best.update(split=split, protocol_version=PROTOCOL_VERSION, point_inference_seconds=inference_seconds, evaluation_total_seconds=time.monotonic()-started,
                protocol="overlapping full-crown polygon IoU >= .5; cardinality-first Hungarian; source-balanced PQ selection")
    (output_dir / f"{split}_metrics.json").write_text(json.dumps(best, indent=2)+"\n")
    model.auxiliary_losses = original_aux
    if hasattr(model.backbone, "shuffle_orders"):
        model.backbone.shuffle_orders = True
    return best


def run_baselines(manifest):
    candidates, history = {}, []
    parameter_paths = list((PROJECT.parent / "run_r").rglob("best_parameters.json"))
    parameter_paths += list((PROJECT.parent / "run_r").rglob("best_refined_parameters.json"))
    routing_path = PROJECT.parent / "run_r/evaluation_for_instance_ideas_als_structural_classes_v4/routing_configuration.json"
    routing = json.loads(routing_path.read_text())
    parameter_paths += [Path(p) for p in routing["parameter_files"]]
    for path in parameter_paths:
        data = json.loads(path.read_text())
        params = data["parameters"]
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
        candidates[key] = params
        historic_metrics = data.get("metrics", {})
        if historic_metrics and not all(isinstance(v, dict) for v in historic_metrics.values()):
            history.append(dict(path=str(path), split="tune", **historic_metrics))
        else:
            for split, values in historic_metrics.items():
                history.append(dict(path=str(path), split=split, **values))
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "historical_baselines.json").write_text(json.dumps(history, indent=2)+"\n")
    results = []
    defaults = {k: getattr(classical, k.upper()) for k in set().union(*(p.keys() for p in candidates.values())) if not k.startswith("lmf_control_")}
    rows = read_manifest(Path(manifest), "val")
    for candidate, params in candidates.items():
        experiment = f"classical_{candidate}"
        destination = ROOT / experiment
        destination.mkdir(exist_ok=True)
        result_path = destination / f"val_metrics_{PROTOCOL_VERSION}.json"
        if result_path.exists():
            results.append(json.loads(result_path.read_text()))
            continue
        apply_parameters(params, defaults)
        record(experiment, status="running", architecture="LMF + CImg watershed (run_r)", parameters=params)
        per_plot = []
        started = time.monotonic()
        for row in rows:
            with rasterio.open(row["chm"]) as source:
                chm = source.read(1).astype(float)
                chm[chm == source.nodata] = np.nan
                smooth = classical.smooth_chm(chm)
                tops, markers = classical.locate_trees_lmf(smooth, source.transform, source.crs)
                crowns = classical.segment_crowns(smooth, source.transform, markers, set(tops.treeID), source.crs)
            gt = gpd.read_file(row["gt_vector"])
            per_plot.append({"dataset_id": row["dataset_id"], "collection": row["collection"], "source_dataset": row["source_dataset"], **plot_metrics(row, gt.geometry, crowns.geometry)})
        result = dict(experiment_id=experiment, protocol_version=PROTOCOL_VERSION, parameters=params, metrics=aggregate(per_plot), per_plot=per_plot, seconds=time.monotonic()-started)
        result_path.write_text(json.dumps(result, indent=2)+"\n")
        results.append(result)
        record(experiment, status="completed_validation", val_metrics=result["metrics"], evaluation_path=str(result_path), seconds=result["seconds"])
        print(experiment, result["metrics"]["source_balanced_pq"], flush=True)
    winner = max(results, key=lambda r: (r["metrics"]["source_balanced_pq"], r["metrics"]["f1"]))
    structural = structural_baseline(manifest, routing, defaults)
    if structural["metrics"]["source_balanced_pq"] > winner["metrics"]["source_balanced_pq"]:
        winner = structural
    (ROOT / "classical_target.json").write_text(json.dumps(winner, indent=2)+"\n")


def structural_baseline(manifest, routing, defaults):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import RobustScaler
    from evaluate_structural_ensemble import chm_features
    destination = ROOT / "classical_structural_v4"
    destination.mkdir(exist_ok=True)
    result_path = destination / f"val_metrics_{PROTOCOL_VERSION}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    train = read_manifest(Path(manifest), "train")
    features = np.asarray([chm_features(Path(row["chm"])) for row in train])
    scaler = RobustScaler().fit(features)
    kmeans = KMeans(n_clusters=3, n_init=50, random_state=20260925).fit(scaler.transform(features))
    order = np.argsort([features[kmeans.labels_ == i, 0].mean() for i in range(3)])
    mapping = {int(raw): int(rank) for rank, raw in enumerate(order)}
    router = dict(center=scaler.center_.tolist(), scale=scaler.scale_.tolist(), cluster_centers=kmeans.cluster_centers_.tolist(), raw_to_ordered=mapping,
                  parameters=routing["parameters"], fit_split="train", seed=20260925)
    (destination / "router.json").write_text(json.dumps(router, indent=2)+"\n")
    scored = []
    started = time.monotonic()
    for row in read_manifest(Path(manifest), "val"):
        raw = kmeans.predict(scaler.transform(chm_features(Path(row["chm"]))[None]))[0]
        params = routing["parameters"][mapping[int(raw)]]
        apply_parameters(params, defaults)
        with rasterio.open(row["chm"]) as source:
            chm = source.read(1).astype(float)
            chm[chm == source.nodata] = np.nan
            smooth = classical.smooth_chm(chm)
            tops, markers = classical.locate_trees_lmf(smooth, source.transform, source.crs)
            crowns = classical.segment_crowns(smooth, source.transform, markers, set(tops.treeID), source.crs)
        gt = gpd.read_file(row["gt_vector"])
        scored.append({"dataset_id": row["dataset_id"], "source_dataset": row["source_dataset"], "collection": row["collection"], **plot_metrics(row, gt.geometry, crowns.geometry)})
    result = dict(experiment_id="classical_structural_v4", protocol_version=PROTOCOL_VERSION, parameters={"router": router}, metrics=aggregate(scored), per_plot=scored, seconds=time.monotonic()-started)
    result_path.write_text(json.dumps(result, indent=2)+"\n")
    record("classical_structural_v4", status="completed_validation", architecture="LMF/CImg + 3-class CHM KMeans router (fit on train only)", parameters={"router": router}, val_metrics=result["metrics"], seconds=result["seconds"])
    print("classical_structural_v4", result["metrics"]["source_balanced_pq"], flush=True)
    return result


def evaluate_offset_model(model, manifest, output_dir, max_points=40000, split="val", config=None, export_outputs=False, tile_size=20.):
    if split == "test" and config is None:
        raise ValueError("Test requires a frozen validation config")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(tile_size=tile_size, overlap=tile_size*.4, max_points=max_points, preserve_height=True)
    model.eval()
    model.backbone.shuffle_orders = False
    device = next(model.parameters()).device
    plots, seconds = [], 0.
    started = time.monotonic()
    for row in read_manifest(Path(manifest), split):
        arrays = load_npz(row["output"])
        raw = predict_offsets(model, arrays, args, device, seed=20260925)
        # This reference-only field is not consumed by postprocessing.
        raw.pop("gt_tree_id", None)
        plots.append((row, gpd.read_file(row["gt_vector"]), raw))
        seconds += float(raw["seconds"])
    configs = [config] if config else [dict(probability=p, vote_smoothing=.5, peak_separation=s,
                     peak_threshold_fraction=.02, assignment_radius=2., min_voxels=12, height_ratio=0.)
                     for p, s in itertools.product((.3, .5, .7), (1., 2.))]
    results = []
    for cfg in configs:
        scored, exports = [], []
        for row, gt, raw in plots:
            predictions = [p for p in cluster_instances(raw, cfg) if p["height"] >= 2.]
            scored.append({"dataset_id": row["dataset_id"], "source_dataset": row["source_dataset"], "collection": row["collection"], **plot_metrics(row, gt.geometry, [p["geometry"] for p in predictions])})
            exports.append((row, predictions, gt.crs))
        results.append(dict(config=cfg, metrics=aggregate(scored), per_plot=scored))
        if export_outputs and config:
            for row, predictions, crs in exports:
                export(output_dir / "Segmentation3", row, predictions, crs)
    best = max(results, key=lambda r: (r["metrics"]["source_balanced_pq"], r["metrics"]["f1"]))
    best.update(split=split, protocol_version=PROTOCOL_VERSION, point_inference_seconds=seconds, evaluation_total_seconds=time.monotonic()-started,
                protocol="full crown polygon IoU >= .5; cardinality-first matching; point offsets + voting + filled convex hull")
    (output_dir / f"{split}_metrics.json").write_text(json.dumps(best, indent=2)+"\n")
    print(f"offset {split} source-balanced PQ={best['metrics']['source_balanced_pq']:.4f}", flush=True)
    model.backbone.shuffle_orders = True
    return best


def apply_parameters(params, defaults):
    for key, value in {**defaults, **params}.items():
        if not key.startswith("lmf_control_"):
            setattr(classical, key.upper(), value)
    if "lmf_control_window_0" in params:
        classical.LMF_CONTROL_HEIGHTS = (params["min_height"], *(params[f"lmf_control_height_{i}"] for i in range(1, 4)))
        classical.LMF_CONTROL_WINDOWS = tuple(params[f"lmf_control_window_{i}"] for i in range(4))
    else:
        classical.LMF_CONTROL_HEIGHTS = classical.LMF_CONTROL_WINDOWS = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baseline", "model"))
    parser.add_argument("--manifest", type=Path, default=PROJECT.parent / "combined_als_crowns_v1/manifest.csv")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "baseline":
        run_baselines(args.manifest)
    else:
        model = load_model(args, torch.device("cuda:0"))
        result = evaluate_model(model, args.manifest, args.output_dir, export_outputs=True)
        print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
