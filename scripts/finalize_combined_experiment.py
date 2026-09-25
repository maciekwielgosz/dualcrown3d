#!/usr/bin/env python3
"""Freeze a validation-selected model and compare once on the reserved split."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import rasterio
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import read_manifest, load_npz
from pointcloud.model import LitePTTreeInstance
from pointcloud.experiment_log import ROOT, record
from scripts.evaluate_pointcloud_mask_decoder import load_model
from scripts.evaluate_combined_full_crowns import (
    aggregate, apply_parameters, classical, evaluate_model, evaluate_offset_model,
    export, plot_metrics, PROTOCOL_VERSION,
)

MANIFEST = Path(os.environ.get("SEGMENTATION_DATA_MANIFEST", PROJECT.parent / "combined_als_crowns_v1/manifest.csv")).resolve()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def load_candidate(weights, experiment):
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    if "offset" in experiment:
        model = LitePTTreeInstance(patch_size=int(args.get("patch_size", 256))).cuda()
        model.load_state_dict(checkpoint["model"])
        evaluator = evaluate_offset_model
    else:
        model = load_model(SimpleNamespace(weights=Path(weights)), torch.device("cuda:0"))
        evaluator = evaluate_model
    model.eval()
    return model, evaluator, args


def assess():
    """Re-evaluate all stopped candidates under the same deterministic code."""
    for path in sorted((ROOT / "records").glob("*.json")):
        info = json.loads(path.read_text())
        experiment = info["experiment_id"]
        if experiment.startswith(("classical", "smoke")) or "training_dir" not in info:
            continue
        if "postprocessing_log" in info:
            print(f"Preserving dedicated scale calibration: {experiment}", flush=True)
            continue
        if info["status"] == "training":
            print(f"Skipping still-active training: {experiment}", flush=True)
            continue
        original_weights = Path(info["checkpoint"])
        paths = [original_weights]
        # Grid correction affects LitePT inference but not point-MLP inference.
        # Revisit saved epoch candidates, not only the old flawed-grid winner.
        if experiment.startswith("litept") and "average" not in experiment:
            paths += [original_weights.parent / name for name in ("best_loss.pt", "best_iou.pt", "last.pt")]
        seen_epochs, results = set(), []
        for weights in paths:
            if not weights.exists():
                continue
            payload = torch.load(weights, map_location="cpu", weights_only=False)
            epoch_key = str(payload.get("epoch", payload.get("source_epochs", "average")))
            del payload
            if epoch_key in seen_epochs:
                continue
            seen_epochs.add(epoch_key)
            destination = ROOT / experiment / "final_validation" / PROTOCOL_VERSION / weights.stem
            cached = destination / "assessment.json"
            weights_hash, data_hash = digest(weights), digest(MANIFEST)
            result = json.loads(cached.read_text()) if cached.exists() else None
            if result is None or result.get("checkpoint_sha256") != weights_hash or result.get("dataset_manifest_sha256") != data_hash:
                torch.manual_seed(20260925)
                model, evaluator, args = load_candidate(weights, experiment)
                result = evaluator(model, MANIFEST, destination, max_points=args["max_points"], tile_size=args["crop_size"])
                result.update(checkpoint=str(weights), checkpoint_sha256=weights_hash, experiment_id=experiment,
                              dataset_manifest_sha256=data_hash, grid_version=2, checkpoint_epoch=epoch_key,
                              max_points=args["max_points"], tile_size=args["crop_size"])
                cached.write_text(json.dumps(result, indent=2)+"\n")
                del model
                torch.cuda.empty_cache()
            results.append(result)
        if not results:
            continue
        winner = max(results, key=lambda r: (r["metrics"]["source_balanced_pq"], r["metrics"]["f1"]))
        (ROOT / experiment / "final_validation/selection.json").write_text(json.dumps(winner, indent=2)+"\n")
        record(experiment, final_val_metrics=winner["metrics"], final_validation_checkpoint=winner["checkpoint"], final_checkpoint_sha256=winner["checkpoint_sha256"], grid_version=2, protocol_version=PROTOCOL_VERSION,
               early_validation_caveat="Training-time full-polygon scores precede class3-ignore protocol; LitePT scores may also use old sparse-grid cache. Use final_val_* for comparison.")


def freeze():
    destination = ROOT / "selected"
    for path in (ROOT / "records").glob("*.json"):
        if json.loads(path.read_text()).get("status") in ("training", "calibrating_validation"):
            raise RuntimeError("Complete the active training runs before freezing test selection")
    if (destination / "frozen_selection.json").exists():
        raise FileExistsError("A selection is already frozen; do not replace it after test exposure")
    candidates = [json.loads(p.read_text()) for p in ROOT.glob("*/final_validation/selection.json")]
    candidates = [c for c in candidates if c.get("protocol_version") == PROTOCOL_VERSION]
    if not candidates:
        raise RuntimeError("Run assess first")
    baseline = json.loads((ROOT / "classical_target.json").read_text())
    if baseline.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Re-evaluate classical candidates under the current protocol")
    eligible = [c for c in candidates if c["metrics"]["source_balanced_pq"] > baseline["metrics"]["source_balanced_pq"] and c["metrics"]["f1"] > baseline["metrics"]["f1"]]
    if not eligible:
        raise RuntimeError("No candidate exceeds classical validation PQ and F1; do not claim success")
    winner = max(eligible, key=lambda r: (r["metrics"]["source_balanced_pq"], r["metrics"]["f1"]))
    if digest(winner["checkpoint"]) != winner["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed after final validation")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(winner["checkpoint"], destination / "best.pt")
    frozen = dict(created_utc=datetime.now(timezone.utc).isoformat(), dataset_manifest_sha256=digest(MANIFEST),
                  model=winner, classical=baseline, test_used_for_selection=False,
                  caveat="Historically exposed plot collection, not a pristine independent publication holdout")
    (destination / "frozen_selection.json").write_text(json.dumps(frozen, indent=2)+"\n")
    record(winner["experiment_id"], status="frozen_for_test", frozen_checkpoint=str(destination / "best.pt"), frozen_selection=str(destination / "frozen_selection.json"))
    print("Frozen", winner["experiment_id"], winner["metrics"]["source_balanced_pq"], flush=True)


def evaluate_classical_test(frozen):
    params = frozen["classical"]["parameters"]
    router = params.get("router")
    if not router:
        apply_parameters(params, {})
    per_plot = []
    seconds = 0.
    folder = ROOT / "selected/classical_test/Segmentation3"
    for row in read_manifest(MANIFEST, "test"):
        start = time.monotonic()
        if router:
            from evaluate_structural_ensemble import chm_features
            features = (chm_features(Path(row["chm"]))-np.asarray(router["center"]))/np.asarray(router["scale"])
            raw = int(np.square(np.asarray(router["cluster_centers"])-features).sum(axis=1).argmin())
            apply_parameters(router["parameters"][router["raw_to_ordered"][str(raw)]], {})
        with rasterio.open(row["chm"]) as source:
            chm = source.read(1).astype(float)
            chm[chm == source.nodata] = np.nan
            smooth = classical.smooth_chm(chm)
            tops, markers = classical.locate_trees_lmf(smooth, source.transform, source.crs)
            crowns = classical.segment_crowns(smooth, source.transform, markers, set(tops.treeID), source.crs)
        seconds += time.monotonic()-start
        gt = gpd.read_file(row["gt_vector"])
        per_plot.append(dict(dataset_id=row["dataset_id"], source_dataset=row["source_dataset"], collection=row["collection"], **plot_metrics(row, gt.geometry, crowns.geometry)))
        tops = tops.set_index("treeID")
        instances = [dict(geometry=c.geometry, confidence=1., height=float(tops.loc[c.treeID, "Z"]), top_x=float(tops.loc[c.treeID].geometry.x), top_y=float(tops.loc[c.treeID].geometry.y)) for c in crowns.itertuples()]
        export(folder, row, instances, gt.crs)
    result = dict(metrics=aggregate(per_plot), per_plot=per_plot, chm_segmentation_seconds=seconds)
    (folder.parent / "test_metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def paired_bootstrap(model_rows, baseline_rows, iterations=2000):
    """Paired spatial-cluster resampling, retaining cross-source group links."""
    rows = read_manifest(MANIFEST, "test")
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row["dataset_id"])
    model_by_id = {r["dataset_id"]: r for r in model_rows}
    baseline_by_id = {r["dataset_id"]: r for r in baseline_rows}
    names = list(groups)
    rng = np.random.default_rng(20260925)
    differences = []
    for _ in range(iterations):
        selected = [identifier for group in rng.choice(names, len(names), replace=True) for identifier in groups[group]]
        m = aggregate([model_by_id[i] for i in selected])
        b = aggregate([baseline_by_id[i] for i in selected])
        differences.append(m["source_balanced_pq"]-b["source_balanced_pq"])
    return dict(iterations=iterations, spatial_groups=len(groups), delta_source_balanced_pq_95ci=np.percentile(differences, [2.5,97.5]).tolist(),
                caveat="Source mixture varies when spatial groups are resampled; small source-specific holdouts limit precision")


def test():
    selected = ROOT / "selected"
    frozen = json.loads((selected / "frozen_selection.json").read_text())
    if digest(MANIFEST) != frozen["dataset_manifest_sha256"] or digest(selected / "best.pt") != frozen["model"]["checkpoint_sha256"]:
        raise RuntimeError("Frozen data or checkpoint checksum mismatch")
    if (selected / "comparison.json").exists():
        raise FileExistsError("Final comparison already exists; do not iterate on this test")
    ledger = selected / "test_started.json"
    entry = dict(frozen_selection_sha256=digest(selected / "frozen_selection.json"), started_utc=datetime.now(timezone.utc).isoformat())
    if ledger.exists() and json.loads(ledger.read_text())["frozen_selection_sha256"] != entry["frozen_selection_sha256"]:
        raise RuntimeError("Test was already accessed by a different selection")
    if not ledger.exists():
        ledger.write_text(json.dumps(entry, indent=2)+"\n")
    experiment = frozen["model"]["experiment_id"]
    torch.manual_seed(20260925)
    model, evaluator, args = load_candidate(selected / "best.pt", experiment)
    # Small warm-up excluded from timing; uses a training plot, never test labels.
    from scripts.evaluate_combined_full_crowns import predict_offsets, predict_plot
    warmup_args = SimpleNamespace(tile_size=frozen["model"]["tile_size"], overlap=frozen["model"]["tile_size"]*.4,
                                 max_points=frozen["model"]["max_points"], preserve_height=True,
                                 raw_object_threshold=.05, raw_mask_threshold=.2)
    warmup_arrays = load_npz(read_manifest(MANIFEST, "train")[0]["output"])
    warmup = predict_offsets if "offset" in experiment else predict_plot
    warmup(model, warmup_arrays, warmup_args, torch.device("cuda:0"), seed=20260925)
    torch.cuda.synchronize()
    model_result = evaluator(model, MANIFEST, selected / "model_test", max_points=frozen["model"]["max_points"], split="test", config=frozen["model"]["config"], export_outputs=True, tile_size=frozen["model"]["tile_size"])
    baseline_result = evaluate_classical_test(frozen)
    mm, bm = model_result["metrics"], baseline_result["metrics"]
    comparison = dict(experiment_id=experiment, model=mm, classical=bm,
                      delta={k: mm[k]-bm[k] for k in ("source_balanced_pq", "source_balanced_f1", "f1", "recall", "precision", "pq")},
                      target_met_on_test=mm["source_balanced_pq"] > bm["source_balanced_pq"] and mm["f1"] > bm["f1"],
                      uncertainty=paired_bootstrap(model_result["per_plot"], baseline_result["per_plot"]),
                      model_point_inference_seconds=model_result["point_inference_seconds"],
                      classical_chm_segmentation_seconds=baseline_result["chm_segmentation_seconds"],
                      timing_caveat="Cached NPZ versus cached CHM; excludes LAS normalization/voxelization; not a whole-pipeline FPS comparison",
                      test_caveat=frozen["caveat"])
    (selected / "comparison.json").write_text(json.dumps(comparison, indent=2)+"\n")
    record(experiment, status="evaluated_on_frozen_test", test_metrics=mm, test_comparison=str(selected / "comparison.json"), target_met_on_test=comparison["target_met_on_test"])
    record(frozen["classical"]["experiment_id"], test_metrics=bm)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("assess", "freeze", "test"))
    arguments = parser.parse_args()
    {"assess": assess, "freeze": freeze, "test": test}[arguments.stage]()
