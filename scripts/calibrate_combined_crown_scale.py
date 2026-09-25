#!/usr/bin/env python3
"""Validation-only sensitivity to whole-crown center clustering scale."""
import hashlib
import itertools
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz, read_manifest
from pointcloud.experiment_log import ROOT, record
from scripts.finalize_combined_experiment import MANIFEST, digest, load_candidate
from scripts.evaluate_combined_full_crowns import PROTOCOL_VERSION, aggregate, plot_metrics, predict_offsets, cluster_instances


def main():
    experiment = "litept_offset_crownscale_v3"
    folder = ROOT / experiment
    if (ROOT / "selected/frozen_selection.json").exists():
        raise RuntimeError("Do not calibrate after freezing the final test model")
    folder.mkdir(exist_ok=True)
    source = ROOT / "litept_offset_average_v1/weights/best.pt"
    weights = folder / "weights/best.pt"
    weights.parent.mkdir(exist_ok=True)
    if weights.exists() and digest(weights) != digest(source):
        raise RuntimeError("Source weights changed")
    if not weights.exists():
        shutil.copy2(source, weights)
    signature = dict(checkpoint_sha256=digest(weights), dataset_manifest_sha256=digest(MANIFEST), protocol_version=PROTOCOL_VERSION)
    torch.manual_seed(20260925)
    model, _, args = load_candidate(weights, experiment)
    model.backbone.shuffle_orders = False
    model.eval()
    prediction_args = SimpleNamespace(tile_size=args["crop_size"], overlap=args["crop_size"]*.4, max_points=args["max_points"], preserve_height=True)
    destination = folder / "raw_validation"
    destination.mkdir(exist_ok=True)
    manifest_path = destination / "signature.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != signature:
        raise RuntimeError("Validation cache does not match frozen inputs")
    manifest_path.write_text(json.dumps(signature, indent=2)+"\n")
    record(experiment, status="calibrating_validation", architecture="LitePT-S averaged semantic/offset heads + validation-selected full-crown clustering scale", checkpoint=str(weights), training_dir=str(folder),
           parameters={"weight_source": str(source), "extra_sgd_epochs": 0, "probability_candidates": [.3,.5,.7], "scale_profiles_m": [[.5,1,2],[.5,2,2],[1,2,4],[1,3,4],[1.5,3,6],[1.5,4,6],[2,4,8],[2,6,10]]})
    plots, seconds = [], 0.
    for i, row in enumerate(read_manifest(MANIFEST, "val")):
        cache = destination / f"{row['dataset_id']}.npz"
        if cache.exists():
            with np.load(cache) as data:
                raw = {k: data[k] for k in data.files}
        else:
            raw = predict_offsets(model, load_npz(row["output"]), prediction_args, torch.device("cuda:0"), seed=20260925)
            raw.pop("gt_tree_id", None)
            np.savez_compressed(cache, **raw)
        seconds += float(raw["seconds"])
        plots.append((row, gpd.read_file(row["gt_vector"]), raw))
        if (i+1)%10 == 0:
            print(f"Scale calibration: predicted {i+1}/79", flush=True)
    del model
    torch.cuda.empty_cache()
    trials_path = folder / "postprocessing_trials.json"
    trials = json.loads(trials_path.read_text()) if trials_path.exists() else []
    profiles = ((.5,1,2),(.5,2,2),(1,2,4),(1,3,4),(1.5,3,6),(1.5,4,6),(2,4,8),(2,6,10))
    for i, (probability, (smoothing, separation, radius)) in enumerate(itertools.product((.3,.5,.7), profiles)):
        cfg = dict(probability=probability, vote_smoothing=smoothing, peak_separation=separation,
                   peak_threshold_fraction=.02, assignment_radius=radius, min_voxels=12, height_ratio=0.)
        if any(t["config"] == cfg for t in trials):
            continue
        started = time.monotonic()
        scored = []
        for row, gt, raw in plots:
            pred = [p for p in cluster_instances(raw, cfg) if p["height"] >= 2.]
            scored.append(dict(dataset_id=row["dataset_id"], source_dataset=row["source_dataset"], collection=row["collection"], **plot_metrics(row, gt.geometry, [p["geometry"] for p in pred])))
        trial = dict(experiment_id=experiment, trial=i, config=cfg, metrics=aggregate(scored), per_plot=scored, calibration_seconds=time.monotonic()-started, **signature)
        trials.append(trial)
        trials_path.write_text(json.dumps(trials, indent=2)+"\n")
        print(f"Scale trial {i+1}/24: {cfg} PQ={trial['metrics']['source_balanced_pq']:.4f}, F1={trial['metrics']['f1']:.4f}", flush=True)
    winner = max(trials, key=lambda r:(r["metrics"]["source_balanced_pq"],r["metrics"]["f1"]))
    winner = dict(winner, checkpoint=str(weights), max_points=args["max_points"], tile_size=args["crop_size"],
                  split="val", grid_version=2, point_inference_seconds=seconds, checkpoint_epoch="average_epochs_10_35")
    validation = folder / "final_validation"
    validation.mkdir(exist_ok=True)
    (validation / "selection.json").write_text(json.dumps(winner, indent=2)+"\n")
    record(experiment, status="completed_validation", final_val_metrics=winner["metrics"], selected_postprocessing=winner["config"],
           final_validation_checkpoint=str(weights), final_checkpoint_sha256=signature["checkpoint_sha256"], protocol_version=PROTOCOL_VERSION, grid_version=2,
           postprocessing_log=str(trials_path), parameter_count=12714809)


if __name__ == "__main__":
    main()
