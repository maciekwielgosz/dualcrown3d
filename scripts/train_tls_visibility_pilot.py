#!/usr/bin/env python3
"""Matched small-data fine-tuning of the retained DualCrown3D checkpoint."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import geopandas as gpd
import numpy as np
from openpyxl import Workbook
import shapely
import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz, move_to_device, read_manifest
from pointcloud.instance_output import merge_masks_with_sources, point_instance_metrics
from pointcloud.joint_training import DrawDataset, fixed_serialization, joint_losses, sampling_weights
from scripts.calibrate_legacy_small_trees import small_hits
from scripts.evaluate_combined_full_crowns import aggregate, metrics
from scripts.predict_dual_head import predict
from scripts.prepare_treescan_helios_dualcrown import sha256, write_csv
from scripts.train_supervision_v4 import INITIAL, REAL, build, configurations

DEFAULT_DATA = PROJECT / "outputs/tls_visibility_pilot_matched_v2/manifest.csv"
DEFAULT_OUTPUT = PROJECT / "outputs/tls_visibility_finetune_pilot_v1"
MODES = ("real_only", "helios", "thinning", "visibility")


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


def score(record: dict) -> float:
    return math.sqrt(record["point_metrics"]["source_balanced_pq"] *
                     record["metrics"]["source_balanced_pq"])


def evaluate(model, rows: list[dict], folder: Path, config: dict) -> dict:
    model.eval()
    fixed_serialization(model)
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.monotonic()
    for count, row in enumerate(rows, 1):
        arrays = load_npz(row["output"])
        raw = predict(model, arrays, owner_only=True)
        labels, _, instances, _ = merge_masks_with_sources(arrays, raw, config)
        reference = gpd.read_file(row["gt_vector"])
        ignore = shapely.from_wkb(Path(row["ignore_vector"]).read_bytes())
        geometries = [entry["geometry"] for entry in instances]
        result = {key: row[key] for key in
                  ("dataset_id", "source_dataset", "collection", "annotation_method")}
        result.update(metrics(reference.geometry, geometries, ignore))
        result.update(small_hits(reference.geometry, geometries))
        result["point_metrics"] = point_instance_metrics(arrays["tree_id"], labels)
        known = arrays["tree_id"] > 0
        result["known_tree_voxel_coverage"] = (
            float(np.mean(labels[known] > 0)) if known.any() else 0.)
        records.append(result)
        print(f"validation {count}/{len(rows)} {row['dataset_id']}", flush=True)
    point_rows = [dict(**{k: record[k] for k in
                        ("dataset_id", "source_dataset", "collection", "annotation_method")},
                       **record["point_metrics"]) for record in records]
    small = {}
    for area in (4, 10):
        ground_truth = sum(r[f"small_{area}_gt"] for r in records)
        true_positive = sum(r[f"small_{area}_tp"] for r in records)
        small[f"up_to_{area}_m2"] = dict(gt=ground_truth, tp=true_positive,
                                         recall=true_positive / max(ground_truth, 1))
    summary = dict(point_metrics=aggregate(point_rows), metrics=aggregate(records),
                   small_crowns=small, per_plot=records,
                   seconds=time.monotonic() - started)
    dump(folder / "metrics.json", summary)
    return summary


def real_weights(rows: list[dict]) -> np.ndarray:
    groups = [(row["source_dataset"], row["collection"]) for row in rows]
    counts = Counter(groups)
    return np.asarray([1. / len(counts) / counts[group] for group in groups])


def run_mode(mode: str, root: Path, source_rows: list[dict], real: list[dict],
             validation: list[dict], baseline: dict, options) -> dict:
    run = root / mode
    if run.exists():
        raise FileExistsError(f"Pilot run protected from replacement: {run}")
    run.mkdir(parents=True)
    (run / "weights").mkdir()
    synthetic = [] if mode == "real_only" else [
        row for row in source_rows if row["pilot_mode"] == mode]
    if mode != "real_only" and len(synthetic) != options.parents:
        raise ValueError(f"{mode}: expected {options.parents} training parents")
    rows = real + synthetic
    write_csv(run / "training_manifest.csv", rows)
    weights = real_weights(rows) if mode == "real_only" else sampling_weights(rows, .5)
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    model, _ = build(INITIAL)
    model.configure_training("partial")
    fixed_serialization(model)
    backbone = [p for n, p in model.named_parameters() if p.requires_grad and
                n.startswith("legacy.backbone.")]
    heads = [p for n, p in model.named_parameters() if p.requires_grad and
             not n.startswith("legacy.backbone.")]
    parameters = [*backbone, *heads]
    optimizer = torch.optim.AdamW([
        dict(params=heads, lr=1e-4), dict(params=backbone, lr=1e-5)], weight_decay=.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=options.epochs, eta_min=5e-7)
    dataset = DrawDataset(rows, options.seed, options.max_points, hard=True)
    config = configurations()["legacy_consensus"]
    initial = dict(checkpoint=str(INITIAL.resolve()),
                   checkpoint_sha256=sha256(INITIAL),
                   synthetic_parent_plots=len(synthetic),
                   source_manifest_sha256=sha256(options.data),
                   real_manifest_sha256=sha256(REAL),
                   seed=options.seed, epochs=options.epochs,
                   samples_per_epoch=options.samples,
                   max_points=options.max_points,
                   real_sample_fraction=1. if mode == "real_only" else .5,
                   initial_score=score(baseline),
                   merge_config=config,
                   validation_plot_count=len(validation),
                   split="train-only synthetic; native-real validation")
    dump(run / "configuration.json", initial)
    best = score(baseline)
    best_epoch = 0
    best_result = baseline
    best_checkpoint = INITIAL
    history = []
    for epoch in range(1, options.epochs + 1):
        model.configure_training("heads" if epoch <= options.warmup_epochs else "partial")
        model.train()
        model.point_decoder.epoch = 100 + epoch
        dataset.epoch = epoch
        rng = np.random.default_rng(options.seed + epoch * 100003)
        draws = [(int(index), step) for step, index in enumerate(
            rng.choice(len(rows), options.samples, p=weights / weights.sum()))]
        loader = DataLoader(dataset, batch_size=None, sampler=draws,
                            num_workers=2, pin_memory=True)
        optimizer.zero_grad(set_to_none=True)
        totals = Counter()
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(loader, 1):
            batch = move_to_device(batch, torch.device("cuda:0"))
            inputs = {key: batch[key] for key in
                      ("coord", "grid_coord", "feat", "offset", "tree_id")}
            losses = joint_losses(model(inputs), batch)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"{mode} epoch {epoch}: nonfinite loss")
            (losses["loss"] / 2.).backward()
            if step % 2 == 0 or step == len(draws):
                torch.nn.utils.clip_grad_norm_(parameters, 2., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            totals.update({key: float(value.detach()) for key, value in losses.items()})
        scheduler.step()
        row = dict(epoch=epoch, mode=mode,
                   train_seconds=time.monotonic() - started,
                   peak_vram_mb=torch.cuda.max_memory_allocated() / 1024**2,
                   **{f"train_{key}": val / len(draws) for key, val in totals.items()})
        if epoch % options.eval_every == 0 or epoch == options.epochs:
            result = evaluate(model, validation,
                              run / "validation" / f"epoch_{epoch:03d}", config)
            current = score(result)
            row.update(validation_score=current,
                       point_sb_pq=result["point_metrics"]["source_balanced_pq"],
                       crown_sb_pq=result["metrics"]["source_balanced_pq"],
                       small_10_tp=result["small_crowns"]["up_to_10_m2"]["tp"])
            if current > best:
                best, best_epoch, best_result = current, epoch, result
                best_checkpoint = run / "weights/best.pt"
                temporary = run / "weights/best.tmp.pt"
                torch.save(dict(model=model.state_dict(), model_args=dict(
                    queries=96, decoder_layers=3, memory_tokens=1024),
                    epoch=epoch, configuration=initial), temporary)
                os.replace(temporary, best_checkpoint)
        history.append(row)
        write_csv(run / "training_log.csv", history)
        print(json.dumps(row), flush=True)
    report = dict(mode=mode, best_epoch=best_epoch, best_score=best,
                  checkpoint=str(best_checkpoint.resolve()),
                  checkpoint_sha256=sha256(best_checkpoint),
                  validation=best_result,
                  accepted=all(best_result[branch][metric] >= baseline[branch][metric]
                    for branch in ("point_metrics", "metrics")
                    for metric in ("source_balanced_pq", "source_balanced_f1")),
                  initial_configuration=initial)
    dump(run / "selected.json", report)
    dump(run / "DONE.json", dict(best_epoch=best_epoch,
                                epochs_completed=options.epochs,
                                selected_checkpoint=report["checkpoint"]))
    del model, optimizer
    torch.cuda.empty_cache()
    return report


def workbook(root: Path, reports: list[dict]) -> None:
    book = Workbook()
    summary = book.active
    summary.title = "comparison"
    summary.append(["mode", "best_epoch", "point_sb_pq", "crown_sb_pq",
                    "point_sb_f1", "crown_sb_f1", "small_4_tp", "small_10_tp",
                    "accepted", "checkpoint"])
    for report in reports:
        value = report["validation"]
        summary.append([report["mode"], report["best_epoch"],
                        value["point_metrics"]["source_balanced_pq"],
                        value["metrics"]["source_balanced_pq"],
                        value["point_metrics"]["source_balanced_f1"],
                        value["metrics"]["source_balanced_f1"],
                        value["small_crowns"]["up_to_4_m2"]["tp"],
                        value["small_crowns"]["up_to_10_m2"]["tp"],
                        report["accepted"], report["checkpoint"]])
    summary.freeze_panes = "A2"
    summary.auto_filter.ref = summary.dimensions
    for mode in MODES:
        path = root / mode / "training_log.csv"
        if not path.exists():
            continue
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        sheet = book.create_sheet(mode[:31])
        sheet.append(list(rows[0]))
        for row in rows:
            sheet.append([float(value) if key != "mode" and value and
                          key != "epoch" else int(value) if key == "epoch"
                          else value for key, value in row.items()])
    temporary = root / "comparison.tmp.xlsx"
    book.save(temporary)
    os.replace(temporary, root / "comparison.xlsx")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--max-points", type=int, default=16000)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--parents", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the training pilot")
    if args.epochs < 1 or args.samples < 1 or args.eval_every < 1:
        raise ValueError("Positive training/evaluation budgets required")
    torch.set_num_threads(4)
    data = args.data.resolve()
    source_rows = read_manifest(data, "train")
    modes = Counter(row.get("pilot_mode") for row in source_rows)
    if any(modes[key] != args.parents for key in MODES[1:]):
        raise ValueError(f"Incomplete synthetic pilot: {modes}")
    parent_sets = [{r["parent_plot"] for r in source_rows if r["pilot_mode"] == key}
                   for key in MODES[1:]]
    if len(set(map(frozenset, parent_sets))) != 1:
        raise ValueError("The three synthetic modes have different parents")
    real = [row for row in read_manifest(REAL, "train")
            if row["train_eligible"] == "true"]
    validation = [row for row in read_manifest(REAL, "val")
                  if row.get("point_eval_eligible", "true") == "true"]
    if any(row["group_id"] in {r["group_id"] for r in validation}
           for row in real):
        raise ValueError("Native train/validation group leakage")
    baseline_path = PROJECT / "outputs/dualcrown3d_legacy_small_tree_calibration_extended/selected.json"
    archived = json.loads(baseline_path.read_text())
    if sha256(INITIAL) != archived["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint differs from verified calibration")
    baseline = archived["baseline"]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    for mode in args.modes:
        reports.append(run_mode(mode, root, source_rows, real,
                                validation, baseline, args))
        workbook(root, reports)
        dump(root / "progress.json", dict(completed=[r["mode"] for r in reports]))
    dump(root / "comparison.json", dict(baseline=baseline,
         modes=reports, selected_by_validation=max(reports, key=lambda r: r["best_score"])["mode"],
         test_used_for_selection=False,
         limitation="Six synthetic parents, one seed, short fine-tuning; screening only"))


if __name__ == "__main__":
    main()
