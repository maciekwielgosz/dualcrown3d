#!/usr/bin/env python3
"""Fine-tune DualCrown3D with HELIOS data and guarded real-data replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from openpyxl import Workbook
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT))

from pointcloud.data import PointCloudCropDataset, load_npz, move_to_device, read_manifest
from pointcloud.dual_head import DualHeadLitePT, dense_losses
from pointcloud.instance_output import merge_masks, point_instance_metrics
from scripts.evaluate_combined_full_crowns import aggregate, plot_metrics
from scripts.predict_dual_head import predict


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        help="Optional prebuilt real+HELIOS manifest; generated in the run directory by default")
    parser.add_argument("--real-manifest", type=Path, default=WORKSPACE / "combined_als_crowns_no_rectangles_v2/manifest.csv")
    parser.add_argument("--helios-manifest", type=Path, default=WORKSPACE / "dualcrown3d_treescan_helios_v1/manifest.csv")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "outputs/dual_head_satv2_litept_v3/weights/best.pt")
    parser.add_argument("--selection", type=Path, default=PROJECT / "configs/dual_head_complete_consensus.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/dualcrown3d_treescan_helios_finetune_v2")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--samples-per-epoch", type=int, default=240)
    parser.add_argument("--max-points", type=int, default=40000)
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--finalize-only", action="store_true",
                        help="Rebuild the final report from an already completed run")
    return parser.parse_args()


def pass_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    sums, batches = {}, 0
    started = time.monotonic()
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = move_to_device(batch, device)
            inputs = {key: batch[key] for key in ("coord", "grid_coord", "feat", "offset")}
            if optimizer is not None:
                inputs["tree_id"] = batch["tree_id"]
            result = model(inputs)
            losses = dense_losses(result, batch)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Non-finite fine-tuning loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.point_decoder.parameters(), 2.0, error_if_nonfinite=True)
                optimizer.step()
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach())
            batches += 1
    return {**{key: value / max(batches, 1) for key, value in sums.items()},
            "seconds": time.monotonic() - started}


def quality(result):
    point = result["point_metrics"]["source_balanced_pq"]
    crown = result["metrics"]["source_balanced_pq"]
    return math.sqrt(max(point, 0.0) * max(crown, 0.0))


def guarded_score(real, helios):
    # Harmonic mean prevents a synthetic-only gain from hiding real-data collapse.
    a, b = quality(real), quality(helios)
    return 2 * a * b / max(a + b, 1e-12)


def save_checkpoint(path, model, epoch, args, optimizer, scheduler, score):
    torch.save({
        "format": "dual_head_litept_v3_helios_finetune",
        "model": model.state_dict(), "epoch": epoch, "score": score,
        "config": {**vars(args), "manifest": str(args.manifest.resolve())},
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
    }, path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summary_row(stage, domain, result):
    crown, point = result["metrics"], result["point_metrics"]
    return {
        "stage": stage, "domain": domain,
        "crown_PQ": crown["pq"], "crown_F1": crown["f1"],
        "crown_source_balanced_PQ": crown["source_balanced_pq"],
        "crown_source_balanced_F1": crown["source_balanced_f1"],
        "point_PQ": point["pq"], "point_F1": point["f1"],
        "point_source_balanced_PQ": point["source_balanced_pq"],
        "point_source_balanced_F1": point["source_balanced_f1"],
        "inference_seconds": result["inference_seconds"],
    }


def write_workbook(path, training_rows, comparison, configuration):
    book = Workbook()
    sheet = book.active; sheet.title = "training"
    fields = list(dict.fromkeys(key for row in training_rows for key in row))
    sheet.append(fields)
    for row in training_rows: sheet.append([row.get(key) for key in fields])
    sheet = book.create_sheet("before_after")
    fields = list(comparison[0])
    sheet.append(fields)
    for row in comparison: sheet.append([row.get(key) for key in fields])
    sheet = book.create_sheet("configuration")
    sheet.append(["parameter", "value"])
    for key, value in configuration.items():
        sheet.append([key, json.dumps(value, default=str) if isinstance(value, (dict, list)) else str(value)])
    for sheet in book:
        sheet.freeze_panes = "A2"; sheet.auto_filter.ref = sheet.dimensions
    temporary = path.with_suffix(".tmp.xlsx")
    book.save(temporary); os.replace(temporary, path)


def finalize_report(output, initial_checkpoint, comparison):
    initial = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)["model"]
    tuned = torch.load(output / "weights/best.pt", map_location="cpu", weights_only=False)["model"]
    legacy_keys = [key for key in initial if key.startswith("legacy.")]
    legacy_unchanged = all(torch.equal(initial[key].cpu(), tuned[key].cpu()) for key in legacy_keys)
    selected = json.loads((output / "selected.json").read_text())
    report = {
        "best_epoch": selected["epoch"], "selection_score": selected["selection_score"],
        "legacy_and_backbone_unchanged": legacy_unchanged,
        "before_after_test": comparison,
        "test_used_for_selection": False,
        "conclusion": {
            "helios_point_PQ_improved": comparison[3]["point_PQ"] > comparison[2]["point_PQ"],
            "helios_crown_PQ_improved": comparison[3]["crown_PQ"] > comparison[2]["crown_PQ"],
            "real_point_PQ_preserved": comparison[1]["point_PQ"] >= comparison[0]["point_PQ"],
            "real_crown_PQ_preserved": comparison[1]["crown_PQ"] >= comparison[0]["crown_PQ"],
        },
    }
    (output / "final_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not legacy_unchanged:
        raise AssertionError("Frozen legacy branch unexpectedly changed")
    return report


def evaluate(model, manifest, folder, config, split):
    """Evaluate the complete two-head production path with frozen postprocessing."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows, results, inference_seconds = read_manifest(Path(manifest), split), [], 0.0
    model.eval(); model.backbone.shuffle_orders = False
    for number, row in enumerate(rows, 1):
        arrays = load_npz(row["output"])
        raw = predict(model, arrays, max_points=40000, owner_only=config.get("owner_only", True))
        labels, _, instances = merge_masks(arrays, raw, config)
        inference_seconds += float(raw["seconds"])
        gt = gpd.read_file(row["gt_vector"])
        metadata = {key: row[key] for key in
                    ("dataset_id", "source_dataset", "collection", "annotation_method")}
        results.append({
            **metadata,
            **plot_metrics(row, gt.geometry, [item["geometry"] for item in instances]),
            "point_metrics": point_instance_metrics(arrays["tree_id"], labels),
        })
        print(f"complete-consensus {split} {number}/{len(rows)}: {row['dataset_id']}", flush=True)
    point_rows = [{**{key: row[key] for key in
                      ("dataset_id", "source_dataset", "collection", "annotation_method")},
                   **row["point_metrics"]} for row in results]
    point = aggregate(point_rows)
    for key in ("mucov", "mwcov"):
        point[key] = float(np.mean([row["point_metrics"][key] for row in results]))
    output = {
        "config": config, "metrics": aggregate(results), "point_metrics": point,
        "per_plot": results, "split": split, "inference_seconds": inference_seconds,
        "protocol": "complete DualCrown3D dual-head consensus; point/crown IoU >= 0.5",
    }
    (folder / f"{split}_metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


def main():
    args = arguments()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    output = args.output_dir.resolve()
    weights = output / "weights"
    if args.finalize_only:
        if not (output / "before_after_test_metrics.csv").exists():
            raise FileNotFoundError("Completed before/after metrics are required for --finalize-only")
        comparison = read_csv(output / "before_after_test_metrics.csv")
        for row in comparison:
            for key, value in list(row.items()):
                if key not in ("stage", "domain") and value != "":
                    row[key] = float(value)
        print(json.dumps(finalize_report(output, args.checkpoint, comparison), indent=2), flush=True)
        return
    output.mkdir(parents=True, exist_ok=False)
    weights.mkdir()
    if args.manifest is None:
        args.manifest = output / "training_manifest.csv"
        # Keep the original split labels. Only train rows are sampled during
        # optimisation, while domain-specific val/test manifests remain fixed.
        write_csv(args.manifest, read_csv(args.real_manifest) + read_csv(args.helios_manifest))
    device = torch.device(args.device)
    selection = json.loads(args.selection.read_text())
    merge_config = selection["config"]
    initial_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DualHeadLitePT().to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.backbone.shuffle_orders = False
    initial_legacy = {key: value.detach().cpu().clone() for key, value in model.legacy.state_dict().items()}

    train = PointCloudCropDataset(args.manifest, "train", max_points=args.max_points, repeats=1,
                                  augment=True, seed=args.seed,
                                  density_keep_fractions=(0.35, 0.5, 0.75, 1.0))
    real_indices = [i for i, row in enumerate(train.rows) if row["source_dataset"] != "TreeScanPL10k_HELIOS"]
    synthetic_indices = [i for i, row in enumerate(train.rows) if row["source_dataset"] == "TreeScanPL10k_HELIOS"]
    if not real_indices or not synthetic_indices:
        raise ValueError("Training manifest must contain both real and TreeScanPL10k_HELIOS rows")
    weights_per_row = [0.5 / len(synthetic_indices) if i in synthetic_indices else 0.5 / len(real_indices)
                       for i in range(len(train.rows))]
    sampler = WeightedRandomSampler(weights_per_row, args.samples_per_epoch, replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(train, batch_size=None, sampler=sampler, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.point_decoder.parameters(), lr=args.lr, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    configuration = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "initial_checkpoint_sha256": initial_sha,
        "trainable_scope": "TreeMaskDecoder only; LitePT-S and legacy centre-vote branch frozen",
        "domain_sampling": "50% HELIOS / 50% real replay",
        "selection": "harmonic mean of real and HELIOS validation sqrt(point_SB_PQ*crown_SB_PQ)",
        "postprocessing_frozen": merge_config,
        "test_used_for_selection": False,
    }
    (output / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")

    baseline_val_real = evaluate(model, args.real_manifest, output / "baseline/real_val", merge_config, "val")
    baseline_val_helios = evaluate(model, args.helios_manifest, output / "baseline/helios_val", merge_config, "val")
    baseline_test_real = evaluate(model, args.real_manifest, output / "baseline/real_test", merge_config, "test")
    baseline_test_helios = evaluate(model, args.helios_manifest, output / "baseline/helios_test", merge_config, "test")

    history, best, best_epoch = [], -math.inf, None
    for epoch in range(1, args.epochs + 1):
        train.set_epoch(epoch); model.point_decoder.epoch = max(4, epoch + int(checkpoint.get("epoch", 4)))
        train_result = pass_epoch(model, loader, device, optimizer)
        scheduler.step()
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train_result.items()},
               "lr": optimizer.param_groups[0]["lr"]}
        save_checkpoint(weights / "last.pt", model, epoch, args, optimizer, scheduler, float("nan"))
        if epoch % args.val_every == 0 or epoch == args.epochs:
            real = evaluate(model, args.real_manifest, output / f"validation/epoch_{epoch:03d}/real", merge_config, "val")
            helios = evaluate(model, args.helios_manifest, output / f"validation/epoch_{epoch:03d}/helios", merge_config, "val")
            score = guarded_score(real, helios)
            row.update(
                selection_score=score,
                real_val_point_SB_PQ=real["point_metrics"]["source_balanced_pq"],
                real_val_crown_SB_PQ=real["metrics"]["source_balanced_pq"],
                helios_val_point_PQ=helios["point_metrics"]["source_balanced_pq"],
                helios_val_crown_PQ=helios["metrics"]["source_balanced_pq"],
            )
            if score > best:
                best, best_epoch = score, epoch
                save_checkpoint(weights / "best.pt", model, epoch, args, optimizer, scheduler, score)
                selected = {"epoch": epoch, "selection_score": score,
                            "real_validation": real, "helios_validation": helios,
                            "checkpoint_sha256": hashlib.sha256((weights / "best.pt").read_bytes()).hexdigest(),
                            "test_used_for_selection": False}
                (output / "selected.json").write_text(json.dumps(selected, indent=2) + "\n")
        history.append(row)
        write_csv(output / "training_log.csv", history)
        print(json.dumps(row), flush=True)

    tuned = torch.load(weights / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(tuned["model"], strict=True)
    tuned_test_real = evaluate(model, args.real_manifest, output / "finetuned/real_test", merge_config, "test")
    tuned_test_helios = evaluate(model, args.helios_manifest, output / "finetuned/helios_test", merge_config, "test")
    comparison = [
        summary_row("before", "real_test", baseline_test_real),
        summary_row("after", "real_test", tuned_test_real),
        summary_row("before", "helios_test", baseline_test_helios),
        summary_row("after", "helios_test", tuned_test_helios),
    ]
    before = {(row["domain"]): row for row in comparison if row["stage"] == "before"}
    for row in comparison:
        if row["stage"] == "after":
            reference = before[row["domain"]]
            for key in list(row):
                if key not in ("stage", "domain") and isinstance(row[key], (int, float)):
                    row[f"delta_{key}"] = row[key] - reference[key]
    write_csv(output / "before_after_test_metrics.csv", comparison)
    write_workbook(output / "finetuning_results.xlsx", history, comparison, configuration)
    report = finalize_report(output, args.checkpoint, comparison)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
