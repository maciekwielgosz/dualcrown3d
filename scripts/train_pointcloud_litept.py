#!/usr/bin/env python3
"""Train LitePT-S directly on tree-labelled point clouds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import hashlib
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pointcloud.data import PointCloudCropDataset, move_to_device  # noqa: E402
from pointcloud.model import LitePTTreeInstance, instance_losses  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_DIR / "manifests" / "pointcloud_dataset_0p25m.csv",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=PROJECT_DIR
        / "models"
        / "litept_nuscenes"
        / "nuscenes-semseg-litept-small-v1m1"
        / "model"
        / "model_best.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "training" / "litept_tree_instance",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--crop-size", type=float, default=20.0)
    parser.add_argument("--max-points", type=int, default=30_000)
    parser.add_argument("--train-repeats", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--density-fractions", default="0.5,0.75,1.0")
    parser.add_argument("--full-val-every", type=int, default=0)
    parser.add_argument("--experiment-id")
    parser.add_argument("--offset-xy-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initial-weights",
        type=Path,
        help="Load model weights only and start a fresh optimizer/schedule.",
    )
    parser.add_argument(
        "--source-dataset",
        choices=("FOR-instance", "ideas_als"),
        help="Optionally restrict both train and validation rows by source dataset.",
    )
    return parser.parse_args()


def epoch_pass(
    model: LitePTTreeInstance,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    class_weights: torch.Tensor,
    amp: bool = True,
    xy_only: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("loss", "semantic_loss", "offset_l1_loss", "offset_cosine_loss")}
    correct = points = 0
    started = time.monotonic()
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            batch = move_to_device(batch, device)
            model_input = {key: batch[key] for key in ("coord", "grid_coord", "feat", "offset")}
            if training:
                optimizer.zero_grad(set_to_none=True)
            # spconv-cu126 has no valid FP16 inference kernel for some sparse
            # shapes on the Blackwell GPU, while FP16 training and FP32
            # inference are both supported.
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda" and training and amp
            ):
                prediction = model(model_input)
            with torch.autocast(device_type=device.type, enabled=False):
                losses = instance_losses(
                    {k: v.float() for k, v in prediction.items()},
                    batch["semantic"],
                    batch["instance_offset"],
                    class_weights,
                    xy_only=xy_only,
                )
            if training:
                assert scaler is not None
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            for key in totals:
                totals[key] += float(losses[key].detach())
            labels = prediction["semantic_logits"].argmax(dim=1)
            correct += int((labels == batch["semantic"]).sum())
            points += len(labels)
    count = max(len(loader), 1)
    return {
        **{key: value / count for key, value in totals.items()},
        "semantic_accuracy": correct / max(points, 1),
        "steps": len(loader),
        "points": points,
        "seconds": time.monotonic() - started,
    }


def write_log(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("GPU requested but unavailable")
    output_dir = args.output_dir.resolve()
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    if (weights_dir / "last.pt").exists() and args.resume is None:
        raise FileExistsError("Use a fresh experiment directory or --resume")

    train_dataset = PointCloudCropDataset(
        args.manifest.resolve(),
        "train",
        crop_size_m=args.crop_size,
        max_points=args.max_points,
        repeats=args.train_repeats,
        augment=True,
        seed=args.seed,
        source_dataset=args.source_dataset,
        density_keep_fractions=tuple(float(v) for v in args.density_fractions.split(",")),
    )
    val_dataset = PointCloudCropDataset(
        args.manifest.resolve(),
        "val",
        crop_size_m=args.crop_size,
        max_points=args.max_points,
        repeats=2,
        augment=False,
        seed=args.seed + 1,
        source_dataset=args.source_dataset,
    )
    loader_kwargs = dict(
        batch_size=None,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        # Workers are recreated each epoch so that set_epoch() changes the
        # deterministic crop and augmentation stream seen by worker copies.
        persistent_workers=False,
    )
    sampler = None
    if args.balanced_sampling:
        collections = Counter((r["source_dataset"], r["collection"]) for r in train_dataset.rows)
        sources = Counter(s for s, c in collections)
        weights = [1./(sources[r["source_dataset"]]*collections[(r["source_dataset"],r["collection"])]) for r in train_dataset.rows]*args.train_repeats
        sampler = WeightedRandomSampler(weights, args.samples_per_epoch or len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, shuffle=sampler is None, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = LitePTTreeInstance(patch_size=args.patch_size).to(device)
    pretrained_status = None
    if args.pretrained.is_file() and args.resume is None and args.initial_weights is None:
        pretrained_status = model.load_pretrained_backbone(args.pretrained.resolve())
        print(f"Loaded pretrained backbone: {pretrained_status}")
    if args.initial_weights is not None:
        initial = torch.load(args.initial_weights.resolve(), map_location=device, weights_only=False)
        model.load_state_dict(initial["model"])
        print(f"Loaded initial model weights: {args.initial_weights.resolve()}")
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {
                "params": list(model.semantic_head.parameters()) + list(model.offset_head.parameters()),
                "lr": args.head_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    best_loss = math.inf
    best_full_pq = -1.
    log_rows: list[dict] = []
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_val_loss"])
        best_full_pq = float(checkpoint.get("best_full_pq", -1.))
        if (output_dir / "training_log.csv").exists():
            with (output_dir / "training_log.csv").open() as stream:
                log_rows = list(csv.DictReader(stream))

    class_weights = torch.tensor([1.0, 1.2], device=device)
    no_improvement = 0
    if args.experiment_id:
        from pointcloud.experiment_log import record
        record(args.experiment_id, status="training", architecture=f"LitePT-S + semantic/{'XY' if args.offset_xy_only else '3D'} center-offset heads + XY vote clustering + filled crown hull", parameters=vars(args), training_dir=str(output_dir), checkpoint=str(weights_dir / "best.pt"), parameter_count=sum(p.numel() for p in model.parameters()), dataset_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    for epoch in range(start_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        train_metrics = epoch_pass(
            model, train_loader, device, optimizer, scaler, class_weights, not args.no_amp, args.offset_xy_only
        )
        val_metrics = epoch_pass(
            model, val_loader, device, None, None, class_weights, False, args.offset_xy_only
        )
        if not math.isfinite(train_metrics["loss"]) or not math.isfinite(val_metrics["loss"]):
            print(
                "Stopping before checkpoint save because a non-finite loss was detected; "
                "the previous best checkpoint remains intact.",
                flush=True,
            )
            break
        scheduler.step()
        row = {"epoch": epoch + 1}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row["backbone_lr"] = optimizer.param_groups[0]["lr"]
        row["head_lr"] = optimizer.param_groups[1]["lr"]
        row["gpu_max_memory_gb"] = (
            torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
        )
        full_result, improved_full = None, False
        if args.full_val_every and ((epoch+1) % args.full_val_every == 0 or epoch+1 == args.epochs):
            from scripts.evaluate_combined_full_crowns import evaluate_offset_model
            full_result = evaluate_offset_model(model, args.manifest, output_dir / f"validation/epoch_{epoch+1:03d}", max_points=args.max_points, tile_size=args.crop_size)
            pq = full_result["metrics"]["source_balanced_pq"]
            row.update(val_full_pq=pq, val_full_f1=full_result["metrics"]["f1"])
            improved_full = pq > best_full_pq+1e-5
            if improved_full:
                best_full_pq = pq
        log_rows.append(row)
        # Full metrics are present only at evaluation epochs.
        fields = list(dict.fromkeys(k for r in log_rows for k in r))
        with (output_dir / "training_log.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(log_rows)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_loss": min(best_loss, val_metrics["loss"]),
            "best_full_pq": best_full_pq,
            "args": vars(args),
            "pretrained_status": pretrained_status,
        }
        torch.save(state, weights_dir / "last.pt")
        improved = val_metrics["loss"] < best_loss - 1e-4
        if improved:
            best_loss = val_metrics["loss"]
            state["best_val_loss"] = best_loss
            torch.save(state, weights_dir / "best_loss.pt")
        if improved_full if args.full_val_every else improved:
            torch.save(state, weights_dir / "best.pt")
            if full_result:
                (output_dir / "selected_validation.json").write_text(json.dumps(full_result, indent=2)+"\n")
            no_improvement = 0
        else:
            no_improvement += 1
        if args.experiment_id:
            values = dict(epoch=epoch+1, best_full_pq=best_full_pq, gpu_max_memory_gb=row["gpu_max_memory_gb"])
            if full_result:
                values["latest_val_metrics"] = full_result["metrics"]
                if improved_full:
                    values.update(selected_val_metrics=full_result["metrics"], selected_epoch=epoch+1, checkpoint_sha256=hashlib.sha256((weights_dir / "best.pt").read_bytes()).hexdigest())
            record(args.experiment_id, **values)
        print(
            f"epoch {epoch + 1:03d}: train={train_metrics['loss']:.4f} "
            f"val={val_metrics['loss']:.4f} acc={val_metrics['semantic_accuracy']:.3f} "
            f"time={train_metrics['seconds'] + val_metrics['seconds']:.1f}s "
            f"memory={row['gpu_max_memory_gb']:.2f}GB{' *' if improved else ''}",
            flush=True,
        )
        if no_improvement >= args.patience:
            print(f"Early stopping after {no_improvement} epochs without improvement")
            break

    summary = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "epochs_completed": len(log_rows),
        "best_val_loss": best_loss,
        "train_plots": len(train_dataset.rows),
        "val_plots": len(val_dataset.rows),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "pretrained": str(args.pretrained.resolve()) if args.pretrained.is_file() else None,
        "pretrained_status": pretrained_status,
        "best_full_pq": best_full_pq,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, default=str, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.experiment_id:
        record(args.experiment_id, status="completed_training", summary=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
