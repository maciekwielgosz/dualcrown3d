#!/usr/bin/env python3
"""Train the LitePT query-mask decoder with point-density augmentation."""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pointcloud.data import PointCloudCropDataset, move_to_device  # noqa: E402
from pointcloud.mask_decoder import LitePTMaskDecoder, mask_decoder_losses  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_DIR / "manifests" / "pointcloud_dataset_0p25m.csv",
    )
    parser.add_argument(
        "--backbone-weights",
        type=Path,
        default=PROJECT_DIR
        / "outputs"
        / "training"
        / "litept_tree_instance"
        / "weights"
        / "best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "training" / "litept_mask_decoder",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--crop-size", type=float, default=20.0)
    parser.add_argument("--max-points", type=int, default=30_000)
    parser.add_argument("--train-repeats", type=int, default=2)
    parser.add_argument("--val-repeats", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--memory-tokens", type=int, default=2048)
    parser.add_argument("--spatial-prior", action="store_true")
    parser.add_argument("--masked-attention", action="store_true")
    parser.add_argument("--auxiliary-losses", action="store_true")
    parser.add_argument("--semantic-head", action="store_true")
    parser.add_argument("--backbone-type", choices=("litept", "point_mlp"), default="litept")
    parser.add_argument("--anchor-mode", choices=("fps", "height_peaks"), default="fps")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--unweighted-object-loss", action="store_true")
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--density-fractions", default="0.025,0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--full-val-every", type=int, default=0)
    parser.add_argument("--experiment-id")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument(
        "--source-dataset", choices=("FOR-instance", "ideas_als")
    )
    return parser.parse_args()


def set_backbone_trainable(model: LitePTMaskDecoder, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(trainable)


def epoch_pass(
    model: LitePTMaskDecoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp: bool = True,
    object_positive_weight: bool = True,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training and not any(p.requires_grad for p in model.backbone.parameters()):
        model.backbone.eval()
    totals = {
        key: 0.0
        for key in (
            "loss",
            "object_loss",
            "mask_bce_loss",
            "mask_dice_loss",
            "matched_mask_iou",
            "center_loss",
            "radius_loss",
            "target_instances",
        )
    }
    points = 0
    valid_steps = 0
    skipped_batches = 0
    started = time.monotonic()
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            batch = move_to_device(batch, device)
            model_input = {
                key: batch[key] for key in ("coord", "grid_coord", "feat", "offset")
            }
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda" and training and amp,
            ):
                prediction = model(model_input)
                required = ("object_logits", "mask_logits")
                if not all(
                    bool(torch.isfinite(prediction[key]).all()) for key in required
                ):
                    if not training:
                        raise FloatingPointError("Non-finite validation prediction")
                    skipped_batches += 1
                    continue
            # Explicitly disable autocast: .float() alone does not protect matmul.
            with torch.autocast(device_type=device.type, enabled=False):
                losses = mask_decoder_losses(prediction, batch["tree_id"], object_positive_weight)
            if not bool(torch.isfinite(losses["loss"])):
                if not training:
                    raise FloatingPointError("Non-finite validation loss")
                skipped_batches += 1
                continue
            if training:
                assert scaler is not None
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
            for key in totals:
                value = losses[key]
                totals[key] += float(value.detach()) if torch.is_tensor(value) else float(value)
            points += len(batch["tree_id"])
            valid_steps += 1
    steps = max(valid_steps, 1)
    if not valid_steps:
        raise FloatingPointError("Epoch had no valid batches")
    return {
        **{key: value / steps for key, value in totals.items()},
        "steps": valid_steps,
        "skipped_batches": skipped_batches,
        "points": points,
        "seconds": time.monotonic() - started,
    }


def write_log(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("GPU requested but unavailable; refusing silent CPU training")
    output_dir = args.output_dir.resolve()
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    if (weights_dir / "last.pt").exists() and args.resume is None:
        raise FileExistsError("Run already contains checkpoints; use --resume or a new output directory")

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
        repeats=args.val_repeats,
        augment=False,
        seed=args.seed + 1,
        source_dataset=args.source_dataset,
    )
    loader_kwargs = dict(
        batch_size=None,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    sampler = None
    if args.balanced_sampling:
        collections = Counter((r["source_dataset"], r["collection"]) for r in train_dataset.rows)
        sources = Counter(source for source, collection in collections)
        weights = [1. / (sources[r["source_dataset"]] * collections[(r["source_dataset"], r["collection"])]) for r in train_dataset.rows] * args.train_repeats
        sampler = WeightedRandomSampler(weights, args.samples_per_epoch or len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, shuffle=sampler is None, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = LitePTMaskDecoder(
        patch_size=args.patch_size,
        queries=args.queries,
        hidden_dim=args.hidden_dim,
        decoder_layers=args.decoder_layers,
        attention_heads=args.attention_heads,
        memory_tokens=args.memory_tokens,
        spatial_prior=args.spatial_prior,
        masked_attention=args.masked_attention,
        auxiliary_losses=args.auxiliary_losses,
        semantic_head=args.semantic_head,
        backbone_type=args.backbone_type,
        anchor_mode=args.anchor_mode,
    ).to(device)
    backbone_status = None
    if args.initial_weights is not None:
        initial = torch.load(args.initial_weights.resolve(), map_location=device, weights_only=False)
        model.load_state_dict(initial["model"])
    elif args.resume is None and not args.random_init and args.backbone_type == "litept":
        backbone_status = model.load_backbone_from_tree_checkpoint(
            args.backbone_weights.resolve()
        )
        print(f"Loaded LitePT backbone: {backbone_status}", flush=True)

    decoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": decoder_parameters, "lr": args.decoder_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    best_loss = math.inf
    best_iou = -math.inf
    no_improvement = 0
    best_full_pq = -1.
    log_rows: list[dict] = []
    if args.resume is not None:
        checkpoint = torch.load(args.resume.resolve(), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_val_loss"])
        best_iou = float(checkpoint.get("best_val_iou", -math.inf))
        best_full_pq = float(checkpoint.get("best_full_pq", -1.))
        no_improvement = int(checkpoint.get("no_improvement", 0))
        log_path = output_dir / "training_log.csv"
        if log_path.exists():
            with log_path.open("r", encoding="utf-8", newline="") as stream:
                log_rows = list(csv.DictReader(stream))

    if args.experiment_id:
        from pointcloud.experiment_log import record
        record(args.experiment_id, status="training", architecture=f"{args.backbone_type} + {'masked' if args.masked_attention else 'global'} query decoder", parameters=vars(args), training_dir=str(output_dir), checkpoint=str(weights_dir / "best.pt"), parameter_count=sum(p.numel() for p in model.parameters()), dataset_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    for epoch in range(start_epoch, args.epochs):
        frozen = epoch < args.freeze_backbone_epochs
        set_backbone_trainable(model, not frozen)
        train_dataset.set_epoch(epoch)
        train_metrics = epoch_pass(model, train_loader, device, optimizer, scaler, not args.no_amp, not args.unweighted_object_loss)
        val_metrics = epoch_pass(model, val_loader, device, None, None, False, not args.unweighted_object_loss)
        if not math.isfinite(train_metrics["loss"]) or not math.isfinite(val_metrics["loss"]):
            print("Stopping before checkpoint save after non-finite loss", flush=True)
            break
        scheduler.step()
        row = {"epoch": epoch + 1, "backbone_frozen": frozen}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row["backbone_lr"] = optimizer.param_groups[0]["lr"]
        row["decoder_lr"] = optimizer.param_groups[1]["lr"]
        row["gpu_max_memory_gb"] = (
            torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
        )
        full_result = None
        improved_full = False
        if args.full_val_every and ((epoch+1) % args.full_val_every == 0 or epoch+1 == args.epochs):
            from scripts.evaluate_combined_full_crowns import evaluate_model
            full_result = evaluate_model(model, args.manifest, output_dir / f"validation/epoch_{epoch+1:03d}", max_points=args.max_points, tile_size=args.crop_size)
            full_pq = full_result["metrics"]["source_balanced_pq"]
            row.update(val_full_pq=full_pq, val_full_f1=full_result["metrics"]["f1"])
            improved_full = full_pq > best_full_pq + 1e-5
            if improved_full:
                best_full_pq = full_pq
                no_improvement = 0
        log_rows.append(row)
        write_log(output_dir / "training_log.csv", log_rows)
        improved_loss = val_metrics["loss"] < best_loss - 1e-4
        improved_iou = val_metrics["matched_mask_iou"] > best_iou + 1e-4
        if improved_loss:
            best_loss = val_metrics["loss"]
        if improved_iou:
            best_iou = val_metrics["matched_mask_iou"]
        if improved_full if args.full_val_every else improved_iou:
            no_improvement = 0
        else:
            no_improvement += 1
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_loss,
            "best_val_iou": best_iou,
            "best_full_pq": best_full_pq,
            "no_improvement": no_improvement,
            "args": vars(args),
            "backbone_status": backbone_status,
        }
        torch.save(state, weights_dir / "last.pt")
        if improved_loss:
            torch.save(state, weights_dir / "best_loss.pt")
        if improved_iou:
            torch.save(state, weights_dir / "best_iou.pt")
        if improved_full if args.full_val_every else improved_iou:
            torch.save(state, weights_dir / "best.pt")
            if full_result:
                (output_dir / "selected_validation.json").write_text(json.dumps(full_result, indent=2)+"\n")
        if args.experiment_id:
            values = dict(epoch=epoch+1, best_crop_iou=best_iou, best_full_pq=best_full_pq, gpu_max_memory_gb=row["gpu_max_memory_gb"])
            if full_result:
                values["latest_val_metrics"] = full_result["metrics"]
                if improved_full:
                    values["selected_val_metrics"] = full_result["metrics"]
                    values["selected_epoch"] = epoch+1
                    values["checkpoint_sha256"] = hashlib.sha256((weights_dir / "best.pt").read_bytes()).hexdigest()
            record(args.experiment_id, **values)
        print(
            f"epoch {epoch + 1:03d}: train={train_metrics['loss']:.4f} "
            f"val={val_metrics['loss']:.4f} valIoU={val_metrics['matched_mask_iou']:.3f} "
            f"targets={train_metrics['target_instances']:.1f} "
            f"time={train_metrics['seconds'] + val_metrics['seconds']:.1f}s "
            f"memory={row['gpu_max_memory_gb']:.2f}GB"
            f"{' frozen' if frozen else ''}{' *' if improved_iou else ''}",
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
        "best_val_iou": best_iou,
        "train_plots": len(train_dataset.rows),
        "val_plots": len(val_dataset.rows),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "backbone_weights": str(args.backbone_weights.resolve()) if backbone_status is not None else None,
        "backbone_status": backbone_status,
        "density_keep_fractions": train_dataset.density_keep_fractions,
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
