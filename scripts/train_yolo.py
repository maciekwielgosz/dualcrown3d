#!/usr/bin/env python3
"""Train YOLO11s-seg with ALS-specific, non-photographic augmentation."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/default.json")
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "dataset/dataset.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="yolo11s_physical")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "Missing DL dependencies. Install requirements-dl.txt first. "
            f"Original error: {error}"
        ) from error

    model_name = args.model or training["model"]
    local_model = PROJECT_DIR / "models" / model_name
    if not Path(model_name).is_absolute() and local_model.is_file():
        model_name = str(local_model)
    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    run_dir = PROJECT_DIR / "outputs" / "training"
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (run_dir / f"{args.name}_environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    model = YOLO(model_name)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs or int(training["epochs"]),
        imgsz=args.imgsz or int(training["image_size"]),
        batch=args.batch or int(training["batch"]),
        patience=args.patience if args.patience is not None else int(training["patience"]),
        workers=args.workers if args.workers is not None else int(training["workers"]),
        device=device,
        project=str(run_dir),
        name=args.name,
        exist_ok=True,
        seed=int(config["seed"]),
        deterministic=True,
        optimizer="AdamW",
        lr0=args.lr0 if args.lr0 is not None else 0.001,
        cos_lr=True,
        close_mosaic=20,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=180.0,
        translate=0.10,
        scale=0.35,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        auto_augment=None,
        amp=torch.cuda.is_available(),
        plots=True,
        save=True,
        save_period=25,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
