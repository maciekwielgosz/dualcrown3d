#!/usr/bin/env python3
"""Validation-only experiment: average best-PQ and best-loss model states."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.experiment_log import ROOT, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "litept_offset_v1")
    parser.add_argument("--experiment-id", default="litept_offset_average_v1")
    args = parser.parse_args()
    paths = [args.source / "weights/best.pt", args.source / "weights/best_loss.pt"]
    states = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    if states[0]["epoch"] == states[1]["epoch"]:
        print("Best PQ and best loss refer to the same epoch; no independent averaging trial")
        return
    averaged = {}
    for key, value in states[0]["model"].items():
        if value.is_floating_point():
            averaged[key] = torch.stack([s["model"][key].float() for s in states]).mean(0).to(value.dtype)
        else:
            averaged[key] = value.clone()
    destination = ROOT / args.experiment_id
    (destination / "weights").mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "weights/best.pt"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    payload = dict(model=averaged, args=states[0]["args"], source_epochs=[s["epoch"]+1 for s in states], inference_only=True)
    torch.save(payload, checkpoint)
    metadata = dict(checkpoints=[str(p) for p in paths], source_epochs=payload["source_epochs"], source_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths], rule="equal average of best validation full-PQ and best validation loss; no test input")
    (destination / "averaging.json").write_text(json.dumps(metadata, indent=2)+"\n")
    record(args.experiment_id, status="completed_training", architecture="LitePT-S semantic/offset, two-checkpoint weight average", parameters=metadata,
           training_dir=str(destination), checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
