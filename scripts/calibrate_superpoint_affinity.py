#!/usr/bin/env python3
"""Recalibrate learned superpoint compression on source-balanced *training* plots."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.superpoints.affinity import EdgeAffinityHead
from scripts.train_superpoint_affinity import (
    OUTPUT as TRAINING, eligibility, summary, threshold_from_train, validation, write_json,
)

OUTPUT = PROJECT / "outputs/dualcrown3d_superpoint_v1/stage2_balanced_calibration"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Calibration protected: {OUTPUT}")
    train_rows, val_rows = eligibility()
    grouped = defaultdict(list)
    for row in train_rows:
        grouped[(row["source_dataset"], row["collection"])].append(row)
    choices = [sorted(rows, key=lambda row: row["dataset_id"])[len(rows) // 2]
               for _, rows in sorted(grouped.items())]
    entries = [dict(dataset_id=row["dataset_id"],
                    cache=str(TRAINING / "cache/train" / f"{row['dataset_id']}.npz"))
               for row in choices]
    val_entries = [dict(dataset_id=row["dataset_id"],
                        source_dataset=row["source_dataset"], collection=row["collection"],
                        cache=str(TRAINING / "cache/val" / f"{row['dataset_id']}.npz"))
                   for row in val_rows]
    if not all(Path(item["cache"]).is_file() for item in entries + val_entries):
        raise FileNotFoundError("Incomplete frozen feature cache")
    payload = torch.load(TRAINING / "affinity_head.pt", map_location="cpu", weights_only=False)
    head = EdgeAffinityHead().cuda().eval()
    head.load_state_dict(payload["model"], strict=True)
    threshold, trials = threshold_from_train(head, entries)
    findings = validation(head, val_entries, threshold)
    comparison = summary(findings)
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "result.json", dict(selected_threshold=threshold,
        train_selection=[item["dataset_id"] for item in entries],
        source_balanced=True, trained_checkpoint=str(TRAINING / "affinity_head.pt"),
        compression_trials=trials, validation=findings, comparison=comparison,
        test_used=False))
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
