#!/usr/bin/env python3
"""Training-only extent ablation for the learned superpoint boundary gate."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.superpoints.affinity import EdgeAffinityHead, bounded_partition
from pointcloud.superpoints.partition import geometric_partition, partition_diagnostics
from scripts.train_superpoint_affinity import (
    OUTPUT as TRAINING, edge_probabilities, eligibility, load_cache, summary, write_json,
)

OUTPUT = PROJECT / "outputs/dualcrown3d_superpoint_v1/stage2_extent_sweep_v2"
EXTENTS = (.6, .8, 1.)
THRESHOLDS = tuple(float(value) for value in np.arange(.35, .901, .05))


def entry(row: dict, split: str) -> dict:
    return dict(dataset_id=row["dataset_id"], source_dataset=row["source_dataset"],
                collection=row["collection"],
                cache=str(TRAINING / "cache" / split / f"{row['dataset_id']}.npz"))


def prepared(head, rows: list[dict], split: str) -> list[dict]:
    outputs = []
    for row in rows:
        item = entry(row, split)
        cache = load_cache(item)
        item["data"] = cache
        item["probability"] = edge_probabilities(head, cache)
        item["fixed"] = geometric_partition(cache["coord"], .5)
        item["target_compression"] = len(cache["coord"]) / len(np.unique(item["fixed"]))
        outputs.append(item)
    return outputs


def diagnostics(item: dict, group: np.ndarray, method: str, extent: float,
                threshold: float) -> dict:
    data = item["data"]
    result = partition_diagnostics(data["coord"], data["tree_id"], group, data["edge"])
    return dict(dataset_id=item["dataset_id"], source_dataset=item["source_dataset"],
                collection=item["collection"], method=method, extent_m=extent,
                threshold=threshold, **{key: value for key, value in result.items()
                                       if key != "oracle"},
                **{f"oracle_{key}": value for key, value in result["oracle"].items()})


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Extent ablation protected: {OUTPUT}")
    train, val = eligibility()
    groups = defaultdict(list)
    for row in train:
        groups[(row["source_dataset"], row["collection"])].append(row)
    choice = [sorted(rows, key=lambda r: r["dataset_id"])[len(rows) // 2]
              for _, rows in sorted(groups.items())]
    payload = torch.load(TRAINING / "affinity_head.pt", map_location="cpu", weights_only=False)
    head = EdgeAffinityHead().cuda().eval()
    head.load_state_dict(payload["model"], strict=True)
    training = prepared(head, choice, "train")
    fixed_train = [diagnostics(item, item["fixed"], "fixed", .5, 0.)
                   for item in training]
    fixed_summary = summary(fixed_train)[0]
    candidates = []
    for extent in EXTENTS:
        trials = []
        for threshold in THRESHOLDS:
            compressions = []
            for item in training:
                data = item["data"]
                group = bounded_partition(data["coord"], data["edge"],
                    item["probability"], threshold=threshold, max_extent_m=extent)
                compressions.append(len(group) / len(np.unique(group)))
            target_mean = float(np.mean([item["target_compression"] for item in training]))
            objective = abs(float(np.mean(compressions)) / target_mean - 1.)
            trials.append(dict(threshold=threshold, extent_m=extent,
                mean_compression=float(np.mean(compressions)),
                mean_fixed_compression=float(np.mean([item["target_compression"]
                                                     for item in training])),
                match_error=objective))
        best = min(trials, key=lambda row: row["match_error"])
        learned_train = []
        for item in training:
            data = item["data"]
            group = bounded_partition(data["coord"], data["edge"],
                item["probability"], threshold=best["threshold"], max_extent_m=extent)
            learned_train.append(diagnostics(item, group, "learned", extent,
                                             best["threshold"]))
        current = summary(learned_train)[0]
        candidate = dict(extent_m=extent, threshold=best["threshold"],
            match_error=best["match_error"], train_fixed=fixed_summary,
            train_learned=current, trials=trials,
            training_gate=(current["mean_oracle_pq"] > fixed_summary["mean_oracle_pq"]
                           and current["mean_boundary_recall"] >
                           fixed_summary["mean_boundary_recall"]
                           and abs(current["mean_compression"] /
                                   fixed_summary["mean_compression"] - 1.) <= .05))
        candidates.append(candidate)
        print(json.dumps({key: candidate[key] for key in
                          ("extent_m", "threshold", "match_error", "train_learned",
                           "training_gate")}), flush=True)
    passing = [row for row in candidates if row["training_gate"]]
    selection = (max(passing, key=lambda row: row["train_learned"]["mean_oracle_pq"] +
                     row["train_learned"]["mean_boundary_recall"]) if passing else None)
    OUTPUT.mkdir(parents=True)
    if selection is None:
        write_json(OUTPUT / "result.json", dict(candidates=candidates, selected=None,
            reason="No extent improved both instance oracle and boundary recall on training",
            validation_not_reused=True, test_used=False))
        print("No training-gate candidate; no new validation selection", flush=True)
        return
    validation_rows = []
    for item in prepared(head, val, "val"):
        data = item["data"]
        learned = bounded_partition(data["coord"], data["edge"], item["probability"],
            threshold=selection["threshold"], max_extent_m=selection["extent_m"])
        validation_rows.append(diagnostics(item, item["fixed"], "fixed", .5, 0.))
        validation_rows.append(diagnostics(item, learned, "learned",
                                           selection["extent_m"], selection["threshold"]))
    comparison = summary(validation_rows)
    write_json(OUTPUT / "result.json", dict(candidates=candidates, selected=selection,
        validation=validation_rows, comparison=comparison,
        selection_used_only_training=True, test_used=False))
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
