#!/usr/bin/env python3
"""Stage-2 ablation: reject affinity bridges with inconsistent frozen center votes."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.superpoints.affinity import bounded_partition
from scripts.sweep_superpoint_partition import diagnostics, prepared
from scripts.train_superpoint_affinity import eligibility, summary, write_json
from scripts.train_supervision_v4 import INITIAL, build

OUTPUT = PROJECT / "outputs/dualcrown3d_superpoint_v1/stage2_vote_coherence"
EXTENTS = (.6, .8)
SIGMAS = (1.5, 3.)
THRESHOLDS = tuple(float(x) for x in np.arange(.3, .801, .05))


@torch.no_grad()
def append_votes(items: list[dict], model) -> None:
    head = model.legacy.offset_head.eval()
    scale = model.legacy.offset_scale_m
    for item in items:
        data = item["data"]
        feature = torch.from_numpy(data["feature"].astype(np.float32)).cuda()
        offset = head(feature).cpu().numpy()[:, :2] * scale
        center = data["coord"][:, :2] + offset
        edges = data["edge"]
        item["center_distance"] = np.linalg.norm(center[edges[:, 0]] -
                                                 center[edges[:, 1]], axis=1)


def adjusted(item: dict, sigma: float) -> np.ndarray:
    distance = item["center_distance"]
    return item["probability"] * np.exp(-.5 * (distance / sigma) ** 2)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Vote ablation protected: {OUTPUT}")
    train, val = eligibility()
    group = defaultdict(list)
    for row in train:
        group[(row["source_dataset"], row["collection"])].append(row)
    chosen = [sorted(rows, key=lambda r: r["dataset_id"])[len(rows) // 2]
              for _, rows in sorted(group.items())]
    from pointcloud.superpoints.affinity import EdgeAffinityHead
    from scripts.train_superpoint_affinity import OUTPUT as TRAINING
    payload = torch.load(TRAINING / "affinity_head.pt", map_location="cpu", weights_only=False)
    affinity = EdgeAffinityHead().cuda().eval()
    affinity.load_state_dict(payload["model"], strict=True)
    model, _ = build(INITIAL)
    model.eval().requires_grad_(False)
    training = prepared(affinity, chosen, "train")
    append_votes(training, model)
    fixed = summary([diagnostics(item, item["fixed"], "fixed", .5, 0.)
                     for item in training])[0]
    candidates = []
    for extent in EXTENTS:
        for sigma in SIGMAS:
            probabilities = [adjusted(item, sigma) for item in training]
            trials = []
            target = float(np.mean([item["target_compression"] for item in training]))
            for threshold in THRESHOLDS:
                compression = []
                for item, probability in zip(training, probabilities):
                    data = item["data"]
                    partition = bounded_partition(data["coord"], data["edge"], probability,
                        threshold=threshold, max_extent_m=extent)
                    compression.append(len(partition) / len(np.unique(partition)))
                trials.append(dict(threshold=threshold,
                    mean_compression=float(np.mean(compression)),
                    relative_error=abs(float(np.mean(compression)) / target - 1.)))
            closest = min(trials, key=lambda row: row["relative_error"])
            records = []
            for item, probability in zip(training, probabilities):
                data = item["data"]
                partition = bounded_partition(data["coord"], data["edge"], probability,
                    threshold=closest["threshold"], max_extent_m=extent)
                records.append(diagnostics(item, partition, "vote_coherent",
                                           extent, closest["threshold"]))
            result = summary(records)[0]
            pass_gate = (result["mean_oracle_pq"] > fixed["mean_oracle_pq"]
                and result["mean_boundary_recall"] > fixed["mean_boundary_recall"]
                and abs(result["mean_compression"] / fixed["mean_compression"] - 1.) <= .05)
            candidate = dict(extent_m=extent, vote_sigma_m=sigma,
                threshold=closest["threshold"], train_fixed=fixed,
                train_candidate=result, training_gate=pass_gate, trials=trials)
            candidates.append(candidate)
            print(json.dumps({key: candidate[key] for key in
                ("extent_m", "vote_sigma_m", "threshold", "train_candidate",
                 "training_gate")}), flush=True)
    passing = [item for item in candidates if item["training_gate"]]
    selected = (max(passing, key=lambda item: item["train_candidate"]["mean_oracle_pq"] +
                    item["train_candidate"]["mean_boundary_recall"]) if passing else None)
    OUTPUT.mkdir(parents=True)
    if selected is None:
        write_json(OUTPUT / "result.json", dict(candidates=candidates, selected=None,
            reason="No matched-compression training improvement in both diagnostics",
            validation_not_reused=True, test_used=False))
        print("No training-gate candidate; validation not reused", flush=True)
        return
    evaluated = prepared(affinity, val, "val")
    append_votes(evaluated, model)
    rows = []
    for item in evaluated:
        data = item["data"]
        p = adjusted(item, selected["vote_sigma_m"])
        partition = bounded_partition(data["coord"], data["edge"], p,
            threshold=selected["threshold"], max_extent_m=selected["extent_m"])
        rows.extend((diagnostics(item, item["fixed"], "fixed", .5, 0.),
                     diagnostics(item, partition, "vote_coherent",
                                 selected["extent_m"], selected["threshold"])))
    comparison = summary(rows)
    gate = (comparison[1]["mean_oracle_pq"] > comparison[0]["mean_oracle_pq"]
        and comparison[1]["mean_boundary_recall"] > comparison[0]["mean_boundary_recall"]
        and abs(comparison[1]["mean_compression"] /
                comparison[0]["mean_compression"] - 1.) <= .05)
    write_json(OUTPUT / "result.json", dict(candidates=candidates, selected=selected,
        validation=rows, comparison=comparison, validation_gate_passed=gate,
        selection_used_only_training=True, test_used=False))
    print(json.dumps(dict(comparison=comparison, validation_gate_passed=gate),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
