#!/usr/bin/env python3
"""Stage-2 real-ALS affinity pilot with frozen retained LitePT features.

This is an engineering pilot, not an ALS-only initialization: the retained
encoder was previously exposed to HELIOS. No test data is read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz, prepare_crop, read_manifest
from pointcloud.superpoints.affinity import EdgeAffinityHead, bounded_partition
from pointcloud.superpoints.partition import (
    geometric_partition, neighbor_pairs, partition_diagnostics,
)
from scripts.train_supervision_v4 import INITIAL, REAL, build

OUTPUT = PROJECT / "outputs/dualcrown3d_superpoint_v1/stage2_affinity_pilot"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_git() -> dict:
    def command(*args):
        return subprocess.run(["git", *args], cwd=PROJECT, check=True,
                              capture_output=True).stdout
    return dict(commit=command("rev-parse", "HEAD").decode().strip(),
                tracked_diff_sha256=hashlib.sha256(command("diff", "--binary")).hexdigest(),
                code_sha256={name: sha256(PROJECT / name) for name in (
                    "pointcloud/superpoints/affinity.py", "pointcloud/superpoints/partition.py",
                    "scripts/train_superpoint_affinity.py")})


def write_json(path: Path, content: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def eligibility() -> tuple[list[dict], list[dict]]:
    train = [row for row in read_manifest(REAL, "train")
             if row["train_eligible"] == "true"]
    val = [row for row in read_manifest(REAL, "val")
           if row["point_eval_eligible"] == "true"]
    train_groups = {row["group_id"] for row in train}
    if train_groups & {row["group_id"] for row in val}:
        raise ValueError("Train/validation parent-group leakage")
    return sorted(train, key=lambda r: r["dataset_id"]), sorted(val, key=lambda r: r["dataset_id"])


def cache_features(model, rows: list[dict], split: str, output: Path,
                   seed: int, max_points: int) -> list[dict]:
    entries = []
    for index, row in enumerate(rows):
        path = output / "cache" / split / f"{row['dataset_id']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Feature cache protected: {path}")
        arrays = load_npz(row["output"])
        rng = np.random.default_rng(seed + index + (0 if split == "train" else 100000))
        # A label-independent anchor makes diagnostics reproducible without GT
        # influencing which geometry the frozen encoder sees.
        anchor = int(rng.integers(len(arrays["coord"])))
        crop = prepare_crop(arrays, rng, 20., max_points, augment=False,
                            preserve_height=bool(row.get("height_normalization")),
                            anchor_index=anchor)
        inputs = {key: crop[key].to("cuda:0") for key in
                  ("coord", "grid_coord", "feat", "offset")}
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.no_grad():
            feature = model.backbone(inputs).feat.detach().cpu().numpy().astype(np.float16)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        coord = crop["coord"].numpy().astype(np.float32)
        edges = neighbor_pairs(coord, radius_m=.75, k=12)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, coord=coord, feature=feature,
                            tree_id=crop["tree_id"].numpy().astype(np.int32), edge=edges)
        os.replace(temporary, path)
        item = dict(dataset_id=row["dataset_id"], source_dataset=row["source_dataset"],
                    collection=row["collection"], group_id=row["group_id"],
                    split=split, cache=str(path), points=len(coord), edges=len(edges),
                    backbone_seconds=elapsed, cache_sha256=sha256(path))
        entries.append(item)
        print(f"cache {split} {index + 1}/{len(rows)} {row['dataset_id']}: "
              f"{len(coord)} points, {len(edges)} edges, {elapsed:.2f}s", flush=True)
    return entries


def load_cache(entry: dict) -> dict:
    with np.load(entry["cache"]) as archive:
        return {name: archive[name] for name in archive.files}


def sample_balanced_edges(ids: np.ndarray, edges: np.ndarray,
                          rng: np.random.Generator, count: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    left, right = ids[edges[:, 0]], ids[edges[:, 1]]
    positive = np.flatnonzero((left == right) & (left > 0))
    cross_tree = np.flatnonzero((left > 0) & (right > 0) & (left != right))
    tree_background = np.flatnonzero(((left > 0) & (right == 0)) |
                                     ((right > 0) & (left == 0)))
    if not len(positive) or not len(cross_tree) + len(tree_background):
        return np.empty((0, 2), np.int32), np.empty(0, np.float32)
    n_positive = count // 2
    # Instance-balanced positives stop the largest crowns monopolising updates.
    frequency = Counter(left[positive].tolist())
    weight = np.asarray([frequency[int(value)] ** -.5 for value in left[positive]])
    chosen_positive = rng.choice(positive, n_positive, replace=True,
                                 p=weight / weight.sum())
    hard = min(count // 4, count - n_positive) if len(cross_tree) else 0
    chosen_hard = rng.choice(cross_tree, hard, replace=True) if hard else np.empty(0, int)
    remainder = count - n_positive - hard
    negative_pool = tree_background if len(tree_background) else cross_tree
    chosen_other = rng.choice(negative_pool, remainder, replace=True)
    indices = np.concatenate((chosen_positive, chosen_hard, chosen_other))
    label = np.concatenate((np.ones(n_positive), np.zeros(count - n_positive))).astype(np.float32)
    permutation = rng.permutation(len(indices))
    return edges[indices[permutation]], label[permutation]


@torch.no_grad()
def edge_probabilities(head, cache: dict, chunk: int = 65536) -> np.ndarray:
    feature = torch.from_numpy(cache["feature"].astype(np.float32)).cuda()
    coord = torch.from_numpy(cache["coord"]).cuda()
    edges = cache["edge"]
    values = []
    for start in range(0, len(edges), chunk):
        edge = torch.from_numpy(edges[start:start + chunk].astype(np.int64)).cuda()
        values.append(head(feature, coord, edge).sigmoid().cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, np.float32)


def threshold_from_train(head, entries: list[dict]) -> tuple[float, list[dict]]:
    # Compression is selected without using train or validation reference IDs.
    candidates = tuple(float(value) for value in np.arange(.35, .901, .05))
    records = []
    for entry in entries[:min(12, len(entries))]:
        cache = load_cache(entry)
        probability = edge_probabilities(head, cache)
        baseline = geometric_partition(cache["coord"], .5)
        target = len(cache["coord"]) / len(np.unique(baseline))
        for threshold in candidates:
            group = bounded_partition(cache["coord"], cache["edge"], probability,
                                      threshold=threshold, max_extent_m=1.)
            compression = len(group) / len(np.unique(group))
            records.append(dict(dataset_id=entry["dataset_id"], threshold=threshold,
                                learned_compression=compression, fixed_compression=target))
    objective = {value: np.mean([abs(np.log(row["learned_compression"] /
                                         row["fixed_compression"]))
                                 for row in records if row["threshold"] == value])
                 for value in candidates}
    return min(objective, key=objective.get), records


def validation(head, entries: list[dict], threshold: float) -> list[dict]:
    results = []
    for item in entries:
        cache = load_cache(item)
        coord, ids, edges = cache["coord"], cache["tree_id"], cache["edge"]
        probability = edge_probabilities(head, cache)
        fixed = geometric_partition(coord, .5)
        learned = bounded_partition(coord, edges, probability,
                                    threshold=threshold, max_extent_m=1.)
        for name, group in (("fixed", fixed), ("learned", learned)):
            diag = partition_diagnostics(coord, ids, group, edges)
            results.append(dict(dataset_id=item["dataset_id"],
                source_dataset=item["source_dataset"], collection=item["collection"],
                method=name, threshold=threshold if name == "learned" else None,
                **{key: value for key, value in diag.items() if key != "oracle"},
                **{f"oracle_{key}": value for key, value in diag["oracle"].items()}))
        print(f"validation {item['dataset_id']}: "
              f"fixed PQ={results[-2]['oracle_pq']:.3f}, "
              f"learned PQ={results[-1]['oracle_pq']:.3f}", flush=True)
    return results


def summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    return [dict(method=name, plots=len(items),
            mean_compression=float(np.mean([item["compression"] for item in items])),
            mean_oracle_pq=float(np.mean([item["oracle_pq"] for item in items])),
            mean_boundary_recall=float(np.mean([item["boundary_recall"]
                for item in items if item["boundary_recall"] is not None])),
            mean_mixed_fraction=float(np.mean([item["mixed_group_fraction"] for item in items])))
            for name, items in grouped.items()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-plots", type=int, default=0,
                        help="Smoke-test cap per split; zero means all eligible plots")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen-feature pilot")
    if args.max_points < 256 or args.epochs < 1:
        raise ValueError("Invalid pilot budget")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Stage-2 output protected: {output}")
    train, val = eligibility()
    if args.max_plots:
        train, val = train[:args.max_plots], val[:args.max_plots]
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = dict(seed=args.seed, max_points=args.max_points, epochs=args.epochs,
                  train_plots=len(train), val_plots=len(val),
                  initialization="retained_HELIOS_exposed_engineering_pilot",
                  checkpoint=str(INITIAL), checkpoint_sha256=sha256(INITIAL),
                  real_manifest_sha256=sha256(REAL), git=current_git(),
                  training_edges_per_plot=1024, crop_m=20.,
                  edge_radius_m=.75, edge_k=12, partition_max_extent_m=1.)
    write_json(output / "config.json", config)
    model, _ = build(INITIAL)
    model.eval()
    model.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    train_entries = cache_features(model, train, "train", output, args.seed, args.max_points)
    val_entries = cache_features(model, val, "val", output, args.seed, args.max_points)
    config["backbone_peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    del model
    torch.cuda.empty_cache()
    head = EdgeAffinityHead().cuda().train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=.01)
    rng = np.random.default_rng(args.seed)
    logs = []
    for epoch in range(1, args.epochs + 1):
        losses = []
        start = time.monotonic()
        for item in rng.permutation(train_entries):
            cache = load_cache(item)
            edge, label = sample_balanced_edges(cache["tree_id"], cache["edge"], rng)
            if not len(edge):
                continue
            feature = torch.from_numpy(cache["feature"].astype(np.float32)).cuda()
            coord = torch.from_numpy(cache["coord"]).cuda()
            edge = torch.from_numpy(edge.astype(np.int64)).cuda()
            label = torch.from_numpy(label).cuda()
            optimizer.zero_grad(set_to_none=True)
            logit = head(feature, coord, edge)
            loss = F.binary_cross_entropy_with_logits(logit, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 2.)
            optimizer.step()
            losses.append(float(loss.detach()))
        record = dict(epoch=epoch, mean_loss=float(np.mean(losses)),
                      updates=len(losses), seconds=time.monotonic() - start)
        logs.append(record)
        print(json.dumps(record), flush=True)
    head.eval()
    weight = output / "affinity_head.pt"
    torch.save(dict(model=head.state_dict(), config=config), weight)
    threshold, compression_trials = threshold_from_train(head, train_entries)
    findings = validation(head, val_entries, threshold)
    comparison = summary(findings)
    config.update(selected_threshold=threshold, checkpoint=str(weight),
                  checkpoint_sha256=sha256(weight),
                  head_peak_vram_mb=torch.cuda.max_memory_allocated() / 1024**2)
    write_json(output / "config.json", config)
    write_json(output / "result.json", dict(comparison=comparison,
        threshold_trials=compression_trials, validation=findings,
        train_history=logs, test_used=False,
        gate="learned boundary/oracle improves fixed at comparable compression"))
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
