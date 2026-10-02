#!/usr/bin/env python3
"""Test Cut Pursuit and EZ-SP kernels with an instance-trained embedding.

The embedding head is trained on real-ALS instance boundaries, not semantic
tree/background classes. This remains a partition-kernel experiment, not a
full SPT/EZ-SP network reproduction or a production model evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.superpoints.instance_embedding import InstanceBoundaryEmbedding, edge_affinity
from pointcloud.superpoints.partition import geometric_partition, partition_diagnostics
from scripts.benchmark_superpoint_algorithms import (
    CACHE, ROOT, PartitionContext, budget_match, load_methods, paired_bootstrap,
    save_json, sha256, summary,
)
from scripts.train_superpoint_affinity import eligibility

DEFAULT_CHECKPOINT = ROOT / "stage2_instance_embedding/embedding.pt"


@torch.no_grad()
def embed_and_weight(model, cache: dict, weight_mode: str):
    torch.cuda.synchronize()
    started = time.monotonic()
    feature = torch.from_numpy(cache["feature"].astype(np.float32)).cuda()
    edge = torch.from_numpy(cache["edge"].astype(np.int64)).cuda()
    embedding = model(feature)
    if weight_mode == "embed_affinity":
        weight = edge_affinity(embedding, edge).cpu().numpy().astype(np.float32)
    else:
        weight = np.ones(len(edge), dtype=np.float32)
    embedding = embedding.cpu().numpy().astype(np.float32)
    torch.cuda.synchronize()
    return embedding, weight, time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--weight-mode", choices=("unit", "embed_affinity"), default="unit")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-plots", type=int, default=0)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Experiment output protected: {output}")
    if args.steps < 2 or args.max_plots < 0 or not torch.cuda.is_available():
        raise ValueError("Need CUDA, positive max-plots and >=2 search steps")
    cp_fn, csr_fn, ez_fn, shim = load_methods()
    checkpoint = args.checkpoint.resolve()
    model = InstanceBoundaryEmbedding().cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    _, val = eligibility()
    if args.max_plots:
        val = val[:args.max_plots]
    output.mkdir(parents=True)
    config = dict(method="SPT Cut Pursuit and EZ-SP contour-prior kernels with instance-trained embedding",
                  exact_full_models=False, checkpoint=str(checkpoint),
                  checkpoint_sha256=sha256(checkpoint), weight_mode=args.weight_mode,
                  graph="cached 0.75m k12", fixed_cell_m=.5,
                  selection="per-crop compression budget only; no validation IDs",
                  steps=args.steps, val_plots=len(val), test_used=False,
                  min_superpoint_size=1, cp_iterations=20, ez_iterations=20,
                  torch_scatter_native_compat=shim,
                  timing_scope="cached feature -> embedding, then partition; no LitePT backbone")
    save_json(output / "config.json", config)
    results, searches = [], []
    for index, row in enumerate(val, 1):
        with np.load(CACHE / f"{row['dataset_id']}.npz") as archive:
            cache = {name: archive[name] for name in archive.files}
        coord, ids, edges = cache["coord"], cache["tree_id"], cache["edge"]
        torch.cuda.reset_peak_memory_stats()
        embedding, weight, embed_seconds = embed_and_weight(model, cache, args.weight_mode)
        started = time.monotonic()
        fixed = geometric_partition(coord, .5)
        fixed_seconds = time.monotonic() - started
        target = len(fixed) / len(np.unique(fixed))
        context = PartitionContext(coord, embedding, edges, weight, csr_fn)
        methods = [("fixed", fixed, dict(regularization=None,
                    final_partition_seconds=fixed_seconds, total_search_seconds=0.))]
        for name, run in (
                ("spt_cut_pursuit", lambda reg: context.run_cut(reg, cp_fn)),
                ("ezsp_contour_prior", lambda reg: context.run_ez(reg, ez_fn))):
            group, chosen, trials = budget_match(run, target, lower=1e-5,
                                                  upper=20., steps=args.steps)
            searches.extend(dict(dataset_id=row["dataset_id"], method=name,
                                 target_compression=target, **trial) for trial in trials)
            methods.append((name, group, chosen))
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        for name, groups, chosen in methods:
            diag = partition_diagnostics(coord, ids, groups, edges)
            results.append(dict(dataset_id=row["dataset_id"],
                                source_dataset=row["source_dataset"],
                                collection=row["collection"], method=name,
                                target_compression=target,
                                regularization=chosen["regularization"],
                                partition_seconds=chosen["final_partition_seconds"],
                                budget_search_seconds=chosen["total_search_seconds"],
                                embedding_seconds=embed_seconds,
                                peak_vram_mb=peak_mb, feature_scale=context.feature_scale,
                                **{key: value for key, value in diag.items() if key != "oracle"},
                                **{f"oracle_{key}": value for key, value in diag["oracle"].items()}))
        save_json(output / "progress.json", dict(results=results, searches=searches))
        shown = " ".join(f"{r['method']}:{r['compression']:.3f}/{r['oracle_pq']:.3f}/"
                         + (f"{r['boundary_recall']:.3f}" if r['boundary_recall'] is not None
                            else "NA") for r in results[-3:])
        print(f"{index}/{len(val)} {row['dataset_id']} target={target:.3f} {shown}", flush=True)
    comparison = [summary(results, name) for name in
                  ("fixed", "spt_cut_pursuit", "ezsp_contour_prior")]
    paired = {name: {key: paired_bootstrap(results, name, key) for key in
                     ("oracle_pq", "boundary_recall")}
              for name in ("spt_cut_pursuit", "ezsp_contour_prior")}
    save_json(output / "result.json", dict(config=config, comparison=comparison,
                                           paired_vs_fixed=paired, per_plot=results,
                                           compression_search=searches))
    print(json.dumps(dict(comparison=comparison, paired_vs_fixed=paired), indent=2), flush=True)


if __name__ == "__main__":
    main()
