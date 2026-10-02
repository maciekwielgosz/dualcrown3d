#!/usr/bin/env python3
"""Compare SPT Cut Pursuit and EZ-SP contour-prior partition kernels.

This is a controlled partition-only pilot, not a reproduction of either full
network. Both kernels receive identical cached LitePT features, graph edges,
and the previously trained instance-affinity scores. A label-free per-crop
regularization search matches the 0.5 m fixed-cell compression budget. Ground
truth is opened only after selecting each partition, for diagnostics.

Run with the isolated optional packages on PYTHONPATH, e.g.::

    PYTHONPATH=/tmp/dualcrown3d_superpoint_deps_20261001 \
      .venv-gpu/bin/python scripts/benchmark_superpoint_algorithms.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from pointcloud.superpoints.affinity import EdgeAffinityHead
from pointcloud.superpoints.partition import geometric_partition, partition_diagnostics
from pointcloud.superpoints.torch_scatter_compat import install as install_scatter_compat
from scripts.train_superpoint_affinity import edge_probabilities, eligibility

ROOT = PROJECT / "outputs/dualcrown3d_superpoint_v1"
CACHE = ROOT / "stage2_affinity_pilot/cache/val"
CHECKPOINT = ROOT / "stage2_affinity_pilot/affinity_head.pt"
DEFAULT_OUTPUT = ROOT / "stage2_spt_ezsp_partition_pilot"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_methods():
    try:
        from pycut_pursuit.cp_d0_dist import cp_d0_dist
        from grid_graph import edge_list_to_forward_star
    except ImportError as error:
        raise RuntimeError("Install isolated pycut-pursuit and pygrid-graph wheels") from error
    try:
        import torch_scatter  # noqa: F401
        shim = False
    except ImportError:
        install_scatter_compat()
        shim = True
    try:
        from torch_graph_components.merge import merge_components_by_contour_prior
    except ImportError as error:
        raise RuntimeError("Install isolated torch-graph-components and torch-geometric") from error
    return cp_d0_dist, edge_list_to_forward_star, merge_components_by_contour_prior, shim


def joint_features(coord: np.ndarray, feature: np.ndarray,
                   edges: np.ndarray) -> tuple[np.ndarray, float]:
    """Match median local feature change to median local spatial distance."""
    feature = feature.astype(np.float32)
    sample = edges[np.linspace(0, len(edges) - 1, min(10000, len(edges))).astype(int)]
    spatial = np.linalg.norm(coord[sample[:, 0]] - coord[sample[:, 1]], axis=1)
    latent = np.linalg.norm(feature[sample[:, 0]] - feature[sample[:, 1]], axis=1)
    scale = float(np.median(spatial) / max(float(np.median(latent)), 1e-6))
    value = np.concatenate((coord - coord.mean(0),
                            (feature - feature.mean(0)) * scale), axis=1)
    return np.ascontiguousarray(value, dtype=np.float32), scale


class PartitionContext:
    def __init__(self, coord, feature, edges, probability, forward_star):
        self.coord = coord
        self.edges = edges
        self.probability = probability
        self.x, self.feature_scale = joint_features(coord, feature, edges)
        self.n = len(coord)
        csr, target, order = forward_star(self.n, edges.astype(np.uint32))
        self.csr = csr.astype(np.uint32)
        self.target = target.astype(np.uint32)
        self.cp_weight = np.ascontiguousarray(np.maximum(probability[order], 1e-4))
        self.cp_input = np.asfortranarray(self.x.T)
        self.cp_coor_weight = np.ones(self.x.shape[1], dtype=np.float32)
        self.gpu_x = torch.from_numpy(self.x).cuda()
        self.gpu_edges = torch.from_numpy(edges.T.astype(np.int64)).cuda()
        self.gpu_weight = torch.from_numpy(probability).cuda()
        self.gpu_size = torch.ones(self.n, device="cuda", dtype=torch.float32)

    def run_cut(self, reg, fn):
        started = time.monotonic()
        groups = fn(self.x.shape[1], self.cp_input, self.csr, self.target,
                    edge_weights=self.cp_weight * np.float32(reg),
                    coor_weights=self.cp_coor_weight, min_comp_weight=0.,
                    cp_it_max=20, verbose=False, max_num_threads=4)[0]
        elapsed = time.monotonic() - started
        _, groups = np.unique(groups, return_inverse=True)
        return groups.astype(np.int32), elapsed

    def run_ez(self, reg, fn):
        torch.cuda.synchronize()
        started = time.monotonic()
        group, _, _ = fn(self.gpu_x, self.gpu_size, self.gpu_edges,
                         self.gpu_weight, reg=float(reg), min_size=1,
                         max_iterations=20)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        return group.cpu().numpy().astype(np.int32), elapsed


def budget_match(run, target: float, *, lower: float, upper: float,
                 steps: int) -> tuple[np.ndarray, dict, list[dict]]:
    """Choose regularization using group count, never reference labels."""
    trials = []
    lo, hi = lower, upper
    for iteration in range(steps):
        reg = float(np.sqrt(lo * hi))
        groups, seconds = run(reg)
        compression = len(groups) / len(np.unique(groups))
        trials.append(dict(iteration=iteration, regularization=reg,
                           compression=compression, partition_seconds=seconds))
        if compression < target:
            lo = reg
        else:
            hi = reg
    best = min(trials, key=lambda row: abs(np.log(row["compression"] / target)))
    groups, seconds = run(best["regularization"])
    best = dict(best, final_partition_seconds=seconds,
                total_search_seconds=sum(row["partition_seconds"] for row in trials) + seconds)
    return groups, best, trials


def summary(rows: list[dict], method: str) -> dict:
    subset = [row for row in rows if row["method"] == method]
    keys = ["compression", "oracle_pq", "oracle_f1", "boundary_recall",
            "mixed_group_fraction", "known_point_purity", "partition_seconds"]
    return dict(method=method, plots=len(subset), **{
        f"mean_{key}": float(np.mean([row[key] for row in subset if row[key] is not None]))
        for key in keys})


def paired_bootstrap(rows: list[dict], method: str, key: str,
                     seed: int = 20261001) -> dict:
    by_plot = {}
    for row in rows:
        by_plot.setdefault(row["dataset_id"], {})[row["method"]] = row
    delta = np.asarray([pair[method][key] - pair["fixed"][key]
                        for pair in by_plot.values() if method in pair
                        and pair[method][key] is not None and pair["fixed"][key] is not None])
    if not len(delta):
        return dict(n=0)
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(delta, (5000, len(delta)), replace=True), axis=1)
    return dict(n=len(delta), mean_delta=float(np.mean(delta)),
                paired_bootstrap_ci95=[float(x) for x in np.quantile(boot, [.025, .975])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-plots", type=int, default=0,
                        help="Smoke-test cap; 0 means all 14 eligible validation plots")
    parser.add_argument("--steps", type=int, default=8,
                        help="Label-free compression matching iterations")
    args = parser.parse_args()
    if args.steps < 2 or args.max_plots < 0 or not torch.cuda.is_available():
        raise ValueError("Need CUDA, positive max-plots and at least two search steps")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Experiment output protected: {output}")
    cp_fn, csr_fn, ez_fn, shim = load_methods()
    _, val = eligibility()
    if args.max_plots:
        val = val[:args.max_plots]
    for row in val:
        if not (CACHE / f"{row['dataset_id']}.npz").exists():
            raise FileNotFoundError(f"Missing cached plot: {row['dataset_id']}")
    weight = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    head = EdgeAffinityHead().cuda().eval()
    head.load_state_dict(weight["model"])
    torch.set_num_threads(4)
    config = dict(description=__doc__.splitlines()[0],
                  full_spt_or_ezsp_model=False,
                  candidate_kernels=["pycut-pursuit.cp_d0_dist", "torch-graph-components.merge_components_by_contour_prior"],
                  cached_encoder="retained_HELIOS_exposed_LitePT_72d",
                  graph="same_cached_0.75m_k12", affinity_checkpoint=str(CHECKPOINT),
                  affinity_checkpoint_sha256=sha256(CHECKPOINT),
                  feature_scale="per-crop median local spatial / feature distance; no labels",
                  edge_weight="same trained sigmoid instance affinity for both methods",
                  cut_pursuit=dict(iterations=20, min_comp_weight=0, threads=4),
                  ezsp=dict(min_size=1, max_iterations=20, native_torch_scatter_shim=shim),
                  fixed_cell_m=0.5, compression_match="per-crop group count, no labels",
                  steps=args.steps, val_plots=len(val), test_used=False,
                  timing_scope="partition only; excludes cached LitePT feature extraction and affinity model",
                  packages={name: importlib.metadata.version(name) for name in
                            ("pycut-pursuit", "pygrid-graph", "torch-graph-components", "torch-geometric")},
                  torch=torch.__version__, numpy=np.__version__,
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT).decode().strip(),
                  script_sha256=sha256(Path(__file__)),
                  compat_sha256=sha256(PROJECT / "pointcloud/superpoints/torch_scatter_compat.py"))
    output.mkdir(parents=True)
    save_json(output / "config.json", config)
    results = []
    searches = []
    for index, row in enumerate(val, 1):
        with np.load(CACHE / f"{row['dataset_id']}.npz") as archive:
            cache = {key: archive[key] for key in archive.files}
        coord, ids, edges = cache["coord"], cache["tree_id"], cache["edge"]
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        probability = edge_probabilities(head, cache).astype(np.float32)
        affinity_seconds = time.monotonic() - started
        started = time.monotonic()
        fixed = geometric_partition(coord, .5)
        fixed_seconds = time.monotonic() - started
        target = len(fixed) / len(np.unique(fixed))
        context = PartitionContext(coord, cache["feature"], edges, probability, csr_fn)
        methods = [
            ("fixed", fixed, dict(regularization=None, compression=target,
                                 final_partition_seconds=fixed_seconds, total_search_seconds=0.)),
        ]
        for name, run, bounds in (
                ("spt_cut_pursuit", lambda reg: context.run_cut(reg, cp_fn), (1e-4, 10.)),
                ("ezsp_contour_prior", lambda reg: context.run_ez(reg, ez_fn), (1e-4, 10.))):
            group, chosen, trials = budget_match(run, target, lower=bounds[0],
                                                  upper=bounds[1], steps=args.steps)
            searches.extend(dict(dataset_id=row["dataset_id"], method=name,
                                 target_compression=target, **trial) for trial in trials)
            methods.append((name, group, chosen))
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        for name, group, chosen in methods:
            diag = partition_diagnostics(coord, ids, group, edges)
            results.append(dict(dataset_id=row["dataset_id"],
                                source_dataset=row["source_dataset"],
                                collection=row["collection"], method=name,
                                target_compression=target,
                                regularization=chosen["regularization"],
                                partition_seconds=chosen["final_partition_seconds"],
                                budget_search_seconds=chosen["total_search_seconds"],
                                affinity_seconds=affinity_seconds,
                                peak_vram_mb=peak_vram_mb,
                                feature_scale=context.feature_scale,
                                **{key: value for key, value in diag.items() if key != "oracle"},
                                **{f"oracle_{key}": value for key, value in diag["oracle"].items()}))
        save_json(output / "progress.json", dict(results=results, searches=searches))
        print(f"{index}/{len(val)} {row['dataset_id']} target={target:.3f} "
              + " ".join(f"{r['method']}:{r['compression']:.3f}/{r['oracle_pq']:.3f}/"
                         + (f"{r['boundary_recall']:.3f}" if r['boundary_recall'] is not None
                            else "NA") for r in results[-3:]), flush=True)
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
