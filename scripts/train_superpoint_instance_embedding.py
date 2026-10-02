#!/usr/bin/env python3
"""Train an EZ-SP-style instance-boundary embedding on cached real ALS crops.

Unlike the published EZ-SP semantic partition CNN, this small head operates on
frozen LitePT features and uses tree IDs to separate adjacent instances. The
last training epoch is kept; validation labels are not used for selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.superpoints.instance_embedding import InstanceBoundaryEmbedding, edge_affinity
from scripts.train_superpoint_affinity import eligibility, sample_balanced_edges

ROOT = PROJECT / "outputs/dualcrown3d_superpoint_v1"
CACHE = ROOT / "stage2_affinity_pilot/cache/train"
OUTPUT = ROOT / "stage2_instance_embedding"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-plots", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    if args.epochs < 1 or args.max_plots < 0 or not torch.cuda.is_available():
        raise ValueError("Need CUDA, positive epochs and nonnegative max-plots")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Experiment output protected: {output}")
    train, _ = eligibility()
    if args.max_plots:
        train = train[:args.max_plots]
    paths = [CACHE / f"{row['dataset_id']}.npz" for row in train]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("Missing Stage-2 frozen-feature training cache")
    output.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    model = InstanceBoundaryEmbedding().cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        losses, same_affinity, other_affinity = [], [], []
        started = time.monotonic()
        for path in rng.permutation(paths):
            with np.load(path) as archive:
                feat = archive["feature"].astype(np.float32)
                ids = archive["tree_id"]
                graph = archive["edge"]
            edge, label = sample_balanced_edges(ids, graph, rng, count=1024)
            if not len(edge):
                continue
            tensor = torch.from_numpy(feat).cuda()
            edge = torch.from_numpy(edge.astype(np.int64)).cuda()
            label = torch.from_numpy(label).cuda()
            optimizer.zero_grad(set_to_none=True)
            embedding = model(tensor)
            affinity = edge_affinity(embedding, edge).clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(affinity, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            losses.append(float(loss.detach()))
            same_affinity.append(float(affinity[label > .5].detach().mean()))
            other_affinity.append(float(affinity[label < .5].detach().mean()))
        record = dict(epoch=epoch, updates=len(losses), mean_loss=float(np.mean(losses)),
                      mean_positive_affinity=float(np.mean(same_affinity)),
                      mean_negative_affinity=float(np.mean(other_affinity)),
                      seconds=time.monotonic() - started)
        history.append(record)
        print(json.dumps(record), flush=True)
    checkpoint = output / "embedding.pt"
    config = dict(method="EZ-SP-style instance-boundary contrastive embedding on frozen LitePT features",
                  exact_ezsp_backbone=False, input_dim=72, output_dim=16,
                  edge_affinity="exp(-Euclidean distance)",
                  target="same positive tree ID vs different tree or known background",
                  unknown_id=-1, tree_background_id=0,
                  train_plots=len(train), val_used_for_selection=False,
                  epochs=args.epochs, seed=args.seed, edges_per_plot=1024,
                  cache_root=str(CACHE), script_sha256=sha256(Path(__file__)))
    torch.save(dict(model=model.cpu().state_dict(), config=config), checkpoint)
    (output / "train.json").write_text(json.dumps(dict(config=config,
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        history=history), indent=2) + "\n")
    print(f"Saved {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
