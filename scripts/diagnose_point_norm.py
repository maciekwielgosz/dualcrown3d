#!/usr/bin/env python3
"""Read-only checkpoint diagnostic: fixed-crop running vs per-crop BatchNorm."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import PointCloudCropDataset, move_to_device
from pointcloud.mask_decoder import mask_decoder_losses
from scripts.evaluate_pointcloud_mask_decoder import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    dataset = PointCloudCropDataset(PROJECT.parent / "combined_als_crowns_v1/manifest.csv", "val", max_points=20000, seed=20260926)
    counts = defaultdict(int)
    batches = []
    for i, row in enumerate(dataset.rows):
        if counts[row["collection"]] < 3:
            batches.append((row["collection"], dataset[i]))
            counts[row["collection"]] += 1
    results = {}
    for mode in ("running_statistics", "per_crop_statistics"):
        model = load_model(SimpleNamespace(weights=args.weights), torch.device("cuda:0"))
        model.auxiliary_losses = False
        if mode == "per_crop_statistics":
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    module.running_mean = module.running_var = None
        scores = defaultdict(list)
        with torch.no_grad():
            for collection, cpu in batches:
                batch = move_to_device(cpu, torch.device("cuda:0"))
                pred = model({k: batch[k] for k in ("coord", "grid_coord", "feat", "offset")})
                loss = mask_decoder_losses(pred, batch["tree_id"], False)
                scores[collection].append(float(loss["matched_mask_iou"]))
        results[mode] = {k: sum(v)/len(v) for k, v in scores.items()}
        del model
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(checkpoint=str(args.weights), note="diagnostic validation point-mask IoU; not full-polygon test", results=results), indent=2)+"\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
