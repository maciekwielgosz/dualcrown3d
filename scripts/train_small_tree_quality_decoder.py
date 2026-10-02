#!/usr/bin/env python3
"""Fine-tune EZ-SP Q128 with duplicate-aware object-score supervision."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.dual_head import dense_losses
from pointcloud.superpoints.model import CachedInstanceModel
from pointcloud.superpoints.quality_loss import calibrated_query_loss
from scripts.benchmark_superpoint_algorithms import save_json, sha256
from scripts.train_superpoint_decoder_pilot import (DEFAULT as PILOT, load_crop, batch_for,
    evaluate, configurations)

INITIAL = PILOT / 'runs/ezsp_large_w192_q128/weights/best.pt'
DEFAULT = PROJECT / 'outputs/dualcrown3d_small_tree_quality_v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--eval-every', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261001)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for decoder fine-tuning')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Protected training run exists: {output}')
    (output/'weights').mkdir(parents=True)
    (output/'validation').mkdir()
    prepared = json.loads((PILOT/'prepared.json').read_text())
    train = [r for r in prepared['entries'] if r['split'] == 'train']
    val = [r for r in prepared['entries'] if r['split'] == 'val']
    if {r['group_id'] for r in train} & {r['group_id'] for r in val}:
        raise ValueError('Train/validation spatial group leakage')
    source = torch.load(INITIAL, map_location='cpu', weights_only=False)
    if source['epoch'] != 12 or source['config']['run_name'] != 'ezsp_large_w192_q128':
        raise ValueError('Initial decoder checkpoint changed')
    model = CachedInstanceModel('ezsp', queries=128, memory_tokens=1536,
                                graph_width=192, graph_layers=4,
                                wide_dim=192, wide_layers=2).cuda()
    model.load_state_dict(source['model'], strict=True)
    model.decoder.epoch = 100
    config = dict(architecture='LitePT72 frozen + EZ-SP graph W192 Q128 + wide mask refiner',
                  initialization=str(INITIAL), initialization_sha256=sha256(INITIAL),
                  prepared_sha256=sha256(PILOT/'prepared.json'), train_crops=len(train),
                  val_crops=len(val), epochs=args.epochs, eval_every=args.eval_every,
                  seed=args.seed, learning_rate=3e-5, weight_decay=.01,
                  extra_loss='0.75x Hungarian matched query BCE; unmatched queries on known support are negatives; small GT upweighted',
                  selected_by='validation crop score with no drop in sparse-tree recall',
                  test_used=False, gpu=torch.cuda.get_device_name())
    save_json(output/'configuration.json', config)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=3e-5, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=5e-6)
    history = []
    best_score = -1.
    base_sparse = None
    for epoch in range(args.epochs + 1):
        entry = dict(epoch=epoch)
        if epoch:
            model.train()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            values = []
            for index in np.random.default_rng(args.seed + epoch).permutation(len(train)):
                batch = batch_for(load_crop(train[index]['cache']), 'ezsp')
                prediction = model(batch)
                base = dense_losses(prediction, batch)
                extra = calibrated_query_loss(prediction['instance_masks'], batch['tree_id'])
                total = base['loss'] + .75 * extra
                if not torch.isfinite(total):
                    raise FloatingPointError(f'Nonfinite loss: epoch={epoch} sample={index}')
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 2., error_if_nonfinite=True)
                optimizer.step()
                values.append((float(total.detach()), float(extra.detach())))
            scheduler.step()
            torch.cuda.synchronize()
            entry.update(seconds=time.monotonic()-started,
                         loss=float(np.mean([x[0] for x in values])),
                         calibrated_query_loss=float(np.mean([x[1] for x in values])),
                         peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2)
        if epoch == 0 or epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, val, configurations())
            save_json(output/'validation'/f'epoch_{epoch:03d}.json', metrics)
            # Small-tree recall is a gate; full-scene validation remains required
            # before any test or production deployment.
            name = max(metrics, key=lambda key: metrics[key]['score'])
            record = metrics[name]
            sparse = record['sparse_tree_recall_le100_voxels']
            if base_sparse is None:
                base_sparse = sparse
            entry.update(validation_score=record['score'],
                         point_pq=record['point']['source_balanced_pq'],
                         crown_pq=record['crown']['source_balanced_pq'],
                         sparse_recall=sparse, thresholds=name)
            if record['score'] > best_score and sparse >= base_sparse:
                best_score = record['score']
                checkpoint = output/'weights/best.pt'
                torch.save(dict(model=model.state_dict(), config=config,
                                epoch=epoch, initial_checkpoint=str(INITIAL)), checkpoint)
                save_json(output/'selected.json', dict(epoch=epoch, validation=record,
                          checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                          base_sparse_recall=base_sparse))
        history.append(entry)
        save_json(output/'history.json', dict(epochs=history))
        print(json.dumps(entry), flush=True)
    save_json(output/'DONE.json', dict(completed=True, selected_score=best_score,
                                      selected=json.loads((output/'selected.json').read_text())['epoch']))


if __name__ == '__main__':
    main()
