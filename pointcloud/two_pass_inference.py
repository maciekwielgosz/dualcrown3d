"""Two-pass inference: frozen Model20 windows, stage-1 consensus, conditioned sweep.

The second sweep reuses Model20's 20 m / 8 m window ownership so every voxel is
owned exactly once. Proposals keep their window's full-resolution mask; IDs are
assigned only by :func:`pointcloud.two_pass_reconcile.reconcile`.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pointcloud.dual_head import DensePointDecoder
from pointcloud.instance_output import merge_masks_with_sources
from pointcloud.two_pass import candidate_competition, stage1_conditioning
from pointcloud.two_pass_reconcile import reconcile

STAGE1_KEYS = ('labels', 'confidence', 'source', 'tree_probability', 'point_probability',
               'vote_xy', 'candidate_count', 'foreign_support')


def stage1_arrays(arrays, raw, labels, confidence, source):
    """Plot-level, label-free per-point arrays the second pass conditions on."""
    count, foreign = candidate_competition(raw, labels)
    return dict(labels=np.asarray(labels, np.int64), confidence=np.asarray(confidence, np.float32),
                source=np.asarray(source, np.uint8),
                tree_probability=raw['tree_probability'].astype(np.float32),
                point_probability=raw['point_probability'].astype(np.float32),
                vote_xy=(raw['shifted_center'][:, :2] - arrays['coord'][:, :2]).astype(np.float32),
                candidate_count=count, foreign_support=foreign)


def stage1_predict(model, arrays, config, raw=None):
    from scripts.predict_dual_head import predict
    if raw is None:
        raw = predict(model, arrays, owner_only=True, raw_object_threshold=.05)
    labels, confidence, _, source = merge_masks_with_sources(arrays, raw, config)
    return raw, labels, confidence, source


def conditioning_subset(arrays, stage1, chosen):
    return stage1_conditioning(arrays['coord'][chosen], stage1['vote_xy'][chosen], stage1['labels'][chosen],
                               stage1['confidence'][chosen], stage1['source'][chosen],
                               stage1['tree_probability'][chosen], stage1['point_probability'][chosen],
                               stage1['candidate_count'][chosen], stage1['foreign_support'][chosen])


@torch.no_grad()
def frozen_window(model, batch):
    """Backbone features, legacy heads and the dense stage-1 hidden state."""
    features = model.legacy.backbone(batch).feat
    logits = model.legacy.semantic_head(features)
    offset = model.legacy.offset_head(features) * model.legacy.offset_scale_m
    dense = DensePointDecoder.forward(model.point_decoder, features, batch, logits, offset)
    return features, dense


@torch.no_grad()
def stage2_sweep(model, refiner, arrays, stage1, max_points=40000, seed=20261002, raw_threshold=.1,
                 size=20., overlap=8., mask_floor=.2):
    from scripts.evaluate_pointcloud_litept import model_input, ownership_intervals, starts_for_axis
    start = time.monotonic()
    xyz = arrays['coord']
    device = next(refiner.parameters()).device
    xs = starts_for_axis(float(xyz[:, 0].min()), float(xyz[:, 0].max()), size, overlap)
    ys = starts_for_axis(float(xyz[:, 1].min()), float(xyz[:, 1].max()), size, overlap)
    xmax, ymax = float(xyz[:, 0].max()), float(xyz[:, 1].max())
    xo = ownership_intervals(xs, size, float(xyz[:, 0].min()), float(np.nextafter(xyz[:, 0].max(), np.float32(np.inf))))
    yo = ownership_intervals(ys, size, float(xyz[:, 1].min()), float(np.nextafter(xyz[:, 1].max(), np.float32(np.inf))))
    index = cKDTree(xyz[:, :2])
    visited = np.zeros(len(xyz), np.uint8)
    need = np.zeros(len(xyz), np.float32)
    members, probabilities, decisions, qualities, seeds, offsets = [], [], [], [], [], [0]
    rng = np.random.default_rng(seed)
    windows = 0
    model.eval()
    refiner.eval()
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            context = np.asarray(sorted(index.query_ball_point([x + size / 2, y + size / 2], size / 2 + 1e-4, p=np.inf)), dtype=np.int64)
            if not len(context):
                continue
            x_end = xmax if xi == len(xs) - 1 else x + size
            y_end = ymax if yi == len(ys) - 1 else y + size
            context = context[(xyz[context, 0] >= x) & (xyz[context, 0] <= x_end)
                              & (xyz[context, 1] >= y) & (xyz[context, 1] <= y_end)]
            a, b = xo[xi]
            c, d = yo[yi]
            owner = context[(xyz[context, 0] >= a) & (xyz[context, 0] < b)
                            & (xyz[context, 1] >= c) & (xyz[context, 1] < d)]
            if not len(owner):
                continue
            for owned in np.array_split(owner, max(1, math.ceil(len(owner) / max_points))):
                extra = np.setdiff1d(context, owned, assume_unique=True)
                capacity = max_points - len(owned)
                if len(extra) > capacity:
                    extra = rng.choice(extra, capacity, replace=False)
                chosen = np.sort(np.concatenate((owned, extra)))
                own_local = np.flatnonzero(np.isin(chosen, owned))
                batch = model_input(xyz[chosen], arrays['grid_coord'][chosen], arrays['intensity'][chosen],
                                    device, preserve_height=True)
                features, dense = frozen_window(model, batch)
                condition = torch.from_numpy(conditioning_subset(arrays, stage1, chosen)).to(device)
                output = refiner(features, dense['features'], batch, condition,
                                 dense['semantic_logits'], dense['offset_m'],
                                 stage1_labels=torch.from_numpy(stage1['labels'][chosen]).to(device))
                need[owned] = output['need_logits'].sigmoid()[own_local].cpu().numpy()
                visited[owned] += 1
                masks = output['mask_logits'].sigmoid().cpu().numpy()
                decision = output['decision_logits'].softmax(1).cpu().numpy()
                quality = output['quality_logits'].sigmoid().cpu().numpy()
                seed_xy = xyz[chosen[output['seed_index'].cpu().numpy()], :2]
                for q in np.flatnonzero(decision[:, 1:].max(1) >= raw_threshold):
                    retained = np.flatnonzero(masks[q] >= mask_floor)
                    if len(retained) < 4:
                        continue
                    center = np.average(xyz[chosen[retained], :2], axis=0, weights=masks[q, retained])
                    if not (a <= center[0] < b and c <= center[1] < d):
                        continue
                    members.append(chosen[retained].astype(np.int32))
                    probabilities.append(masks[q, retained].astype(np.float16))
                    decisions.append(decision[q])
                    qualities.append(float(quality[q]))
                    seeds.append(seed_xy[q])
                    offsets.append(offsets[-1] + len(retained))
                windows += 1
    if not np.all(visited == 1):
        raise AssertionError(f'Ownership failed: missing={int((visited == 0).sum())}, repeated={int((visited > 1).sum())}')
    return dict(candidate_offset=np.asarray(offsets, np.int64),
                point_index=np.concatenate(members) if members else np.empty(0, np.int32),
                point_score=np.concatenate(probabilities) if probabilities else np.empty(0, np.float16),
                decision=np.asarray(decisions, np.float32).reshape(-1, 3),
                quality=np.asarray(qualities, np.float32),
                seed_xy=np.asarray(seeds, np.float32).reshape(-1, 2),
                need_probability=need, seconds=np.float64(time.monotonic() - start), windows=np.int64(windows))


def two_pass_predict(model, refiner, arrays, stage1_config, reconcile_config=None, raw=None,
                     stage1=None, proposals=None, max_points=40000, seed=20261002):
    """Full pipeline on one prepared plot; cached intermediates may be supplied."""
    timing = {}
    if stage1 is None:
        start = time.monotonic()
        raw, labels, confidence, source = stage1_predict(model, arrays, stage1_config, raw)
        stage1 = stage1_arrays(arrays, raw, labels, confidence, source)
        timing['stage1_seconds'] = time.monotonic() - start
    if proposals is None:
        proposals = stage2_sweep(model, refiner, arrays, stage1, max_points=max_points, seed=seed)
        timing['stage2_seconds'] = float(proposals['seconds'])
    start = time.monotonic()
    labels, confidence, source, records = reconcile(arrays, stage1['labels'], stage1['confidence'],
                                                    stage1['source'], proposals, reconcile_config)
    timing['reconcile_seconds'] = time.monotonic() - start
    return dict(labels=labels, confidence=confidence, source=source, records=records,
                stage1=stage1, proposals=proposals, raw=raw, timing=timing)
