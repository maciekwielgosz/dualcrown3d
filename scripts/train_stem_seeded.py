#!/usr/bin/env python3
"""Fine-tune Model20 with the stem-seeded decoder (one query per stem foot).

    .venv-gpu/bin/python scripts/train_stem_seeded.py --run ss_v1 [--epochs 40 --samples 160]
    .venv-gpu/bin/python scripts/train_stem_seeded.py --run smoke --smoke

Starts from the Model20 checkpoint (every weight reused; only the ``point_decoder.stem_*`` modules
are new) and trains on crops whose feet come from ``stems.csv`` (``scripts/prepare_stem_feet.py``),
corrupted at training time (dropped, shifted, false and duplicated feet).

Training data (train splits only): supervision-v4 real plots (FOR-instance + IDEAS-ALS,
``train_eligible``; proxy feet from the lowest points), TreeScanPL10k HELIOS plots and their flight
variants (TLS-traced feet), Weiser real scans (reference feet) and Weiser simulated tiles. Share of
the draws per group: ``--sources for_instance=1,helios=1,weiser_real=1,weiser_sim=0.5``.

Validation every ``--val-every`` epochs on the val splits: supervision-v4 (``point_eval_eligible``),
HELIOS and the real Weiser val plots (BR02, BR08; 14 pts/m2 seed 0), with the labelled feet as
queries, clean and corrupted (fixed seed per plot). Metric: source-balanced point PQ at IoU 0.5
(``evaluate_combined_full_crowns.aggregate``, the Model20 protocol) of the per-point argmax
assignment. Selection = corrupted-feet PQ (what a detector would deliver); clean = the ceiling with
perfect feet. Model20 with its production consensus is scored once on the same plots with the same
metric (``<root>/baseline_model20``). No test plot is read.

``--resume`` continues an interrupted run from ``weights/last.pt`` (short backfill jobs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz, move_to_device, read_manifest  # noqa: E402
from pointcloud.dual_head import DualHeadLitePT  # noqa: E402
from pointcloud.instance_output import merge_masks, point_instance_metrics  # noqa: E402
from pointcloud.joint_training import fixed_serialization, joint_losses  # noqa: E402
from pointcloud.paths import remap  # noqa: E402
from pointcloud.stem_data import StemDrawDataset, read_stems, source_weights, stem_group, stems_path  # noqa: E402
from pointcloud.stem_seeded import corrupt_stems, predict_stems  # noqa: E402
from scripts.evaluate_combined_full_crowns import aggregate  # noqa: E402

WORKSPACE = PROJECT.parent
DEFAULT_ROOT = PROJECT / 'outputs/stem_seeded_v1'
MODEL20 = PROJECT / 'outputs/dualcrown3d_joint_campaign_v1/runs/repeat_20260930/weights/best.pt'
HELIOS = WORKSPACE / 'dualcrown3d_treescan_helios_v1/manifest.csv'
AUGMENT = WORKSPACE / 'TreeScanPL10k_HELIOS_ALS_flight_augmentation_v2/manifest.csv'
WEISER = WORKSPACE / 'dualcrown3d_weiser_v1/manifest.csv'
MODEL20_SELECTION = PROJECT / 'outputs/dualcrown3d_joint_campaign_v1/runs/repeat_20260930/selected.json'
REAL = WORKSPACE / 'combined_als_crowns_supervision_v4/manifest.csv'
DATA_KEYS = ('coord', 'grid_coord', 'feat', 'offset', 'tree_id', 'stems', 'stem_tree_id', 'stem_score')


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + '\n')
    os.replace(tmp, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_weights(text):
    out = {}
    for item in filter(None, text.split(',')):
        key, value = item.split('=')
        out[key.strip()] = float(value)
    return out


def build(payload, model_args=None):
    """Model20 weights into the stem-seeded architecture; only ``stem_*`` parameters start fresh."""
    kwargs = dict(model_args or payload.get('model_args', dict(queries=96, decoder_layers=3, memory_tokens=1024)))
    kwargs.pop('decoder_policy', None)
    model = DualHeadLitePT(decoder_policy='stem_seeded', **kwargs)
    state = {k: v for k, v in payload['model'].items()}
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = [k for k in incompatible.unexpected_keys]
    missing = [k for k in incompatible.missing_keys if not k.startswith('point_decoder.stem_')]
    if unexpected or missing:
        raise ValueError(f'Checkpoint mismatch: unexpected {unexpected[:5]}, missing {missing[:5]}')
    fixed_serialization(model)
    return model, {**kwargs, 'decoder_policy': 'stem_seeded'}


def training_rows(weights, augment):
    rows = []
    if weights.get('helios', 0) > 0 and HELIOS.exists():
        rows += read_manifest(HELIOS, 'train')
        if augment and AUGMENT.exists():
            rows += read_manifest(AUGMENT, 'train')
    if (weights.get('weiser_real', 0) > 0 or weights.get('weiser_sim', 0) > 0) and WEISER.exists():
        rows += [r for r in read_manifest(WEISER, 'train') if weights.get(stem_group(r), 0) > 0]
    if weights.get('for_instance', 0) > 0 and REAL.exists():
        rows += [r for r in read_manifest(REAL, 'train') if r.get('train_eligible', 'true').lower() == 'true']
    missing = [r['dataset_id'] for r in rows if not stems_path(r).exists()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} training rows without stems.csv (e.g. {missing[:3]}); '
                                'run scripts/prepare_stem_feet.py')
    if not rows:
        raise FileNotFoundError('No training rows')
    return rows


def validation_rows():
    rows = []
    if REAL.exists():
        rows += [r for r in read_manifest(REAL, 'val') if r.get('point_eval_eligible', 'true').lower() == 'true']
    if HELIOS.exists():
        rows += read_manifest(HELIOS, 'val')
    if WEISER.exists():
        rows += [r for r in read_manifest(WEISER, 'val') if r['source_dataset'] == 'WEISER_2022_ALS']
    missing = [r['dataset_id'] for r in rows if not stems_path(r).exists()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} validation rows without stems.csv (e.g. {missing[:3]})')
    return rows


def metric_row(row, truth, labels):
    return dict(dataset_id=row['dataset_id'], source_dataset=row['source_dataset'],
                collection=row.get('collection', row['source_dataset']), group=stem_group(row),
                **point_instance_metrics(truth, labels))


def summarise(rows):
    if not rows:
        return dict(source_balanced_pq=0., by_group={})
    result = aggregate(rows)
    groups = sorted({r['group'] for r in rows})
    result['by_group'] = {g: aggregate([r for r in rows if r['group'] == g])['source_balanced_pq'] for g in groups}
    return result


def model20_feet(rows, folder, max_points=40000):
    """One foot per Model20 instance (production consensus) on every val plot, cached as
    ``<folder>/<dataset_id>.npy`` (world x, y, h): the realistic feet available today."""
    folder.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (folder / f"{r['dataset_id']}.npy").exists()]
    if todo:
        from pointcloud.dual_head import DualHeadLitePT as Legacy
        from scripts.predict_dual_head import predict
        from scripts.prepare_stem_feet import proxy_feet
        payload = torch.load(MODEL20, map_location='cpu', weights_only=False)
        config = json.loads(MODEL20_SELECTION.read_text())['config']
        model = Legacy(**payload.get('model_args', {})).cuda()
        model.load_state_dict(payload['model'], strict=True)
        fixed_serialization(model); model.eval()
        for row in todo:
            arrays = load_npz(row['output'])
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(7001)
                raw = predict(model, arrays, max_points=max_points, owner_only=config.get('owner_only', True))
            labels, _, _ = merge_masks(arrays, raw, config)
            feet = proxy_feet(arrays['coord'] + arrays['source_origin'], labels.astype(np.int64))
            np.save(folder / f"{row['dataset_id']}.npy", np.array([[f['x'], f['y'], f['h']] for f in feet]).reshape(-1, 3))
            print(f"Model20 feet {row['dataset_id']}: {len(feet)}", flush=True)
        del model; torch.cuda.empty_cache()
    return {r['dataset_id']: np.load(folder / f"{r['dataset_id']}.npy") for r in rows}


def validate(model, rows, folder, max_points, m20_feet=None):
    folder.mkdir(parents=True, exist_ok=True)
    clean_rows, noisy_rows, feet = [], [], []
    m20_rows = {0.: [], .5: []}
    with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
        torch.manual_seed(7001)
        for row in rows:
            arrays = load_npz(row['output'])
            stems, ids, quality = read_stems(row)
            rng = np.random.default_rng(int(hashlib.sha256(row['dataset_id'].encode()).hexdigest()[:8], 16))
            canopy = (arrays['coord'] + arrays['source_origin'])[arrays['coord'][:, 2] >= 2., :2]
            noisy, noisy_ids, score = corrupt_stems(stems, ids, rng, xy_canopy=canopy)
            clean_labels, _, _ = predict_stems(model, arrays, stems, max_points=max_points)
            noisy_labels, noisy_score, _ = predict_stems(model, arrays, noisy, stem_score=score, max_points=max_points)
            clean_rows.append(metric_row(row, arrays['tree_id'], clean_labels))
            noisy_rows.append(metric_row(row, arrays['tree_id'], noisy_labels))
            if m20_feet is not None:
                for t in m20_rows:
                    m20_labels, _, _ = predict_stems(model, arrays, m20_feet[row['dataset_id']], max_points=max_points,
                                                     object_threshold=t)
                    m20_rows[t].append(metric_row(row, arrays['tree_id'], m20_labels))
            feet.append(dict(dataset_id=row['dataset_id'], stems=int(len(stems)), false_feet=int((noisy_ids == 0).sum()),
                             false_mean_score=float(np.nanmean(noisy_score[noisy_ids == 0])) if (noisy_ids == 0).any() else None,
                             true_mean_score=float(np.nanmean(noisy_score[noisy_ids > 0])) if (noisy_ids > 0).any() else None))
            print(f"val {row['dataset_id']}: clean tp/fp/fn {clean_rows[-1]['tp']}/{clean_rows[-1]['fp']}/{clean_rows[-1]['fn']}, "
                  f"corrupted {noisy_rows[-1]['tp']}/{noisy_rows[-1]['fp']}/{noisy_rows[-1]['fn']}", flush=True)
    clean, noisy = summarise(clean_rows), summarise(noisy_rows)
    false_scores = [f['false_mean_score'] for f in feet if f['false_mean_score'] is not None]
    true_scores = [f['true_mean_score'] for f in feet if f['true_mean_score'] is not None]
    m20 = {t: summarise(r) for t, r in m20_rows.items() if r}
    by_group = {g: dict(clean=clean['by_group'][g], corrupted=noisy['by_group'][g],
                        **{f'ext_feet_t{t:g}': m['by_group'][g] for t, m in m20.items()}) for g in clean['by_group']}
    m20_score = max((m['source_balanced_pq'] for m in m20.values()), default=None)
    result = dict(score=noisy['source_balanced_pq'], clean_score=clean['source_balanced_pq'], ext_feet_score=m20_score,
                  ext_feet={f't{t:g}': m['source_balanced_pq'] for t, m in m20.items()},
                  by_group=by_group,
                  false_feet_score=float(np.mean(false_scores)) if false_scores else None,
                  true_feet_score=float(np.mean(true_scores)) if true_scores else None,
                  clean=clean, corrupted=noisy, per_plot=dict(clean=clean_rows, corrupted=noisy_rows, feet=feet))
    (folder / 'val_metrics.json').write_text(json.dumps(result, indent=2, default=float) + '\n')
    return result


def baseline_model20(rows, folder, max_points):
    """Model20 with its production consensus on the same val plots and metric (computed once)."""
    path = folder / 'val_metrics.json'
    if path.exists():
        return json.loads(path.read_text())
    from pointcloud.dual_head import DualHeadLitePT as Legacy
    from scripts.predict_dual_head import predict
    payload = torch.load(MODEL20, map_location='cpu', weights_only=False)
    config = json.loads(MODEL20_SELECTION.read_text())['config']
    model = Legacy(**payload.get('model_args', {})).cuda()
    model.load_state_dict(payload['model'], strict=True)
    fixed_serialization(model); model.eval()
    out = []
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(7001)
        for row in rows:
            arrays = load_npz(row['output'])
            raw = predict(model, arrays, max_points=max_points, owner_only=config.get('owner_only', True))
            labels, _, _ = merge_masks(arrays, raw, config)
            out.append(metric_row(row, arrays['tree_id'], labels))
            print(f"Model20 val {row['dataset_id']}: tp/fp/fn {out[-1]['tp']}/{out[-1]['fp']}/{out[-1]['fn']}", flush=True)
    summary = summarise(out)
    result = dict(score=summary['source_balanced_pq'], by_group=summary['by_group'], summary=summary, per_plot=out,
                  checkpoint=str(MODEL20), merge_strategy=config.get('merge_strategy'))
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=float) + '\n')
    del model; torch.cuda.empty_cache()
    return result


def smoke(model, rows, args):
    model.cuda(); model.configure_training(args.scope); fixed_serialization(model); model.train()
    dataset = StemDrawDataset(rows[:1], 123, args.max_points, corrupt=True, min_quality=args.min_quality)
    batch = move_to_device(dataset[(0, 0)], torch.device('cuda:0'))
    torch.cuda.reset_peak_memory_stats()
    losses = joint_losses(model({k: batch[k] for k in DATA_KEYS}), batch)
    losses['loss'].backward()
    stem_grads = {n: bool(p.grad is not None and p.grad.abs().sum() > 0)
                  for n, p in model.named_parameters() if n.startswith('point_decoder.stem_')}
    report = dict(loss=float(losses['loss']), mask_loss=float(losses['mask_loss']),
                  assignment_loss=float(losses.get('assignment_loss', 0.)), mask_iou=float(losses['mask_iou']),
                  stems=int(len(batch['stems'])), false_feet=int((batch['stem_tree_id'] == 0).sum()),
                  stem_gradients=stem_grads, peak_vram_mb=torch.cuda.max_memory_allocated() / 1024 ** 2)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--run', required=True)
    ap.add_argument('--initial', type=Path, default=MODEL20)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--samples', type=int, default=160, help='crops per epoch')
    ap.add_argument('--max-points', type=int, default=16000)
    ap.add_argument('--scope', choices=('mask', 'heads', 'partial'), default='partial')
    ap.add_argument('--sources', default='for_instance=1,helios=1,weiser_real=1,weiser_sim=0.5')
    ap.add_argument('--min-quality', type=float, default=0.)
    ap.add_argument('--no-corrupt', action='store_true')
    ap.add_argument('--no-augment', action='store_true', help='leave out the HELIOS flight-augmentation variants')
    ap.add_argument('--resume', action='store_true', help='continue an interrupted run from weights/last.pt')
    ap.add_argument('--skip-baseline', action='store_true')
    ap.add_argument('--objectness', action='store_true', help='v2: BCE real/false foot as the object score')
    ap.add_argument('--fragment', type=float, default=0., help='share of true feet given a false fragment foot 1.5-4 m away')
    ap.add_argument('--jitter', type=float, nargs='+', default=[.3, .6, 1.])
    ap.add_argument('--detector', default=None, help='detector feet name (stems_<name>.csv, scripts/cache_detector_feet.py)')
    ap.add_argument('--detector-probability', type=float, default=.7)
    ap.add_argument('--height-scale', type=float, nargs=2, default=[.85, 1.15], help='uniform height factor of corrupted feet')
    ap.add_argument('--val-feet', default='model20',
                    help="external feet scored in validation: 'model20' (computed and cached) or a feet-cache folder name, "
                         "e.g. stemfield_sf_v1 (outputs/stem_seeded_v1/feet_cache/<name>/<dataset_id>.npy)")
    ap.add_argument('--select', choices=('corrupted', 'model20', 'external', 'mean'), default='corrupted',
                    help='selection score: corrupted labelled feet, external feet (model20 = alias; best of t=0/0.5), or the mean of clean, corrupted and external')
    ap.add_argument('--val-max-points', type=int, default=40000)
    ap.add_argument('--min-stop-epoch', type=int, default=16)
    ap.add_argument('--lr-heads', type=float, default=5e-5)
    ap.add_argument('--lr-backbone', type=float, default=5e-6)
    ap.add_argument('--val-every', type=int, default=4)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--seed', type=int, default=20261014)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    payload = torch.load(args.initial, map_location='cpu', weights_only=False)
    model, model_args = build(payload)
    model.point_decoder.objectness = bool(args.objectness)
    corruption = dict(jitter=tuple(args.jitter), fragment=args.fragment, height_scale=tuple(args.height_scale))
    weights = parse_weights(args.sources)
    rows = training_rows(weights, not args.no_augment)
    if args.smoke:
        root = args.root / 'smoke'; root.mkdir(parents=True, exist_ok=True)
        dump(root / 'smoke.json', smoke(model, rows, args)); return
    run = args.root / 'runs' / args.run
    if (run / 'DONE.json').exists():
        print(f'already complete: {run}'); return
    resume = args.resume and (run / 'weights/last.pt').exists()
    if run.exists() and not resume:
        if not args.resume:
            raise FileExistsError(f'Incomplete run protected: {run}; pass --resume or remove it')
        import shutil
        shutil.rmtree(run)  # died before the first epoch: nothing to keep
        print(f'restarting {run} from scratch (no last.pt)', flush=True)
    (run / 'weights').mkdir(parents=True, exist_ok=True)
    val_rows = validation_rows()
    if not args.skip_baseline:
        base = baseline_model20(val_rows, args.root / 'baseline_model20', 40000)
        print(json.dumps(dict(model20_val_pq=base['score'], by_group=base['by_group'])), flush=True)
    m20_feet = None
    if args.select != 'corrupted':
        if args.val_feet == 'model20':
            m20_feet = model20_feet(val_rows, args.root / 'baseline_model20' / 'feet')
        else:
            folder = args.root / 'feet_cache' / args.val_feet
            m20_feet = {r['dataset_id']: np.load(folder / f"{r['dataset_id']}.npy") for r in val_rows}

    def selection(val):
        if args.select in ('model20', 'external'):
            return val['ext_feet_score']
        if args.select == 'mean':
            return float(np.mean([val['clean_score'], val['score'], val['ext_feet_score']]))
        return val['score']

    sampling = source_weights(rows, weights)
    dataset = StemDrawDataset(rows, args.seed, args.max_points, corrupt=not args.no_corrupt, min_quality=args.min_quality,
                              corruption=corruption, detector=args.detector, detector_probability=args.detector_probability)
    model.cuda(); model.configure_training(args.scope); fixed_serialization(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    backbone = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('legacy.backbone.')]
    heads = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('legacy.backbone.')]
    groups = [dict(params=heads, lr=args.lr_heads, name='heads')]
    if backbone:
        groups.append(dict(params=backbone, lr=args.lr_backbone, name='backbone'))
    optimizer = torch.optim.AdamW(groups, weight_decay=.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-7)
    config = dict(name=args.run, scope=args.scope, sources=weights, min_quality=args.min_quality, corrupt=not args.no_corrupt,
                  augment=not args.no_augment, seed=args.seed, model_args=model_args, epochs_max=args.epochs,
                  samples_per_epoch=args.samples, max_points=args.max_points, accumulation=2, warmup_heads_epochs=4,
                  validation_every=args.val_every, patience_epochs=args.patience, lr_heads=args.lr_heads,
                  lr_backbone=args.lr_backbone, initial_checkpoint=str(args.initial), initial_sha256=digest(args.initial),
                  training_rows=len(rows), training_groups=dict(Counter(stem_group(r) for r in rows)),
                  validation_rows=[r['dataset_id'] for r in val_rows],
                  trainable_parameters=sum(p.numel() for p in parameters),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  objectness=bool(args.objectness), corruption=corruption, select=args.select, val_feet=args.val_feet,
                  detector=args.detector, detector_probability=args.detector_probability if args.detector else 0.,
                  val_max_points=args.val_max_points,
                  selection=f'source-balanced point PQ, {args.select} (val splits only)',
                  test_used_for_selection=False)

    def save(path, epoch, score, best_epoch):
        torch.save(dict(model=model.state_dict(), model_args=model_args, epoch=epoch, score=score, best_epoch=best_epoch,
                        format='dualcrown3d_stem_seeded_v1', config=config,
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), path)

    if resume:
        state = torch.load(run / 'weights/last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        start_epoch, best, best_epoch = int(state['epoch']) + 1, float(state['score']), int(state.get('best_epoch', 0))
        history = []
        if (run / 'training_log.csv').exists():
            import csv as _csv
            with (run / 'training_log.csv').open() as f:
                history = [dict(r) for r in _csv.DictReader(f) if int(r['epoch']) < start_epoch]
        print(f'resumed {run} at epoch {start_epoch} (best {best:.4f} at {best_epoch})', flush=True)
    else:
        dump(run / 'configuration.json', config)
        with (run / 'training_manifest.csv').open('w') as f:
            f.write('dataset_id,group,output\n')
            for r in rows:
                f.write(f"{r['dataset_id']},{stem_group(r)},{remap(r['output'])}\n")
        initial = validate(model, val_rows, run / 'validation/epoch_000', args.val_max_points, m20_feet)
        best, best_epoch, history, start_epoch = selection(initial), 0, [], 1
        save(run / 'weights/best.pt', 0, best, 0)
        dump(run / 'selected.json', dict(epoch=0, score=best, clean_score=initial['clean_score'], ext_feet=initial['ext_feet'], by_group=initial['by_group']))
    started = time.monotonic()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        model.configure_training('heads' if args.scope == 'partial' and epoch <= 4 else args.scope)
        fixed_serialization(model); model.train(); dataset.epoch = epoch
        rng = np.random.default_rng(args.seed + epoch * 100003)
        draws = [(int(i), step) for step, i in enumerate(rng.choice(len(rows), args.samples, p=sampling / sampling.sum()))]
        loader = DataLoader(dataset, batch_size=None, sampler=draws, num_workers=args.workers, pin_memory=True)
        total = Counter(); optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
        epoch_start = time.monotonic()
        for step, batch in enumerate(loader, 1):
            batch = move_to_device(batch, torch.device('cuda:0'))
            losses = joint_losses(model({k: batch[k] for k in DATA_KEYS}), batch)
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError('Non-finite loss')
            (losses['loss'] / 2).backward()
            if step % 2 == 0 or step == len(draws):
                torch.nn.utils.clip_grad_norm_(parameters, 2., error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            total.update({k: float(v.detach()) for k, v in losses.items() if torch.is_tensor(v)})
        scheduler.step()
        log = dict(epoch=epoch, scope=model.training_scope, seconds=time.monotonic() - epoch_start,
                   peak_vram_mb=torch.cuda.max_memory_allocated() / 1024 ** 2,
                   **{f'train_{k}': v / len(draws) for k, v in total.items()})
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val = validate(model, val_rows, run / f'validation/epoch_{epoch:03d}', args.val_max_points, m20_feet)
            log.update(val_selection=selection(val), val_ext_feet=val['ext_feet_score'],
                       val_score=val['score'], val_clean=val['clean_score'], val_false_feet_score=val['false_feet_score'],
                       val_true_feet_score=val['true_feet_score'],
                       **{f'val_{g}_{k}': v for g, d in val['by_group'].items() for k, v in d.items()})
            if selection(val) > best:
                best, best_epoch = selection(val), epoch
                save(run / 'weights/best.pt', epoch, best, best_epoch)
                dump(run / 'selected.json', dict(epoch=epoch, score=best, clean_score=val['clean_score'], ext_feet=val['ext_feet'], by_group=val['by_group']))
        save(run / 'weights/last.pt', epoch, best, best_epoch)
        history.append(log)
        with (run / 'training_log.csv').open('w') as f:
            keys = list(dict.fromkeys(k for h in history for k in h))
            f.write(','.join(keys) + '\n')
            for h in history:
                f.write(','.join(str(h.get(k, '')) for k in keys) + '\n')
        print(json.dumps(log), flush=True)
        if epoch >= args.min_stop_epoch and epoch - best_epoch >= args.patience:
            break
    dump(run / 'DONE.json', dict(name=args.run, score=best, best_epoch=best_epoch, epochs_completed=epoch,
                                 seconds=time.monotonic() - started, checkpoint=str(run / 'weights/best.pt'),
                                 checkpoint_sha256=digest(run / 'weights/best.pt')))


if __name__ == '__main__':
    main()
