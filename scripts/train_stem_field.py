#!/usr/bin/env python3
"""Train StemField, the stem-foot detector on raw points (``pointcloud/stem_field.py``).

    $PY scripts/train_stem_field.py --run sf_v1 [--epochs 40 --samples 2000]
    $PY scripts/train_stem_field.py --run smoke --smoke

Data: the train splits of the stem-seeded training (supervision-v4 real plots with proxy feet,
TreeScanPL10k HELIOS plots and flight variants with TLS-traced feet, Weiser real scans with measured
feet, Weiser simulated tiles), feet from ``stems.csv``. Crops of 24 m (Model20's crop sampler, with
rotation, scale and point dropout), rasterised on a 28 m grid; cells without points and cells under
mostly unannotated vegetation are ignored by the heat loss. Half of the crops denser than the target
are thinned to 6-40 points/m2, so the detector also learns the sparse airborne regime.

Validation every ``--val-every`` epochs on the val plots (supervision-v4 point-eval plots, HELIOS
val, Weiser BR02/BR08 d14 s0): whole-plot detection, one-to-one matching to the labelled feet within
1.5 m, detections in mostly unannotated vegetation not counted as false positives. Selection =
mean over the three groups of the F1 at the best of ``--thresholds``. With a feet cache
(``scripts/cache_feet_sources.py``) Model20's and v7's feet are scored the same way once
(``<root>/baselines.json``). Test splits are never read. ``--resume`` continues from ``last.pt``.
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz, prepare_crop  # noqa: E402
from pointcloud.data import read_manifest  # noqa: E402
from pointcloud.stem_data import read_stems, read_stems_ignore, source_weights, stem_group  # noqa: E402
from pointcloud.stem_field import (CELL, StemFieldNet, annotated_area, detect_plot, field_loss, heat_targets,  # noqa: E402
                                   ignore_mask, match_feet, rasterise)
from pointcloud.stem_seeded import stems_for_crop  # noqa: E402
from scripts.train_stem_seeded import parse_weights, training_rows, validation_rows  # noqa: E402

DEFAULT_ROOT = PROJECT / 'outputs/stem_field_v1'
FEET_CACHE = PROJECT / 'outputs/stem_seeded_v1/feet_cache'
BRANDENBURG = PROJECT.parent / 'dualcrown3d_brandenburg_stems_v1/manifest.csv'
IGNORE_RADIUS = 1.5  # m around a stem flagged ignore (understorey)
CROP = 24.
GRID = 56  # cells of 0.5 m = 28 m, a multiple of 8
TARGETS = ('volume', 'heat', 'offset', 'log_h', 'positive', 'known', 'ignore')


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=float, allow_nan=False) + '\n')
    os.replace(tmp, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coverage(coord, origin, shape, dilate=2):
    X, Y = shape
    ix = np.floor((coord[:, 0] - origin[0]) / CELL).astype(np.int64)
    iy = np.floor((coord[:, 1] - origin[1]) / CELL).astype(np.int64)
    ok = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y)
    occupied = np.zeros((X, Y), np.float32)
    occupied[ix[ok], iy[ok]] = 1.
    k = 2 * dilate + 1
    return F.max_pool2d(torch.from_numpy(occupied)[None, None], k, 1, dilate)[0, 0].numpy() > 0


class StemFieldDataset(Dataset):
    """Index = (row, draw)."""

    def __init__(self, rows, seed, thin_probability=.5, thin_range=(6., 40.)):
        if any(r['model_split'] != 'train' for r in rows):
            raise ValueError('train rows only')
        self.rows, self.seed, self.epoch = rows, seed, 0
        self.thin_probability, self.thin_range = thin_probability, thin_range
        self._stems = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row_index, draw = index
        row = self.rows[row_index]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, draw, row_index]))
        arrays = load_npz(row['output'])
        crop = prepare_crop(arrays, rng, CROP, 10 ** 7, True, density_keep_fractions=(.5, .75, 1.),
                            preserve_height=bool(row.get('height_normalization')), return_transform=True)
        coord = crop['coord'].numpy().astype(np.float64)
        intensity = crop['feat'][:, 3].numpy()
        tree_id = crop['tree_id'].numpy()
        origin = np.array([-GRID * CELL / 2, -GRID * CELL / 2])
        covered = coverage(coord, origin, (GRID, GRID))
        area = max(covered.sum() * CELL * CELL, 1.)
        density = len(coord) / area
        if rng.random() < self.thin_probability:
            target = rng.uniform(*self.thin_range)
            if density > target and len(coord) > 512:
                keep = rng.random(len(coord)) < target / density
                coord, intensity, tree_id = coord[keep], intensity[keep], tree_id[keep]
                density = len(coord) / area
        key = row['output']
        if key not in self._stems:
            self._stems[key] = read_stems_ignore(row)
        world, ids, _, flagged = self._stems[key]
        local, _ = stems_for_crop(world[~flagged], ids[~flagged], crop['local_to_world'].numpy(), coord, margin=0.)
        hidden, _ = stems_for_crop(world[flagged], ids[flagged], crop['local_to_world'].numpy(), coord, margin=2.)
        if len(local):
            i = np.floor((local[:, 0] - origin[0]) / CELL).astype(np.int64)
            j = np.floor((local[:, 1] - origin[1]) / CELL).astype(np.int64)
            inside = (i >= 0) & (i < GRID) & (j >= 0) & (j < GRID)
            local = local[inside][covered[i[inside], j[inside]]]
        ignore = ignore_mask(coord, tree_id, origin, (GRID, GRID)) | ~covered
        heat, offset, log_h, positive, known, ignore = heat_targets(local, origin, (GRID, GRID), ignore)
        if len(hidden):  # stems the detector cannot be asked to see: neither positive nor negative
            gx, gy = np.meshgrid(origin[0] + (np.arange(GRID) + .5) * CELL, origin[1] + (np.arange(GRID) + .5) * CELL, indexing='ij')
            for x, y, _ in hidden:
                ignore |= ((gx - x) ** 2 + (gy - y) ** 2 <= IGNORE_RADIUS ** 2) & ~positive
        volume = rasterise(coord, intensity.astype(np.float64), origin, (GRID, GRID), density=density)
        return dict(volume=torch.from_numpy(volume), heat=torch.from_numpy(heat), offset=torch.from_numpy(offset),
                    log_h=torch.from_numpy(log_h), positive=torch.from_numpy(positive), known=torch.from_numpy(known),
                    ignore=torch.from_numpy(ignore))


def plot_detection(model, row, threshold_floor=.1):
    arrays = load_npz(row['output'])
    local, scores = detect_plot(model, arrays['coord'].astype(np.float64), arrays['intensity'].astype(np.float64),
                                threshold=threshold_floor)
    labelled, _, _, flagged = read_stems_ignore(row)
    shift = arrays['source_origin'][:2]
    truth = np.column_stack((labelled[~flagged, :2] - shift, labelled[~flagged, 2]))
    hidden = labelled[flagged, :2] - shift
    area = annotated_area(arrays['coord'], arrays['tree_id'])
    if not len(hidden):
        return local, scores, truth, area
    from scipy.spatial import cKDTree
    near = cKDTree(hidden)

    def excuse(xy):
        d, _ = near.query(np.asarray(xy)[:, :2])
        return area(xy) | (d <= IGNORE_RADIUS)
    return local, scores, truth, excuse


def summarise(records, thresholds):
    groups = sorted({r['group'] for r in records})
    table = {}
    for t in thresholds:
        per = {}
        for g in groups + ['all']:
            sub = [r for r in records if g == 'all' or r['group'] == g]
            tp, fp, fn = (sum(r[t][k] for r in sub) for k in ('tp', 'fp', 'fn'))
            d = np.concatenate([r[t]['distances'] for r in sub]) if sub else np.zeros(0)
            per[g] = dict(recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1), f1=2 * tp / max(2 * tp + fp + fn, 1),
                          median_distance_m=float(np.median(d)) if len(d) else None, tp=tp, fp=fp, fn=fn)
        per['score'] = float(np.mean([per[g]['f1'] for g in groups])) if groups else 0.
        table[f'{t:g}'] = per
    best = max(table, key=lambda k: table[k]['score'])
    return dict(score=table[best]['score'], threshold=float(best), by_threshold=table)


def validate(model, rows, folder, thresholds):
    records = []
    for row in rows:
        local, scores, truth, excuse = plot_detection(model, row, min(thresholds))
        record = dict(dataset_id=row['dataset_id'], group=stem_group(row), detections=int(len(local)), truth=int(len(truth)))
        for t in thresholds:
            m = match_feet(local[scores >= t], truth, ignore_xy=excuse)
            record[t] = dict(tp=m['tp'], fp=m['fp'], fn=m['fn'], distances=m['distances'])
        records.append(record)
    result = summarise(records, thresholds)
    folder.mkdir(parents=True, exist_ok=True)
    per_plot = [{k: (v if not isinstance(k, float) else {kk: vv for kk, vv in v.items() if kk != 'distances'})
                 for k, v in r.items()} for r in records]
    dump(folder / 'val_metrics.json', dict(**result, per_plot=[{str(k): v for k, v in p.items()} for p in per_plot]))
    return result


def baselines(rows, root, thresholds):
    """Feet of Model20 and v7 (cache) matched like StemField's; computed once."""
    path = root / 'baselines.json'
    if path.exists() or not FEET_CACHE.exists():
        return json.loads(path.read_text()) if path.exists() else None
    out = {}
    for source in ('model20', 'v7', 'v7fb'):
        records = []
        for row in rows:
            f = FEET_CACHE / source / f"{row['dataset_id']}.npy"
            if not f.exists():
                return None
            arrays = load_npz(row['output'])
            feet = np.load(f)
            local = np.column_stack((feet[:, :2] - arrays['source_origin'][:2], feet[:, 2]))
            labelled, _, _ = read_stems(row)
            truth = np.column_stack((labelled[:, :2] - arrays['source_origin'][:2], labelled[:, 2]))
            m = match_feet(local, truth, ignore_xy=annotated_area(arrays['coord'], arrays['tree_id']))
            records.append({'group': stem_group(row), 0.: dict(tp=m['tp'], fp=m['fp'], fn=m['fn'], distances=m['distances'])})
        out[source] = summarise(records, [0.])
    dump(path, out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--run', required=True)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--samples', type=int, default=2000, help='crops per epoch')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--width', type=int, default=24)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--sources', default='for_instance=1,helios=1,weiser_real=1,weiser_sim=0.5')
    ap.add_argument('--thin-probability', type=float, default=.5)
    ap.add_argument('--thresholds', type=float, nargs='+', default=[.2, .3, .4, .5])
    ap.add_argument('--val-every', type=int, default=4)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--seed', type=int, default=20261015)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--bb-fold', choices=('A', 'B'), default=None,
                    help='add the Brandenburg tuning plots of this fold (measured stems); the other fold is logged as a test')
    ap.add_argument('--bb-weight', type=float, default=1.)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    weights = parse_weights(args.sources)
    rows = training_rows(weights, True)
    bb_test = []
    if args.bb_fold:
        bb = read_manifest(BRANDENBURG, 'train')
        rows += [r for r in bb if r['fold'] == args.bb_fold]
        bb_test = [r for r in bb if r['fold'] != args.bb_fold and r['density'] == 'd14' and str(r['seed']) == '0']
        weights['brandenburg'] = args.bb_weight
    sampling = source_weights(rows, weights)
    dataset = StemFieldDataset(rows, args.seed, args.thin_probability)
    rows_for_test = bb_test
    model = StemFieldNet(args.width).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * max(1, args.samples // args.batch),
                                                    pct_start=.1)
    if args.smoke:
        loader = DataLoader(dataset, batch_size=args.batch, sampler=[(i, i) for i in range(args.batch)], num_workers=0)
        batch = {k: v.cuda() for k, v in next(iter(loader)).items()}
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(batch['volume'])
        losses = field_loss({k: v.float() for k, v in out.items()}, *(batch[k] for k in TARGETS[1:]))
        losses['loss'].backward()
        report = {k: float(v) if torch.is_tensor(v) else v for k, v in losses.items()}
        report.update(peak_vram_mb=torch.cuda.max_memory_allocated() / 2 ** 20, ignored_share=float(batch['ignore'].float().mean()))
        print(json.dumps(report, indent=2)); return
    run = args.root / 'runs' / args.run
    if (run / 'DONE.json').exists():
        print(f'already complete: {run}'); return
    resume = args.resume and (run / 'weights/last.pt').exists()
    if run.exists() and not resume:
        if not args.resume:
            raise FileExistsError(f'Incomplete run protected: {run}; pass --resume or remove it')
        import shutil
        shutil.rmtree(run)
    (run / 'weights').mkdir(parents=True, exist_ok=True)
    val_rows = validation_rows()
    base = baselines(val_rows, args.root, args.thresholds)
    if base:
        print(json.dumps({s: dict(score=v['score'], all=v['by_threshold']['0']['all']) for s, v in base.items()}, default=float), flush=True)
    config = dict(name=args.run, width=args.width, lr=args.lr, epochs=args.epochs, samples_per_epoch=args.samples, batch=args.batch,
                  sources=weights, bb_fold=args.bb_fold, bb_test_plots=sorted({r['parent_plot'] for r in bb_test}),
                  thin_probability=args.thin_probability, thresholds=args.thresholds, crop_m=CROP,
                  grid_cells=GRID, cell_m=CELL, training_rows=len(rows), training_groups=dict(Counter(stem_group(r) for r in rows)),
                  validation_rows=[r['dataset_id'] for r in val_rows], parameters=sum(p.numel() for p in model.parameters()),
                  selection='mean over val groups of the detection F1 at the best threshold', test_used_for_selection=False)

    def save(path, epoch, score, best_epoch, threshold):
        torch.save(dict(model=model.state_dict(), width=args.width, epoch=epoch, score=score, best_epoch=best_epoch,
                        threshold=threshold, config=config, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        format='stem_field_v1'), path)

    if resume:
        state = torch.load(run / 'weights/last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        start_epoch, best, best_epoch = state['epoch'] + 1, state['score'], state['best_epoch']
        threshold = state.get('threshold', .3)
        print(f'resumed at epoch {start_epoch}', flush=True)
    else:
        dump(run / 'configuration.json', config)
        start_epoch, best, best_epoch, threshold = 1, -1., 0, .3
    history = []
    epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); dataset.epoch = epoch
        rng = np.random.default_rng(args.seed + epoch * 100003)
        draws = [(int(i), s) for s, i in enumerate(rng.choice(len(rows), args.samples, p=sampling / sampling.sum()))]
        loader = DataLoader(dataset, batch_size=args.batch, sampler=draws, num_workers=args.workers, pin_memory=True,
                            drop_last=True, persistent_workers=False)
        total, steps, t0 = Counter(), 0, time.monotonic()
        for batch in loader:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(batch['volume'])
            losses = field_loss({k: v.float() for k, v in out.items()}, *(batch[k] for k in TARGETS[1:]))
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError('Non-finite loss')
            optimizer.zero_grad(set_to_none=True)
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); scheduler.step()
            total.update({k: float(v) for k, v in losses.items()}); steps += 1
        log = dict(epoch=epoch, seconds=time.monotonic() - t0, lr=scheduler.get_last_lr()[0],
                   **{f'train_{k}': v / max(steps, 1) for k, v in total.items()})
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val = validate(model, val_rows, run / f'validation/epoch_{epoch:03d}', args.thresholds)
            t = f"{val['threshold']:g}"
            if rows_for_test:  # the other Brandenburg fold: logged, never used for selection
                bbv = validate(model, rows_for_test, run / f'validation/epoch_{epoch:03d}_bb_other_fold', args.thresholds)
                log.update(**{f'bb_test_{k}': v for k, v in bbv['by_threshold'][t]['all'].items()
                              if k in ('recall', 'precision', 'f1', 'median_distance_m')})
            log.update(val_score=val['score'], val_threshold=val['threshold'],
                       **{f'val_{g}_{k}': v for g, d in val['by_threshold'][t].items() if g != 'score'
                          for k, v in d.items() if k in ('recall', 'precision', 'f1', 'median_distance_m')})
            if val['score'] > best:
                best, best_epoch, threshold = val['score'], epoch, val['threshold']
                save(run / 'weights/best.pt', epoch, best, best_epoch, threshold)
                dump(run / 'selected.json', dict(epoch=epoch, score=best, threshold=threshold, by_threshold=val['by_threshold']))
        save(run / 'weights/last.pt', epoch, best, best_epoch, threshold)
        history.append(log)
        with (run / 'training_log.jsonl').open('a') as f:
            f.write(json.dumps(log, default=float) + '\n')
        print(json.dumps(log, default=float), flush=True)
        if epoch >= 16 and epoch - best_epoch >= args.patience:
            break
    dump(run / 'DONE.json', dict(name=args.run, score=best, best_epoch=best_epoch, threshold=threshold, epochs_completed=epoch,
                                 checkpoint=str(run / 'weights/best.pt'), checkpoint_sha256=digest(run / 'weights/best.pt')))


if __name__ == '__main__':
    main()
