#!/usr/bin/env python3
"""Scaled stage-1 fine-tune of Model20 for sharper centre votes.

Starts from the production checkpoint and trains with a much larger budget than
the 6,400 crops of the joint campaign. Everything that differs from that recipe is
a switch: the trainable part (``partial`` = enc4 + decoder stages + heads as
Model20, ``full`` = the whole LitePT backbone), learning rates with warm-up and
cosine decay, crops per run, voxels per crop, accumulation, mixed precision for
the backbone, vertical supervision of the votes, a vote-assignment loss and strong
density thinning. Checkpoints are selected on the clustering-free vote-quality
validation of ``scripts/validate_stage1_votes.py`` (single-pass votes; dense and
thinned real validation plots). The test splits are never read.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import move_to_device, read_manifest
from pointcloud.dual_head import DualHeadLitePT
from pointcloud.joint_training import fixed_serialization, sampling_weights
from pointcloud.paths import remap
from pointcloud.stage1_scaled import ScaledDrawDataset, forward_stage1, scaled_losses

WORKSPACE = PROJECT.parent
DEFAULT_ROOT = PROJECT / 'outputs/dualcrown3d_stage1_scaled_v1'
INITIAL = PROJECT / 'outputs/dualcrown3d_joint_campaign_v1/runs/repeat_20260930/weights/best.pt'
REAL = {'v4': WORKSPACE / 'combined_als_crowns_supervision_v4/manifest.csv',
        'v2': WORKSPACE / 'combined_als_crowns_no_rectangles_v2/manifest.csv'}
HELIOS = WORKSPACE / 'dualcrown3d_treescan_helios_v1/manifest.csv'
AUGMENT = WORKSPACE / 'TreeScanPL10k_HELIOS_ALS_flight_augmentation_v2/manifest.csv'
DIFFICULTY = PROJECT / 'outputs/dualcrown3d_joint_campaign_v1/training_difficulty.json'
IDENTITY = ('dataset_id', 'group_id', 'source_sha256', 'npz_sha256', 'coordinates_sha256', 'original_las')
INPUT_KEYS = ('coord', 'grid_coord', 'feat', 'offset', 'tree_id')


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    os.replace(temporary, path)


def digest(path):
    with Path(remap(path)).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def all_rows(path):
    with Path(remap(path)).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    temporary = Path(path).with_suffix('.tmp.csv')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def training_rows(real_source):
    """Training rows exactly as the joint campaign builds them (plus the v4 eligibility filter)."""
    real = read_manifest(REAL[real_source], 'train')
    if real_source == 'v4':
        real = [r for r in real if r['train_eligible'] == 'true']
    helios = read_manifest(HELIOS, 'train')
    readiness = json.loads(Path(remap(AUGMENT.with_name('READY.json'))).read_text())
    if readiness['simulations'] != 86:
        raise ValueError('The complete 86-flight augmentation is required')
    extra = read_manifest(AUGMENT, 'train')
    parents = {r['dataset_id'].removeprefix('treescan_helios__') for r in helios}
    if any(r['parent_plot'] not in parents for r in extra):
        raise ValueError('Augmentation split leakage')
    return real + helios + extra


def leakage_check(rows, verify_hashes=True):
    """Prove that no training plot is a validation or test plot of the real or simulated manifests.

    Compares every identity field (dataset id, spatial group, source-file and voxel-file
    hashes) of the training rows with all non-train rows of both real manifests and of
    the HELIOS manifest, checks that every training file lives outside ``val``/``test``
    folders and that its content hash equals the manifest's. Only manifests and training
    files are read; no held-out data file is opened.
    """
    held = {}
    for name, path in (('real_v4', REAL['v4']), ('real_v2', REAL['v2']), ('helios', HELIOS)):
        if Path(remap(path)).exists():
            held[name] = [r for r in all_rows(path) if r['model_split'] != 'train']
    if 'real_v4' not in held or 'helios' not in held:
        raise FileNotFoundError('The v4 and HELIOS manifests are required for the leakage check')
    eligible = {s: [r for r in held['real_v4'] if r['model_split'] == s and r.get('point_eval_eligible') == 'true']
                for s in ('val', 'test')}
    if (len(eligible['val']), len(eligible['test'])) != (14, 24):
        raise AssertionError('Expected 14 validation and 24 test plots in the v4 manifest')
    report = dict(training_rows=len(rows), held_out_rows={k: dict(Counter(r['model_split'] for r in v)) for k, v in held.items()},
                  v4_eligible_validation_plots=14, v4_eligible_test_plots=24, collisions={}, fields=list(IDENTITY))
    if any(r['model_split'] != 'train' for r in rows):
        raise AssertionError('A training row is not in the train split')
    for name, others in held.items():
        for field in IDENTITY:
            values = {r[field] for r in others if r.get(field)}
            hits = sorted({r['dataset_id'] for r in rows if r.get(field) and r[field] in values})
            report['collisions'][f'{name}:{field}'] = hits
    held_parents = {r['dataset_id'].removeprefix('treescan_helios__') for r in held['helios']}
    report['collisions']['helios:parent_plot'] = sorted({r['dataset_id'] for r in rows if r.get('parent_plot') in held_parents})
    bad_paths = [r['dataset_id'] for r in rows if any(part in ('val', 'test') for part in Path(r['output']).parts)]
    report['collisions']['path_in_val_or_test_folder'] = bad_paths
    mismatched, content_hits = [], []
    if verify_hashes:
        # Content of the training files themselves: the stored coordinates must hash to the manifest's
        # coordinate hash (real plots; the v4 files changed labels only, so their file hash differs from
        # the inherited ``npz_sha256``) or the file to ``npz_sha256`` (simulations), and must not equal the
        # coordinates of any held-out plot.
        held_coordinates = {r['coordinates_sha256'] for others in held.values() for r in others if r.get('coordinates_sha256')}
        for r in rows:
            if r.get('coordinates_sha256'):
                with np.load(r['output']) as archive:
                    actual = hashlib.sha256(archive['coord'].tobytes()).hexdigest()
                if actual != r['coordinates_sha256']:
                    mismatched.append(r['dataset_id'])
                if actual in held_coordinates:
                    content_hits.append(r['dataset_id'])
            elif r.get('npz_sha256') and digest(r['output']) != r['npz_sha256']:
                mismatched.append(r['dataset_id'])
    report['collisions']['file_content_equals_held_out_coordinates'] = content_hits
    report['training_file_hash_mismatches'] = mismatched
    report['training_file_hashes_verified'] = bool(verify_hashes)
    report['passed'] = not any(report['collisions'].values()) and not mismatched
    return report


def build(payload):
    model = DualHeadLitePT(**payload['model_args'])
    model.load_state_dict(payload['model'], strict=True)
    fixed_serialization(model)
    return model


def configure(model, scope):
    """``partial``: Model20's trainable set. ``full``: additionally the whole LitePT backbone
    (batch-norm statistics stay frozen in both, as ``DualHeadLitePT.train`` keeps them)."""
    model.configure_training('partial')
    if scope == 'full':
        model.legacy.backbone.requires_grad_(True)
    elif scope != 'partial':
        raise ValueError(scope)
    fixed_serialization(model)


class EpochSampler:
    def __init__(self):
        self.draws = []

    def __iter__(self):
        return iter(self.draws)

    def __len__(self):
        return len(self.draws)


def schedule(step, warmup, total, floor):
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1., (step - warmup) / max(total - warmup, 1))
    return floor + (1 - floor) * .5 * (1 + np.cos(np.pi * progress))


def benchmark(model, dataset, rows, args, device):
    """Speed of float32 against mixed precision, and how far the votes of the two differ."""
    report = {}
    rng = np.random.default_rng(1)
    real = [i for i, r in enumerate(rows) if r['source_dataset'] != 'TreeScanPL10k_HELIOS']
    batches = [move_to_device(dataset[(int(i), n, 0)], device) for n, i in enumerate(rng.choice(real, 12))]
    report['voxels'] = [int(len(b['coord'])) for b in batches]
    parameters = [p for p in model.parameters() if p.requires_grad]
    def timed(amp):
        model.train()
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
        optimizer = torch.optim.SGD(parameters, lr=0.)          # lets the scaler adapt without changing the weights
        norms, skipped = [], 0
        for number, batch in enumerate(batches[:1] + batches + batches):       # the first pass only warms up
            if number == 1:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
            losses = scaled_losses(forward_stage1(model, {k: batch[k] for k in INPUT_KEYS}, amp), batch,
                                   args.z_weight, args.assign_weight, args.assign_temperature)
            scaler.scale(losses['loss']).backward()
            scaler.unscale_(optimizer)
            norm = float(torch.sqrt(sum(p.grad.float().square().sum() for p in parameters if p.grad is not None)))
            if number:
                skipped += not np.isfinite(norm)
                norms.append(norm if np.isfinite(norm) else None)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return dict(seconds_per_crop=(time.monotonic() - start) / (2 * len(batches)), loss=float(losses['loss'].detach()),
                    gradient_norms=norms, nonfinite_steps=int(skipped), steps=len(norms), final_scale=float(scaler.get_scale()),
                    peak_vram_mb=torch.cuda.max_memory_allocated() / 1024 ** 2)

    report['fp32'] = timed(False)
    try:
        report['amp'] = timed(True)
        model.eval()
        differences = []
        with torch.no_grad():
            for batch in batches:
                data = {k: batch[k] for k in INPUT_KEYS[:4]}
                a, b = forward_stage1(model, dict(data), False), forward_stage1(model, dict(data), True)
                delta = (a['offset_m'] - b['offset_m']).norm(dim=1)
                differences.append(dict(vote_difference_median_m=float(delta.median()), vote_difference_max_m=float(delta.max()),
                                        probability_difference_max=float((a['semantic_logits'].softmax(1) - b['semantic_logits'].softmax(1)).abs().max())))
        report['fp32_vs_amp_inference'] = differences
        report['gradient_norm_ratio_amp_over_fp32'] = [a / max(b, 1e-12) if a is not None and b is not None else None
                                                       for a, b in zip(report['amp']['gradient_norms'], report['fp32']['gradient_norms'])]
    except Exception as error:      # recorded, not hidden: mixed precision is then unavailable
        model.zero_grad(set_to_none=True)
        report['amp'] = dict(error=f'{type(error).__name__}: {str(error)[-400:]}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--run', required=True)
    parser.add_argument('--initial', type=Path, default=INITIAL)
    parser.add_argument('--real-source', choices=sorted(REAL), default='v4',
                        help='v4: repaired annotation, 79 native 3-D plots; v2: the manifest Model20 was trained on')
    parser.add_argument('--scope', choices=('partial', 'full'), default='partial')
    parser.add_argument('--lr-heads', type=float, default=1e-4)
    parser.add_argument('--lr-backbone', type=float, default=2e-5)
    parser.add_argument('--lr-floor', type=float, default=.02, help='final learning rate as a fraction of the peak')
    parser.add_argument('--warmup-steps', type=int, default=500, help='optimizer steps of linear warm-up')
    parser.add_argument('--weight-decay', type=float, default=.02)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--samples', type=int, default=8000, help='crops per epoch')
    parser.add_argument('--max-points', type=int, default=16000, help='voxels per crop (inference uses 40000)')
    parser.add_argument('--accumulation', type=int, default=2, help='crops per optimizer step')
    parser.add_argument('--clip', type=float, default=2.)
    parser.add_argument('--amp', action='store_true', help='float16 autocast for the backbone')
    parser.add_argument('--z-weight', type=float, default=0., help='weight of the vertical offset loss (0: XY only, as Model20)')
    parser.add_argument('--assign-weight', type=float, default=0., help='weight of the vote-assignment cross-entropy')
    parser.add_argument('--assign-temperature', type=float, default=.5, help='metres; softness of the vote-assignment cross-entropy')
    parser.add_argument('--thin-probability', type=float, default=0., help='share of real crops thinned to --thin-range')
    parser.add_argument('--thin-range', type=float, nargs=2, default=(6., 25.), help='voxels/m2, log-uniform')
    parser.add_argument('--real-fraction', type=float, default=.5)
    parser.add_argument('--no-hard', action='store_true', help='disable hard-crop sampling')
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--val-every', type=int, default=4)
    parser.add_argument('--val-sets', nargs='+', default=['dense', 'thin10', 'helios'])
    parser.add_argument('--val-jobs', type=int, default=14)
    parser.add_argument('--time-limit-hours', type=float, default=0., help='stop cleanly after this wall time (0: none)')
    parser.add_argument('--resume', action='store_true', help='continue an interrupted run from weights/last.pt')
    parser.add_argument('--check-only', action='store_true', help='run the leakage check and exit')
    parser.add_argument('--benchmark', action='store_true', help='time float32 against mixed precision and exit')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = training_rows(args.real_source)
    if args.check_only:
        report = leakage_check(rows)
        dump(root / f'leakage_check_{args.real_source}.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'collisions'}, indent=2))
        print('collisions:', {k: v for k, v in report['collisions'].items() if v} or 'none')
        if not report['passed']:
            raise SystemExit('LEAKAGE CHECK FAILED')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    device = torch.device('cuda:0')
    if args.smoke:
        args.epochs, args.samples, args.val_every, args.run = 2, 48, 1, args.run + '_smoke'
    run = root / 'runs' / args.run
    initial = Path(remap(args.initial.resolve()))
    payload = torch.load(initial, map_location='cpu', weights_only=False)
    payload.setdefault('model_args', dict(queries=96, decoder_layers=3, memory_tokens=1024))
    difficulty = json.loads(Path(remap(DIFFICULTY)).read_text())
    dataset = ScaledDrawDataset(rows, args.seed, args.max_points, hard=not args.no_hard, difficulty=difficulty,
                                thin_probability=args.thin_probability, thin_range=args.thin_range)
    model = build(payload).to(device)
    configure(model, args.scope)
    if args.benchmark:
        report = benchmark(model, dataset, rows, args, device)
        dump(root / f'benchmark_{args.run}.json', report)
        print(json.dumps(report, indent=2))
        return
    resume = None
    if (run / 'DONE.json').exists():
        print((run / 'DONE.json').read_text())
        return
    if run.exists():
        if not (args.resume and (run / 'weights/last.pt').exists()) and not args.smoke:
            raise FileExistsError(f'Incomplete run protected: {run}; pass --resume to continue from last.pt')
        if args.resume and (run / 'weights/last.pt').exists():
            resume = torch.load(run / 'weights/last.pt', map_location='cpu', weights_only=False)
    (run / 'weights').mkdir(parents=True, exist_ok=True)
    random_state = args.seed
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

    leakage = leakage_check(rows, verify_hashes=resume is None)
    dump(run / 'leakage_check.json', leakage)
    if not leakage['passed']:
        raise AssertionError('Leakage check failed; see leakage_check.json')
    write_csv(run / 'training_manifest.csv', rows)
    weights = sampling_weights(rows, args.real_fraction)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    parameters = [p for p in model.parameters() if p.requires_grad]
    backbone = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('legacy.backbone.')]
    heads = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('legacy.backbone.')]
    optimizer = torch.optim.AdamW([dict(params=heads, lr=args.lr_heads, name='heads'),
                                   dict(params=backbone, lr=args.lr_backbone, name='backbone')], weight_decay=args.weight_decay)
    steps_per_epoch = int(np.ceil(args.samples / args.accumulation))
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: schedule(s, args.warmup_steps, total_steps, args.lr_floor))
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    config = dict(
        name=args.run, initial_checkpoint=str(initial), initial_sha256=digest(initial), real_source=args.real_source,
        scope=args.scope, lr_heads=args.lr_heads, lr_backbone=args.lr_backbone, lr_floor=args.lr_floor,
        warmup_steps=args.warmup_steps, weight_decay=args.weight_decay, schedule='linear warm-up, cosine per optimizer step',
        epochs_max=args.epochs, samples_per_epoch=args.samples, crops_total=args.epochs * args.samples,
        max_points=args.max_points, accumulation=args.accumulation, clip=args.clip, amp_backbone=args.amp,
        z_weight=args.z_weight, assign_weight=args.assign_weight, assign_temperature=args.assign_temperature,
        thin_probability=args.thin_probability,
        thin_range=list(args.thin_range), keep_fractions=[.5, .75, 1.], real_fraction=args.real_fraction,
        hard=not args.no_hard, augment=True, seed=args.seed, model_args=payload['model_args'],
        training_rows=len(rows), real_rows=sum(r['source_dataset'] != 'TreeScanPL10k_HELIOS' for r in rows),
        manifest_sha256=digest(run / 'training_manifest.csv'),
        trainable_parameters=sum(p.numel() for p in parameters), total_parameters=sum(p.numel() for p in model.parameters()),
        offset_loss='smooth-L1 (beta 0.5), mean per tree then over trees; XY' + (f' + {args.z_weight} x Z' if args.z_weight else ' only'),
        serialization='fixed at every pooling stage', batch_norm='statistics frozen',
        selection='vote-quality validation, single-pass votes: mean over dense and thin10 of sqrt(point x crown oracle SB-PQ)',
        validation_sets=args.val_sets, test_used_for_selection=False, command=' '.join(sys.argv),
        slurm_job=os.environ.get('SLURM_JOB_ID'), torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    if resume is None:
        dump(run / 'configuration.json', config)
    else:
        dump(run / f'configuration_resume_{os.environ.get("SLURM_JOB_ID", int(time.time()))}.json', config)

    from scripts.validate_stage1_votes import evaluate_checkpoint, load_plots, selection_score
    loaded = {name: load_plots(name, 2 if args.smoke else None) for name in args.val_sets}

    def validate(epoch):
        model.eval()
        fixed_serialization(model)
        with torch.random.fork_rng(devices=[0]):
            summaries, per_plot, seconds = evaluate_checkpoint(model, args.val_sets, ('single',), args.val_jobs, loaded=loaded)
        dump(run / f'validation/epoch_{epoch:03d}.json', dict(epoch=epoch, summaries=summaries, per_plot=per_plot, seconds=seconds))
        record = dict(val_score=selection_score(summaries), val_seconds=sum(seconds.values()))
        for name, value in summaries.items():
            single = value['single']
            record.update({f'{name}_oracle_point': single['oracle']['point_sb_pq'], f'{name}_oracle_crown': single['oracle']['crown_sb_pq'],
                           f'{name}_centroid_oracle_point': single['oracle_centroid']['point_sb_pq'],
                           f'{name}_centroid_oracle_crown': single['oracle_centroid']['crown_sb_pq'],
                           f'{name}_tree_xy': single['tree_horizontal']['median'], f'{name}_tree_z': single['tree_vertical']['median'],
                           f'{name}_voxel_xy': single['voxel_horizontal']['median'], f'{name}_voxel_z': single['voxel_vertical']['median'],
                           f'{name}_semantic_accuracy': single['semantic']['accuracy'],
                           f'{name}_purity': single['nearest_centre_purity']['voxel']})
        return record

    def save(path, epoch, score, full=False):
        state = dict(model=model.state_dict(), model_args=payload['model_args'], epoch=epoch, score=score,
                     format='dualcrown3d_joint_v1',
                     config={**config, 'legacy_cluster_config': payload.get('config', {}).get('legacy_cluster_config')})
        if full:
            state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                         history=history, best=best, best_epoch=best_epoch)
        temporary = path.with_suffix('.tmp.pt')
        torch.save(state, temporary)
        os.replace(temporary, path)

    started = time.monotonic()
    history, best, best_epoch, first = [], None, 0, 1
    if resume is not None:
        model.load_state_dict(resume['model'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        scaler.load_state_dict(resume['scaler'])
        history, best, best_epoch, first = resume['history'], resume['best'], resume['best_epoch'], resume['epoch'] + 1
        print(f'resumed after epoch {resume["epoch"]}', flush=True)
    else:
        record = dict(epoch=0, **validate(0))
        best = record['val_score']
        history.append(record)
        save(run / 'weights/best.pt', 0, best)
        dump(run / 'selected.json', dict(epoch=0, score=best, record=record, test_used_for_selection=False))
        print(json.dumps(record), flush=True)
    sampler = EpochSampler()
    loader = DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    stopped = None
    for epoch in range(first, args.epochs + 1):
        configure(model, args.scope)
        model.train()
        model.point_decoder.epoch = 100 + epoch
        rng = np.random.default_rng(args.seed + epoch * 100003)
        sampler.draws = [(int(i), step, epoch) for step, i in enumerate(rng.choice(len(rows), args.samples, p=weights / weights.sum()))]
        totals, voxels = Counter(), 0
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        epoch_start = time.monotonic()
        for step, batch in enumerate(loader, 1):
            batch = move_to_device(batch, device)
            prediction = forward_stage1(model, {k: batch[k] for k in INPUT_KEYS}, args.amp)
            losses = scaled_losses(prediction, batch, args.z_weight, args.assign_weight, args.assign_temperature)
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch}, step {step}')
            scaler.scale(losses['loss'] / args.accumulation).backward()
            if step % args.accumulation == 0 or step == len(sampler):
                scaler.unscale_(optimizer)
                # Under mixed precision a rare overflowing step is skipped by the scaler instead of aborting.
                torch.nn.utils.clip_grad_norm_(parameters, args.clip, error_if_nonfinite=not args.amp)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            totals.update({k: float(v.detach()) for k, v in losses.items()})
            voxels += len(batch['coord'])
        record = dict(epoch=epoch, crops=epoch * args.samples, seconds=time.monotonic() - epoch_start,
                      lr_heads=optimizer.param_groups[0]['lr'], lr_backbone=optimizer.param_groups[1]['lr'],
                      mean_voxels=voxels / args.samples, peak_vram_mb=torch.cuda.max_memory_allocated() / 1024 ** 2,
                      **{f'train_{k}': v / args.samples for k, v in totals.items()})
        elapsed = (time.monotonic() - started) / 3600
        out_of_time = bool(args.time_limit_hours) and elapsed + 1.5 * record['seconds'] / 3600 > args.time_limit_hours
        if epoch % args.val_every == 0 or epoch == args.epochs or out_of_time:
            record.update(validate(epoch))
            save(run / f'weights/epoch_{epoch:03d}.pt', epoch, record['val_score'])
            if record['val_score'] > best:
                best, best_epoch = record['val_score'], epoch
                save(run / 'weights/best.pt', epoch, best)
                dump(run / 'selected.json', dict(epoch=epoch, score=best, record=record, test_used_for_selection=False))
        history.append(record)
        save(run / 'weights/last.pt', epoch, best, full=True)
        write_csv(run / 'training_log.csv', history)
        print(json.dumps(record), flush=True)
        if out_of_time and epoch < args.epochs:
            stopped = f'time limit of {args.time_limit_hours} h reached after epoch {epoch}'
            dump(run / 'STOPPED.json', dict(reason=stopped, epoch=epoch, resume='rerun the same command with --resume'))
            print(stopped, flush=True)
            break
    final = torch.load(run / 'weights/last.pt', map_location='cpu', weights_only=False)['model']
    changed = [n for n, p in final.items() if not torch.equal(p, payload['model'][n])]
    illegal = [n for n in changed if n not in trainable]
    if illegal:
        raise AssertionError(f'Frozen parameters or buffers changed: {illegal[:8]}')
    if not args.smoke and stopped is None:
        dump(run / 'DONE.json', dict(
            name=args.run, epochs_completed=epoch, crops=epoch * args.samples, best_epoch=best_epoch, best_score=best,
            seconds=time.monotonic() - started, checkpoint=str(run / 'weights/best.pt'),
            checkpoint_sha256=digest(run / 'weights/best.pt'), last_sha256=digest(run / 'weights/last.pt'),
            changed_tensors=len(changed), frozen_parameters_unchanged=True, test_used_for_selection=False))
    print(json.dumps(dict(run=args.run, best_epoch=best_epoch, best_score=best, stopped=stopped)), flush=True)


if __name__ == '__main__':
    main()
