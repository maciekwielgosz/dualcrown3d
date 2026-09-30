#!/usr/bin/env python3
"""Time repeated full-area DualCrown3D inference without re-exporting files."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.dual_head import DualHeadLitePT
from scripts.predict_dual_head import predict


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'output_20_dualcrown3d_joint_finetune')
    parser.add_argument('--prepared-dir', type=Path, default=PROJECT / 'output_15_litept_v2_no_rectangles_pointcloud/work')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.runs < 2:
        parser.error('At least two measured runs are required')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for this benchmark')

    output = args.output_dir.resolve()
    final = output / f'inference_timing_{args.runs}runs.json'
    progress = output / f'inference_timing_{args.runs}runs.progress.json'
    if final.exists() or progress.exists():
        raise FileExistsError(f'Timing output already exists: {final} or {progress}')
    original = json.loads((output / 'inference_report.json').read_text())
    checkpoint_path = Path(original['checkpoint'])
    selection_path = Path(original['selection'])
    selected = json.loads(selection_path.read_text())
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if digest != original['checkpoint_sha256'] or digest != selected['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the frozen output_20 selection')
    with np.load(args.prepared_dir / 'benchmark_pointcloud.npz') as data:
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads((args.prepared_dir / 'preparation.json').read_text())
    area_ha = sum((tile['bounds'][2] - tile['bounds'][0]) *
                  (tile['bounds'][3] - tile['bounds'][1]) for tile in metadata['tiles']) / 10000
    if area_ha <= 0:
        raise ValueError('Nonpositive output tile area')
    torch.set_num_threads(4)
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = DualHeadLitePT(**payload.get('model_args', {})).to(args.device)
    model.load_state_dict(payload['model'], strict=True)
    for module in model.modules():
        if hasattr(module, 'shuffle_orders'):
            module.shuffle_orders = False
    model.eval()

    measurements = []
    header = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=digest,
                  selection=str(selection_path), prepared_dir=str(args.prepared_dir.resolve()),
                  output_dir=str(output), device=args.device, runs=args.runs,
                  area_ha=area_ha, voxel_count=len(arrays['coord']),
                  original_inference_seconds=original['inference_seconds'],
                  protocol='Full prepared four-tile cloud; each run repeats predict() with one loaded model. '
                           'Timing includes window preparation, network inference and raw mask extraction; '
                           'excludes loading, mask merging, polygon creation and LAZ export.')
    for number in range(1, args.runs + 1):
        gc.collect()
        torch.cuda.synchronize()
        started = time.perf_counter()
        raw = predict(model, arrays, owner_only=selected['config'].get('owner_only', True))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        windows = int(raw['windows'])
        if windows != original['windows']:
            raise AssertionError(f'Changed window count: {windows} != {original["windows"]}')
        row = dict(run=number, wall_seconds=elapsed, seconds_per_hectare=elapsed / area_ha,
                   internal_predict_seconds=float(raw['seconds']), windows=windows,
                   candidate_masks=len(raw['object_score']))
        measurements.append(row)
        write_json(progress, {**header, 'measurements': measurements})
        print(json.dumps(row), flush=True)
        del raw

    durations = [row['wall_seconds'] for row in measurements]
    summary = dict(mean_seconds=statistics.mean(durations),
                   std_seconds=statistics.stdev(durations),
                   median_seconds=statistics.median(durations),
                   min_seconds=min(durations), max_seconds=max(durations),
                   mean_seconds_per_hectare=statistics.mean(durations) / area_ha,
                   std_seconds_per_hectare=statistics.stdev(durations) / area_ha)
    write_json(final, {**header, 'measurements': measurements, 'summary': summary})
    progress.unlink()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
