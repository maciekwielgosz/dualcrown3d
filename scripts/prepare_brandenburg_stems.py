#!/usr/bin/env python3
"""Brandenburg tuning plots as StemField training data (measured stems, no point labels).

    .venv-gpu/bin/python scripts/prepare_brandenburg_stems.py [--out ../dualcrown3d_brandenburg_stems_v1]

Real leaf-off state ALS (``trunk_reconstruction/data/paired/brandenburg``) over 50 x 50 m TLS plots;
each scan covers the plot plus a 25 m buffer. Only the 16 tuning plots: the held-out plots of
``trunk_reconstruction/configs/real_holdout.json`` are refused. Per plot and scan variant (d14 seeds
0-2, d10 seed 0, native):

- ``points_0p25m.npz`` in the DualCrown3D layout (0.25 m voxels, heights above ground, intensity
  scaled to the 1st-99th percentile). ``tree_id``: 0 = ground, 1 = vegetation inside the plot box (every
  stem there is measured), -1 = vegetation in the buffer (stems unknown), so StemField's ignore mask
  covers the buffer. These ids are NOT instances; the rows are for the stem detector only.
- ``stems.csv``: every measured stem (x, y, h = TLS height, quality 1); ``ignore`` = 1 for understorey
  stems whose reference height is not their own (height > 150 x DBH, the protocol-v2 rule): they stand
  under the canopy, so the detector is neither rewarded nor punished there.

Folds: plots sorted by name and alternated into ``fold`` A and B (8 + 8), so a detector trained on one
fold can be tested end to end on the other.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.prepare_treescan_helios_dualcrown import sha256, voxelise, write_csv  # noqa: E402

TRUNK = PROJECT.parent / 'trunk_reconstruction'
BUFFER = 25.
SLENDER_MAX = 150.
VARIANTS = (('d14', 0), ('d14', 1), ('d14', 2), ('d10', 0), ('native', 0))
GROUND_HAG = .3


def convert(las_path, reference, target, voxel_size=.25):
    las = laspy.read(las_path)
    hag = np.asarray(las.height_agl, np.float64)
    xyz = np.column_stack((las.x, las.y, hag))
    ground = (np.asarray(las.classification) == 2) | (hag < GROUND_HAG)
    lo, hi = np.array([las.x.min(), las.y.min()]), np.array([las.x.max(), las.y.max()])
    box_lo, box_hi = lo + BUFFER, hi - BUFFER
    inside = ((xyz[:, :2] >= box_lo) & (xyz[:, :2] <= box_hi)).all(1)
    tree_id = np.where(ground, 0, np.where(inside, 1, -1)).astype(np.int64)
    finite = np.isfinite(xyz).all(1)
    vox, inten, tid, voxel_origin = voxelise(xyz[finite], np.asarray(las.intensity, np.float32)[finite], tree_id[finite],
                                             voxel_size, False)
    if inten.max() > 0:
        a, b = np.percentile(inten, (1, 99))
        inten = np.clip((inten - a) / max(float(b - a), 1.), 0, 1).astype(np.float32)
    xy_origin = np.floor(vox[:, :2].min(0) * 2) / 2
    source_origin = np.asarray([xy_origin[0], xy_origin[1], 0.], np.float64)
    coord = (vox - source_origin).astype(np.float32)
    grid = np.floor((vox - voxel_origin) / voxel_size).astype(np.int32)
    target.mkdir(parents=True, exist_ok=True)
    npz = target / 'points_0p25m.npz'
    np.savez_compressed(npz, coord=coord, grid_coord=grid, intensity=inten, tree_id=tid.astype(np.int32),
                        instance_offset=np.zeros_like(coord), source_origin=source_origin, voxel_origin=voxel_origin,
                        voxel_size=np.float32(voxel_size), raw_points=np.int64(len(xyz)))
    ref = reference[np.isfinite(reference[['x', 'y']]).all(1)]
    dbh_m = ref.dbh_ref_cm.values / 100.
    h = ref.h_ref_m.values
    understorey = np.isfinite(h) & np.isfinite(dbh_m) & (h > SLENDER_MAX * dbh_m)
    pd.DataFrame(dict(tree_id=ref.tree_id.values, x=ref.x.values, y=ref.y.values, h=np.nan_to_num(h),
                      quality=1., source='brandenburg_tls', ignore=understorey.astype(int))).to_csv(target / 'stems.csv', index=False)
    return npz, dict(voxels=len(coord), stems=int(len(ref)), understorey=int(understorey.sum()),
                     plot_box=[*box_lo.tolist(), *box_hi.tolist()])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=PROJECT.parent / 'dualcrown3d_brandenburg_stems_v1')
    args = ap.parse_args()
    held = set(json.loads((TRUNK / 'configs/real_holdout.json').read_text())['brandenburg'])
    src = TRUNK / 'data/paired/brandenburg'
    plots = sorted(p.name for p in src.iterdir() if p.is_dir() and p.name.startswith('BB_') and p.name not in held)
    folds = {p: 'AB'[i % 2] for i, p in enumerate(plots)}
    rows = []
    for plot in plots:
        if plot in held:
            raise SystemExit(f'{plot} is held out')
        reference = pd.read_csv(src / plot / 'reference.csv')
        for density, seed in VARIANTS:
            las = src / plot / f'{plot}_{density}_s{seed}.laz'
            if not las.exists():
                continue
            dataset_id = f'brandenburg__{plot}_{density}_s{seed}'
            npz, info = convert(las, reference, args.out / 'train' / dataset_id)
            rows.append(dict(dataset_id=dataset_id, source_dataset='BRANDENBURG_ALS', collection='BRANDENBURG',
                             model_split='train', parent_plot=plot, fold=folds[plot], density=density, seed=seed,
                             output=str(npz.resolve()), height_normalization='provided_height_agl', voxel_size=.25,
                             source_las=str(las.resolve()), npz_sha256=sha256(npz), **{k: v for k, v in info.items() if k != 'plot_box'},
                             plot_box=json.dumps(info['plot_box'])))
            print(dataset_id, info, flush=True)
    write_csv(args.out / 'manifest.csv', rows)
    (args.out / 'READY.json').write_text(json.dumps(dict(rows=len(rows), plots=plots, folds=folds, held_out_refused=sorted(held)),
                                                    indent=2) + '\n')


if __name__ == '__main__':
    main()
