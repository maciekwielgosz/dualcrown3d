#!/usr/bin/env python3
"""Tree instances from given stem feet with the stem-seeded decoder, for the paired test scans.

    .venv-gpu/bin/python scripts/predict_stem_seeded.py --checkpoint outputs/stem_seeded_v1/runs/ss_v1/weights/best.pt \
        --manifest ../trunk_reconstruction/data/paired/weiser/manifest.csv --densities d14 \
        --stems reference --out ../trunk_reconstruction/data/paired/weiser_stemseed_ref

For every scan of a ``trunk_reconstruction`` paired manifest (LAZ with ``height_agl``): voxelise at
0.25 m, decode one mask per stem foot, assign every point to the stem of highest probability, and
write ``<out>/<scan>/trees_<scan>.laz`` (``tree_id`` = stem index + 1, ``height_agl``) plus the
scan's ``reference.csv`` and a ``<out>/manifest.csv``, the layout ``tools/run_dualcrown3d.py``
produces, so ``tools/evaluate_paired.py`` scores it unchanged. ``stems_<scan>.csv`` lists every
foot with its object score and point count.

Stem sources (``--stems``): ``reference`` = the measured feet of ``reference.csv`` (the oracle
ceiling of the assignment); ``<dir>`` = ``<dir>/<scan>.csv`` with ``x, y, h[, score]`` in the
scan's CRS (a detector's or GraphStem's feet).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.dual_head import DualHeadLitePT  # noqa: E402
from pointcloud.joint_training import fixed_serialization  # noqa: E402
from pointcloud.paths import remap  # noqa: E402
from pointcloud.stem_seeded import predict_stems  # noqa: E402
from scripts.prepare_treescan_helios_dualcrown import voxelise  # noqa: E402

GROUND_HAG = .3
VOXEL = .25
COPY_DIMS = ('X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns', 'classification', 'scan_angle',
             'point_source_id', 'gps_time')


def load_model(checkpoint, device):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    kwargs = dict(payload['model_args']); kwargs.pop('decoder_policy', None)
    model = DualHeadLitePT(decoder_policy='stem_seeded', **kwargs)
    model.load_state_dict(payload['model'], strict=True)
    fixed_serialization(model)
    return model.to(device).eval(), payload


def scan_arrays(las):
    """DualCrown3D voxel arrays of one paired scan and the voxel index of every point."""
    hag = np.asarray(las.height_agl, np.float64)
    xyz = np.column_stack((las.x, las.y, hag))
    ground = (np.asarray(las.classification) == 2) | (hag < GROUND_HAG)
    intensity = np.asarray(las.intensity, np.float32)
    finite = np.isfinite(xyz).all(1)
    tree_id = np.where(ground, 0, -1).astype(np.int64)
    vox, inten, tid, voxel_origin = voxelise(xyz[finite], intensity[finite], tree_id[finite], VOXEL, False)
    if inten.max() > 0:
        lo, hi = np.percentile(inten, (1, 99))
        inten = np.clip((inten - lo) / max(float(hi - lo), 1.), 0, 1).astype(np.float32)
    xy_origin = np.floor(vox[:, :2].min(0) * 2) / 2
    source_origin = np.asarray([xy_origin[0], xy_origin[1], 0.], np.float64)
    grid = np.floor((vox - voxel_origin) / VOXEL).astype(np.int32)
    arrays = dict(coord=(vox - source_origin).astype(np.float32), grid_coord=grid, intensity=inten,
                  source_origin=source_origin, voxel_origin=voxel_origin, voxel_size=np.float32(VOXEL))
    # every original point -> its voxel row
    grid_all = np.floor((xyz - voxel_origin) / VOXEL).astype(np.int64)
    extent = np.maximum(grid.max(0), grid_all.max(0)).astype(np.int64) + 1
    encode = lambda g: g[:, 0] + extent[0] * (g[:, 1] + extent[1] * g[:, 2])
    keys = encode(grid.astype(np.int64)); order = np.argsort(keys)
    wanted = encode(grid_all)
    position = np.searchsorted(keys[order], wanted)
    good = position < len(order)
    good[good] &= keys[order[position[good]]] == wanted[good]
    lookup = np.where(good, order[np.minimum(position, len(order) - 1)], -1)
    return arrays, lookup, ground


def stems_of(scan_path, reference, source):
    if source == 'reference':
        h = reference['h_ref_m'] if 'h_ref_m' in reference else reference.get('h_als_m', pd.Series(np.zeros(len(reference))))
        frame = pd.DataFrame(dict(x=reference.x, y=reference.y, h=pd.to_numeric(h, errors='coerce').fillna(0.), score=1.))
    else:
        frame = pd.read_csv(Path(source) / f"{scan_path.stem.removeprefix('trees_')}.csv")
        if 'score' not in frame:
            frame['score'] = 1.
        if 'h' not in frame:
            frame['h'] = frame['h_tree'] if 'h_tree' in frame else 0.
    frame = frame[np.isfinite(frame[['x', 'y']]).all(1)].reset_index(drop=True)
    return frame


def write_laz(las, path, tree_id, ground):
    header = laspy.LasHeader(point_format=las.header.point_format.id, version='1.4')
    header.offsets, header.scales = las.header.offsets, las.header.scales
    out = laspy.LasData(header)
    for dim in COPY_DIMS:
        if dim in set(las.point_format.dimension_names):
            out[dim] = np.asarray(las[dim])
    out.add_extra_dim(laspy.ExtraBytesParams(name='tree_id', type=np.uint32))
    out.add_extra_dim(laspy.ExtraBytesParams(name='height_agl', type=np.float32))
    ids = tree_id.astype(np.uint32); ids[ground] = 0
    out.tree_id = ids
    out.height_agl = np.asarray(las.height_agl, np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write(path)


def run_scan(model, scan_path, reference, stems_source, out_dir, device, mask_threshold, object_threshold, max_points):
    t0 = time.monotonic()
    las = laspy.read(scan_path)
    arrays, lookup, ground = scan_arrays(las)
    stems = stems_of(scan_path, reference, stems_source)
    world = stems[['x', 'y', 'h']].values.astype(np.float64)
    labels, score, probability = predict_stems(model, arrays, world, stem_score=stems.score.values.astype(np.float32),
                                               max_points=max_points, mask_threshold=mask_threshold,
                                               object_threshold=object_threshold, device=device)
    point_labels = np.where(lookup >= 0, labels[np.maximum(lookup, 0)], 0)
    name = scan_path.stem.removeprefix('trees_')
    write_laz(las, out_dir / f'trees_{name}.laz', point_labels, ground)
    counts = np.bincount(point_labels, minlength=len(stems) + 1)[1:]
    stems_out = stems.assign(stem_index=np.arange(len(stems)), tree_id=np.arange(1, len(stems) + 1),
                             object_score=score, points=counts)
    stems_out.to_csv(out_dir / f'stems_{name}.csv', index=False)
    summary = dict(scan=name, points=int(len(las.points)), voxels=int(len(arrays['coord'])), stems=int(len(stems)),
                   stems_with_points=int((counts > 0).sum()), labelled_points=int((point_labels > 0).sum()),
                   canopy_points_labelled=float((point_labels[~ground] > 0).mean()) if (~ground).any() else 0.,
                   seconds=time.monotonic() - t0)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True, help='trunk_reconstruction paired manifest (path, plot, density, seed)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--densities', nargs='+', default=['d14'])
    ap.add_argument('--seeds', nargs='*', type=int, default=None)
    ap.add_argument('--plots', nargs='*', default=None)
    ap.add_argument('--stems', default='reference', help="'reference' or a directory of <scan>.csv feet")
    ap.add_argument('--mask-threshold', type=float, default=.5)
    ap.add_argument('--object-threshold', type=float, default=0., help='drop feet whose object score is below this')
    ap.add_argument('--max-points', type=int, default=40000)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    model, payload = load_model(args.checkpoint, args.device)
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest.density.isin(args.densities)]
    if args.seeds is not None:
        manifest = manifest[manifest.seed.isin(args.seeds)]
    if args.plots:
        manifest = manifest[manifest['plot'].isin(args.plots)]
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    rows, summaries = [], []
    for r in manifest.itertuples():
        scan = Path(remap(r.path)) if Path(r.path).is_absolute() else (args.manifest.parent / r.path).resolve()
        if not scan.exists():
            scan = (args.manifest.parents[2] / r.path).resolve()
        name = scan.stem.removeprefix('trees_')
        d = out / name
        final = d / f'trees_{name}.laz'
        reference = pd.read_csv(scan.parent / 'reference.csv')
        if not final.exists():
            summaries.append(run_scan(model, scan, reference, args.stems, d, args.device, args.mask_threshold,
                                      args.object_threshold, args.max_points))
            print(json.dumps(summaries[-1]), flush=True)
        shutil.copy(scan.parent / 'reference.csv', d / 'reference.csv')
        rows.append({**r._asdict(), 'path': str(final)})
    pd.DataFrame(rows).drop(columns='Index').to_csv(out / 'manifest.csv', index=False)
    (out / 'run.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint.resolve()), epoch=payload.get('epoch'),
                                                  stems=args.stems, densities=args.densities, mask_threshold=args.mask_threshold,
                                                  object_threshold=args.object_threshold, scans=summaries), indent=2) + '\n')
    print(f"wrote {out / 'manifest.csv'} ({len(rows)} scans)")


if __name__ == '__main__':
    main()
