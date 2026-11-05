#!/usr/bin/env python3
"""Stem feet per tree for every DualCrown3D plot (``stems.csv`` next to ``points_0p25m.npz``).

    .venv-gpu/bin/python scripts/prepare_stem_feet.py [--manifests ...] [--check]

Sources, by dataset:

- TreeScanPL10k HELIOS plots: the TLS-traced trunks of ``trunk_reconstruction/data/labels/<plot>/trunks.csv``
  (same local frame and tree ids as the HELIOS scans; quality from the label, foot = ``x, y``);
  flight-augmentation variants use their parent plot's labels.
- Weiser real scans: ``trunk_reconstruction/data/labels_weiser/<plot>/trunks.csv`` (EPSG:25832,
  measured or TLS-traced feet; quality 1).
- Weiser simulated tiles: the plot's ``labels_weiser/<plot>/trunks.csv`` shifted into the tile frame
  (scene origin from ``coverage.npz`` plus the offsets in the window name), so edge trees whose foot
  lies outside the window keep a foot; fallback ``labels_weiser_tiles[_v3]/<window>/trunks.csv``.
- FOR-instance / IDEAS-ALS plots (no stem labels): proxy foot = mean xy of the tree's points in its
  lowest metre above 0.5 m when at least 5 exist, else of the points below 0.3 h, else the crown
  centroid; quality 0.5 / 0.35 / 0.2 so training can exclude them (``--min-quality``).

Columns: ``tree_id, x, y, h, quality, source``; x, y in the frame of ``coord + source_origin``.
``--check`` only reports, for every plot, the share of labelled trees with a foot and the median
distance between the foot and the lowest-points proxy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.data import load_npz, read_manifest  # noqa: E402
from pointcloud.paths import remap  # noqa: E402
from pointcloud.stem_data import STEM_FILE  # noqa: E402

WORKSPACE = PROJECT.parent
TRUNK = WORKSPACE / 'trunk_reconstruction'
MANIFESTS = [WORKSPACE / 'dualcrown3d_treescan_helios_v1/manifest.csv',
             WORKSPACE / 'TreeScanPL10k_HELIOS_ALS_flight_augmentation_v2/manifest.csv',
             WORKSPACE / 'dualcrown3d_weiser_v1/manifest.csv',
             WORKSPACE / 'combined_als_crowns_supervision_v4/manifest.csv']


def proxy_feet(coord, tree_id):
    """Foot proxies from the points themselves (dense data without stem labels)."""
    rows = []
    for tid in np.unique(tree_id[tree_id > 0]):
        m = tree_id == tid
        pts = coord[m]
        h = float(pts[:, 2].max())
        low = pts[(pts[:, 2] >= .5) & (pts[:, 2] < 1.5)]
        if len(low) >= 5:
            xy, quality = low[:, :2].mean(0), .5
        else:
            base = pts[pts[:, 2] < .3 * h]
            if len(base) >= 5:
                xy, quality = base[:, :2].mean(0), .35
            else:
                xy, quality = pts[:, :2].mean(0), .2
        rows.append(dict(tree_id=int(tid), x=float(xy[0]), y=float(xy[1]), h=h, quality=quality, source='proxy_lowest_points'))
    return rows


def labelled_feet(trunks, frame_offset=(0., 0.), source='label'):
    rows = []
    for t in trunks.itertuples():
        quality = float(t.quality) if 'quality' in trunks else 1.
        rows.append(dict(tree_id=int(t.tree_id), x=float(t.x) + frame_offset[0], y=float(t.y) + frame_offset[1],
                         h=float(t.h_tree), quality=max(quality, .6) if quality > 0 else .6, source=source))
    return rows


def trunks_for(row):
    """Trunk label table and its frame for one manifest row, or None."""
    source = row['source_dataset']
    if source == 'TreeScanPL10k_HELIOS':
        plot = row.get('parent_plot') or row['dataset_id'].removeprefix('treescan_helios__')
        path = TRUNK / 'data/labels' / plot / 'trunks.csv'
        return (pd.read_csv(path), 'tls_fit') if path.exists() else None
    if source == 'WEISER_2022_ALS':
        path = TRUNK / 'data/labels_weiser' / row['parent_plot'] / 'trunks.csv'
        return (pd.read_csv(path), 'weiser_reference') if path.exists() else None
    if source == 'WEISER_2022_SIM_V2':
        # Tile frame: window corner = scene origin + the offsets in the window name (simulate_weiser_tiles.py).
        # The plot-level labels shifted into that frame cover every tree of the tile; the per-tile
        # trunk files hold only trunks whose axis at 1.3 m lies inside the window.
        _, plot, dx, dy = row['window'].split('_')
        plot_dir = TRUNK / 'data/labels_weiser' / plot
        if (plot_dir / 'trunks.csv').exists() and (plot_dir / 'coverage.npz').exists():
            origin = np.load(plot_dir / 'coverage.npz')['origin']
            trunks = pd.read_csv(plot_dir / 'trunks.csv').copy()
            trunks['x'] -= float(origin[0]) + float(dx)
            trunks['y'] -= float(origin[1]) + float(dy)
            return trunks, 'weiser_reference_shifted'
        for folder in ('labels_weiser_tiles', 'labels_weiser_tiles_v3'):
            path = TRUNK / 'data' / folder / row['window'] / 'trunks.csv'
            if path.exists():
                return pd.read_csv(path), 'weiser_tile_reference'
        return None
    return None


def feet_for(row, arrays):
    coord = arrays['coord'] + arrays['source_origin']
    found = trunks_for(row)
    if found is None:
        return proxy_feet(coord, arrays['tree_id']), 'proxy'
    trunks, source = found
    present = set(np.unique(arrays['tree_id'][arrays['tree_id'] > 0]).tolist())
    trunks = trunks[trunks.tree_id.isin(present)]
    feet = labelled_feet(trunks, source=source)
    # Labelled trees without a trunk label (QA failure, no reference foot) get a proxy foot, so every
    # labelled tree has a query and validation compares like with like against Model20.
    missing = present - set(trunks.tree_id.astype(int).tolist())
    if missing:
        ids = np.where(np.isin(arrays['tree_id'], list(missing)), arrays['tree_id'], 0)
        feet += [{**f, 'quality': min(f['quality'], .3)} for f in proxy_feet(coord, ids)]
    return feet, source


def check(rows_out, arrays):
    coord = arrays['coord'] + arrays['source_origin']
    tree_id = arrays['tree_id']
    labelled = np.unique(tree_id[tree_id > 0])
    feet = {r['tree_id']: (r['x'], r['y']) for r in rows_out}
    distances = []
    for tid in labelled:
        if tid not in feet:
            continue
        m = tree_id == tid
        low = m & (coord[:, 2] < 3.)
        proxy = coord[low, :2].mean(0) if low.sum() >= 3 else coord[m, :2].mean(0)
        distances.append(float(np.hypot(*(proxy - feet[tid]))))
    return dict(trees=int(len(labelled)), with_foot=int(sum(t in feet for t in labelled)),
                median_proxy_distance_m=float(np.median(distances)) if distances else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifests', nargs='*', type=Path, default=MANIFESTS)
    ap.add_argument('--splits', nargs='*', default=['train', 'val'])
    ap.add_argument('--check', action='store_true', help='report only, write nothing')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    report = []
    for manifest in args.manifests:
        if not manifest.exists():
            print(f'skip (missing): {manifest}', flush=True)
            continue
        rows = [r for split in args.splits for r in read_manifest(manifest, split)]
        for row in rows:
            target = Path(remap(row['output'])).with_name(STEM_FILE)
            if target.exists() and not args.overwrite and not args.check:
                continue
            arrays = load_npz(row['output'])
            feet, source = feet_for(row, arrays)
            summary = dict(manifest=manifest.name, dataset_id=row['dataset_id'], source=source, **check(feet, arrays))
            report.append(summary)
            if not args.check:
                pd.DataFrame(feet, columns=['tree_id', 'x', 'y', 'h', 'quality', 'source']).to_csv(target, index=False)
            print(json.dumps(summary), flush=True)
    frame = pd.DataFrame(report)
    if len(frame):
        print(frame.groupby(['manifest', 'source']).agg(plots=('dataset_id', 'size'), trees=('trees', 'sum'),
                                                       with_foot=('with_foot', 'sum'),
                                                       median_proxy_distance_m=('median_proxy_distance_m', 'median')).to_string())
    if not args.check:
        (PROJECT / 'outputs' / 'stem_feet_report.json').parent.mkdir(exist_ok=True)
        (PROJECT / 'outputs' / 'stem_feet_report.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
