#!/usr/bin/env python3
"""New HELIOS flights over train scenes only; preserve source plot groups."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import laspy
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.prepare_treescan_helios_dualcrown import convert_plot, write_csv, sha256

BASE = PROJECT.parent / 'TreeScanPL10k_HELIOS_ALS_v1'
OUTPUT = PROJECT.parent / 'TreeScanPL10k_HELIOS_ALS_flight_augmentation_v2'
VARIANTS = [dict(name='cross_sparse', heading=90., altitude=650., speed=95., angle=25., offsets=[0., -230.]),
            dict(name='diagonal_dense', heading=45., altitude=400., speed=65., angle=18., offsets=[0., -105.])]


def generate_survey(plot, variant, folder):
    folder.mkdir(parents=True, exist_ok=True)
    scene = ET.parse(BASE / 'scenes' / f'{plot}.xml')
    for param in scene.findall('.//param'):
        if param.get('key') in ('filepath', 'matfile'):
            param.set('value', str(BASE / param.get('value')))
    scene_path = folder / 'scene.xml'
    scene.write(scene_path, encoding='utf-8', xml_declaration=True)
    survey = ET.parse(BASE / 'surveys' / f'{plot}.xml')
    s = survey.find('survey')
    s.set('scene', f'{scene_path}#{plot}')
    for key in ('platform', 'scanner'):
        path, reference = s.get(key).split('#')
        s.set(key, f'{BASE / path}#{reference}')
    survey.find('scannerSettings').set('scanAngle_deg', str(variant['angle']))
    theta = np.deg2rad(variant['heading'])
    for i, leg in enumerate(s.findall('leg')):
        direction = (1 if i in (0, 3) else -1)
        along = -95. * direction
        across = variant['offsets'][i // 2]
        settings = leg.find('platformSettings')
        settings.set('x', str(along * np.cos(theta) - across * np.sin(theta)))
        settings.set('y', str(along * np.sin(theta) + across * np.cos(theta)))
        settings.set('z', str(variant['altitude']))
        settings.set('movePerSec_m', str(variant['speed']))
    path = folder / 'survey.xml'
    survey.write(path, encoding='utf-8', xml_declaration=True)
    return path


def export(plot, variant, raw, folder):
    meta = json.loads((BASE / 'assets' / plot / 'scene_metadata.json').read_text())
    assert meta['split'] == 'train'
    strips = sorted(raw.rglob('strip*_points.laz'))
    if len(strips) != 2:
        raise ValueError(f'Expected exactly two strips, found {len(strips)} in {raw}')
    cfg = json.loads((BASE / 'config/calibration.json').read_text())
    origin = np.array([meta['center_x'], meta['center_y'], meta['z_reference']])
    clouds = [laspy.read(p) for p in strips]
    xyz = np.concatenate([np.column_stack((c.x, c.y, c.z)) for c in clouds]) + origin
    hits = np.concatenate([np.asarray(c.hitObjectId) for c in clouds]).astype(np.int32)
    ids = np.zeros(len(xyz), np.int32)
    species = np.zeros(len(xyz), np.uint16)
    for part in meta['tree_parts']:
        take = hits == part['global_tree_id']
        ids[take], species[take] = part['tree_id'], part['species_code']
    unknown = (hits != 0) & (ids == 0)
    if unknown.any():
        raise ValueError('Unmapped non-ground HELIOS object IDs')
    grid = np.load(BASE / 'assets' / plot / 'ground_grid.npz')
    ix = np.clip(np.rint((xyz[:, 0]-grid['xs'][0]) / np.diff(grid['xs'])[0]).astype(int), 0, len(grid['xs'])-1)
    iy = np.clip(np.rint((xyz[:, 1]-grid['ys'][0]) / np.diff(grid['ys'])[0]).astype(int), 0, len(grid['ys'])-1)
    hag = xyz[:, 2] - grid['z'][iy, ix]
    hag[ids == 0] = 0
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.scales = [.001]*3
    for key, dtype in [('treeID', np.int32), ('treeSP', np.uint16), ('completelyInside', np.uint8),
                       ('height_agl', np.float32), ('hitObjectId', np.int32)]:
        header.add_extra_dim(laspy.ExtraBytesParams(name=key, type=dtype))
    result = laspy.LasData(header)
    result.x, result.y, result.z = xyz.T
    result.treeID, result.treeSP = ids, species
    result.completelyInside = (ids > 0).astype(np.uint8)
    result.height_agl, result.hitObjectId = hag.astype(np.float32), hits
    intensity = np.concatenate([np.asarray(c.intensity) for c in clouds])
    result.intensity = np.clip(np.rint(np.interp(intensity, cfg['target']['helios_intensity_quantiles_10cm'],
                                              cfg['target']['intensity_quantiles'])), 0, 65535).astype(np.uint16)
    result.classification = np.where(ids > 0, 5, 2).astype(np.uint8)
    for key in ('return_number', 'number_of_returns', 'gps_time'):
        result[key] = np.concatenate([np.asarray(c[key]) for c in clouds])
    result.point_source_id = np.concatenate([np.full(len(c.points), 117+i, np.uint16) for i,c in enumerate(clouds)])
    theta = np.deg2rad(variant['heading'])
    across = -(xyz[:, 0]-origin[0])*np.sin(theta) + (xyz[:, 1]-origin[1])*np.cos(theta)
    flight_offset = np.where(np.asarray(result.point_source_id)==117, *variant['offsets'])
    result.scan_angle_rank = np.rint(np.rad2deg(np.arctan2(across-flight_offset, variant['altitude']-(xyz[:, 2]-origin[2])))).astype(np.int8)
    folder.mkdir(parents=True, exist_ok=True)
    result.write(folder / f'{plot}_ALS.laz')
    shutil.copy2(BASE / 'assets' / plot / 'crowns_full.gpkg', folder / 'crowns_full.gpkg')
    return dict(points=len(ids), trees_with_returns=int(len(np.unique(ids[ids>0]))),
                full_crowns=meta['tree_count'], density_points_m2=len(ids)/900., variant=variant,
                parent_plot=plot, split='train')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    plots = sorted((BASE / 'prepared_dataset/train').iterdir())[:args.limit]
    rows, reports = [], []
    for plot in plots:
        for variant in VARIANTS:
            folder = root / variant['name'] / plot.name
            completed = folder / 'ready.json'
            if completed.exists():
                done = json.loads(completed.read_text())
                # Match the canonical base dataset's source-plot group exactly.
                import hashlib
                done['row']['group_id'] = hashlib.sha256(plot.name.encode()).hexdigest()[:16]
                (folder/'model/metadata.json').write_text(json.dumps(done['row'], indent=2)+'\n')
                completed.write_text(json.dumps(done, indent=2)+'\n')
                rows.append(done['row']); reports.append(done['report'])
                continue
            survey = generate_survey(plot.name, variant, folder)
            raw = folder / 'raw'
            command = [str(BASE / 'env/bin/helios'), str(survey), '--assets', str(BASE),
                       '--output', str(raw), '--lasOutput', '--zipOutput', '--seed', '20260929',
                       '--rebuildScene', '-j', '6', '-q']
            with (folder / 'simulation.log').open('w') as log:
                subprocess.run(command, cwd=BASE, stdout=log, stderr=subprocess.STDOUT, check=True)
            prepared = folder / 'prepared'
            report = export(plot.name, variant, raw, prepared)
            row = convert_plot(prepared, folder / 'model', 'train', .25, allow_unobserved_training_crowns=True)
            import hashlib
            row.update(dataset_id=f'treescan_helios__{plot.name}__{variant["name"]}',
                       group_id=hashlib.sha256(plot.name.encode()).hexdigest()[:16],
                       parent_plot=plot.name, flight_variant=variant['name'])
            (folder/'model/metadata.json').write_text(json.dumps(row, indent=2)+'\n')
            completed.write_text(json.dumps(dict(row=row, report=report), indent=2)+'\n')
            rows.append(row); reports.append(report)
            write_csv(root / 'manifest_progress.csv', rows)
            print(json.dumps(report), flush=True)
    write_csv(root / 'manifest.csv', rows)
    (root / 'READY.json').write_text(json.dumps(dict(plots=len(plots), simulations=len(rows),
        train_only=True, variants=VARIANTS, manifest_sha256=sha256(root/'manifest.csv'), reports=reports), indent=2)+'\n')


if __name__ == '__main__':
    main()
