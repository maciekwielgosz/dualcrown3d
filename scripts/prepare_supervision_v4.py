#!/usr/bin/env python3
"""Versioned annotation repair. Raw data, old caches and split groups are immutable."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import geopandas as gpd
import laspy
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy.ndimage import distance_transform_edt, median_filter, minimum_filter
import shapely

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
from pointcloud.supervision import annotation_targets
from prepare_pointcloud_dataset import voxel_indices

PDAL = Path('/home/maciej.wielgosz/.local/share/micromamba/envs/qgis/bin/pdal')
VERSION = 'annotation_v4_tree_background_unknown'


def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, records):
    fields = list(dict.fromkeys(k for r in records for k in r))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader(); writer.writerows(records)


def recover_classification(row, arrays):
    """Recreate exactly the historical representatives; reject any misalignment."""
    cloud = laspy.read(row['source_las'])
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float64)
    classification = np.asarray(cloud.classification)
    finite = np.isfinite(xyz).all(1)
    xyz, classification = xyz[finite], classification[finite]
    with rasterio.open(row['chm']) as raster:
        h, w = raster.shape
        rr, cc = rasterio.transform.rowcol(raster.transform, xyz[:, 0], xyz[:, 1])
    rr, cc = np.clip(rr, 0, h-1), np.clip(cc, 0, w-1)
    cell = rr*w+cc
    ground = classification == 2
    if xyz[:, 2].min() <= 2:
        dtm = np.zeros((h, w))
    elif ground.any():
        sums = np.bincount(cell[ground], weights=xyz[ground, 2], minlength=h*w)
        counts = np.bincount(cell[ground], minlength=h*w)
        dtm = (sums/np.maximum(counts, 1)).reshape(h, w)
        nearest = distance_transform_edt(counts.reshape(h, w) == 0, return_distances=False, return_indices=True)
        dtm = dtm[tuple(nearest)]
    else:
        low = np.full(h*w, np.inf)
        np.minimum.at(low, cell, xyz[:, 2])
        low = low.reshape(h, w)
        nearest = distance_transform_edt(~np.isfinite(low), return_distances=False, return_indices=True)
        dtm = median_filter(minimum_filter(low[tuple(nearest)], size=5), size=9)
    xyz[:, 2] -= dtm[rr, cc]
    selected, _ = voxel_indices(xyz, .25)
    reconstructed = (xyz[selected]-arrays['source_origin']).astype(np.float32)
    if not np.array_equal(reconstructed, arrays['coord']):
        raise AssertionError(f'Representative alignment failed: {row["dataset_id"]}')
    return classification[selected]


def ignore_geometry(row, arrays):
    """Unknown non-ground cells with no known non-ground observations.

    This uses explicit point annotation validity, never the GT polygon union.
    Mixed cells remain scored; duplicate predictions over labelled trees count.
    """
    with rasterio.open(row['chm']) as raster:
        h, w, transform = raster.height, raster.width, raster.transform
    xyz = arrays['coord']; xy = xyz[:, :2].astype(float)+arrays['source_origin'][:2]
    rr, cc = rasterio.transform.rowcol(transform, xy[:, 0], xy[:, 1])
    cell = np.clip(rr, 0, h-1)*w+np.clip(cc, 0, w-1)
    unknown, known = np.zeros(h*w, bool), np.zeros(h*w, bool)
    above = xyz[:, 2] > .5
    unknown[cell[above & (arrays['tree_id'] < 0)]] = True
    known[cell[above & (arrays['tree_id'] >= 0)]] = True
    mask = (unknown & ~known).reshape(h, w).astype(np.uint8)
    return shapely.union_all([shapely.geometry.shape(g) for g, v in shapes(mask, mask=mask.astype(bool), transform=transform) if v])


def weak_ecodse(row, folder, arrays):
    """SMRF terrain audit; crown labels stay 2-D and never supervise 3-D IDs."""
    target = folder/'smrf_hag.las'
    pipeline = [dict(type='readers.las', filename=row['source_las']),
                dict(type='filters.assign', value='Classification = 0'),
                dict(type='filters.smrf', cell=1., window=16., slope=.15, threshold=.5, scalar=1.25),
                dict(type='filters.hag_nn'),
                dict(type='writers.las', filename=str(target), extra_dims='all', forward='all')]
    (folder/'terrain_pipeline.json').write_text(json.dumps(pipeline, indent=2)+'\n')
    if not target.exists():
        subprocess.run([str(PDAL), 'pipeline', '--stdin'], input=json.dumps(pipeline), text=True, check=True, capture_output=True)
    cloud = laspy.read(target)
    xyz = np.column_stack((cloud.x, cloud.y, cloud.HeightAboveGround)).astype(float)
    selected, origin = voxel_indices(xyz, .25)
    coord = (xyz[selected]-arrays['source_origin']).astype(np.float32)
    classification = np.asarray(cloud.classification)[selected]
    gt_projected = np.asarray(cloud.treeID)[selected].astype(np.int32)
    arrays = dict(coord=coord, grid_coord=np.floor((xyz[selected]-origin)/.25).astype(np.int32),
                  intensity=np.zeros(len(selected), np.float32), tree_id=np.full(len(selected), -1, np.int32),
                  semantic_target=np.where(classification == 2, 0, -1).astype(np.int8),
                  instance_offset=np.zeros_like(coord), source_origin=arrays['source_origin'],
                  voxel_origin=origin, voxel_size=np.float32(.25), projected_crown_id=gt_projected,
                  raw_points=np.int64(len(xyz)))
    plot = row['dataset_id'].split('__')[-1].removeprefix('plot_').removesuffix('_annotated')
    official = ROOT/f'ideas_als/_raw/ECODSE/ECODSEdataset/RSdata/chm/{plot}_chm.tif'
    audit = dict(ground_points=int((np.asarray(cloud.classification)==2).sum()),
                 height_min=float(xyz[:, 2].min()), height_q95=float(np.percentile(xyz[:, 2], 95)),
                 below_minus_1m_fraction=float((xyz[:, 2]<-1).mean()),
                 terrain_method='PDAL SMRF + HAG NN; estimated, not independently verified DTM',
                 point_supervision='excluded: polygons do not establish 3D instances or annotation completeness')
    if official.exists():
        with rasterio.open(official) as raster:
            rr, cc = rasterio.transform.rowcol(raster.transform, xyz[:, 0], xyz[:, 1])
            rr, cc = np.asarray(rr), np.asarray(cc)
            valid=(rr>=0)&(rr<raster.height)&(cc>=0)&(cc<raster.width)
            observed=np.full(raster.height*raster.width, -np.inf)
            np.maximum.at(observed,rr[valid]*raster.width+cc[valid],xyz[valid,2])
            reference=raster.read(1,masked=True).filled(np.nan).ravel()
            use=np.isfinite(observed)&np.isfinite(reference)&(reference>=0)
            if use.any():
                audit['official_chm_median_absolute_error_m']=float(np.median(np.abs(observed[use]-reference[use])))
                audit['official_chm_p90_absolute_error_m']=float(np.percentile(np.abs(observed[use]-reference[use]),90))
    return arrays, audit


def prepare(row, output, taxonomy):
    row = dict(row); folder = output/row['model_split']/row['dataset_id']; folder.mkdir(parents=True, exist_ok=True)
    done = folder/'metadata.json'
    if done.exists():
        result=json.loads(done.read_text())
        if result['annotation_protocol'] != VERSION: raise ValueError('Incompatible cache')
        return result
    with np.load(row['output']) as data: arrays={k:data[k] for k in data.files}
    original_ids=arrays['tree_id'].copy()
    gt=gpd.read_file(row['gt_vector'])
    audit={}
    if row['collection']=='ECODSE':
        arrays,audit=weak_ecodse(row,folder,arrays)
        row.update(train_eligible='false', point_eval_eligible='false', crown_eval_eligible='false',
                   supervision_kind='weak_crown_2d', height_normalization='smrf_hag_nn_audited')
        excluded=[]
    else:
        classes=recover_classification(row,arrays)
        plot=row['dataset_id'].split('__')[-1].removeprefix('plot_').removesuffix('_annotated')
        excluded=taxonomy.get(plot,[]) if row['collection']=='WILDFOREST3D' else []
        arrays['tree_id'],arrays['semantic_target']=annotation_targets(original_ids,classes,
            partial_zero=row['collection']=='WILDFOREST3D',bush_ids=excluded,height=arrays['coord'][:,2])
        gt=gt[~gt.treeID.isin(excluded)].copy()
        arrays['instance_offset'][arrays['tree_id']<=0]=0
        audit.update(original_outside_voxels=int((classes==3).sum()), bush_instances=len(excluded),
                     geometry_and_features_unchanged=True)
        row.update(train_eligible='true',point_eval_eligible='true',crown_eval_eligible='true',supervision_kind='native_3d')
    npz=folder/'points_0p25m.npz'; np.savez_compressed(npz,**arrays)
    vector=folder/'crowns_full.gpkg'; gt.to_file(vector,layer='crowns_gt',driver='GPKG')
    ignore=folder/'annotation_ignore.wkb'; ignore.write_bytes(shapely.to_wkb(ignore_geometry(row,arrays)))
    audit.update(unknown_voxels=int((arrays['tree_id']<0).sum()),background_voxels=int((arrays['tree_id']==0).sum()),
                 tree_voxels=int((arrays['tree_id']>0).sum()), semantic_unknown=int((arrays['semantic_target']<0).sum()),
                 label_hash=hashlib.sha256(arrays['tree_id'].tobytes()).hexdigest())
    (folder/'annotation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    row.update(original_prepared=row['output'],original_gt_vector=row['gt_vector'],output=str(npz),
               gt_vector=str(vector),ignore_vector=str(ignore),instances=len(gt),voxels=len(arrays['coord']),
               tree_voxels=audit['tree_voxels'],annotation_protocol=VERSION)
    done.write_text(json.dumps(row,indent=2)+'\n')
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'combined_als_crowns_supervision_v4')
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args(); output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    taxonomy={}
    for r in rows(ROOT/'ideas_als/WILDFOREST3D/tree_data_WILDFOREST3D.csv'):
        if r['objectType']=='BUSH':taxonomy.setdefault(r['plotID'],[]).append(int(r['treeID']))
    source=rows(ROOT/'combined_als_crowns_no_rectangles_v2/manifest.csv')
    results=[]
    for i,row in enumerate(source[:args.limit] if args.limit else source,1):
        results.append(prepare(row,output,taxonomy))
        print(f'{i}/{len(source)} {row["dataset_id"]}',flush=True)
    write_csv(output/('manifest_preview.csv' if args.limit else 'manifest.csv'),results)
    if not args.limit:
        for split in ('train','val','test'):write_csv(output/f'manifest_{split}.csv',[r for r in results if r['model_split']==split])
        summary=dict(protocol=VERSION,plots=len(results),split_groups_unchanged=True,source_data_unchanged=True,
                     native_train=sum(r['model_split']=='train' and r['train_eligible']=='true' for r in results),
                     native_val=sum(r['model_split']=='val' and r['point_eval_eligible']=='true' for r in results),
                     weak_ecodse_excluded_from_3d=sum(r['supervision_kind']=='weak_crown_2d' for r in results),
                     warning='New protocol: compare checkpoints on this same manifest, not against old published numbers.')
        (output/'READY.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__': main()
