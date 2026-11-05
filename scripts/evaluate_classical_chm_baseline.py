#!/usr/bin/env python3
"""Classical CHM baseline (LMF treetops + marker-controlled watershed) on the labelled plots.

Runs the production algorithm of ``skrypt/pcopw_chunks_500m_Segmentacja.R`` (lidR
``locate_trees``/``lmf`` + ``ForestTools::mcws``, production parameters, no tuning)
on the CHM of every point-eval-eligible validation and test plot and scores it with
the metric code used for the deep models:

(a) native crown polygons   -> scripts.evaluate_combined_full_crowns.metrics/aggregate
(b) voxels labelled by the polygon containing their xy
                            -> scripts.train_vote_field.evaluate_field/summarize_field
                               (= train_two_pass_refiner.evaluate_plot/summarize plus the
                               annotated-area point metric)

Nothing is selected or tuned here; test is scored once with fixed settings.
The script refuses to overwrite a completed output (one that has metrics.json).

    CUDA_VISIBLE_DEVICES= .venv-gpu/bin/python scripts/evaluate_classical_chm_baseline.py
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import sys
sys.dont_write_bytecode = True          # never write .pyc into run_r/code
import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.shutil
import shapely
import torch
from rasterio.features import rasterize

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))
torch.set_num_threads(4)

from pointcloud.data import load_npz, read_manifest
from scripts.cache_stage1_predictions import eligible_rows
from scripts.evaluate_combined_full_crowns import aggregate, apply_parameters, classical, metrics
from scripts.train_supervision_v4 import REAL
from scripts.train_vote_field import evaluate_field, summarize_field

NAMES = ('dataset_id', 'source_dataset', 'collection')
DEFAULT_OUTPUT = PROJECT / 'outputs/classical_chm_baseline_v1'
PRODUCTION_SCRIPT = ROOT / 'skrypt/pcopw_chunks_500m_Segmentacja.R'
CONDA = ROOT / '.tools/miniforge3/bin/conda'
R_LIBS = ROOT / 'run_r/Rlib'
TUNED_PARAMETERS = ROOT / 'run_r/optimization_ideas_als_source_balanced_loso_v2/best_parameters.json'
TUNED_STUDY = ROOT / 'run_r/optimization_ideas_als_dev_weighted_pq_v1/study_configuration.json'
CHM_RESOLUTION = .5
HEIGHT_THRESHOLDS = (.5, 2.)            # voxel-labelling floors; the first is primary
PRIMARY_THRESHOLD = .5                  # fixed a priori: the voxel domain the deep models label
MODEL20 = dict(val=dict(point_sb_pq=.4051, crown_sb_pq=.4442, small_10_tp=6, small_10_gt=161, large_recall=.565),
               test=dict(point_sb_pq=.418, crown_sb_pq=.422, small_10_tp=21, small_10_gt=241, large_recall=.622))
PRODUCTION_PORT_DEFAULTS = dict(MIN_HEIGHT=2., MEDIAN_FILTER_SIZE=3, CHM_PIT_FILL_SIZE=1, CHM_PIT_DEPTH=0.,
                                CHM_GAUSSIAN_SIGMA=0., LMF_LOW_HEIGHT_LIMIT=10., LMF_HIGH_HEIGHT_LIMIT=25.,
                                LMF_LOW_WINDOW_SIZE=2., LMF_MID_SLOPE=.1, LMF_MID_INTERCEPT=.3, LMF_HIGH_SLOPE=.15,
                                LMF_HIGH_INTERCEPT=1., WATERSHED_CONNECTIVITY=8, POLYGON_CONNECTIVITY=4,
                                LMF_CONTROL_HEIGHTS=None, LMF_CONTROL_WINDOWS=None)

# The per-CHM body is the production script's process_seg_chunk, line for line. Differences, all outside
# the algorithm: (1) plots are standalone rasters, so there is no VRT/25 m neighbour buffer to crop;
# (2) the output keeps the CHM's own CRS instead of being relabelled EPSG:2180; (3) a CHM without a CRS
# (local plot coordinates) has terra's lon/lat guess cleared so all geometry stays planar, as in production;
# (4) errors are reported per plot instead of being swallowed.
R_RUNNER = r'''
options(encoding = "UTF-8")
suppressPackageStartupMessages({library(lidR); library(terra); library(ForestTools); library(sf)})
args <- commandArgs(trailingOnly = TRUE)
jobs <- read.csv(args[1], stringsAsFactors = FALSE)
dbh_naslund <- function(H, a = 10, b = 0.6) {
  Hm <- pmax(H - 1.3, 0); A <- -Hm * b; C <- -Hm * a; disc <- A^2 - 4 * C
  return((-A + sqrt(disc)) / 2)
}
status <- data.frame(dataset_id = jobs$dataset_id, status = "", trees = 0L, crowns = 0L, stringsAsFactors = FALSE)
for (i in seq_len(nrow(jobs))) {
  result <- tryCatch({
    chm_r <- terra::rast(jobs$chm[i])
    if (jobs$planar[i] == 1) terra::crs(chm_r) <- ""
    core_ext <- terra::ext(chm_r)
    chm_chunk <- chm_r
    if (terra::global(chm_chunk, "max", na.rm = TRUE)[1, 1] < 2) stop("NO_TREES")
    chm_spikefree <- terra::focal(chm_chunk, w = 3, fun = "median", na.rm = TRUE)
    f_ws <- function(z) ifelse(z < 10, 2, ifelse(z < 25, z * 0.1 + 0.3, z * 0.15 + 1))
    ttops <- lidR::locate_trees(chm_spikefree, lidR::lmf(ws = f_ws, hmin = 2))
    if (nrow(ttops) == 0) stop("NO_TREES")
    ttops_sf <- sf::st_as_sf(ttops)
    coords <- sf::st_coordinates(ttops_sf)
    inside_core <- coords[, 1] >= core_ext[1] & coords[, 1] < core_ext[2] &
      coords[, 2] >= core_ext[3] & coords[, 2] < core_ext[4]
    ttops_final <- ttops_sf[inside_core, ]
    if (nrow(ttops_final) == 0) stop("NO_TREES")
    ttops_final$dbh <- round(dbh_naslund(ttops_final$Z), 2)
    crowns <- ForestTools::mcws(treetops = ttops, CHM = chm_spikefree, minHeight = 2, format = "polygons")
    crowns_sf <- sf::st_as_sf(crowns)
    crowns_final <- crowns_sf[crowns_sf$treeID %in% ttops_final$treeID, ]
    crowns_final$area_m2 <- as.numeric(sf::st_area(crowns_final))
    sf::st_write(ttops_final, jobs$ttops[i], quiet = TRUE, append = FALSE, delete_dsn = TRUE)
    sf::st_write(crowns_final, jobs$crowns[i], quiet = TRUE, append = FALSE, delete_dsn = TRUE)
    c("ok", nrow(ttops_final), nrow(crowns_final))
  }, error = function(e) c(if (conditionMessage(e) == "NO_TREES") "no_trees" else paste("error:", conditionMessage(e)), 0, 0))
  status$status[i] <- result[1]; status$trees[i] <- as.integer(result[2]); status$crowns[i] <- as.integer(result[3])
  cat(sprintf("[%d/%d] %s: %s (%s treetops, %s crowns)\n", i, nrow(jobs), jobs$dataset_id[i], result[1], result[2], result[3]))
}
versions <- sapply(c("lidR", "terra", "ForestTools", "sf", "imager"), function(p) as.character(packageVersion(p)))
write.csv(status, args[2], row.names = FALSE)
writeLines(c(R.version.string, paste(names(versions), versions)), args[3])
'''


def rscript(*arguments, cwd=None):
    environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    command = [str(CONDA), 'run', '--no-capture-output', '-n', 'treescan', 'env', f'R_LIBS_USER={R_LIBS}', 'Rscript',
               *map(str, arguments)]
    done = subprocess.run(command, cwd=cwd, env=environment, text=True, capture_output=True)
    if done.returncode:
        raise RuntimeError(f'Rscript failed ({done.returncode}):\n{done.stdout}\n{done.stderr}')
    return done.stdout + done.stderr


def split_rows(split):
    if split == 'val':
        return eligible_rows('val')
    return [r for r in read_manifest(REAL, split) if r.get('point_eval_eligible') == 'true']


def rebuild_chm(row, arrays, path):
    """Fallback only: max voxel height per 0.5 m cell (used when the manifest CHM is missing/unusable)."""
    xy = arrays['coord'][:, :2].astype(np.float64) + arrays['source_origin'][:2]
    left = np.floor(xy[:, 0].min() / CHM_RESOLUTION) * CHM_RESOLUTION
    top = np.ceil(xy[:, 1].max() / CHM_RESOLUTION) * CHM_RESOLUTION
    cols = np.floor((xy[:, 0] - left) / CHM_RESOLUTION).astype(int)
    rows = np.floor((top - xy[:, 1]) / CHM_RESOLUTION).astype(int)
    height, width = int(rows.max()) + 1, int(cols.max()) + 1
    chm = np.full(height * width, -9999., np.float32)
    np.maximum.at(chm, rows * width + cols, np.maximum(arrays['coord'][:, 2], 0))
    crs = gpd.read_file(row['gt_vector']).crs
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', driver='GTiff', width=width, height=height, count=1, dtype='float32',
                       transform=rasterio.transform.from_origin(left, top, CHM_RESOLUTION, CHM_RESOLUTION),
                       crs=crs, nodata=-9999, compress='deflate') as target:
        target.write(chm.reshape(height, width), 1)


def chm_inventory(rows, output):
    """Check every manifest CHM; rebuild from the voxel cloud only when it cannot be used."""
    inventory = {}
    for row in rows:
        path, reason = Path(row['chm']) if row.get('chm') else None, None
        if path is None or not path.exists():
            reason = 'missing'
        else:
            with rasterio.open(path) as source:
                if not np.allclose(source.res, CHM_RESOLUTION):
                    reason = f'resolution {source.res}'
        if reason:
            path = output / 'rebuilt_chm' / f"chm_{row['dataset_id']}.tif"
            rebuild_chm(row, load_npz(row['output']), path)
        with rasterio.open(path) as source:
            inventory[row['dataset_id']] = dict(path=str(path), rebuilt=reason, resolution=list(source.res),
                                                shape=list(source.shape), crs=source.crs.to_string() if source.crs else None,
                                                nodata_fraction=float(np.mean(source.read(1, masked=True).mask)))
    return inventory


def run_production_r(rows_by_split, inventory, output):
    script, jobs, status_path, versions = (output / 'run_production_mcws.R', output / 'r_jobs.csv',
                                           output / 'r_status.csv', output / 'r_versions.txt')
    script.write_text(R_RUNNER)
    with jobs.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['split', 'dataset_id', 'chm', 'planar', 'crowns', 'ttops'])
        for split, rows in rows_by_split.items():
            (output / split).mkdir(parents=True, exist_ok=True)
            for row in rows:
                item = inventory[row['dataset_id']]
                writer.writerow([split, row['dataset_id'], item['path'], int(item['crs'] is None),
                                 output / split / f"crowns_{row['dataset_id']}.gpkg",
                                 output / split / f"ttops_{row['dataset_id']}.gpkg"])
    print(rscript(script, jobs, status_path, versions), flush=True)
    with status_path.open() as stream:
        status = {r['dataset_id']: r for r in csv.DictReader(stream)}
    failed = {k: v['status'] for k, v in status.items() if v['status'].startswith('error')}
    if failed:
        raise RuntimeError(f'Production R run failed on {failed}')
    return status, versions.read_text().splitlines()


def read_crowns(path):
    path = Path(path)
    if not path.exists():
        return gpd.GeoDataFrame(dict(treeID=np.zeros(0, np.int32)), geometry=[])
    return gpd.read_file(path)


def run_port(row, inventory):
    """Python port (run_r/code) with whatever parameters are currently applied to the module."""
    with rasterio.open(inventory[row['dataset_id']]['path']) as source:
        chm = source.read(1).astype(float)
        chm[chm == source.nodata] = np.nan
        smooth = classical.smooth_chm(chm)
        tops, markers = classical.locate_trees_lmf(smooth, source.transform, source.crs)
        if not len(tops):
            return tops, gpd.GeoDataFrame(dict(treeID=np.zeros(0, np.int32)), geometry=[], crs=source.crs)
        return tops, classical.segment_crowns(smooth, source.transform, markers, set(tops.treeID), source.crs)


def crown_raster(crowns, chm_path):
    with rasterio.open(chm_path) as source:
        transform, shape = source.transform, source.shape
    raster = np.zeros(shape, np.int32)
    if len(crowns):
        raster = rasterize(zip(crowns.geometry, crowns.treeID.astype(int)), out_shape=shape, transform=transform,
                           fill=0, dtype='int32')
    return raster, transform


def voxel_labels(arrays, crowns, chm_path, minimum_height):
    """ID of the crown polygon containing each voxel's xy (polygons are unions of CHM cells)."""
    raster, transform = crown_raster(crowns, chm_path)
    xy = arrays['coord'][:, :2].astype(np.float64) + arrays['source_origin'][:2]
    cols, rows = (~transform) * (xy[:, 0], xy[:, 1])
    rows, cols = np.floor(rows).astype(int), np.floor(cols).astype(int)
    inside = (rows >= 0) & (rows < raster.shape[0]) & (cols >= 0) & (cols < raster.shape[1])
    labels = np.zeros(len(xy), np.uint32)
    labels[inside] = raster[rows[inside], cols[inside]]
    labels[arrays['coord'][:, 2] < minimum_height] = 0
    return labels


def by_source(records, key=None):
    rows = [{**{k: r[k] for k in NAMES}, **(r[key] if key else {k: r[k] for k in ('tp', 'fp', 'fn', 'iou_sum')})}
            for r in records]
    return {name: {k: value[k] for k in ('tp', 'fp', 'fn', 'pq', 'f1', 'precision', 'recall')}
            for name, value in aggregate(rows)['by_source'].items()}


def evaluate(rows, inventory, crowns_of, thresholds):
    """Score one set of crown polygons with both protocols."""
    native, point = [], {t: [] for t in thresholds}
    for row in rows:
        arrays = load_npz(row['output'])
        gt = gpd.read_file(row['gt_vector'])
        data = Path(row['ignore_vector']).read_bytes() if row.get('ignore_vector') else b''
        ignore = shapely.from_wkb(data) if data else None
        crowns = crowns_of(row)
        native.append({**{k: row[k] for k in NAMES}, **metrics(gt.geometry, list(crowns.geometry), ignore),
                       'predicted_polygons': int(len(crowns)), 'gt_crowns': int(len(gt))})
        for threshold in thresholds:
            labels = voxel_labels(arrays, crowns, inventory[row['dataset_id']]['path'], threshold)
            confidence = (labels > 0).astype(np.float32)
            point[threshold].append(evaluate_field(row, arrays, gt, ignore, labels, confidence))
    result = dict(native_polygons=dict(summary=aggregate(native), per_plot=native))
    for threshold, records in point.items():
        summary = summarize_field(records)
        summary['point_by_source'] = by_source(records, 'point_metrics')
        summary['crown_hull_by_source'] = by_source(records)
        summary['annotated_by_source'] = by_source(records, 'annotated')
        result[f'voxels_ge_{threshold:g}m'] = dict(summary=summary, per_plot=records)
    return result


def compare_with_port(rows, inventory, output_split):
    """Cell-level agreement of the R result with the Python port at production parameters."""
    report = dict(plots=len(rows), plots_same_treetops=0, plots_same_partition=0, r_treetops=0, port_treetops=0,
                  canopy_cells=0, cells_differing=0)
    port = {}
    for row in rows:
        tops, crowns = run_port(row, inventory)
        port[row['dataset_id']] = crowns
        r_tops = read_crowns(output_split / f"ttops_{row['dataset_id']}.gpkg")
        r_crowns = read_crowns(output_split / f"crowns_{row['dataset_id']}.gpkg")
        a = {(round(p.x, 3), round(p.y, 3)) for p in r_tops.geometry}
        b = {(round(p.x, 3), round(p.y, 3)) for p in tops.geometry}
        report['r_treetops'] += len(a)
        report['port_treetops'] += len(b)
        report['plots_same_treetops'] += a == b
        chm = inventory[row['dataset_id']]['path']
        first, transform = crown_raster(r_crowns, chm)
        second, _ = crown_raster(crowns, chm)
        # Compare partitions through the treetop cell so differing ID numbering does not matter.
        def canonical(raster, points):
            lookup = np.zeros(int(raster.max()) + 1, np.int64)
            for p in points:
                col, r = (~transform) * (p.x, p.y)
                value = raster[int(np.floor(r)), int(np.floor(col))]
                if value:
                    lookup[value] = int(np.floor(r)) * raster.shape[1] + int(np.floor(col)) + 1
            return lookup[raster]
        one, two = canonical(first, r_tops.geometry), canonical(second, tops.geometry)
        different = int((one != two).sum())
        report['canopy_cells'] += int(((first > 0) | (second > 0)).sum())
        report['cells_differing'] += different
        report['plots_same_partition'] += different == 0
    return report, port


def check_production_script(rows, inventory, output_split):
    """Run the unmodified production script (only its two paths and n_cores are substituted) per plot."""
    source = PRODUCTION_SCRIPT.read_text(encoding='utf-8')
    report = dict(plots=len(rows), identical=0, crowns_compared=0, max_symmetric_difference_m2=0., mismatches=[])
    with tempfile.TemporaryDirectory(prefix='classical_chm_check_') as temporary:
        for row in rows:
            name = row['dataset_id']
            folder = Path(temporary) / name
            (folder / 'in').mkdir(parents=True)
            os.symlink(inventory[name]['path'], folder / 'in' / f'chm_{name}.tif')
            rasterio.shutil.copy(folder / 'in' / f'chm_{name}.tif', folder / 'in' / '00_CHM_Full.vrt', driver='VRT')
            text, counts = source, []
            for pattern, replacement in ((r'input_chm_dir <- "[^"]*"', f'input_chm_dir <- "{folder / "in"}"'),
                                         (r'output_dir <- "[^"]*"', f'output_dir <- "{folder / "out"}"'),
                                         (r'n_cores\s*<- 10', 'n_cores     <- 1')):
                text, count = re.subn(pattern, lambda _: replacement, text)
                counts.append(count)
            if counts != [1, 1, 1]:
                raise RuntimeError(f'Production script layout changed: substitutions {counts}')
            (folder / 'production.R').write_text(text, encoding='utf-8')
            rscript(folder / 'production.R', cwd=folder)
            theirs = read_crowns(folder / 'out/Segmentation3' / f'crowns_{name}.gpkg')
            ours = read_crowns(output_split / f'crowns_{name}.gpkg')
            left = dict(zip(theirs.treeID.astype(int), theirs.geometry))
            right = dict(zip(ours.treeID.astype(int), ours.geometry))
            difference = (max((left[k].symmetric_difference(right[k]).area for k in left), default=0.)
                          if left.keys() == right.keys() else float('inf'))
            report['crowns_compared'] += len(right)
            if difference < 1e-6:
                report['identical'] += 1
                report['max_symmetric_difference_m2'] = max(report['max_symmetric_difference_m2'], difference)
            else:
                report['mismatches'].append(dict(dataset_id=name, production=len(left), runner=len(right),
                                                 difference=None if difference == float('inf') else difference))
    return report


def tuned_overlap(rows_by_split):
    """Which of our plots were in the 193-plot IDEAS-ALS dev set the tuned parameters were selected on."""
    study = json.loads(TUNED_STUDY.read_text())['split']
    used = {name: part for part, names in study.items() for name in names}
    result = {}
    for split, rows in rows_by_split.items():
        hits = {r['dataset_id']: used.get(r['dataset_id'].replace('ideas__', '').replace('__', '_')) for r in rows}
        result[split] = dict(plots=len(rows), in_tuning_set=sum(v is not None for v in hits.values()),
                             tuning_role={k: v for k, v in hits.items() if v is not None})
    return result


def table_row(result, primary):
    point, native = result[primary]['summary'], result['native_polygons']['summary']
    return dict(point_sb_pq=point['point_sb_pq'], crown_hull_sb_pq=point['crown_sb_pq'],
                native_polygon_sb_pq=native['source_balanced_pq'], native_polygon_sb_f1=native['source_balanced_f1'],
                point_precision=point['point_precision'], point_recall=point['point_recall'],
                crown_hull_precision=point['crown_precision'], crown_hull_recall=point['crown_recall'],
                native_polygon_precision=native['precision'], native_polygon_recall=native['recall'],
                small_10=f"{point['small_10_tp']}/{point['small_10_gt']}", large_recall=point['large_recall'],
                oversplit_gt=point['oversplit_gt'], undersegmented_pred=point['undersegmented_pred'],
                predicted_instances=point['predicted_instances'], native_predicted_polygons=native['tp'] + native['fp'] + native['ignored_predictions'],
                annotated_point_sb_pq=point['annotated_point_sb_pq'], annotated_point_precision=point['annotated_point_precision'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--skip-production-script-check', action='store_true')
    parser.add_argument('--skip-tuned', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if PROJECT not in output.parents:
        raise SystemExit('Output must be inside DL_model_version')
    if (output / 'metrics.json').exists():
        raise SystemExit(f'{output}/metrics.json exists: refusing to overwrite a completed output')
    for name, value in PRODUCTION_PORT_DEFAULTS.items():
        if getattr(classical, name) != value:
            raise SystemExit(f'Python port default {name}={getattr(classical, name)!r} is not the production value {value!r}')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows_by_split = {split: split_rows(split) for split in ('val', 'test')}
    inventory = chm_inventory([r for rows in rows_by_split.values() for r in rows], output)
    print({s: len(r) for s, r in rows_by_split.items()}, 'rebuilt CHMs:',
          [k for k, v in inventory.items() if v['rebuilt']], flush=True)

    status, versions = run_production_r(rows_by_split, inventory, output)
    primary = f'voxels_ge_{PRIMARY_THRESHOLD:g}m'
    results, checks, table = {}, {}, {}
    for split, rows in rows_by_split.items():
        folder = output / split
        results.setdefault('production_r', {})[split] = evaluate(
            rows, inventory, lambda row: read_crowns(folder / f"crowns_{row['dataset_id']}.gpkg"), HEIGHT_THRESHOLDS)
        print(split, 'production R', json.dumps(table_row(results['production_r'][split], primary)), flush=True)
        report, port = compare_with_port(rows, inventory, folder)
        checks.setdefault('python_port_vs_r', {})[split] = report
        results.setdefault('python_port_production_parameters', {})[split] = evaluate(
            rows, inventory, lambda row: port[row['dataset_id']], (PRIMARY_THRESHOLD,))
        print(split, 'port check', report, flush=True)
        if not args.skip_production_script_check:
            checks.setdefault('production_script_vs_runner', {})[split] = check_production_script(rows, inventory, folder)
            print(split, 'production script check', checks['production_script_vs_runner'][split], flush=True)

    tuned = None
    if not args.skip_tuned:
        tuned = json.loads(TUNED_PARAMETERS.read_text())
        parameters = tuned['parameters']
        defaults = {k: getattr(classical, k.upper()) for k in parameters}
        apply_parameters(parameters, defaults)
        try:
            for split, rows in rows_by_split.items():
                folder = output / f"tuned_ideas_{tuned['candidate_id']}" / split
                folder.mkdir(parents=True, exist_ok=True)
                crowns = {}
                for row in rows:
                    crowns[row['dataset_id']] = run_port(row, inventory)[1]
                    if len(crowns[row['dataset_id']]):
                        crowns[row['dataset_id']].to_file(folder / f"crowns_{row['dataset_id']}.gpkg", driver='GPKG')
                results.setdefault('tuned_ideas_python_port', {})[split] = evaluate(
                    rows, inventory, lambda row: crowns[row['dataset_id']], (PRIMARY_THRESHOLD,))
                print(split, 'tuned', json.dumps(table_row(results['tuned_ideas_python_port'][split], primary)), flush=True)
        finally:
            apply_parameters({}, defaults)
        tuned = dict(candidate_id=tuned['candidate_id'], parameters=parameters, source=str(TUNED_PARAMETERS),
                     overlap_with_tuning_plots=tuned_overlap(rows_by_split),
                     optimistic='ideas_als plots (CEDAR_CYPRESS, FGI_EMIT, WILDFOREST3D) overlap the tuning set')

    for variant, value in results.items():
        table[variant] = {split: table_row(value[split], primary) for split in value}
    table['production_r_voxels_ge_2m'] = {split: table_row(results['production_r'][split], 'voxels_ge_2m')
                                          for split in rows_by_split}
    payload = dict(
        description='Classical CHM baseline (lidR lmf + ForestTools mcws, production parameters, no tuning)',
        manifest=str(REAL), plots={s: [r['dataset_id'] for r in rows] for s, rows in rows_by_split.items()},
        production_parameters=dict(median_filter='terra::focal w=3 median na.rm=TRUE', hmin=2, minHeight=2,
                                   f_ws='ifelse(z<10, 2, ifelse(z<25, z*0.1+0.3, z*0.15+1))'),
        primary_voxel_height_threshold_m=PRIMARY_THRESHOLD, voxel_height_thresholds_m=list(HEIGHT_THRESHOLDS),
        r_versions=versions, r_status=status, chm=inventory,
        chm_resolutions=sorted({tuple(v['resolution']) for v in inventory.values()}),
        rebuilt_chm=[k for k, v in inventory.items() if v['rebuilt']],
        implementation_checks=checks, tuned_variant=tuned, table=table, reference_model20=MODEL20,
        results=results, seconds=time.monotonic() - started)
    (output / 'metrics.json').write_text(json.dumps(payload, indent=2, default=float) + '\n')
    print(json.dumps(table, indent=2), flush=True)


if __name__ == '__main__':
    main()
