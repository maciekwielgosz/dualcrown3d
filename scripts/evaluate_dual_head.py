"""Validation of the new instance masks using the same decoder/export path."""
import json
import time
from pathlib import Path
from types import SimpleNamespace
import geopandas as gpd
import numpy as np
from pointcloud.data import read_manifest, load_npz
from pointcloud.dual_head import MaskBranchView
from pointcloud.instance_output import merge_masks, point_instance_metrics
from scripts.evaluate_pointcloud_mask_decoder import predict_plot
from scripts.evaluate_combined_full_crowns import plot_metrics, aggregate


def evaluate_masks(model, manifest, folder, max_points=40000, configs=None, split='val'):
    if split == 'test' and not configs:
        raise ValueError('Test requires a validation-frozen configuration')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.backbone.shuffle_orders = False
    configs = configs or [dict(object_threshold=s, mask_threshold=m, minimum_voxels=12, merge_overlap=.5)
                          for s in (.1, .3, .5) for m in (.4, .5)]
    args = SimpleNamespace(tile_size=20., overlap=8., max_points=max_points,
                           preserve_height=True, raw_object_threshold=.05, raw_mask_threshold=.2,
                           retain_all_context_masks=any(not c.get('owner_only', True) for c in configs))
    device = next(model.parameters()).device
    all_rows, seconds = [[] for _ in configs], 0.
    started = time.monotonic()
    for i, row in enumerate(read_manifest(manifest, split)):
        arrays = load_npz(row['output'])
        raw = predict_plot(MaskBranchView(model), arrays, args, device, 20260925)
        seconds += float(raw['seconds'])
        gt = gpd.read_file(row['gt_vector'])
        metadata = {k: row[k] for k in ('dataset_id', 'source_dataset', 'collection', 'annotation_method')}
        for config, results in zip(configs, all_rows):
            labels, confidence, instances = merge_masks(arrays, raw, config)
            results.append({**metadata, **plot_metrics(row, gt.geometry, [p['geometry'] for p in instances]),
                            'point_metrics': point_instance_metrics(arrays['tree_id'], labels)})
        print(f'mask validation plot {i+1}: {row["dataset_id"]}', flush=True)
    trials = []
    for config, rows in zip(configs, all_rows):
        point_rows = [{**{k: r[k] for k in ('dataset_id', 'source_dataset', 'collection', 'annotation_method')},
                       **r['point_metrics']} for r in rows]
        native = [r for r in point_rows if r['annotation_method'] == 'point_native']
        trials.append(dict(config=config, metrics=aggregate(rows), point_metrics=aggregate(point_rows),
                           native_point_metrics=aggregate(native) if native else {}, per_plot=rows))
    best = max(trials, key=lambda r: (r['point_metrics']['source_balanced_pq'], r['metrics']['source_balanced_pq']))
    best.update(split=split, inference_seconds=seconds, seconds=time.monotonic()-started,
                selection_metric='source-balanced point-instance PQ@0.5',
                point_annotation_caveat='Mixed native and polygon-projected IDs; native subset reported separately.')
    (folder / f'{split}_metrics.json').write_text(json.dumps(best, indent=2) + '\n')
    (folder / 'configuration_trials.json').write_text(json.dumps(trials, indent=2) + '\n')
    return best
