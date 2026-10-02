#!/usr/bin/env python3
"""Two-pass inference on the prepared reference scene: full-resolution LAZ and GeoPackages.

Stage 1 reuses the frozen Model20 raw predictions of an existing Model20 output
folder (hash-checked). Stage 2 detects tree centres in the vote space and
reassigns votes with the configuration frozen on validation. Outputs follow the
Model20 export conventions; this scene has no ground truth.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.instance_output import merge_masks_with_sources
from pointcloud.vote_centers import (SOURCE_ADDED, SOURCE_RELOCATED, SOURCE_STAGE1_CENTRE, VoteCenterNet,
                                     detect_centres, merge_with_stage1, second_pass_labels, select_centres,
                                     stage1_centres)
from scripts.cache_stage1_predictions import DEFAULT_OUTPUT, sha256, write_json
from scripts.predict_dual_head import export_laz, write_vectors


def core_instances(labels, confidence, source, instances, metadata):
    """Keep instances whose treetop lies in a core tile; renumber consecutively."""
    kept = [p for p in instances if any(t['bounds'][0] <= p['top_x'] < t['bounds'][2]
                                        and t['bounds'][1] <= p['top_y'] < t['bounds'][3] for t in metadata['tiles'])]
    renumber = np.zeros(int(labels.max(initial=0)) + 1, np.uint32)
    for i, item in enumerate(kept, 1):
        renumber[item['tree_id']] = i
        item['tree_id'] = i
    labels = renumber[labels]
    confidence = np.where(labels > 0, confidence, 0.).astype(np.float32)
    source = np.where(labels > 0, source, 0).astype(np.uint8)
    return labels, confidence, source, kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run', required=True)
    parser.add_argument('--weights', default='best.pt')
    parser.add_argument('--prepared-dir', type=Path, default=PROJECT / 'output_15_litept_v2_no_rectangles_pointcloud/work')
    parser.add_argument('--stage1-output', type=Path, default=PROJECT / 'output_20_dualcrown3d_joint_finetune')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is required')
    torch.set_num_threads(4)
    output = args.output_dir.resolve()
    if (output / 'inference_report.json').exists():
        raise FileExistsError('Completed outputs already exist')
    root = args.root.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    weights = root / 'runs' / args.run / 'weights' / args.weights
    payload = torch.load(weights, map_location='cpu', weights_only=False)
    config = payload.get('selected_config')
    if config is None or args.weights != 'best.pt':
        raise ValueError('Scene inference requires the validation-selected best.pt')
    stage1_report = json.loads((args.stage1_output / 'inference_report.json').read_text())
    if stage1_report['checkpoint_sha256'] != protocol['stage1_sha256'] or stage1_report['mask_config'] != protocol['stage1_config']:
        raise ValueError('Stage-1 output folder does not match the frozen protocol')
    with np.load(args.prepared_dir / 'benchmark_pointcloud.npz') as archive:
        arrays = {k: archive[k] for k in archive.files}
    metadata = json.loads((args.prepared_dir / 'preparation.json').read_text())
    with np.load(stage1_report['raw_predictions']) as archive:
        raw = {k: archive[k] for k in archive.files}
    if str(raw['checkpoint_sha256']) != protocol['stage1_sha256'] or len(raw['tree_probability']) != len(arrays['coord']):
        raise ValueError('Raw stage-1 predictions do not match the checkpoint or the prepared cloud')
    output.mkdir(parents=True, exist_ok=True)
    network = VoteCenterNet(width=payload['width'], dropout=payload.get('dropout', 0.)).cuda()
    network.load_state_dict(payload['network'])
    timing = dict(stage1_forward_reused=float(raw['seconds']))
    start = time.monotonic()
    labels1, confidence1, instances1, source1 = merge_masks_with_sources(arrays, raw, protocol['stage1_config'])
    timing['stage1_consensus'] = time.monotonic() - start
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    centres, scores = detect_centres(network, arrays, raw['shifted_center'], raw['tree_probability'],
                                     labels1.astype(np.int64), threshold=.1, flips=True)
    timing['stage2_detect'] = time.monotonic() - start
    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 2
    start = time.monotonic()
    anchors = stage1_centres(raw['shifted_center'], labels1)
    if 'add_threshold' in config:
        chosen, info = select_centres(centres, scores, anchors, config['threshold'], config['add_threshold'])
        kinds = info.pop('kinds')
    else:
        kept = centres[scores >= config['threshold']]
        chosen, retained = merge_with_stage1(kept, anchors) if config.get('fallback') else (kept, np.zeros(len(anchors), bool))
        kinds = np.r_[np.full(len(kept), SOURCE_RELOCATED), np.full(int(retained.sum()), SOURCE_STAGE1_CENTRE)].astype(np.uint8)
        info = dict(detected=int(len(kept)), stage1_kept=int(retained.sum()), stage1_total=int(len(anchors)))
    labels2, confidence2, instances2, source2 = second_pass_labels(arrays, raw, protocol['stage1_config'], chosen,
                                                                  config['assign'], config['consensus'], kinds=kinds)
    timing['stage2_assign'] = time.monotonic() - start
    labels1, confidence1, source1, instances1 = core_instances(labels1, confidence1, source1, instances1, metadata)
    labels2, confidence2, source2, instances2 = core_instances(labels2, confidence2, source2, instances2, metadata)
    start = time.monotonic()
    vectors = write_vectors(instances2, metadata, output / 'Segmentation3')
    stage1_vectors = write_vectors(instances1, metadata, output / 'Stage1_Model20/Segmentation3')
    clouds = export_laz(output, arrays, metadata, labels2, labels1, confidence2,
                        (raw['point_probability'] >= .3).astype(np.uint8), source2)
    timing['export'] = time.monotonic() - start
    identifiers, first = np.unique(labels2, return_index=True)
    per_tree = source2[first[identifiers > 0]]
    provenance = {name: int((per_tree == code).sum())
                  for name, code in (('stage1_centre_kept', SOURCE_STAGE1_CENTRE), ('relocated_or_split', SOURCE_RELOCATED),
                                     ('added', SOURCE_ADDED))}
    area = [p['geometry'].area for p in instances2]
    area1 = [p['geometry'].area for p in instances1]
    report = dict(stage1_checkpoint=protocol['stage1_checkpoint'], stage1_sha256=protocol['stage1_sha256'],
                  stage1_config=protocol['stage1_config'], stage1_raw_predictions=stage1_report['raw_predictions'],
                  stage2_weights=str(weights), stage2_sha256=sha256(weights), stage2_epoch=payload['epoch'],
                  stage2_config=config, stage2_config_name=payload.get('selected_config_name'),
                  centre_selection=info, timing_seconds=timing, stage2_peak_vram_mb=peak_vram,
                  trees=len(instances2), stage1_trees=len(instances1), tree_provenance=provenance,
                  crowns_up_to_10_m2=int(sum(a <= 10. for a in area)), stage1_crowns_up_to_10_m2=int(sum(a <= 10. for a in area1)),
                  median_crown_area_m2=float(np.median(area)) if area else 0.,
                  stage1_median_crown_area_m2=float(np.median(area1)) if area1 else 0.,
                  labelled_voxels=int((labels2 > 0).sum()), stage1_labelled_voxels=int((labels1 > 0).sum()),
                  vectors=vectors, stage1_vectors=stage1_vectors, clouds=clouds, voxel_size=.25, window_size=20., overlap=8.,
                  ground_truth_available=False)
    write_json(output / 'inference_report.json', report)
    (output / 'README.md').write_text(
        '# DualCrown3D two-pass (vote-centre) inference on the reference scene\n\n'
        'Stage 1 is the frozen Model20 (same raw predictions as `output_20_dualcrown3d_joint_finetune`). Stage 2 detects '
        'tree centres in Model20\'s 3-D vote space and reassigns the votes. The configuration was frozen on validation; '
        'this scene has no ground truth, so nothing here is a measured accuracy.\n\n'
        'Open `PointClouds/trees_*.laz` in CloudCompare, accept the Global Shift and colour by RGB.\n\n'
        '- `tree_id`: final two-pass instance ID, unique across tiles; matches `treeID` in `Segmentation3/crowns_*.gpkg` and `ttops_*.gpkg`.\n'
        '- `legacy_tree_id`: **Model20 (stage-1) final ID** for the same point, for side-by-side comparison; its crowns are in `Stage1_Model20/Segmentation3`.\n'
        '- `assignment_source`: 0 = unassigned, 3 = tree whose stage-1 centre was kept, 8 = tree relocated or split by stage 2, 9 = tree added by stage 2.\n'
        '- `segmentation_status`: 0 = predicted background/ground, 1 = assigned instance, 2 = predicted tree without an instance.\n'
        '- `tree_confidence`: stage-1 tree probability of the point (uncalibrated).\n'
        '- Z is the original elevation; `height_agl` is the height above ground. CRS EPSG:2180.\n\n'
        'Crown polygons are convex hulls of the instance voxels, the same exporter as the Model20 final crowns, so the two '
        'folders are directly comparable. See `inference_report.json` for hashes, counts and timing.\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('trees', 'stage1_trees', 'tree_provenance', 'crowns_up_to_10_m2',
                                             'stage1_crowns_up_to_10_m2', 'median_crown_area_m2',
                                             'stage1_median_crown_area_m2', 'centre_selection', 'timing_seconds',
                                             'stage2_peak_vram_mb')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
