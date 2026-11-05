#!/usr/bin/env python3
"""v7 two-pass inference on the prepared reference scene: full-resolution LAZ and GeoPackages.

Stage 1 reuses the frozen Model20 raw predictions of an existing Model20 output
folder (hash-checked) for its labels. The votes handed to stage 2 are recomputed
as a rotation ensemble of the same checkpoint and cached next to the output.
Stage 2 is the vote-field detector with the configuration frozen on validation.
Outputs follow the Model20 export conventions; this scene has no ground truth.
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
from pointcloud.dual_fusion import instance_records
from pointcloud.instance_output import merge_masks_with_sources
from pointcloud.vote_ensemble import ensemble_votes
from pointcloud.vote_field import SOURCE_ADDED, SOURCE_RELOCATED, SOURCE_STAGE1_CENTRE, second_pass
from scripts.cache_stage1_predictions import DEFAULT_OUTPUT, sha256, write_json
from scripts.predict_dual_head import export_laz, write_vectors
from scripts.predict_two_pass_scene import core_instances
from scripts.train_supervision_v4 import build
from scripts.build_two_pass_pipeline import load_assign_network
from scripts.train_centre_verifier import load_verifier
from scripts.validate_vote_field import load_network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run', required=True)
    parser.add_argument('--weights', default='best.pt')
    parser.add_argument('--prepared-dir', type=Path, default=PROJECT / 'output_15_litept_v2_no_rectangles_pointcloud/work')
    parser.add_argument('--stage1-output', type=Path, default=PROJECT / 'output_20_dualcrown3d_joint_finetune')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ensemble-cache', type=Path, default=DEFAULT_OUTPUT / 'scene_ensemble_votes.npz')
    parser.add_argument('--no-flips', action='store_true')
    parser.add_argument('--fallback', type=float, default=0.,
                        help='coverage fallback: keep a Model20 tree when the second pass left at least this share of its '
                             'voxels unlabelled (0 = off). Evaluated on validation only; not part of the tested candidate.')
    parser.add_argument('--stage2-only', type=Path, help='compute the second-pass labels, save them to this npz and stop '
                                                         '(for running stage 2 on another machine; no export)')
    parser.add_argument('--stage2-from', type=Path, help='use second-pass labels computed with --stage2-only')
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
    from pointcloud.paths import remap
    with np.load(remap(stage1_report['raw_predictions'])) as archive:
        raw = {k: archive[k] for k in archive.files}
    if str(raw['checkpoint_sha256']) != protocol['stage1_sha256'] or len(raw['tree_probability']) != len(arrays['coord']):
        raise ValueError('Raw stage-1 predictions do not match the checkpoint or the prepared cloud')
    output.mkdir(parents=True, exist_ok=True)
    symmetries = int(payload['symmetries'])
    timing = dict(stage1_forward_reused=float(raw['seconds']))
    ensemble = None
    if args.ensemble_cache.exists():
        with np.load(args.ensemble_cache) as archive:
            if (str(archive['checkpoint_sha256']) == protocol['stage1_sha256'] and int(archive['symmetries']) == symmetries
                    and len(archive['shifted_center']) == len(arrays['coord'])):
                ensemble = {k: archive[k].astype(np.float32) for k in ('shifted_center', 'tree_probability', 'vote_spread')}
                timing['vote_ensemble_reused'] = float(archive['seconds'])
    if ensemble is None:
        model, _ = build(remap(protocol['stage1_checkpoint']))  # protocol paths are the workstation's
        start = time.monotonic()
        ensemble = ensemble_votes(model, arrays, symmetries)
        timing['vote_ensemble'] = time.monotonic() - start
        np.savez_compressed(args.ensemble_cache, checkpoint_sha256=np.asarray(protocol['stage1_sha256']),
                            symmetries=np.int64(symmetries), seconds=np.float64(timing['vote_ensemble']),
                            shifted_center=ensemble['shifted_center'], tree_probability=ensemble['tree_probability'],
                            vote_spread=ensemble['vote_spread'].astype(np.float16))
        del model
        torch.cuda.empty_cache()
    start = time.monotonic()
    labels1, confidence1, instances1, source1 = merge_masks_with_sources(arrays, raw, protocol['stage1_config'])
    timing['stage1_consensus'] = time.monotonic() - start
    signature = dict(stage2_sha256=sha256(weights), voxels=int(len(arrays['coord'])), flips=not args.no_flips)
    if args.stage2_from:
        with np.load(args.stage2_from, allow_pickle=False) as archive:
            stored = json.loads(str(archive['signature']))
            if stored != signature:
                raise ValueError(f'Stored second-pass labels do not match this run: {stored} vs {signature}')
            if not np.array_equal(archive['stage1_labels'], labels1):
                raise ValueError('Stored second-pass labels were computed from different stage-1 labels')
            labels2, confidence2, source2 = archive['labels'], archive['confidence'], archive['source']
            info = json.loads(str(archive['info']))
            timing['stage2'], peak_vram = float(archive['seconds']), float(archive['peak_vram_mb'])
            timing['stage2_machine'] = str(archive['machine'])
    else:
        network = load_network(payload)
        verifier = load_verifier(payload)
        assign_network = load_assign_network(payload)
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        labels2, confidence2, source2, info = second_pass(network, arrays, ensemble, labels1, config, flips=not args.no_flips,
                                                          verifier=verifier, assign_network=assign_network)
        timing['stage2'] = time.monotonic() - start
        peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 2
        if args.stage2_only:
            import platform
            np.savez_compressed(args.stage2_only, labels=labels2, confidence=confidence2, source=source2,
                                stage1_labels=labels1, signature=np.asarray(json.dumps(signature)),
                                info=np.asarray(json.dumps(info)), seconds=np.float64(timing['stage2']),
                                peak_vram_mb=np.float64(peak_vram), machine=np.asarray(platform.node()))
            print(json.dumps(dict(saved=str(args.stage2_only), seconds=timing['stage2'], info=info)))
            return
    fallback_instances = 0
    if args.fallback > 0:
        from pointcloud.vote_field import fallback_to_stage1
        labels2, source2, fallback_instances = fallback_to_stage1(labels2, source2, labels1, args.fallback)
        confidence2 = np.where(labels2 > 0, ensemble['tree_probability'], 0.).astype(np.float32)
    instances2 = instance_records(arrays, labels2, confidence2)
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
                                     ('added', SOURCE_ADDED), ('stage1_fallback', 10))}
    area = [p['geometry'].area for p in instances2]
    area1 = [p['geometry'].area for p in instances1]
    report = dict(stage1_checkpoint=protocol['stage1_checkpoint'], stage1_sha256=protocol['stage1_sha256'],
                  stage1_config=protocol['stage1_config'], stage1_raw_predictions=stage1_report['raw_predictions'],
                  vote_ensemble_symmetries=symmetries, stage2_weights=str(weights), stage2_sha256=sha256(weights),
                  stage2_epoch=payload['epoch'], stage2_config=config, stage2_config_name=payload.get('selected_config_name'),
                  centre_selection=info, timing_seconds=timing, stage2_peak_vram_mb=peak_vram,
                  coverage_fallback=dict(threshold=args.fallback, instances=fallback_instances,
                                         note='validation-only option; the held-out tests were run without it'),
                  trees=len(instances2), stage1_trees=len(instances1), tree_provenance=provenance,
                  crowns_up_to_10_m2=int(sum(a <= 10. for a in area)), stage1_crowns_up_to_10_m2=int(sum(a <= 10. for a in area1)),
                  median_crown_area_m2=float(np.median(area)) if area else 0.,
                  stage1_median_crown_area_m2=float(np.median(area1)) if area1 else 0.,
                  labelled_voxels=int((labels2 > 0).sum()), stage1_labelled_voxels=int((labels1 > 0).sum()),
                  vectors=vectors, stage1_vectors=stage1_vectors, clouds=clouds, voxel_size=.25, window_size=20., overlap=8.,
                  ground_truth_available=False)
    write_json(output / 'inference_report.json', report)
    (output / 'README.md').write_text(
        '# DualCrown3D two-pass v7 (vote-field detector) on the reference scene\n\n'
        'Stage 1 is the frozen Model20 (same raw predictions as `output_20_dualcrown3d_joint_finetune`). Stage 2 averages '
        'Model20\'s centre votes over four rotations, detects tree centres and their reach in the vote space and '
        'reassigns the votes. The configuration was frozen on validation; this scene has no ground truth, so nothing '
        'here is a measured accuracy.\n\n'
        'Open `PointClouds/trees_*.laz` in CloudCompare, accept the Global Shift and colour by RGB.\n\n'
        '- `tree_id`: final two-pass instance ID, unique across tiles; matches `treeID` in `Segmentation3/crowns_*.gpkg` and `ttops_*.gpkg`.\n'
        '- `legacy_tree_id`: **Model20 (stage-1) final ID** for the same point, for side-by-side comparison; its crowns are in `Stage1_Model20/Segmentation3`.\n'
        '- `assignment_source`: 0 = unassigned, 3 = tree whose stage-1 centre was kept, 8 = tree whose centre stage 2 placed near a stage-1 centre (relocated or split), 9 = tree added by stage 2, 10 = Model20 tree kept by the coverage fallback (only with `--fallback`).\n'
        '- `segmentation_status`: 0 = predicted background/ground, 1 = assigned instance, 2 = predicted tree without an instance.\n'
        '- `tree_confidence`: tree probability of the point (uncalibrated).\n'
        '- Z is the original elevation; `height_agl` is the height above ground. CRS EPSG:2180.\n\n'
        'Crown polygons are convex hulls of the instance voxels, the same exporter as the Model20 final crowns, so the two '
        'folders are directly comparable. See `inference_report.json` for hashes, counts and timing.\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('trees', 'stage1_trees', 'tree_provenance', 'crowns_up_to_10_m2',
                                             'stage1_crowns_up_to_10_m2', 'median_crown_area_m2',
                                             'stage1_median_crown_area_m2', 'centre_selection', 'timing_seconds',
                                             'stage2_peak_vram_mb')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
