# DualCrown3D

Research code for individual-tree instance segmentation directly from LiDAR
point clouds, exporting labelled LAZ files and filled crown polygons (GeoPackage).

## Current model

- **Input:** XYZ with height above ground + intensity; 0.25 m voxels,
  20 × 20 m inference windows with 8 m overlap.
- **Shared LitePT-S backbone** with a centre-vote MLP branch and a
  SegmentAnyTreeV2-inspired transformer mask branch: 3 layers, 4 attention
  heads, up to 96 queries per window.
- **Dual-head consensus:** combines vote instances, complementary masks,
  independent mask-only trees and bounded local point recovery.

The network has 13.78 million parameters. The latest correction changes
postprocessing, not trained weights. This is an adaptation, not an exact
SegmentAnyTreeV2 reproduction. See the [architecture](reports/dual_head_satv2_design.md)
and [consensus report](reports/dual_consensus_fix.md).

## Run

Run from this repository's root using the configured `.venv-gpu` environment.
A compatible CUDA/PyTorch environment and the [point-cloud dependencies](requirements-pointcloud.txt)
are required. The current runner expects local assets **not included in a clone**:
the checkpoint at `outputs/dual_head_satv2_litept_v3/weights/best.pt` and the
prepared scene at `output_15_litept_v2_no_rectangles_pointcloud/work/`.
Paths can be overridden through CLI arguments; see `--help`.

```bash
.venv-gpu/bin/python scripts/predict_dual_head.py --device cuda:0 --output-dir output_dualcrown3d
.venv-gpu/bin/python scripts/validate_dual_outputs.py --output-dir output_dualcrown3d
.venv-gpu/bin/python -m unittest discover -s tests -v
```

Use a fresh output directory; completed exports are protected from replacement.
The selected settings are in [dual_head_complete_consensus.json](configs/dual_head_complete_consensus.json).
Training and cached-prediction reproduction commands are in the linked reports.

## Outputs

Latest local result: `output_19_dual_head_complete_consensus/`.

- `PointClouds/trees_*.laz`: original coordinates and source classification,
  final `tree_id`, confidence and assignment provenance.
- `PointHead/Segmentation3/`: final crown/treetop GeoPackages matching `tree_id`.
- `Segmentation3/`: unchanged legacy-branch polygons for comparison.

In CloudCompare, use RGB for instance colours. `tree_id = 0` means unassigned,
including ground; `segmentation_status = 2` identifies predicted tree points
still without an instance. `assignment_source` distinguishes recovery methods.
IDs and colours are not stable between model versions.

## Evaluation and scope

On 18 validation plots, source-balanced PQ at IoU 0.50 improved from
**0.1505 to 0.2545** for point instances and **0.2029 to 0.3388** for crowns.
The current update did not evaluate the test set. Quality varies by collection,
and unsegmented regions remain; coverage is not accuracy.
[Full metrics, limitations and experiment-log locations](reports/dual_consensus_fix.md).

Git contains code, configurations, tests and reports—not datasets, checkpoints,
environments or generated geospatial outputs. Weight hashes are recorded in
[CHECKSUMS.sha256](CHECKSUMS.sha256); vendored LitePT retains its
[upstream license](vendor/LitePT/LICENSE). Historical YOLO experiments are not
the current point-cloud model.
