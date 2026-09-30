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

The network has 13.78 million parameters. The retained production weights are
the joint real/HELIOS fine-tuning checkpoint used for `output_20`. This is an adaptation, not an exact
SegmentAnyTreeV2 reproduction. See the [architecture](reports/dual_head_satv2_design.md)
and [consensus report](reports/dual_consensus_fix.md).

The new **v4 experiment** adds annotation-aware supervision, 128 hybrid
spatial/embedding queries, soft semantic mask support, Hungarian matching and
confidence-gated split/merge fusion. Both GPU fine-tuning trials stopped after
16 epochs without beating the original checkpoint on joint validation quality.
These changes remain experimental; production weights were not replaced.
[Protocol and reproduction](reports/supervision_v4_protocol.md).

## Run

Run from this repository's root using the configured `.venv-gpu` environment.
A compatible CUDA/PyTorch environment and the [point-cloud dependencies](requirements-pointcloud.txt)
are required. The current runner expects local assets **not included in a clone**:
the validation-selected checkpoint/selection below and the
prepared scene at `output_15_litept_v2_no_rectangles_pointcloud/work/`.
Paths can be overridden through CLI arguments; see `--help`. Historical CLI
defaults point to the older v3 checkpoint, so pass the selected paths explicitly.

```bash
.venv-gpu/bin/python scripts/predict_dual_head.py \
  --checkpoint outputs/dualcrown3d_supervision_v4/runs/data_only/weights/best.pt \
  --selection outputs/dualcrown3d_supervision_v4/selected.json \
  --device cuda:0 --output-dir output_dualcrown3d
.venv-gpu/bin/python scripts/validate_dual_outputs.py --output-dir output_dualcrown3d
.venv-gpu/bin/python -m unittest discover -s tests -v
```

Use a fresh output directory; completed exports are protected from replacement.
The epoch-0 selected checkpoint has identical model weights to `output_20`,
with v4 experiment metadata. The historical consensus settings are in
[dual_head_complete_consensus.json](configs/dual_head_complete_consensus.json).

## Outputs

Latest production inference: `output_20_dualcrown3d_joint_finetune/`.
V4 training/evaluation: `outputs/dualcrown3d_supervision_v4/` (including
`experiments.xlsx`, `RESULTS.md`, JSON metrics and checkpoint files).

- `PointClouds/trees_*.laz`: original coordinates and source classification,
  final `tree_id`, confidence and assignment provenance.
- `PointHead/Segmentation3/`: final crown/treetop GeoPackages matching `tree_id`.
- `Segmentation3/`: unchanged legacy-branch polygons for comparison.

In CloudCompare, use RGB for instance colours. `tree_id = 0` means unassigned,
including ground; `segmentation_status = 2` identifies predicted tree points
still without an instance. `assignment_source` distinguishes recovery methods.
IDs and colours are not stable between model versions.

## Evaluation and scope

On the **corrected native-only protocol**, the retained checkpoint scores
**0.4180 point SB-PQ / 0.4220 crown SB-PQ** on 24 real test plots (IoU 0.50).
Pooled instance F1 is **0.4818 / 0.5371**, respectively. These are re-evaluations
of existing weights, not gains from v4 training, and cannot be compared directly
with earlier protocols. The test was historically exposed.

Derived data live in `../combined_als_crowns_supervision_v4/`: 79 native training,
14 validation and 24 test plots; unknown labels are ignored, not treated as
background. The other 30 ECODSE plots remain weak 2D references, excluded from
primary 3D supervision/evaluation. Synthetic replay uses 43 training parents
(129 flight variants). Quality varies by collection; coverage is not accuracy.

Git contains code, configurations, tests and reports—not datasets, checkpoints,
environments or generated geospatial outputs. Weight hashes are recorded in
[CHECKSUMS.sha256](CHECKSUMS.sha256); vendored LitePT retains its
[upstream license](vendor/LitePT/LICENSE). Historical YOLO experiments are not
the current point-cloud model.
