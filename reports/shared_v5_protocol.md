# Shared-instance v5 experiment

Status: experimental. The retained production checkpoint and output_20 are unchanged.

## Design

One LitePT-S feature backbone feeds a transformer instance decoder with 128
queries. Each query predicts (1) a mask over ALS voxels, (2) a complete crown
mask on a 0.5 m BEV grid, (3) objectness, and (4) estimated mask quality.
The point and crown masks share the same query embedding and therefore the
same instance identity. The older centre-vote and semantic branches remain
auxiliary; they no longer assign exported tree IDs.

Training matches predicted queries to annotated trees jointly in 3-D points
and complete 2-D crown polygons. The polygon target is rasterized after the
exact training-crop transform. Unknown regions are ignored. Candidate queries
are seeded from canopy predictions at inference, not from ground truth. A
pairwise window merge joins duplicate proposals, retains spatially distinct
ones with unique support, then resolves disputed points by proposal score.
Each accepted query produces one ID in both LAZ and GeoPackage.

## Reproduction

From the repository root with `.venv-gpu` and local data/checkpoints present:

```bash
.venv-gpu/bin/python scripts/train_shared_instances.py \
  --output outputs/dualcrown3d_shared_v5_new --epochs 16
.venv-gpu/bin/python scripts/calibrate_shared_instances.py \
  --checkpoint outputs/dualcrown3d_shared_v5_new/run/weights/best.pt \
  --output outputs/dualcrown3d_shared_v5_new_calibration
.venv-gpu/bin/python scripts/predict_shared_instances.py \
  --checkpoint outputs/dualcrown3d_shared_v5_new/run/weights/best.pt \
  --selection outputs/dualcrown3d_shared_v5_new_calibration/selected.json \
  --output-dir output_21_shared_instances_v5_experimental
```

The actual two-stage pilot used the first 16-epoch run in
`outputs/dualcrown3d_shared_v5_full_retry/` and an 8-epoch quality-head
warm-start in `outputs/dualcrown3d_shared_v5_quality_fix/`. Its final
checkpoint and validation-only threshold selection are in
`outputs/dualcrown3d_shared_v5_quality_fix/run/weights/best.pt` and
`outputs/dualcrown3d_shared_v5_calibration_geomfix/selected.json`.
Training and calibration logs, parameters, and checkpoints remain in these
ignored generated-output directories; no checkpoint is committed to Git.
The training script refuses to overwrite an existing run. Use a new output
path to repeat a run.

## Validation outcome

The frozen comparison uses the same 14 real native validation plots as the
v4 baseline; the held-out test set was not used for selecting this variant.

| Model | Point SB-PQ | Crown SB-PQ | Point SB-F1 | Crown SB-F1 |
| --- | ---: | ---: | ---: | ---: |
| Retained consensus baseline | 0.4051 | 0.4442 | 0.5165 | 0.6195 |
| v5, validation-calibrated | 0.1610 | 0.2066 | 0.2329 | 0.3218 |

The current v5 fails the acceptance gate, which requires non-inferiority on
both SB-PQ and SB-F1 for both outputs. Do not use it as a quality upgrade.
The main observed issue is false positive instance proposals: in the selected
configuration, point precision is 0.112 and crown precision is 0.185. The
shared-query design fixes output identity consistency, but it has **not** yet
fixed instance detection quality. More epochs or looser thresholds alone
cannot be assumed to solve this; objectness and query matching need further
work, potentially with hard-negative mining and source-balanced calibration.

The `Segmentation3` and `PointHead/Segmentation3` outputs of v5 intentionally
contain the same IDs and polygons. `tree_id=0` in the LAZ means unassigned;
`pred_semantic=1` with no ID still marks a potential missed tree. These
semantics differ from the retained dual-head output_20, where the two vector
folders represent different branches.

The end-to-end export test is `output_21_shared_instances_v5_experimental/`:
four LAZ tiles and paired GeoPackages with 12,857 unique IDs. Every exported
polygon ID occurs in the LAZ; both vector folders have identical geometries
and IDs. The 7,724,226-voxel scene took 353.3 s for GPU prediction over 5,398
windows and another 255.7 s for merging and export, about 609 s total on the
local RTX PRO 500. This is a functional test, not a benchmark comparison to
output_20, and the high instance count is consistent with weak precision.
This particular output used the earlier threshold selection before correcting
the raster-to-polygon component index. The corrected full validation picked a
stricter object threshold (0.4 versus 0.1). The functional export is therefore
