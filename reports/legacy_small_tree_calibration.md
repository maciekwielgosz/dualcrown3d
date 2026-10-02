# Retained DualCrown3D: small-tree threshold calibration

The retained `output_20` checkpoint was frozen. We calibrated only its legacy
instance postprocessing on the corrected, native-only **validation** split:
14 plots from the combined dataset, with 39 reference crowns up to 4 m² and
161 up to 10 m². The test split and production exports were not used for
selection or modified. Crown recall below uses one-to-one IoU ≥ 0.5 matching
against reference polygons; point and crown SB-PQ/F1 use the established
evaluation protocol.

We tested 15 configurations, changing minimum instance voxels, height and
area; mask/object thresholds; vote probability; and peak threshold. The
backward-compatible code defaults remain 12 voxels, 2 m and 0.75 m². Raw
network predictions were cached with checkpoint and input hashes, then
re-merged for every configuration. Re-running the baseline reproduced the
archived validation SB-PQ/F1 within 1e-6.

| Validation setting | Point SB-PQ | Crown SB-PQ | Point SB-F1 | Crown SB-F1 | Recall ≤4 m² | Recall ≤10 m² |
|---|---:|---:|---:|---:|---:|---:|
| Retained baseline | 0.4051 | 0.4442 | 0.5165 | 0.6195 | 0/39 | 6/161 |
| 6 voxels, 1 m, 0.25 m², mask 0.3 | 0.4080 | 0.4448 | 0.5201 | 0.6209 | 0/39 | 6/161 |
| 8 voxels, 1.5 m, 0.4 m², vote probability 0.2 | 0.4020 | 0.4378 | 0.5108 | 0.6086 | 0/39 | 7/161 |

The middle setting offers a **small aggregate metric gain**, but does not
recover an additional small reference crown. Its crown SB-PQ difference is
only +0.0006; a stratified plot bootstrap (2,000 resamples within the five
source collections) gave a 95% interval of roughly [-0.0003, +0.0027]. It
is an exploratory quality-only candidate, not evidence that the small-tree
problem is fixed. Lowering the vote threshold finds one more crown up to
10 m² but reduces both point and crown quality. No setting passed the
predefined promotion rule: improved ≤10 m² recall **and** non-inferior
source-balanced PQ/F1 for both outputs. The retained baseline was selected.

The diagnostic signal points upstream of the final size filter: lowering
minimum size alone has virtually no effect. A separate inspection of the
cached raw proposals found that most small annotated trees had semantic
foreground support, but very few individual window mask proposals covered
a whole small tree at point IoU ≥ 0.5. This is not a formal stitched-output
metric; it suggests fragmented/merged instances or missing proposals rather
than simply a threshold set too high. Substantially improving small-tree
recall likely requires better proposals and training supervision, then a new
validation-only calibration. Relaxing thresholds further cannot be assumed
to preserve precision.

The full per-plot results, all configurations, selected baseline and Excel
table are in `outputs/dualcrown3d_legacy_small_tree_calibration_extended/`.
To reproduce the sweep from the frozen checkpoint (requires the local dataset
and CUDA environment), use a **fresh** output directory:

```bash
.venv-gpu/bin/python scripts/calibrate_legacy_small_trees.py \
  --output outputs/dualcrown3d_legacy_small_tree_calibration_recheck
```
