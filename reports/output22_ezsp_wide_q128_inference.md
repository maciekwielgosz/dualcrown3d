# Output 22: wide EZ-SP q128 full-scene inference

Date: 2026-10-01. The selected epoch-12 checkpoint from the wide decoder pilot
was applied to the same four native ALS tiles used by
`output_21_shared_instances_v5_experimental`. The model was not retrained for
this scene. No reference tree IDs or polygons entered inference.

Output: [`output_22_ezsp_wide_q128_experimental/`](../output_22_ezsp_wide_q128_experimental/)

| Tile | Original points in LAZ | Crown polygons |
| --- | ---: | ---: |
| 764000_197000 | 1,987,560 | 6,957 |
| 764000_197500 | 1,615,120 | 1,421 |
| 764500_197000 | 1,895,902 | 2,156 |
| 764500_197500 | 2,570,585 | 2,925 |
| **Total** | **8,069,167** | **13,459** |

The frozen LitePT backbone generated 72-D features in 5,398 overlapping
20 m windows. An EZ-SP partition and the trained four-layer, width-192 graph
fed the 128-query decoder with two additional width-192 mask-attention layers.
Its validation-selected object/mask thresholds are 0.1/0.5. Window merging
used pairwise candidate sorting, overlap suppression and limited point-support
fusion. Crown polygons were reconstructed from final point IDs using 0.5 m
occupancy, small-gap closing and hole filling. This model does **not** have the
independent full-crown raster head used by shared-v5 output_21.

The GPU prediction pass took approximately 1,221 s according to its progress
log; collecting proposals, global merging and export took 63.8 s. This is
roughly 21.4 minutes end-to-end on the local RTX PRO 500. These are this
specific experiment's elapsed times, not a controlled throughput benchmark.
The run considered 171,162 local proposals, yielding 13,459 global IDs. The
instance count alone is not an accuracy measure; the pilot still had low
precision and weak small-tree recall.

## Contents and verification

- [`PointClouds/`](../output_22_ezsp_wide_q128_experimental/PointClouds/): four LAZ tiles with original XYZ, intensity, classification and extra `tree_id`, confidence, `pred_semantic`, `height_agl`, assignment source and segmentation status. `tree_id=0` is unassigned.
- [`Segmentation3/`](../output_22_ezsp_wide_q128_experimental/Segmentation3/): full-crown support polygons and treetops as GeoPackages, with benchmark-compatible tile names and `treeID`.
- [`PointHead/Segmentation3/`](../output_22_ezsp_wide_q128_experimental/PointHead/Segmentation3/): intentionally identical polygons and IDs, provided for the familiar folder layout; it is not a second independent head.
- [`inference_report.json`](../output_22_ezsp_wide_q128_experimental/inference_report.json): checkpoint hashes, parameters, tile counts and export timing.
- `work/stripes/`: 72 resumable prediction shards; leave them in place for provenance or repeat export.

Independent verification compared all four output_22 LAZ files with output_21:
point counts and original X/Y/Z, intensity and classification match exactly.
All 13,459 polygon IDs occur in the LAZ union; matching crown and treetop IDs
and valid crown geometries were checked for each tile. The two vector folders
have equal geometries and IDs. Five focused tests passed.

This is an **experimental whole-scene transfer**. The thresholds were selected
on 20 m validation crops, not calibrated on stitched scenes. Full-scene PQ,
F1 and small-tree recall cannot be inferred from the number of polygons; the
four output tiles have no matching complete reference labels in this run. A
visual comparison with output_21 is useful, but it does not establish a quality
ranking. Output_21 itself used an earlier shared-v5 threshold selection.

Reproduce prediction and export without overwriting this output by choosing a
new output directory:

```bash
PYTHONPATH=/tmp/dualcrown3d_superpoint_deps_20261001 \
  .venv-gpu/bin/python scripts/predict_superpoint_wide_scene.py \
  --output-dir NEW_OUTPUT --stage predict
PYTHONPATH=/tmp/dualcrown3d_superpoint_deps_20261001 \
  .venv-gpu/bin/python scripts/predict_superpoint_wide_scene.py \
  --output-dir NEW_OUTPUT --stage export
```
