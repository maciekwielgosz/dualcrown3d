# Support-preserving mask merging

The previous merger rejected an entire candidate whenever more than 15% of its
points were already labelled. Complementary points in a suppressed mask were
therefore lost even if they belonged to an accepted tree.

`support_fusion_v2` retains the original accepted masks as immutable anchors.
For each overlapping candidate it requires:

- point-set IoU of at least 0.30 with one anchor;
- at least 80% of the candidate's already-labelled support belonging to that
  anchor, to reject ambiguous masks spanning multiple trees;
- mask probability of at least 0.50 for every recovered point;
- distance of at most 1.5 m from an original anchor point in 3D;
- an existing legacy XY centre vote within 2 m of the anchor's median XY vote.

Only previously unassigned points are recovered. Competing offers are resolved
by mask probability times object score times anchor IoU. Existing anchor labels
cannot be overwritten, and recovered points cannot create new matches or
transitive expansion. Both polygons and treetops are rebuilt from final point
support. Historical configurations without `merge_strategy` retain the exact
legacy algorithm; the runner now defaults to `configs/dual_head_inference.json`.

## Validation

The epoch-4 checkpoint is unchanged (SHA256
`5c253e8bbf8dd77ef8acf603695a08df565b26424447536fc808a8745db140a5`).
All comparisons use the same cached predictions from the production window
runner on 18 validation plots. No test plots were used. These are paired,
recomputed metrics: the historical validation runner sampled dense windows
differently, so its original report is not the control for this correction.

| Metric | Legacy suppression | Selected support fusion |
|---|---:|---:|
| Source-balanced point PQ@0.50 | 0.143660 | 0.150508 |
| Pooled point F1@0.50 | 0.119636 | 0.128739 |
| Pooled point precision | 0.128492 | 0.138268 |
| Pooled point recall | 0.111922 | 0.120438 |
| Source-balanced crown PQ@0.50 | 0.188052 | 0.202945 |
| Pooled crown F1@0.50 | 0.179541 | 0.189283 |
| Labelled reference-foreground fraction | 0.710886 | 0.741045 |

Boundary expansion without centre-vote checks increased coverage but degraded
PQ, and was not selected. Interior-only recovery preserved the crown outlines
but also slightly reduced source-balanced point PQ. The selected vote-guided
variant improved both point and crown PQ. The model remains experimental and
this correction does not recover every missing tree or guarantee full point
coverage. Coverage is not a substitute for instance accuracy.

## Benchmark re-export

The four corrected LAZ tiles in `output_17_dual_head_support_fusion` preserve
all 8,069,167 original points, their XYZ/intensity/classification, normalized
heights, and the legacy branch's point labels. Validation passed for all eight
crown/treetop pairs and all four LAZ files. The 26 unit/protocol tests passed.

The correction assigns 57,356 previously unassigned points and loses zero
existing assignments. Coverage of non-ground points at least 2 m above terrain
increases from 69.28% to 72.27%; the mask branch still has 4,801 tree instances.
One tree's highest labelled point moves across a tile edge, changing the first
two polygon counts by -1/+1 while preserving the global tree count. This is
consistent with the existing treetop-based ownership contract.

| Tile | Previous canopy coverage | Corrected canopy coverage |
|---|---:|---:|
| 764000_197000 | 69.21% | 71.79% |
| 764000_197500 | 70.11% | 73.45% |
| 764500_197000 | 68.36% | 71.85% |
| 764500_197500 | 69.65% | 73.22% |

The export took 34.8 seconds using the existing prediction cache. The reported
265.3-second inference time belongs to the original cached inference. The
benchmark lacks reference tree instances; these coverage figures are not F1
or PQ. Machine-readable checks and coverage tables (JSON, CSV and Excel) are
beside the new point-cloud outputs.

Each family has its complete parameter/metric log and Excel workbook under
`outputs/dual_head_support_fusion_v2/`: the initial boundary trials at the root,
interior trials in `interior/`, and the selected vote-guided trials in `vote/`.
`vote/selected.json` contains the full paired evaluation and checkpoint hash.
Architecture details remain in `dual_head_satv2_design.md`.

## Reproduction

Run from `DL_model_version` with the prepared data and the checkpoint available:

```bash
.venv-gpu/bin/python scripts/calibrate_dual_support_fusion.py cache
.venv-gpu/bin/python scripts/calibrate_dual_support_fusion.py evaluate \
  --family vote --output-dir outputs/dual_head_support_fusion_v2/vote \
  --cache-dir outputs/dual_head_support_fusion_v2/raw_val
.venv-gpu/bin/python scripts/predict_dual_head.py --stage export \
  --raw-predictions output_16_dual_head_satv2_pointcloud/work/dual_predictions.npz
.venv-gpu/bin/python scripts/validate_dual_outputs.py \
  --output-dir output_17_dual_head_support_fusion
.venv-gpu/bin/python scripts/compare_dual_point_coverage.py
```

The new output directory contains `PointClouds/trees_*.laz`, the legacy vectors
in `Segmentation3/`, and mask-branch vectors in `PointHead/Segmentation3/`.
The original outputs remain available for comparison. Re-exporting the cached
benchmark predictions does not run the network again. The inference report
separates the historical inference time from this export's elapsed time.

For CloudCompare select **Properties > Colors > RGB** to show instance colours.
When using the `tree_id` scalar field, restore the full **displayed** ID range
on each cloud. Changing only the saturation range does not restore hidden
points. ID zero denotes an unassigned point, including ground points.
