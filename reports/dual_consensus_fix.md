# Dual-head consensus and residual canopy recovery

## Scope and cause

`output_17_dual_head_support_fusion` recovered complementary mask support but
kept the mask decoder's accepted instances as immutable anchors. It could not
recover an entire missing tree. In `764000_197000`, 288,142 non-ground returns
at height >= 2 m remained unassigned (28.2%). Of these, 93.8% were predicted
as trees and 169,295 already had a legacy-branch instance assignment.

This correction changes postprocessing, not the trained architecture or weights.
The epoch-4 checkpoint SHA256 remains
`5c253e8bbf8dd77ef8acf603695a08df565b26424447536fc808a8745db140a5`.
Predictions made on the GPU are reused; no additional training is claimed.

## Implemented algorithm

`pointcloud/dual_fusion.py` implements both fusion directions. Validation found
the centre-vote branch to be the stronger instance partition, so the selected
configuration uses its instances as anchors, with masks providing complementary
support. This is not the old mask partition with renamed IDs.

1. Generate vote peaks in world-aligned 100 m cores with 8 m context; peaks
   belong to one core. Assign points against the global peak set, so labels
   do not stop at processing-core edges. Use the same algorithm for validation
   plots and benchmark scenes. This differs from the historical per-tile
   legacy clustering, whose separate output remains unchanged.
2. Match each complementary mask to the fixed anchors by shared point indices,
   requiring at least four shared voxels and 80% dominance among claimed points.
   Integer IDs from the two branches are never assumed to correspond.
3. Recover matching unassigned points from already accepted masks if
   the predicted XY centre is within 2 m of the anchor's median centre vote,
   and the point is within 1.5 m of existing support in 3D.
4. Recover an independent mask instance when no more than 5% of its points
   overlap anchors, at least 12 unassigned voxels remain, height reaches 2 m,
   and crown area is at least 0.75 m². These masks already passed the original
   object/mask filters (object >= 0.1, point mask >= 0.5). Do not require the
   other head's semantic probability or centre vote to confirm a mask-only
   tree: doing so made its supposedly independent fallback ineffective.
5. Run one further, non-iterative completion pass from frozen support. Require
   tree probability >= 0.5, height >= 0.5 m, spatial distance < 1.5 m,
   predicted-centre distance <= 4 m and a 0.25 m margin over the second-best
   centre. No growth propagates through previously recovered points in that pass.
6. Rebuild filled, hole-free crown polygons and treetops from final membership.
   Preserve all original LAS points/coordinates/attributes and zero ground
   class-2 instance IDs at export.

The intermediate `output_18` improved average coverage but removed much of a
young stand. A sampled rectangle there had median legacy tree probability
0.144 versus 0.388 from the newer head; only 6.1% of canopy voxels passed the
legacy 0.3 threshold, versus 85.5% for the newer head. This diagnostic motivated
the independent fallback. Its parameters were selected on validation, not
against target-site ground truth (which is unavailable). This was a
target-informed engineering iteration, not an untouched benchmark evaluation.
There is no requirement that all ground/understory/uncertain points receive IDs.

## Validation and selection

All 18 validation plots from the no-rectangles manifest were used, with the
same cached network predictions for every variant. Input/checkpoint hashes
are verified. Ground truth is used only in scoring, never in fusion. No test
plots were evaluated for this correction. Annotation remains a mixture of
native point instances and polygon-projected labels, with the existing crown
ignore-area protocol preserved.

The growth ablation table contains 34 configurations, including controls and
some deliberately equivalent ablations; the subsequent fallback table has
11 configurations including both controls and the intermediate selection.
The initial point-PQ-only selection
favoured a conservative 82.2%-coverage configuration. For the two-output task,
the final selection uses the geometric mean of source-balanced **point and
crown PQ**, subject to both PQs and both pooled F1s not regressing versus
`output_17`, plus an increase in labelled reference-foreground coverage.
The final fallback study allows 0.001 absolute joint-PQ tolerance, then chooses
higher labelled reference-foreground coverage among near-best eligible trials.
The tolerance was set before running that study. This criterion change and
the visual diagnostic that prompted the additional study are explicit.

| Metric, IoU >= 0.50 | Previous support fusion | Vote-only control | Selected consensus |
|---|---:|---:|---:|
| Source-balanced point PQ | 0.150508 | 0.249639 | 0.254455 |
| Pooled point F1 | 0.128739 | 0.182832 | 0.183621 |
| Pooled point precision | 0.138268 | 0.131621 | 0.131510 |
| Pooled point recall | 0.120438 | 0.299270 | 0.304136 |
| Source-balanced crown PQ | 0.202945 | 0.327423 | 0.338821 |
| Pooled crown F1 | 0.189283 | 0.236568 | 0.242280 |
| Labelled reference-foreground fraction | 74.10% | 81.70% | 92.47% |

Most of the accuracy gain comes from choosing the stronger anchor branch.
Completion and independent fallback then increase coverage and crown quality.
Final point/crown PQ and F1 exceed both controls, but precision is slightly
below the previous mask partition as recall increases substantially. The best
strict-joint-PQ fallback had point PQ 0.254569 / crown PQ 0.339169 and 92.445%
coverage; the selected near-best variant has 92.472% coverage and retains full
accepted mask support rather than applying a second semantic threshold.
Quality remains uneven across collections; for example ECODSE point F1 is
only 0.001845. These validation estimates are not a fresh held-out test result.

Full parameters, per-plot metrics, checkpoint provenance and the selection rule:

- `configs/dual_head_complete_consensus.json`
- `outputs/dual_head_consensus_v3/mask_fallback/configuration_trials.json`
- `outputs/dual_head_consensus_v3/mask_fallback/selected.json`
- `outputs/dual_head_consensus_v3/mask_fallback/experiments.xlsx`

The earlier `growth/` and `bidirectional/` folders retain their own ablations.
`configs/dual_head_consensus.json` is the frozen intermediate control.

## Output and inspection

Final outputs are under `output_19_dual_head_complete_consensus`; earlier output folders
are preserved. `PointClouds/trees_*.laz` carries final `tree_id`, and
`PointHead/Segmentation3` has matching `treeID` crown/treetop pairs. The separate
`Segmentation3` and `legacy_tree_id` preserve the historical legacy branch.
IDs and RGB colours are not stable between output_17 and output_19 because
the anchor partition changed. Do not compare trees by equal numeric IDs.

Additional LAZ scalar fields:

- `segmentation_status`: 0 predicted background/ground, 1 assigned instance,
  2 predicted tree still without an instance. This is a diagnostic, not truth.
- `assignment_source`: 0 unassigned, 1 mask anchor, 2 cross-head completion,
  3 vote-head instance, 4 local recovery, 5 added mask-only instance.

For the final vote-anchored fallback configuration, `tree_confidence` is an
uncalibrated mask-times-quality score for sources 2/5 and legacy semantic
probability for sources 3/4. These are not comparable accuracy estimates.

## Final benchmark audit

The final four LAZ files preserve **8,069,167** source points and their original
XYZ, intensity, classification and normalized height. All eight legacy vector
files have identical feature attributes/geometries to output_17. All final
GeoPackage crown/treetop pairs have matching IDs, valid filled polygons and
EPSG:2180, and every exported point instance has a corresponding crown.
All **39 tests** pass, including independent fallback, ambiguous boundaries,
world-aligned vote cores, historical API compatibility and LAZ provenance.

| Tile | Previous canopy coverage | Final canopy coverage |
|---|---:|---:|
| 764000_197000 | 71.79% | 84.09% |
| 764000_197500 | 73.45% | 77.69% |
| 764500_197000 | 71.85% | 82.04% |
| 764500_197500 | 73.22% | 86.33% |
| All four | 72.27% | 83.78% |

Here canopy is a proxy: non-ground returns with height >= 2 m, not reference
tree labels. Coverage is not accuracy. Unassigned canopy falls from 28.21%
to 15.91% on the pictured tile, but is not eliminated. Remaining difficult
areas, including the young stand, still need model improvements; this is not
a guarantee that every crown is complete or correctly separated.

The changed partition recovers **386,241** previously unassigned points while
**191,615** previously assigned points become unassigned (all heights), a net
gain of **194,626** labels. Thus this is not a monotonic label-only fill, and
some local areas can regress despite better per-tile coverage. There are
8,240 final crown instances versus 4,801 in output_17. No claim is made that
this count is the true tree count.

Re-export took 48.3 s, including fusion, legacy vectors, final vectors and
full-resolution LAZ writing/round-trip checks, with cached network outputs.
The separately reported 265.3 s inference time is historical, not a newly
measured GPU run or a full end-to-end pipeline measurement.

`output_19_dual_head_complete_consensus` contains `validation.json`,
`coverage_comparison.json/csv/xlsx` and `coverage_map_764000_197000.png`.
The spatial map was inspected: the large young-stand regression in the
intermediate output_18 is removed, though smaller residual gaps remain.

## Reproduction

Run in `DL_model_version` using the existing `.venv-gpu` environment:

```bash
.venv-gpu/bin/python scripts/calibrate_dual_consensus.py \
  --growth-study --selection-objective balanced \
  --output-dir outputs/dual_head_consensus_v3/growth
.venv-gpu/bin/python scripts/calibrate_dual_consensus.py \
  --fallback-study --joint-pq-tolerance .001 \
  --output-dir outputs/dual_head_consensus_v3/mask_fallback
.venv-gpu/bin/python scripts/predict_dual_head.py --stage export \
  --selection configs/dual_head_complete_consensus.json \
  --raw-predictions output_16_dual_head_satv2_pointcloud/work/dual_predictions.npz \
  --output-dir output_19_dual_head_complete_consensus
.venv-gpu/bin/python scripts/validate_dual_outputs.py \
  --output-dir output_19_dual_head_complete_consensus
.venv-gpu/bin/python scripts/compare_dual_point_coverage.py \
  --before output_17_dual_head_support_fusion --after output_19_dual_head_complete_consensus
.venv-gpu/bin/python scripts/visualize_dual_coverage.py \
  --before output_17_dual_head_support_fusion --after output_19_dual_head_complete_consensus
.venv-gpu/bin/python -m unittest discover -s tests -v
```

The exporter refuses to overwrite a completed output. Use a fresh output
directory when reproducing. Validation caches can be regenerated with
`scripts/calibrate_dual_support_fusion.py cache`. `--select-only` on the new
calibrator re-selects from a complete, metadata-checked validation ablation
table; a smoke subset cannot be used for selection.
