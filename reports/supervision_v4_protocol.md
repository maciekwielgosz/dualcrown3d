# DualCrown3D v4: annotation repair and decoder ablations

The source datasets and the previous outputs are unchanged. Derived data are in
`../combined_als_crowns_supervision_v4`. All 147 plots retain their previous
train/validation/test and spatial-group assignments. Local-coordinate datasets
retain their original unspecified CRS; no invented georeferencing is added.

## Annotation contract

- `tree_id > 0`: annotated tree instance, including annotated dead trees.
- `tree_id = 0`: known background or explicitly labelled bush.
- `tree_id = -1`: unknown instance annotation; excluded from instance losses and
  point-instance evaluation. Unknown does not mean non-tree.
- A separate `semantic_target` permits known tree components without a known
  instance ID to supervise tree/background classification.
- FOR-instance class 3 is outside the annotated area and is ignored. These are
  the dataset's component classes, not generic ASPRS classifications.
- WildForest3D zero IDs are unknown. Its documented zero-height ground is known
  background. Explicit BUSH instances are non-tree targets and are removed from
  the new tree-crown reference; original polygons and taxonomy remain preserved.

Native point coordinates/features are byte-identical to the previous prepared
inputs. Reconstructed original voxel representatives must exactly match before
component labels are transferred. No nearest-neighbour label guessing is used.

ECODSE's 30 plots contain polygon-projected IDs, not independently annotated 3D
instances. They are retained as weak 2D references, excluded from 3D training
and primary metrics. Heights are re-estimated using PDAL SMRF and HAG NN and
audited against the supplied CHM. This remains an estimated terrain surface.
Incomplete annotation coverage prevents a defensible all-scene precision/F1;
supplemental weak-crown results must therefore be labelled recall-only.

The resulting native primary sets contain 79 training, 14 validation and 24 test
plots. Synthetic replay uses the existing 43 training parents and 129 flight
variants; validation/test parents are never used for training.

## Model experiments

Both trials start from output_20's checkpoint `repeat_20260930/weights/best.pt`.
`data_only` retains the existing 96-query decoder. `hybrid_queries` uses 128
queries, 1536 memory tokens, 3 attention layers and the same transferable weight
shapes. Half the seed budget is spatial FPS, the rest is embedding FPS. Spatial
cell representatives reduce density bias before the memory budget is applied.

The hybrid decoder scores masks on all crop points. Semantic probability is a
soft prior rather than a hard mask veto. Seed teacher forcing decreases to zero
over eight epochs; evaluation never reads reference IDs. Hungarian matching
supervises distinct masks even when their initial seeds miss a tree. Unknown
points and unmatched unknown-seed proposals do not become negative labels.

The deepest encoder stage and feature upsampling are trainable after four head
warm-up epochs; early encoder layers and normalization buffers remain fixed.
Training uses CUDA, 160 draws/epoch, 16k-point crops, two-crop gradient accumulation,
AdamW, head LR 1e-4, backbone LR 1e-5 and at most 40 epochs. Early stopping uses
12 epochs without validation improvement after at least 16 epochs.

## Fusion and evaluation

The historical merger remains supported. New split/merge proposals can revise
existing vote instances using high-confidence masks, mutual coverage and bounded
3D residual assignment. Each point still has at most one ID. Final IDs are
consecutive and polygons remain filled. Assignment provenance 6/7 records a
split/merge. Validation compares historical, split, split+merge, strict revision
and mask-anchor configurations; no test results select a configuration.

Primary selection is the geometric mean of collection-balanced point/crown PQ
at IoU 0.5. Point metrics ignore explicitly unknown annotation. Crown metrics
ignore unmatched predictions dominated by cells containing only unknown
non-ground point labels. Mixed cells remain scored; duplicates over annotated
reference crowns are not forgiven. Ignore masks do not derive from the union of
GT crown polygons. Test results remain a historically exposed regression test.

The original checkpoint is scored under the same new annotation protocol before
fine-tuning. Old and new protocol scores are not interchangeable. A higher score
caused by correcting evaluation is not itself a model improvement. One training
seed is used per variant; the previous campaign's three-seed uncertainty does
not apply to these trials.

## Reproduction

From `DL_model_version`:

```bash
.venv-gpu/bin/python scripts/prepare_supervision_v4.py
.venv-gpu/bin/python -m unittest discover -s tests -v
.venv-gpu/bin/python scripts/train_supervision_v4.py --smoke
.venv-gpu/bin/python scripts/train_supervision_v4.py
.venv-gpu/bin/python scripts/report_supervision_v4.py
```

Outputs, training logs, selections, model arguments, checkpoint paths/hashes and
Excel logs are under `outputs/dualcrown3d_supervision_v4`. Incomplete training
directories are protected from silent overwrite.

## Observed outcome

Both GPU trials stopped after 16 epochs. Selection retained `data_only` epoch 0:
its model tensors are exactly equal to the original output_20 checkpoint.
Non-regression acceptance therefore does **not** mean that fine-tuning improved
the model. Existing production exports were not replaced.

| Validation-frozen model | Val point SB-PQ | Val crown SB-PQ | Test point SB-PQ | Test crown SB-PQ |
|---|---:|---:|---:|---:|
| Original / retained | 0.4051 | 0.4442 | 0.4180 | 0.4220 |
| Hybrid queries, epoch 4 | 0.4075 | 0.4383 | 0.4238 | 0.4187 |

These values use the corrected native-only annotation protocol. The hybrid's
real-test point F1 is 0.4827 versus 0.4818; crown F1 is 0.5305 versus 0.5371.
Its test result is a descriptive ablation, evaluated after validation selection,
not a reason to change the selected checkpoint. The point PQ difference is
+0.0058, but crown PQ is -0.0033: this is a trade-off, not an overall gain.

The new revision code performs actual splits and merges with the legacy decoder
(recorded in `validation_ablations.csv`). At the tested confidence thresholds,
it made **no revisions** for the hybrid decoder. Its masks did not pass all
revision gates; confidence calibration and mask coherence remain limitations.
The v4 experiment does not demonstrate a benefit from adaptive fusion yet.

Independent verification: 52 unit tests pass; all 147 plot split/group/source
hash assignments remain unchanged; native coordinates and input features are
unchanged. All 6,510 retained reference crown geometries exactly match their
originals; 411 explicitly labelled WildForest BUSH references are excluded.
No early-encoder or normalization-buffer changes occurred during fine-tuning.
Detailed per-collection scores, paired bootstrap intervals, weak-reference
recall, timing, epoch logs and checkpoint hashes are in the output directory.
