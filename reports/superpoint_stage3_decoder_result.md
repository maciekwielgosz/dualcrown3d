# Stage-3 decoder pilot: completed GPU comparison

Date: 2026-10-01. The downstream experiment was completed after the user
approved proceeding despite the failed partition-only gate. The implementation
and protocol are described in [the Stage-3 protocol](superpoint_stage3_decoder_protocol.md).

## Result

EZ-SP plus revisable point masks outperformed geometric grouping after the same
12-epoch training, and improved crown PQ relative to the equally trained
no-graph control. It did **not** improve both point and crown quality over the
common starting decoder. Joint checkpoint selection retained epoch zero for
all three arms. No new production checkpoint or whole-scene output was promoted.

These are **actual predicted instance metrics** on 14 native-ALS validation
crops, not the earlier oracle partition scores. Values below are source-balanced
PQ/F1 at IoU 0.5; they are not comparable directly with full-scene production
metrics or the classical full-scene baseline.

| Variant | Epoch | Point PQ | Point F1 | Crown PQ | Crown F1 | Joint score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Common hybrid decoder before training | 0 | 0.1862 | 0.2533 | **0.2256** | **0.3442** | **0.2050** |
| No-graph control | 12 | 0.2214 | **0.3108** | 0.1522 | 0.2354 | 0.1836 |
| Fixed 0.5 m groups + graph | 12 | 0.2001 | 0.2689 | 0.1574 | 0.2460 | 0.1775 |
| EZ-SP groups + graph | 12 | **0.2259** | 0.3034 | 0.1828 | 0.2910 | 0.2032 |

Joint score is the geometric mean of point/crown PQ. Among the trained arms,
EZ-SP improved crown PQ by 0.0306 over the control (exploratory paired within-
source plot bootstrap interval 0.0162 to 0.0402). Its point-PQ difference was
only 0.00446 (interval -0.0155 to 0.0164), and its point F1 was lower. This is
one seed on an already-explored validation set, not evidence of held-out-test
superiority. CULS and SCION each have one validation crop, so the within-source
bootstrap does not characterize their between-site uncertainty.

The original retained mask head, evaluated with the common pilot exporter,
scored point/crown PQ 0.1825/0.1588. That reference is **not** the retained
production dual-consensus pipeline: its other head and full-scene fusion are
absent from this crop experiment.

## What changed and what the experiment reveals

The implemented model is frozen LitePT72 -> superpoint pooling -> three graph
layers of width 96 -> point-specific residual -> pretrained 128-D instance
decoder with 96 queries and three masked transformer layers. Original point
features are retained. The output is predicted at point resolution, rather
than imposing one label on every superpoint. The zero-initialized residual
makes all trainable variants identical at epoch zero; observed validation
outputs confirmed this. After training, graph output weights have nonzero norms
(Fixed 0.5467, EZ-SP 0.5425).

The EZ-SP final predictions assign multiple labels within 2,122 validation
superpoints (including point/background boundaries). This verifies that mixed
groups are revisable. It does not prove that every reassignment is correct.
The point assignment stage compares duplicates pairwise and lets disputed
points compete again after removal of an undersized instance.

The remaining issue is not simply unassigned coverage: the EZ-SP arm labels
98.70% of annotated tree points, yet its pooled point matching has 87 true
positives, 349 false positives and 135 false negatives. High coverage therefore
coexists with poor instance delineation. Crown fragmentation and mismatched
instances need attention before increasing training scale. The no-graph control
shows the same tradeoff, so a partition-only diagnostic did not capture the
main downstream limitation.

This pilot's training loss supervises point semantics, offsets, embeddings and
instance masks. The complete crown polygons are used for evaluation/selection,
**not a direct crown loss**. A following bounded experiment should test explicit
full-crown consistency on the same queries, query duplicate/object-quality
supervision, and more than one training crop per parent. These are candidates
for further experiments, not proven causes or guaranteed fixes.

## Runtime, validation and artifacts

Each arm trained for 12 epochs on 79 real-ALS crops on the NVIDIA RTX PRO 500
Blackwell GPU. LitePT was frozen and its existing features cached. Training-loop
times were approximately 69 s (control), 108 s (Fixed) and 103 s (EZ-SP), excluding
preparation, validation and export. Peak allocated training VRAM was 375/1226/
1235 MiB, respectively. These are not complete backbone training or end-to-end inference times.

Mean group compression was 1.638x (Fixed) and 1.630x (EZ-SP). The EZ-SP crop
preparation includes an approximately 0.236 s label-free regularization search,
0.004 s embedding computation and 0.077 s group-graph construction. Decoder
timing alone cannot substantiate an inference speedup or seconds per hectare.

Validation included 13 passing tests, a GPU forward/backward smoke test, exact
coordinate/label alignment for all 93 prepared crops, disjoint parent train/val
groups, and matching instance IDs across LAZ, crown and treetop exports. The
96-query budget exceeds the maximum observed reference count (80 training,
68 validation trees per crop). The held-out test split was not used.

Experiment root:
`outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot/`

- [Excel log](../outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot/experiments.xlsx):
  configurations, checkpoints, epochs, validation grids, final-epoch results,
  timing, paired intervals and export checks.
- [Selected checkpoint report](../outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot/final_report.json).
- [Trained epoch-12 report](../outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot/trained_checkpoint_report.json).
- `runs/{control,fixed,ezsp}/weights/best.pt` is **epoch zero**. Use
  `weights/last.pt` to inspect the actually trained epoch-12 checkpoint.
- `runs/<method>/validation_exports/` contains selected-checkpoint outputs;
  `runs/<method>/last_epoch_validation_exports/` contains trained outputs, each
  with `PointClouds/`, `Segmentation3/` and an export manifest.
- [Trained EZ-SP exports](../outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot/runs/ezsp/last_epoch_validation_exports/).

LAZ exports contain retained input voxels from 20 m validation crops, with
`tree_id`, `reference_tree_id`, `height_agl`, confidence and categorical RGB.
Gray indicates unassigned points; Z is height AGL, not absolute elevation.
Eight source plots use local coordinates with no declared CRS in either their
native LAS headers or reference polygons. Their exports retain that local
coordinate system instead of guessing an EPSG code. Crown footprints retain
disconnected parts, close small raster gaps and fill holes without a convex
hull joining remote components.

The crop decoder pilot is complete. Whole-scene persistent-ID/overlap fusion,
independent validation, TLS/simulation transfer and teacher distillation remain
subsequent stages; this run does not claim to have completed those stages.

## Reproduction

Use a new output directory: existing experiments and exports are protected.

```bash
PYTHONPATH=/tmp/dualcrown3d_superpoint_deps_20261001 \
  .venv-gpu/bin/python scripts/prepare_superpoint_decoder_pilot.py --output NEW_RUN
for method in retained control fixed ezsp; do
  .venv-gpu/bin/python scripts/train_superpoint_decoder_pilot.py \
    --output NEW_RUN --method "$method" --epochs 12 --eval-every 4 || exit "$?"
done
.venv-gpu/bin/python scripts/report_superpoint_decoder_pilot.py --output NEW_RUN
.venv-gpu/bin/python scripts/review_superpoint_decoder_training.py --output NEW_RUN
```

The isolated partition packages and native-scatter compatibility shim are the
same as in the earlier SPT/EZ-SP test. Training/export use `.venv-gpu` directly.
