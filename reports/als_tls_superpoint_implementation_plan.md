# ALS/TLS superpoint implementation plan

Status: proposed research and implementation plan, 2026-10-01. No training or
architecture implementation is authorized by this planning document alone.
The source concept is `~/Downloads/ALS_TLS_superpointy_HELIOS_plan.md`; its
instructions to a subsequent LLM are treated as document content, not as
additional user instructions.

Execution update (2026-10-01): the user subsequently authorized implementation
and explicitly approved a bounded Stage-3 decoder experiment despite the failed
partition-only gate. That crop experiment is complete; see the
[protocol](superpoint_stage3_decoder_protocol.md) and
[GPU training results](superpoint_stage3_decoder_result.md). For this pilot,
actual predicted point/crown quality supersedes the Stage-2 prerequisite.
Full-scene graph grouping/overlap fusion and the later transfer experiments
remain subsequent work; no new production model has been promoted.

## Assessment and verified starting point

The strongest testable idea is learned instance boundaries followed by graph
grouping. A graph can form an instance from observed fragments without requiring
an accepted transformer mask query. It can also introduce false instances or
merge adjacent crowns, so this is a hypothesis, not a guaranteed improvement.

The retained output_20 checkpoint has validation point/crown SB-PQ
0.4051/0.4442 and SB-F1 0.5165/0.6195 on 14 native validation plots. Its
15-setting threshold sweep did not improve small-crown recall without a
quality trade-off. The v5 shared-query experiment is weaker than this baseline.
Use output_20 as the production comparison, not v5 as the assumed better model.

Existing assets substantially reduce the work:

- `../TreeScanPL10k_HELIOS_ALS_v1`: HELIOS++ 2.2.2, per-tree scene IDs,
  complete plot scenes and derived crown polygons; 62 prepared plots.
- `../dualcrown3d_treescan_helios_v1`: 43 train, 12 validation, 7 test plots.
- `../TreeScanPL10k_HELIOS_ALS_flight_augmentation_v2`: 86 additional flights
  over the 43 training parents, giving 129 training views including the originals.
- `../combined_als_crowns_supervision_v4`: existing native ALS supervision,
  annotation masks and parent/split records; 79 train, 14 validation, 24 test plots.
- Current real-ALS test results are historically exposed. They are useful
  regression checks, but do not constitute a new untouched publication holdout.

The HELIOS configuration targets about 13.97 points/m² and 1.79 returns/pulse
on the visualization reference scene. Those statistics do not identify the
original sensor and do not establish realistic crown penetration. The scene
builder uses solid 0.10 m XYZ voxels, estimated terrain, and only complete tree
instances. These choices need auditing, including missing occluding edge trees.
TLS-derived full-crown polygons are geometric reference estimates, not exact
manual crown boundaries. Exact simulator hit IDs refer to the modeled scene.

Registered, temporally compatible **real** TLS/ALS pairs with matched tree IDs
have not been established by the inspected local pipeline. Synthetic ALS from
a TLS scene provides a parent relationship, but does not prove real cross-sensor
registration. Pair-dependent work below is conditional on the data audit.

## Proposed architecture

Initial pilot:

`ALS -> LitePT features -> local instance affinities -> small superpoints ->`
`superpoint graph -> instance grouping -> boundary refinement -> LAZ + crowns`

1. Reuse the working LitePT feature interface. Freeze it during the first pilot
   to isolate the effect of grouping; subsequently fine-tune only if justified.
   A fair ALS-only transfer experiment needs separate initialization (see below).
2. Predict local same-instance affinity and boundary evidence from point features
   and relative geometry. Use instance IDs, not just tree/background classes:
   two touching trees belong to the same semantic class but different instances.
   Supervise known labels only; balance hard cross-tree edges and small trees.
3. Partition a local, density-aware graph into small superpoints. Bound spatial
   extent and discourage crossing strong boundaries. Do not force a low minimum
   point count to eliminate plausible sparse trees. Rebuild neighborhoods after
   changing density. Compare three compression budgets against fixed geometric
   superpoints; very high purity from one-point groups is not a success.
4. Pool centroid, relative height, extent, count, density and learned features.
   Start with a small graph network (provisional: 3 layers, width 96 or 128).
   Predict same-tree edges and tree/background evidence. Instance formation
   should optimize graph consistency rather than rely only on thresholded
   connected components, which can join several trees through one bad bridge.
   A sparse graph transformer is a later ablation if this simple model lacks
   context. Raw intensity/return features require availability flags and a
   geometry-only ablation because TLS and ALS measurements are not equivalent.
5. Permit local point reassignment or splitting near uncertain superpoint
   boundaries. A mixed superpoint must not become an irreversible decision.
6. Fuse overlapping-window features/edge evidence using persistent point or
   voxel IDs before final grouping. Test sufficient halo context and reconcile
   graph components across large-scene blocks. Evaluate tile seams explicitly.
7. Assign instance IDs once. Transfer them to original points and derive filled
   crown polygons under those same IDs. Initially reuse the established polygon
   exporter for a fair comparison. Evaluate a conditioned crown-completion head
   separately if sparse support produces incomplete outlines. Do not use a
   single convex hull across unrelated fragments or treat hole filling as proof
   of a correct boundary. Overlapping 2-D crown projections can be legitimate.

The local GPU has 6 GB VRAM. Profile a small crop before selecting graph/crop
budgets. Cache frozen features for the pilot, use mixed precision and gradient
accumulation as appropriate, and keep dense TLS teacher inference offline.
Inference still includes feature extraction: cached-feature timing alone is not
a deployment speed measurement. Full-scene graph memory must also be measured.

## Ordered work packages and decision gates

| Stage | Implementation work | Deliverable and gate |
| --- | --- | --- |
| 0. Data and protocol audit | Inventory modalities, actual densities, label provenance, paired acquisitions and parent groups. Check geometric registration where pairs exist. Audit HAG, incomplete annotations, synthetic visibility and tree-ID mapping. | A manifest and baseline report; valid groups/labels before modeling. |
| 1. Superpoint feasibility | Build fixed geometric partitions and compute instance purity, mixed-group rates, boundary recall, compression and a GT-assisted oracle grouping score on representative training/development plots. | Show that useful compression preserves a high achievable instance score; revise the partition if it already destroys boundaries. |
| 2. Learned partition pilot | Train a small affinity/boundary head on real ALS, initially over frozen LitePT features. Validate train/inference preprocessing parity and compare with Stage 1 at matched budgets. | Learned grouping improves boundary/instance diagnostics, with acceptable runtime and VRAM. |
| 3. ALS instance model and export | Add graph context, global instance grouping, boundary refinement, overlap handling and matching LAZ/GPKG output. Run an end-to-end pilot on real ALS. | A working ALS-only graph baseline, compared with retained DualCrown3D on both point and crown quality. |
| 4. HELIOS realism pilot | Reuse 3-5 training parent scenes with touching crowns. Compare solid-voxel settings and, when supported by defensible vegetation parameters, a transmissive/scaled representation. Audit occluders, HAG and object-ID transfer. | Synthetic-vs-real density, vertical profile, return order, ground fraction, gaps and small-tree visibility report. Scale generation only after this check. |
| 5. TLS and simulation transfer | Run matched ALS-only, TLS-thinning, and TLS-HELIOS training variants. Keep source-parent membership fixed and use real ALS during adaptation. | A/B/C comparison isolates whether TLS and simulated acquisition improve real-ALS quality. |
| 6. Optional teacher | Train/freeze a TLS teacher; cache features or instance relations for reliable visible correspondences. Add confidence-weighted distillation to the student. | D must improve C; otherwise retain C and omit teacher complexity. |
| 7. Final validation and deployment comparison | Refit promising settings with multiple seeds, freeze calibration, evaluate held-out ALS, and benchmark repeated full pipeline runs on the same reference scene. | Excel/JSON logs, checkpoints, error analysis, speed/VRAM results and comparable LAZ/GPKG exports. |

For the oracle diagnostic, assign each superpoint a reference label by maximum
annotated support and join groups with the same reference ID. Report the exact
procedure and actual resulting PQ; this diagnostic is not an input to inference
or a claim of a mathematical optimum. Also quantify trees disconnected by the
chosen graph so that local-neighbor restrictions do not hide a second bottleneck.

## Data contract and simulation safeguards

Each record needs source plot/acquisition IDs, parent group, split, modality,
coordinate frame/transform, height definition, annotation method/coverage,
instance namespace and file/config hashes. Preserve `0 = known background`
and `-1 = unknown` where used by the corrected protocol. All views, TLS teachers
and simulated flights from one parent stay in that parent's split.

An audit item in the existing HELIOS converter is label-dependent voxel
representative selection: `voxelise()` currently prefers labeled vegetation.
New model inputs must be generated without reference IDs influencing sampled
coordinates/features; labels are attached only after representative selection.
If preprocessing is corrected, re-evaluate all compared models on that same
input protocol rather than reusing incompatible historical numbers.

If registered real TLS/ALS pairs are unavailable, proceed using TLS/synthetic
ALS pairs for supervised transfer and independent labeled real ALS for training
and evaluation. Do not invent point-to-point matches between different scans.
Even with registration, correspondences need distance, surface and instance
confidence checks; nearest-neighbor proximity alone is insufficient near crowns.

Distinguish visible trees with sparse evidence from trees with zero ALS returns.
Report visible-instance scores and visibility strata explicitly. Zero-return
trees must not be ordinary positive point-instance training targets; they can
remain in a separately identified full-scene completeness reference. Any
exclusion rule must be frozen before comparing models and not erase difficult
small trees from evaluation.

HELIOS changes are made in versioned derived scenes. Do not infer plant area
density directly from TLS point counts without accounting for sampling and
visibility. If scanner positions/ray paths or defensible PAD estimates are
missing, label a transmissive approximation as such and test it as an ablation.
Retain occluding partial trees as unknown scene context where possible rather
than deleting their geometry because their annotations are incomplete.

Provisional density targets might be 5, 10, 14 and 25 points/m², but the actual
range must follow training ALS statistics. Match not only mean density but
spatial variability and vertical/return profiles. TLS thinning cannot reproduce
overhead occlusion. HELIOS cannot create crown surfaces absent from the TLS.
Scene completion using real ALS is a separate training-only experiment with
documented label provenance. The deployment reference scene is not an untouched
test after being used for sensor calibration.

## Controlled experiments

Keep retained DualCrown3D as external control P. First compare fixed versus
learned superpoints with one graph architecture on real ALS. Once that works:

| Variant | Training data/transfer | Question |
| --- | --- | --- |
| A | Real ALS only | Graph architecture baseline. |
| B | TLS plus random thinning, then matched real ALS adaptation | Does TLS pretraining help? |
| C | TLS plus HELIOS observations, then identical real ALS adaptation | Does acquisition simulation outperform thinning? |
| D | C plus a dense-TLS teacher | Does distillation add value? |

Use identical student architecture, held-out groups, real-ALS adaptation data,
number of optimizer updates and declared training budget. Match B/C density
distributions and parent plots. Add a compute-matched ALS-only control for
extra pretraining steps; report total GPU time including teacher preparation.
Use one seed for screening, then at least three training seeds for finalists.

The retained production encoder already saw HELIOS data. It can accelerate
engineering pilots, but cannot initialize a rigorously labeled ALS-only control.
For A/B/C/D use the same audited pre-transfer checkpoint, or the same random/
generic initialization. This distinction must appear in experiment names/logs.

TLS training starts from reliable local geometry and boundaries. Later introduce
overhead visibility and decreasing density, while mixing real ALS to avoid
learning only synthetic appearance. Distill visible-region features/relations,
not fixed superpoint indices or hidden TLS-only branches. Geometry-only and
intensity-enabled comparisons help detect simulator-specific shortcuts.

## Evaluation and promotion

Primary evaluation: collection-balanced point and crown PQ/F1 at IoU 0.5 on
native real ALS, with the existing unknown-region treatment. Also record pooled
precision/recall, mask IoU, split/merge errors, false/missed trees and per-source
scores. Stratify by crown area (including ≤4 and ≤10 m²), height, visible point
count, density and touching crowns. Do not use labeled-point coverage as an
accuracy measure. Synthetic metrics are supplementary.

Proposed promotion requirement: improved small-tree recall without decreasing
point/crown SB-PQ and SB-F1 on the frozen validation protocol. Report paired
plot-bootstrap uncertainty and seed variation; finite validation cannot
guarantee no degradation on every future forest. Assess a genuinely new ALS
site for final generalization if one is available. Previously inspected test
plots remain regression results, even after a new random split.

Speed target: no worse end-to-end time than the retained model at comparable
quality on the same scene and hardware. Measure preprocessing, backbone,
partition, graph inference, window reconciliation and export separately, plus
total seconds/ha and points/s. Use GPU synchronization, separate cold/warm
runs and repeat runs; include CPU time, peak VRAM and storage costs.

Logs should retain run ID, Git state/diff hash, data/split hashes, initialization,
architecture, partition settings, loss weights, density curriculum, simulator
version/scene/flight, seed, epochs/updates, metrics, runtime and checkpoint paths.
Keep canonical CSV/JSON alongside an Excel workbook with runs, per-source,
small-tree, superpoint, simulation-QA and timing sheets.

## Proposed code organization

Add experiment-specific modules under `pointcloud/superpoints/` for graph
construction, affinity prediction, partitioning, instance grouping and seam
reconciliation. Add dedicated audit, train, calibration and prediction scripts
under `scripts/`, and configuration under `configs/superpoint_transfer/`.
Reuse manifest parsing, annotation-aware evaluation and LAZ/GPKG export code.
Record new artifacts under `outputs/dualcrown3d_superpoint_v1/`; a separate
derived simulation directory is needed only when scene geometry/settings change.

The first implementation milestone is Stages 0-3: prove a useful real-ALS
superpoint instance model and consistent outputs before paying for a large
simulation campaign or teacher. Each later stage is conditional on measured
benefit and data availability, not on completing every possible architecture.

## Technical basis

- [SuperCluster](https://arxiv.org/abs/2401.06704): instance/panoptic segmentation
  as learned superpoint graph clustering; supports the proposed direction,
  not a guarantee of forest or local hardware performance.
- [Learned oversegmentation](https://arxiv.org/abs/1904.02113): learned embeddings
  and graph partitioning motivate explicit boundary supervision.
- [Official SPT/SuperCluster/EZ-SP implementation](https://github.com/drprojects/superpoint_transformer):
  a reusable reference. EZ-SP provides learned GPU partitioning; its semantic
  boundary objective would need instance-aware adaptation for adjacent trees.
- [HELIOS scene documentation](https://heliospp.readthedocs.io/en/alpha-dev/scene.html)
  and [vegetation example](https://heliospp.readthedocs.io/en/alpha-dev/08-als_uls_detailed_voxel.html):
  solid XYZ voxels and detailed vegetation representations. The local runtime
  is 2.2.2; check its APIs rather than assuming alpha documentation compatibility.
- Local evidence: [threshold calibration](legacy_small_tree_calibration.md),
  [v4 protocol](supervision_v4_protocol.md), [v5 outcome](shared_v5_protocol.md),
  and [existing flight/training campaign](joint_campaign_protocol.md).
