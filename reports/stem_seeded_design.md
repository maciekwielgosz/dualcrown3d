# Stem-seeded instance segmentation (Model20 decoder with one query per stem foot)

Status 2026-10-14 (night). Runs `ss_v1` (job 3365646) and `ss_v2` (job 3365694, 78 min, early stop
at epoch 28, best epoch 12, sha256 `a0ba0d75...`) done; feet evaluations 3365658 / 3365695 done.
Outputs on the Athena mirror under `outputs/stem_seeded_v1/runs/`. **Verdict so far: the decoder is a
gain only with good feet; it does not refine Model20 on Model20's own feet (v2 0.341 against 0.396).**

### Result of `ss_v1` (val splits only; source-balanced point PQ at IoU 0.5, Model20 protocol)

Model20 with its production consensus on the same 28 val plots reproduces its known supervision-v4
val score (0.4055 against 0.4051), so the metric path is the same.

| method (`feet_eval`, 40k points per window) | FOR-inst./IDEAS (14) | HELIOS (12) | Weiser real (2) | all | recall | precision |
|---|---:|---:|---:|---:|---:|---:|
| Model20 | 0.406 | 0.524 | 0.216 | **0.396** | 0.517 | 0.445 |
| stem-seeded, labelled feet | 0.572 | 0.606 | 0.734 | **0.600** | 0.718 | 0.718 |
| stem-seeded, Model20 feet | 0.334 | 0.487 | 0.177 | **0.333** | 0.483 | 0.375 |
| stem-seeded, Model20 feet, object score >= 0.35 | 0.286 | 0.430 | 0.283 | 0.306 | 0.366 | 0.401 |

- With correct feet the decoder beats Model20; the fair comparison is HELIOS (fully annotated):
  0.606 against 0.524. On the partly annotated sets the gain is inflated (no feet, hence no false
  positives, on unannotated trees).
- With Model20's own instances as feet it is worse than Model20 (0.333 against 0.396): it reproduces
  Model20's split crowns (two pieces, two feet) and its object score does not reject the extra foot
  (mean score of false feet 0.27 against 0.35 for true ones in the corrupted-feet validation).
- So the gain sits in the feet: the decoder is ready for a good stem detector, but is not by itself a
  refinement of Model20.

### `ss_v2` (changes against v1)

Object score trained as real/false foot (BCE, `--objectness`) instead of a regression to the mask IoU;
training corruption adds a false "fragment" foot 1.5-4 m from 20 % of the true feet (a crown cut in
two, as Model20 does) and shifts feet by up to 1.5 m; 60 epochs x 640 crops; validation also runs the
decoder on Model20's feet (cached per val plot) at object thresholds 0 and 0.5, and selection is that
score (`--select model20`), the configuration that can be deployed today. Local check (600 crops,
3 small plots): false/true foot score 0.39 / 0.58 (v1 0.27 / 0.35).

Result (`ss_v2/feet_eval`): labelled feet 0.606 (HELIOS 0.622); Model20 feet 0.341 at object threshold 0,
0.342 at 0.5 (recall 0.49, precision 0.41) against Model20 0.396. Objectness now separates feet
(false / true 0.23 / 0.67 at epoch 28) but dropping feet does not recover what Model20's split
instances cost. Next cheap test: feet from the two-pass v7 centres (`ens4_b`), a detector that already
beats Model20 (test point PQ 0.512 against 0.418); then StemField.

## Idea

The trunk pipeline's largest remaining error is the instance it is handed (held-out Weiser:
DBH 11.6 cm with Model20 + merge against 8.5 cm with the reference segments; 76 % against 98 % of
trees found; see `trunk_reconstruction/reports/RESULTS.md`). Model20 decides "how many trees and
where" from crown shapes, with farthest-point seeds and a consensus merge. Here the stems are given
(a stem-foot detector, GraphStem feet, or for the ceiling the measured feet), and the decoder answers
only "which points belong to each stem": one query per foot, one mask per query, assignment by
per-point argmax, **no NMS and no merge step**. A tree can no longer be split into two instances or
glued to its neighbour unless the feet themselves are wrong, and the decoder is trained with wrong
feet so that it can reject them (object score) and shift a mask off a misplaced foot (up to 1 m).

This is idea 3 of `trunk_reconstruction/reports/LITERATURE_SEGMENTATION.md` (instance-aware
seeding) applied to Model20, and step 4b of the stem-first plan (`stem field -> assignment ->
GraphStem`); the cone rule is step 4a and the baseline this must beat.

## What was built

| file | content |
|---|---|
| `pointcloud/stem_seeded.py` | `StemSeededDecoder` (policy `stem_seeded`, subclass of `TreeMaskDecoder`: Model20's weights reused, new modules `stem_token`, `stem_height`, `stem_score_embed`, `stem_column`, `stem_shift`); `stem_mask_losses` (dice + focal per real foot, focal on false feet, score regression to the mask IoU, per-point cross-entropy over stems + background); `assign_points`; `corrupt_stems` (drop up to 25 %, shift 0.3-1 m, height noise, unknown height, false feet at mid-points and in the canopy, duplicates); `stems_for_crop` / `stems_to_local`; `predict_stems` (20 m windows every 6 m with Model20's point ownership: each point is assigned in the one window whose central 6 m cell owns it, by the argmax over every foot inside that window; dense windows covered in chunks) |
| `pointcloud/dual_head.py` | `decoder_policy='stem_seeded'` and the loss dispatch (two small additions; `legacy`, `hybrid_v4`, `shared_v5` unchanged) |
| `pointcloud/stem_data.py` | `StemDrawDataset`: Model20's crops (`prepare_crop`, same augmentation) plus the feet moved into the crop frame and corrupted; `source_weights` per source group |
| `scripts/prepare_stem_feet.py` | `stems.csv` next to every `points_0p25m.npz`: HELIOS plots from the TLS-traced trunks (`trunk_reconstruction/data/labels`), Weiser real scans from `labels_weiser`, Weiser simulated tiles from the plot labels shifted into the tile frame, FOR-instance/IDEAS plots from a lowest-points proxy (quality 0.2-0.5, excludable with `--min-quality 0.6`); `--check` reports coverage without writing |
| `scripts/train_stem_seeded.py` | fine-tuning from Model20 (`partial` scope as the campaign: decoder, legacy heads, enc4 and upsampling; 4 warm-up epochs heads only) on every train split available: supervision-v4 real plots (79 `train_eligible` FOR-instance/IDEAS plots, proxy feet), HELIOS (43 plots + 86 flight variants), Weiser real (35 scans) and Weiser simulated tiles (298); draw shares for_instance 1 / helios 1 / weiser_real 1 / weiser_sim 0.5. Validation on the 28 val plots (14 supervision-v4, 12 HELIOS, Weiser BR02/BR08 d14 s0) with labelled feet clean and corrupted, source-balanced point PQ (Model20 protocol, `evaluate_combined_full_crowns.aggregate`); Model20 + production consensus scored once on the same plots (`baseline_model20/`); selection = corrupted-feet PQ; `--resume`, `--smoke` |
| `scripts/predict_stem_seeded.py` | paired scans -> `trees_<scan>.laz` (+ `reference.csv`, `manifest.csv`) in the layout of `trunk_reconstruction/tools/run_dualcrown3d.py`, so `evaluate_paired.py` and `segmentation_errors.py` score it unchanged; feet from `reference.csv` (oracle) or a directory of `<scan>.csv` |
| `slurm/train_stem_seeded.sbatch` | Athena job (`sbatch --job-name=ss_v1 slurm/train_stem_seeded.sbatch ss_v1 --epochs 40`) |
| `tests/test_stem_seeded.py` | shapes, loss classes (real / false / ignored feet), gradients reach the new modules and not the legacy branch, Model20 state loads with only `stem_*` missing, a 40-step toy fit separates two trees, assignment and corruption helpers |

### Query construction

For foot `s` at local `(x, y)` with tree height `h` (0 = unknown) and input score `c`:

```
q_s = stem_token + position([x, y, 0]) + stem_height([h/30, h>0]) + stem_score_embed(c) + stem_column(column_s)
column_s = Gaussian-weighted (sigma 1.5 m, cut 4 m, all heights) mean of the memory tokens around the foot
```

Masks as in Model20 (`8 * cos(query, point)` + locality prior `2.5 - 0.5 d^2 / r^2` + 0.25 x semantic
logit), with the locality centre at `foot + 1 m * tanh(stem_shift(q))` and the learned radius head;
three masked cross-attention layers over the 0.75 m-deduplicated memory (1,024 tokens). Points of
all heights can receive a mask (as in `hybrid_v4`). The object score is trained to the mask IoU
(0 for false feet), so a detector's false feet can be dropped by `--object-threshold`.

### Supervision classes of a foot (`stem_tree_id`)

- `> 0`: a tree with at least 4 labelled points in the crop -> its mask is the target; the first foot
  of a tree owns its points in the assignment cross-entropy (a duplicate shares the mask target).
- `0`: false foot -> empty mask, score 0.
- `-1`: foot of a tree without labelled points in the crop (edge, unknown) -> query only, no loss.
Points with `tree_id -1` (unlabelled vegetation) are excluded from every loss, as in Model20.

## How to run (when decided)

```bash
cd DL_model_version
.venv-gpu/bin/python tests/test_stem_seeded.py                                  # CPU, seconds (10 tests)
.venv-gpu/bin/python scripts/prepare_stem_feet.py --check --manifests ../dualcrown3d_weiser_v1/manifest.csv
.venv-gpu/bin/python scripts/prepare_stem_feet.py                               # writes stems.csv (all four manifests)
.venv-gpu/bin/python scripts/train_stem_seeded.py --run smoke --smoke           # one crop, VRAM, gradients
sbatch --job-name=ss_v1 slurm/train_stem_seeded.sbatch ss_v1 --epochs 40        # Athena (data copy needs stems.csv too)
# oracle ceiling on the paired tuning plots (feet = measured stems), then the trunk-side scores:
.venv-gpu/bin/python scripts/predict_stem_seeded.py --checkpoint outputs/stem_seeded_v1/runs/ss_v1/weights/best.pt \
    --manifest ../trunk_reconstruction/data/paired/weiser/manifest.csv --densities d14 --stems reference \
    --out ../trunk_reconstruction/data/paired/weiser_stemseed_ref
cd ../trunk_reconstruction && python tools/segmentation_errors.py ...   # split / swallowed / kept against Model20
```

Decisions on the tuning plots only; the held-out plots of `configs/real_holdout.json` are in the
`test` split of `dualcrown3d_weiser_v1` and are never read by the training or validation code.

## Gates (tuning plots, before any held-out read)

1. **Oracle feet** (reference stems as queries): split share and swallowed share clearly below
   Model20's 39 % / 24 % on Weiser, and GraphStem DBH on the instances approaching the
   reference-segment value (8.5 cm on the holdout; on the tuning plots 6.2 against 10.7 with Model20).
   If this fails, the assignment itself is the problem and no detector will fix it.
2. **Corrupted feet** (validation protocol): corrupted-feet PQ within 0.05 of the clean PQ, false feet
   scored below 0.3 on average, true feet above 0.7.
3. **Detector feet** (two-pass v7 centres, GraphStem feet on Model20 instances, or StemField once it
   exists): `segmentation_gate.py` against Model20 + merge 0.8, and the cone-rule assignment (step 4a)
   on the same feet. Adopt only if it beats the cone.

## Known limits of this implementation

- **Validation favours the stem model on precision in partly annotated plots** (Weiser real, many
  FOR-instance/IDEAS plots): feet exist only for annotated trees, while Model20 also segments the
  unannotated trees and some of those instances count as false positives. Compare recall and the
  trunk-side scores (`evaluate_paired.py`), not only PQ, when the feet come from labels.

- FOR-instance / IDEAS feet are proxies from the lowest points; in dense data they are usually
  within a metre of the stem, but they are not measured. Train with `--sources ...,for_instance=0`
  or `--min-quality 0.6` to exclude them.
- Brandenburg plots are not in any DualCrown3D dataset (no point-level instance labels), so they
  cannot supervise masks; they remain evaluation-only through `predict_stem_seeded.py`.
- `predict_stems` decodes each foot once, in the window that owns it; a crown that crosses the
  window's context edge (10 m from the foot) is cut there. Model20's predictor has the same 20 m
  windows.
- The memory of the cross-attention is capped at 1,024 tokens per window as in Model20; masks are
  evaluated on all points.

## Stem-first pipeline (2026-10-15): StemField detector and the end-to-end chain

Built after the v2 result (the decoder needs good feet). All runs on Athena (`plgdyplomanci7-gpu-a100`).

| file | content |
|---|---|
| `pointcloud/stem_field.py` | StemField: points -> 3-D grid (0.5 x 0.5 x 1 m, 48 m; log count, intensity, occupancy, layer height, window density) -> 3-D encoder with per-level height pooling -> 2-D U-Net -> heat / sub-cell offset / log tree height at 0.5 m (3.4 M parameters). CenterNet focal loss; cells under mostly unannotated vegetation and cells without points ignored. `detect_plot` (32 m tiles, central 20 m kept), `match_feet` (Hungarian within 1.5 m, detections in unannotated vegetation excused) |
| `scripts/train_stem_field.py` | 24 m crops of every train split with the feet of `stems.csv`, half of the dense crops thinned to 6-40 pts/m2; validation = whole-plot detection F1 per group, selection = group mean at the best threshold |
| `scripts/cache_feet_sources.py` | feet (and labels) of Model20, two-pass v7 `ens4_b` and v7 + fallback 0.7 on the 28 val plots |
| `scripts/evaluate_stem_seeded_feet.py` | per foot source: detection against labelled feet, PQ of the source's own segmentation, PQ of the stem-seeded decoder on its feet |
| `scripts/predict_stem_field.py` | feet for paired real scans: StemField, or the instances of a scan |
| `../trunk_reconstruction/slurm/stem_chain.sbatch` | end to end on the paired tuning scans (Brandenburg 16 plots, protocol v2; Weiser BR02, BR08): Model20 baseline, stem-seeded decoder on StemField / Model20 / measured feet (and v7 + fallback on Brandenburg), GraphStem round 3, `evaluate_paired.py --match position` |

Jobs of 2026-10-15: `feet_cache` 3367803, `sf_v1` 3367804, `ss_v2_v7` 3367805, `ss_v2_all` 3367806, `stem_chain` 3367807.

### Results of 2026-10-15 (Athena jobs 3367803-07, all completed)

StemField `sf_v1`: 36 epochs (early stop), best epoch 24, threshold 0.3, sha256 `4f853f45...`; 26 min.

Feet against the labelled feet on the 28 val plots (one-to-one within 1.5 m; detections in unannotated vegetation excused):

| source | recall | precision | F1 | median distance |
|---|---:|---:|---:|---:|
| Model20 instances | 0.556 | 0.697 | 0.619 | 0.57 m |
| two-pass v7 instances | 0.622 | 0.852 | 0.719 | 0.44 m |
| v7 + fallback | 0.649 | 0.796 | 0.715 | 0.45 m |
| **StemField** | **0.810** | 0.779 | **0.794** | **0.32 m** |

StemField is the best stem detector on every val group (HELIOS F1 0.855 against 0.650 / 0.716), but the
stem-seeded decoder on its feet segments worse than on any other source: point PQ 0.339 (labelled feet
0.606, v7 feet 0.430, Model20 feet 0.341, v7's own segmentation 0.533, Model20 0.396). On HELIOS, where
StemField's feet are close to the labels in count and position, PQ is 0.492 against 0.622 with labels:
diagnostic `scripts/diagnose_feet_variants.py` (job 3368001) swaps positions and heights between the two.

End to end (GraphStem round 3, `evaluate_paired.py --match position`, d14 seed 0; `stemchain_sf_v1/SUMMARY.md`):

| set | segmentation | matched | DBH RMSE | DBH bias | position | n |
|---|---|---:|---:|---:|---:|---:|
| Brandenburg 16 tuning plots, protocol v2 | Model20 | **0.532** | 7.21 | -1.01 | **1.20** | 884 |
| | v7 + fallback | 0.510 | 6.80 | +1.12 | 1.24 | 848 |
| | stem-seeded on Model20 feet | **0.546** | 7.12 | -0.36 | 1.25 | 908 |
| | stem-seeded on v7 + fallback feet | 0.526 | **6.69** | +0.93 | 1.23 | 874 |
| | stem-seeded on StemField feet | 0.412 | 8.90 | +3.40 | 1.27 | 685 |
| | stem-seeded on measured feet (ceiling) | 0.702 | 7.03 | +1.14 | 1.05 | 1166 |
| Weiser BR02 + BR08 | Model20 | 0.909 | 9.54 | -6.36 | 0.72 | 50 |
| | stem-seeded on Model20 feet | 0.909 | **6.78** | -3.69 | 0.70 | 50 |
| | stem-seeded on StemField feet | 0.655 | 7.56 | +1.79 | **0.45** | 36 |
| | stem-seeded on measured feet (ceiling) | 0.964 | 7.64 | +2.21 | 0.61 | 53 |

(`matched` here is the share of all reference trees, understorey included in the denominator; 27 % of the
Brandenburg trees are understorey under protocol v2.) On real Brandenburg StemField under-detects (7,223
feet against 8,464 for Model20; 12.7 % of measured stems end without an instance against 1.6 %): no real
pine/oak stand at this density is in its training data. Weiser has only 2 plots / 50 trees here.

Diagnostic (job 3368001, `outputs/stem_seeded_v1/runs/ss_v2/feet_diagnose/diagnose.json`), 12 HELIOS val plots, stem-seeded `ss_v2`:

| feet | point PQ | recall | precision | tree points unassigned |
|---|---:|---:|---:|---:|
| labelled xy + labelled h | 0.622 | 0.826 | 0.829 | 8.3 % |
| labelled xy + StemField h | 0.541 | 0.729 | 0.734 | 8.2 % |
| StemField xy + labelled h | 0.534 | 0.719 | 0.719 | 9.9 % |
| StemField xy + StemField h | 0.489 | 0.662 | 0.662 | 8.9 % |
| StemField xy + unknown h | 0.466 | 0.642 | 0.640 | 2.4 % |
| StemField feet matched to a label only | 0.528 | 0.666 | 0.780 | 12.8 % |

StemField heights against the labels: median 22.8 against 21.7 m, mean ratio 1.17, MAE 3.0 m. The decoder
loses about 0.08 PQ from 0.3 m position errors and about 0.08 from height errors; false feet cost about
0.04. Its training noise (Gaussian shifts, +-15 % height) does not resemble a detector's errors, so it
learned to rely on exact feet. Next: train the decoder on StemField's own feet on the training plots
(matched feet keep their tree, unmatched ones are false), with stronger height noise or a calibrated
height head; for Brandenburg, add the measured stems of the tuning plots to StemField's training
(feet only, ignore outside the plot) with plot-wise splits.

### Round 3 (2026-10-15/16): Brandenburg in StemField, decoder on detector feet

- Data: `scripts/prepare_brandenburg_stems.py` -> `../dualcrown3d_brandenburg_stems_v1` (16 tuning plots x d14 s0-2,
  d10 s0, native; folds A/B of 8 plots; measured stems inside the 50 x 50 m plot box, buffer = unknown, understorey
  stems = ignore). Held-out plots refused.
- `sf_v2A` / `sf_v2B` (StemField + one Brandenburg fold): val F1 0.738 / 0.730 (sf_v1 0.739). **On real
  Brandenburg the scores collapse** (median 0.19, against 0.28 for sf_v1): at threshold 0.1 sf_v2A finds 110 of
  131 canopy stems of BB_08 (sf_v1: 100), at the val-selected 0.2-0.3 almost none. The measured stems lie about
  0.9 m from the detections (0.3 m elsewhere), so the heat is spread. Fix without retraining:
  `scripts/calibrate_stem_field.py` picks threshold and peak window on the detector's own training fold
  (`calibration_bb.json`); the chain (`stem_chain.sbatch`, 4th argument) applies it to the other fold.
- `ss_v3` (decoder trained on StemField sf_v1 feet, `scripts/cache_detector_feet.py`, 70 % of crops; heights x0.7-1.4):
  best checkpoint = epoch 0 under the 3-group selection, but by group (StemField feet, t = 0) HELIOS 0.489 -> 0.517
  (fully annotated, fair), FOR 0.333 -> 0.332, Weiser real (2 partly annotated plots) 0.216 -> 0.115. The chain
  decides on measured trees: runs `ss_v2_sf2` (decoder ss_v2) and `ss_v3last_sf2` (ss_v3 last.pt, epoch 24).
- Lesson: selection on PQ over partly annotated plots is unreliable (Weiser real); use HELIOS + FOR or trunk-side scores.

Results of round 3 (jobs 3369190, 3369656-59, all completed 2026-10-16). Calibration on own fold: sf_v2A threshold
0.16 / window 5, sf_v2B 0.16 / 3. Val plots, sf_v2A feet against labels: F1 0.777, median 0.35 m (Model20 0.619,
v7 0.719). End to end (GraphStem round 3, d14 s0; Brandenburg out-of-fold, protocol v2):

| set | segmentation | matched | DBH RMSE | DBH bias | position | n |
|---|---|---:|---:|---:|---:|---:|
| Brandenburg 16 | Model20 | 0.532 | 7.21 | -1.01 | 1.20 | 884 |
| | v7 + fallback | 0.510 | 6.80 | +1.12 | 1.24 | 848 |
| | ss_v2 on StemField v2 feet | 0.502 | 7.82 | +2.06 | 1.26 | 835 |
| | ss_v3 (last) on StemField v2 feet | 0.496 | 7.12 | +0.64 | 1.25 | 824 |
| | ss_v3 (last) on Model20 feet | **0.554** | 7.08 | +0.00 | 1.27 | 921 |
| Weiser BR02+BR08 | Model20 | 0.909 | 9.54 | -6.36 | 0.72 | 50 |
| | ss_v2 on StemField v2 feet | 0.800 | 8.76 | +2.86 | 0.65 | 44 |
| | ss_v3 (last) on StemField v2 feet | 0.800 | **6.73** | +0.06 | **0.51** | 44 |
| | ss_v3 (last) on Model20 feet | **0.927** | 6.78 | -1.70 | 0.72 | 51 |

Training the decoder on detector feet (ss_v3) helps end to end (StemField feet: Brandenburg DBH 7.82 -> 7.12 cm,
Weiser 8.76 -> 6.73 cm), so the val-PQ drop on Weiser was the partial-annotation artefact. None of this is a
held-out read; Weiser has 2 plots / 50 trees.
