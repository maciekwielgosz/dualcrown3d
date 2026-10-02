# DualCrown3D two-pass model: vote-centre second pass

Written 2026-10-02 as the follow-up to `AGENT_HANDOFF_2026-10-02.md`, section 11 (staged segmentation).
Status: **experimental candidate. At the full density of the annotated plots it passes the preregistered validation gate and improves the held-out test. At the density of the reference scene (about 10 voxels/m2) it shows no demonstrated benefit. Not promoted.** Model20 and every existing output folder are unchanged. Nothing from this session is committed to Git.

## 1. Summary

The handoff proposed a learned second pass that re-analyses what the first pass got wrong. That is what was built, but the diagnosis moved the second pass to a different place than planned.

- A second pass that **generates masks** (three decoder variants) did not work. It is kept as a recorded negative result (section 9).
- The decisive finding: **Model20's frozen 3-D centre votes already separate most trees.** Assigning each voxel to the nearest *true* tree centroid in vote space finds 523 of 673 validation trees. Model20's own clustering finds 299. The bottleneck is centre detection, not features and not masks.
- The second pass is therefore a small 3-D CNN that **detects tree centres in the vote space**, conditioned on the stage-1 result, followed by nearest-centre assignment of the votes.
- It is **corrective**: where the detector is silent, the stage-1 tree stands. Large crowns cannot be removed by a silent or unsure second pass.

## 2. Diagnosis

All numbers: 14 native validation plots, frozen Model20, point IoU >= 0.5.

**What Model20 misses.** 374 of 673 trees are missed; 260 of those already have >= 80% of their voxels assigned to another instance. Two failure modes dominate:

| Mode | Example | Missed trees look like |
|---|---|---|
| Understory under tall canopy | FGI_EMIT 1018: 160 of 216 missed | median height 11.7 m against 25.3 m for found trees; host instance 12x larger and 2.2x taller |
| Codominant neighbours merged | CEDAR_CYPRESS Saiki_3: 47 of 64 missed | same height as found trees; host about 2.5x their size |

**Votes are not the problem.** Median vote error is 1.01 m for missed trees and 1.07 m for found ones. What differs is spacing: missed trees stand 1.9 m (median) from their nearest neighbour, found trees 3.1 m. The vote's height component is accurate to about 1.4 m, far less than the roughly 10 m between an understory and an overstory centroid, and stage 1 ignores it.

**Oracle: same frozen votes, true centres.**

| Assignment | Trees found | Point SB-PQ | Crown SB-PQ | Small crowns <= 10 m2 | Large-tree recall |
|---|---:|---:|---:|---:|---:|
| Model20 (2-D density peaks + consensus) | 299 | 0.405 | 0.444 | 6 / 161 | 0.565 |
| True centres, 2-D votes | 422 | 0.571 | | | |
| True centres, 3-D votes (height x 0.33), radius 3 m | 523 | 0.670 | 0.599 | 57 / 161 | 0.884 |
| True centres, 3-D votes, radius 2 m | | 0.582 | 0.545 | 81 / 161 | 0.823 |

The same oracle on a third of the training plots gives 0.61, so training-plot votes are not cleaner than validation votes.

**Masks are the weak branch.** Model20's raw masks contain a good mask for only 247 of 673 trees, and for 4 of the 96 missed trees under 300 voxels.

## 3. Architecture

```text
voxels 0.25 m (XYZ-HAG + intensity)
        |
STAGE 1  frozen Model20: LitePT-S -> semantic + 3-D centre votes + masks -> consensus tree_id
        |
        |  per 20 m window (8 m overlap), votes within the window +- 4 m
        v
rasterise votes into an 80 x 80 x 40 grid (0.25 m x 0.25 m x 1 m of centroid height), 9 channels
        |
STAGE 2  VoteCenterNet: 3-D U-Net, width 24, two down/up levels  ->  centre heatmap
        |
3-D non-maximum suppression (0.5 m, 1 m), heatmap averaged over 4 flips
        |
corrective selection against the stage-1 centres (section 4)
        |
nearest-centre assignment of votes in (x, y, 0.33 z), height-dependent radius
        |
final tree_id  ->  LAZ + GeoPackage (same exporters as Model20)
```

Input channels: log vote count, mean tree probability, mean voter height, mean horizontal and vertical offset length, share of voters assigned in stage 1, stage-1 centre marker, number of distinct stage-1 instances voting into the cell, and the point occupancy of the same grid.

Target: one Gaussian per ground-truth tree at its centroid. Loss: CenterNet penalty-reduced focal loss. Cells dominated by votes of unannotated voxels are ignored. Because every tree is one peak, a 30-voxel tree and a 5000-voxel tree weigh the same.

The detector has about 1.0 M parameters and trains from cached Model20 predictions, so one epoch takes about 40 s on the 6 GB laptop GPU.

## 4. Corrective selection and assignment

- A detected centre **near a stage-1 centre** (within 1.5 m horizontally and 4 m in centroid height) relocates or splits an existing tree. It needs heatmap score >= 0.4 (v5; the selected v6 setting uses 0.3).
- A detected centre with **no stage-1 centre nearby** creates a brand-new tree. It needs more evidence: score >= 0.55.
- Every stage-1 centre that no kept detection replaces is **retained**.
- Votes go to the nearest centre in (x, y, 0.33 z). The reach is `clip(1.5 + 0.1 * centroid height, 2 m, 3 m)`: compact for small trees, wide for tall crowns.
- Minimum size, height and area filters are Model20's.

The LAZ field `assignment_source` records provenance: 3 = stage-1 centre kept, 8 = relocated or split by stage 2, 9 = added by stage 2.

## 5. Protocol and gate

Frozen in `outputs/dualcrown3d_two_pass_v1/protocol.json` **before** any training: stage-1 checkpoint and config hashes, manifest hash, baseline numbers and the gate. Training scripts re-score the cached stage-1 labels and abort unless they reproduce the Model20 baseline to 1e-6.

A candidate passes only if, on all 14 validation plots: point and crown SB-PQ >= Model20 - 0.005, pooled point and crown precision >= Model20 - 0.01, large-tree recall (>= 500 voxels) >= Model20 - 0.01, and at least 16 of 161 small crowns (<= 10 m2). The test split is read only by `scripts/evaluate_two_pass_test.py`, which refuses checkpoints without a validation-selected configuration.

Training data: 79 native training plots, source-balanced by collection, 800 random rotated and flipped windows per epoch, half of them anchored on a stage-1-missed tree. ECODSE, HELIOS and the test split are not used.

## 6. Results at full density

Two detector runs passed the gate. **v5** (`vote_centres_v5`, 20 epochs, dense votes only) and **v6** (`vote_centres_v6`, 24 epochs, half of the training windows from density-thinned plots, section 7).

**Validation, 14 plots (selection split).**

| Metric | Model20 | Two-pass v5 | Two-pass v6 |
|---|---:|---:|---:|
| Point SB-PQ | 0.4051 | 0.4164 | 0.4191 |
| Crown SB-PQ | 0.4442 | 0.4461 | 0.4405 |
| Point / crown SB-F1 | 0.517 / 0.620 | 0.531 / 0.628 | 0.535 / 0.624 |
| Pooled point / crown precision | 0.417 / 0.499 | 0.420 / 0.494 | 0.433 / 0.501 |
| Small crowns <= 10 m2 | 6 / 161 | 18 / 161 | 16 / 161 |
| Large-tree recall | 0.565 | 0.625 | 0.641 |
| Over-split GT / merged predictions | 125 / 102 | 137 / 76 | 126 / 67 |

**Held-out test, 24 plots.** Frozen checkpoints and configurations, each evaluated once.

| Metric | Model20 | Two-pass v5 | Two-pass v6 |
|---|---:|---:|---:|
| Point SB-PQ | 0.418 | **0.462** | 0.448 |
| Crown SB-PQ | 0.422 | **0.449** | 0.437 |
| Point / crown SB-F1 | 0.519 / 0.587 | 0.570 / 0.625 | 0.554 / 0.605 |
| Pooled point / crown precision | 0.448 / 0.547 | 0.469 / 0.547 | 0.462 / 0.536 |
| Pooled point / crown recall | 0.521 / 0.528 | 0.602 / 0.599 | 0.587 / 0.583 |
| Small crowns <= 10 m2 | 21 / 241 | 46 / 241 | 40 / 241 |
| Small crowns <= 4 m2 | 0 / 65 | 1 / 65 | 0 / 65 |
| Large-tree recall | 0.622 | 0.716 | 0.694 |
| Over-split GT / merged predictions | 140 / 150 | 163 / 117 | 165 / 114 |

Per collection on test (v5): point PQ rises in all six collections; crown PQ rises in five and is flat on FGI_EMIT (0.467 to 0.465). The largest crown gains are NIBIO (158 to 183 true positives), CEDAR_CYPRESS (133 to 152) and SCION (18 to 24).

Reading: at full density the second pass finds more small trees **and** more large trees, at equal or better precision. The cost is more over-split reference trees (140 to 163 on test), partly offset by fewer merged predictions (150 to 117). Crowns of 4 m2 or less remain essentially undetected.

## 7. Density and the reference scene

The annotated plots hold roughly 70 to 380 voxels/m2. The user's reference scene holds about 10. The first scene inference (v5, `output_28_...`) changed only 219 of 9,660 trees: the detector was silent, so the corrective fallback returned Model20's trees.

To measure this instead of guessing, plots were randomly thinned to 10 voxels/m2 and Model20 was rerun on them (`scripts/cache_stage1_thinned.py`).

**Validation thinned to 10 voxels/m2.**

| | Point SB-PQ | Crown SB-PQ | Point / crown precision | Small crowns | Large-tree recall |
|---|---:|---:|---:|---:|---:|
| Model20 | 0.394 | 0.314 | 0.344 / 0.365 | 12 / 161 | 0.654 |
| Two-pass v5 (10 centre changes of 1,098) | 0.404 | 0.333 | 0.347 / 0.371 | 6 / 161 | 0.654 |
| Two-pass v6 (472 centre changes) | 0.401 | 0.336 | 0.362 / 0.403 | 16 / 161 | 0.654 |
| True centres, same votes (separate thinning draw) | 0.658 | 0.511 | 0.838 / 0.791 | 40 / 161 | 0.862 |

**Held-out test thinned to 10 voxels/m2** (`output_30_...`).

| | Point SB-PQ | Crown SB-PQ | Point / crown precision | Small crowns | Large-tree recall | Over-split GT |
|---|---:|---:|---:|---:|---:|---:|
| Model20 | 0.390 | 0.322 | 0.381 / 0.432 | 24 / 241 | 0.500 | 181 |
| Two-pass v6 | 0.394 | 0.328 | 0.395 / 0.432 | 20 / 241 | 0.469 | 218 |

Conclusions:

- Mixed-density training made the detector active on sparse clouds, and thinned validation looked modestly better. **The thinned test did not confirm it**: PQ is flat, small crowns and large-tree recall are slightly lower, over-splitting is higher.
- **At the reference scene's density the two-pass model has no demonstrated benefit.** Model20 stays the right model for that scene today.
- The oracle shows the votes are still informative at 10 voxels/m2 (0.658 / 0.511). The detector, not the votes, is what fails in the sparse regime. A 0.25 m vote grid is probably too fine when a small tree casts only a few dozen votes.
- Random thinning is a crude proxy for a sparser scan. It ignores occlusion and pulse geometry.

Scene outputs exist for viewing only (no ground truth): `output_28_two_pass_vote_centres_scene` (v5, 219 trees differ from Model20) and `output_31_two_pass_v6_scene` (v6, 1,325 of 9,753 trees relocated, split or added). In both, `legacy_tree_id` holds the Model20 ID of the same point and `assignment_source` marks what stage 2 changed.

## 8. Cost

| | Stage-1 forward | Stage-1 consensus | Stage-2 detection | Stage-2 assignment |
|---|---:|---:|---:|---:|
| 24 test plots, full density | 45 s | 29 s | 37 s | 9 s |
| Reference scene, 78 ha, 7.7 M voxels | 228 s (reused) | 18 s | 469 s | 5 s |

Stage-2 peak GPU memory is 1.5 GB. Stage-2 detection is not optimised: windows are rasterised in NumPy and the heatmap is averaged over four flips. On the sparse scene it currently costs twice the stage-1 forward pass. Training one detector takes 35 to 50 minutes including validation sweeps.

## 9. Negative result: a mask-generating second pass

Before the vote diagnosis, three versions of a conditioned query decoder (`pointcloud/two_pass.py`, `scripts/train_two_pass_refiner.py`) were trained on the frozen Model20 features. Each query predicted a mask and a decision (background, new tree, existing tree), and `pointcloud/two_pass_reconcile.py` let a new mask reclaim voxels from a stage-1 instance without taking its treetop or a majority of it.

| Run | Change | Outcome on validation |
|---|---|---|
| `residual_v1` | 20 stage-1 conditioning features, Hungarian targets | nothing selected; best quality-preserving setting 7 / 161 small crowns |
| `residual_v2` | + sub-canopy features, same-instance prior, vertical lid, seed-containment targets | nothing selected; accepted masks lowered PQ |
| `residual_v3` | + Model20 decoder warm start, learned centre shift | stopped at epoch 8, same behaviour as v2 |

Why it failed: mask quality. Training mask IoU stayed near 0.46 on trees stage 1 had found and 0.30 on missed trees. An audit at epoch 9 found a good mask for only 47 of 374 missed trees, and 625 poor masks with a high "new tree" probability. The conditioning features, corruption augmentation, reconciliation rules and diagnostics from this work are tested and reusable; the decoder itself should not be pursued further on these frozen features.

## 10. Limitations

- **Validation was reused heavily**, in this session and before it. The asymmetric thresholds and the height-dependent radius were designed after inspecting validation errors, and the 0.55 addition threshold was added to the sweep after coarser steps. The gate margin on small crowns is slim (16 to 18 against 16 required).
- **The test split was read twice** (v5, then v6). Recommending v5 over v6 is informed by those test results, so the v5 test numbers are a slightly optimistic estimate. The test split had also been inspected in earlier sessions.
- **One training seed per run.** No confidence intervals.
- **Pooled crown precision is depressed by unannotated regions.** In CEDAR_CYPRESS Saiki_3, half the plot is unlabelled and 61 of 83 detected "false positives" lie mostly on unlabelled voxels. A higher-recall model is penalised for this more than Model20.
- **FGI_EMIT understory is ambiguous.** On plot 1020 most added instances are low fragments of annotated trees, while on plot 1018 comparable instances are annotated as separate small trees.
- **Stage-1 votes used for training are in-sample.** The true-centre oracle is not higher on training plots than on validation, which argues against memorised votes, but out-of-fold predictions were not produced.
- **Crown polygons are convex hulls**, as in Model20's final exporter, so the comparison is like for like but outlines are not the filled raster footprints the handoff prefers.
- No comparison against the classical R pipeline on a common protocol was attempted.

## 11. Files, artefacts and commands

| Path | Content |
|---|---|
| `pointcloud/vote_centers.py` | Vote rasterisation, `VoteCenterNet`, focal loss, peak extraction, corrective selection, assignment |
| `scripts/train_vote_centers.py` | Training, dual-density validation sweep, gate and selection |
| `scripts/validate_vote_centers.py`, `scripts/select_vote_centers.py` | Re-run a sweep on a saved checkpoint; freeze the gate-passing configuration |
| `scripts/evaluate_two_pass_test.py` | Held-out comparison with LAS/LAZ and GeoPackage export; optional thinning |
| `scripts/predict_two_pass_scene.py` | Reference-scene inference with Model20-compatible exports |
| `scripts/cache_stage1_predictions.py`, `scripts/cache_stage1_thinned.py` | Frozen protocol, stage-1 caches and diagnostics |
| `pointcloud/two_pass*.py`, `scripts/train_two_pass_refiner.py`, `scripts/audit_two_pass_proposals.py` | Mask second pass (negative result), shared evaluation helpers |
| `tests/test_vote_centers.py`, `tests/test_two_pass.py` | 6 + 13 synthetic tests |
| `outputs/dualcrown3d_two_pass_v1/protocol.json` | Frozen hashes, baseline and gate |
| `outputs/dualcrown3d_two_pass_v1/stage1/diagnostics.xlsx` | Missed, absorbed and over-split trees per plot |
| `outputs/dualcrown3d_two_pass_v1/runs/<run>/` | Configuration, training log, per-epoch validation JSON, `experiments.xlsx`, weights, `selected.json` |
| `output_27_two_pass_vote_centres_test`, `output_29_two_pass_v6_test` | Held-out test, v5 and v6, with `comparison.xlsx` and per-plot LAS/LAZ and GPKG for both models |
| `output_30_two_pass_v6_test_thinned10` | Held-out test thinned to 10 voxels/m2, v6 |
| `output_28_two_pass_vote_centres_scene`, `output_31_two_pass_v6_scene` | Reference scene, v5 and v6 |

```bash
cd /home/maciej.wielgosz/Projects/segm/DL_model_version
.venv-gpu/bin/python -m unittest discover -s tests -p "test_vote_centers.py" -v
.venv-gpu/bin/python scripts/cache_stage1_predictions.py            # frozen protocol + stage-1 cache
.venv-gpu/bin/python scripts/cache_stage1_thinned.py                # optional, for mixed-density training
.venv-gpu/bin/python scripts/train_vote_centers.py --run <name> --epochs 20 --samples 800 --val-every 2 \
    --dropout 0.1 --weight-decay 1e-3 --only adaptive
.venv-gpu/bin/python scripts/evaluate_two_pass_test.py --run <name> --output <new folder>
.venv-gpu/bin/python scripts/predict_two_pass_scene.py --run <name> --output-dir <new folder>
```

Checkpoints: v5 `runs/vote_centres_v5/weights/best.pt` (epoch 20, `asym_r0.4_a0.55_adaptive`); v6 `runs/vote_centres_v6/weights/best.pt` (epoch 24, `asym_r0.3_a0.55_adaptive`). Hashes are in each run's `selected.json` and in the output folders' `run.json`.

## 12. Recommended next steps

1. **Make the detector work on sparse clouds.** The oracle says the votes allow it. Try a coarser or multi-resolution vote grid, input normalisation by local vote density, and training on data that is sparse for real: the HELIOS set calibrated to about 14 points/m2 and any labelled low-density ALS. Judge it on a thinned and a truly sparse validation set, not only on the dense one.
2. **Decide on promotion for dense data.** v5 improves every headline metric on the held-out test. Before promoting it, repeat training with three seeds and confirm on a site that was never inspected.
3. **Protect large crowns explicitly.** Over-split reference trees rise from 140 to 163 on test. A split of a stage-1 tree could require both resulting centres to be confident.
4. **Cut the stage-2 cost.** Rasterise on the GPU and test whether flip averaging is needed.
5. **Fix the precision measurement.** Report precision restricted to annotated area alongside the pooled figure.
6. **Run the classical R pipeline on the same plots and metrics**, the comparison the project still lacks.
