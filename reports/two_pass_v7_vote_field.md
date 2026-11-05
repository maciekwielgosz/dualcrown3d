# DualCrown3D two-pass v7: vote-field detector, centre verifier and what limits the model next

Written 2026-10-02 (evening session). Follow-up to `two_pass_refinement_design.md` (v4-v6) and `AGENT_HANDOFF_2026-10-02.md`.
Status: **experimental candidate that passes every preregistered validation gate with one configuration, including the gate at the reference-scene density. Held-out results are in section 6. Not promoted; Model20 and all earlier output folders are unchanged.**

## 1. Summary

- The v5/v6 centre detector did not help at the density of the reference scene (about 10 voxels/m2). The cause was the detector, not the votes.
- v7 rebuilds the second pass around four measured findings (section 2) and replaces Model20's clustering completely: every tree centre now comes from stage 2.
- On validation the final candidate (four detectors averaged, plus a verifier) reaches 0.508 / 0.513 point / crown SB-PQ at full density against Model20's 0.405 / 0.444, and 0.470 / 0.390 against 0.394 / 0.314 on the same plots thinned to 10 voxels/m2. Small crowns go from 6 to 34 of 161, large-tree recall from 0.565 to 0.719, and over-split reference trees fall from 125 to 65.
- The classical CHM pipeline was scored on the same plots with the same code for the first time (section 7): 0.313 / 0.315 on validation.
- Two further stages were explored on the Athena cluster: a learned voxel-to-centre assignment (stage 3, section 9) and a scaled retraining of stage 1 (section 10).

## 2. What was measured before redesigning

All on the 14 validation plots unless stated.

| Finding | Evidence | Consequence |
|---|---|---|
| The v5 detector underfitted | centre recall / precision at score 0.3: 0.55 / 0.44 on *training* plots, 0.63 / 0.39 on validation | more data would not help; the task definition had to change |
| The target was badly posed | 141 of 251 missed centres had a peak in the right place with a score below 0.3; the target was one 0.25 m cell at the geometric centroid, which the 1 m-noisy votes do not mark | detect where the *votes* of a tree concentrate, on a coarser output grid |
| Votes get sharper for free | averaging Model20's votes over four 90-degree rotations raises the true-centre oracle from 0.671 / 0.602 to 0.705 / 0.627 point / crown SB-PQ; Model20's own clustering does not benefit | stage 2 uses rotation-ensembled votes; stage-1 labels stay single-pass |
| Each tree needs its own reach | with the true reach of every tree's votes the oracle rises to 0.744 / 0.674 and finds 132 instead of 90 of 161 small crowns | the detector predicts an extent per centre |
| Model20's centres are poor at low density | at 10 voxels/m2 its centres have recall 0.50 and precision 0.28 (1,098 instances for 624 trees) | "keep the stage-1 centre where stage 2 is silent" (v5) preserves the errors; v7 replaces them |
| The protocol metric charges for unlabelled trees | half of the canopy voxels of a typical real plot are unannotated; 262 of Model20's 418 point "false positives" lie mostly on unannotated voxels | an annotated-area point PQ is reported next to the protocol metric (gates still use the protocol metric) |

## 3. Architecture

```text
voxels 0.25 m
   |
STAGE 1   frozen Model20 (LitePT-S): semantic + centre votes + masks -> consensus tree_id   (unchanged)
   |      + the same network run on 4 lattice rotations: averaged votes, vote spread
   v
vote volume per 20 m window: 80 x 80 x 40 cells (0.25 m x 0.25 m x 1 m of centroid height), 14 channels
   |
STAGE 2a  VoteFieldNet: 3-D U-Net, three down levels, heads at 0.5 m:
   |        centre heat  |  sub-cell offset  |  extent (reach of the tree's votes)
   |      four independently trained detectors, outputs averaged, each over 4 window flips
   v
candidates = heat peaks (score >= 0.1)  +  stage-1 centres no peak sits on
   |
STAGE 2b  centre verifier (gradient-boosted trees, 40 set-level features per candidate)
   |      final score = mean(verifier probability, heat)
   v
keep candidates with score >= 0.7   ->   every vote goes to the centre with the smallest
                                         distance / reach among the centres whose reach covers it
                                         (reach = 1.5 x predicted extent, between 1 and 4 m)
   v
Model20's size filters -> final tree_id -> LAZ + GeoPackage (same exporters as Model20)
```

Details that matter:

- **Targets.** The centre of a tree is the median of its own votes, its extent the 90th percentile of the distance of its votes to that median, both in the space (x, y, 0.33 z) in which votes are assigned.
- **Input channels.** The nine v4 channels, plus the mean direction the votes of a cell arrive from (x and y), a canopy-height column, the window density, and the spread of the vote across the four rotations.
- **Training data.** Three regimes mixed 40 / 40 / 20: the 79 dense training plots, six randomly thinned versions of each (5 to 80 voxels/m2, Model20 rerun on each), and 129 HELIOS simulations (about 14 points/m2). Rotation, flip, vote jitter and voter dropout as before.
- **Verifier features.** Things a local 3-D view cannot see: a stronger candidate claiming the same votes, whether the candidate sits in the middle of its voters or on their flank, how many votes it wins exclusively, whether it is a fragment of a stage-1 tree. It scores stage-1 centres too, so stage 2 can add, relocate and reject in one decision.
- **Provenance** (`assignment_source` in the LAZ): 3 = a stage-1 centre the verifier kept, 8 = stage-2 centre next to a stage-1 centre, 9 = stage-2 centre with no stage-1 centre nearby.

## 4. Protocol

`outputs/dualcrown3d_two_pass_v1/protocol_v7.json` was frozen before the first v7 detector was trained. It adds two validation sets to the 14 dense plots: the same plots thinned to 10 voxels/m2 (the draw used since v6) and 12 HELIOS validation plots. Thinning to 25 voxels/m2 is reported but not gated.

| Set | Gate against Model20 on the same set |
|---|---|
| dense | point and crown SB-PQ not more than 0.005 lower, pooled precision not more than 0.01 lower, large-tree recall not more than 0.01 lower, at least 16 of 161 small crowns (the v1 gate) |
| thinned to 10 voxels/m2 | point **and** crown SB-PQ at least **0.02 higher**, precision and large-tree recall not more than 0.01 lower, at least as many small crowns (12) |
| HELIOS | point and crown SB-PQ not more than 0.005 lower, precision and large-tree recall not more than 0.01 lower |

One configuration must pass all three. Among those, the one with the largest summed gain in sqrt(point SB-PQ x crown SB-PQ) over dense and thinned is selected. The held-out plan (`test_plan_v7.json`) was written before any v7 test evaluation and names a primary and one secondary candidate.

## 5. Validation results

Final candidate: run `ens4_b`, configuration `mean_heat_t0.7_nms0_extent1.5_norm`.

| Set | Model | Point SB-PQ | Crown SB-PQ | Annotated-area point SB-PQ | Point / crown precision | Small crowns | Large-tree recall | Over-split GT | Instances |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | Model20 | 0.405 | 0.444 | 0.518 | 0.417 / 0.498 | 6 / 161 | 0.565 | 125 | 931 |
| dense | two-pass v7 | **0.508** | **0.513** | **0.624** | 0.552 / 0.611 | 34 / 161 | 0.719 | 65 | 874 |
| thinned 10 | Model20 | 0.394 | 0.314 | 0.490 | 0.343 / 0.365 | 12 / 161 | 0.654 | 144 | 1098 |
| thinned 10 | two-pass v7 | **0.470** | **0.390** | **0.567** | 0.554 / 0.539 | 16 / 161 | 0.731 | 62 | 706 |
| thinned 25 | Model20 | 0.410 | 0.388 | 0.529 | 0.393 / 0.454 | 8 / 161 | 0.745 | 127 | 1030 |
| thinned 25 | two-pass v7 | **0.495** | **0.441** | **0.603** | 0.545 / 0.573 | 25 / 161 | 0.758 | 79 | 842 |
| HELIOS | Model20 | 0.524 | 0.458 | 0.524 | 0.732 / 0.693 | 28 / 81 | 0.556 | 50 | 261 |
| HELIOS | two-pass v7 | **0.718** | **0.623** | **0.718** | 0.959 / 0.927 | 51 / 81 | 1.000 | 18 | 246 |

How the gain is built up (summed gain over dense and thinned, the selection measure; Model20 = 0):

| Step | Gain | Dense point / crown | Thinned-10 point / crown | Small crowns dense |
|---|---:|---:|---:|---:|
| v5 (reference, earlier session) | 0.02 | 0.416 / 0.446 | 0.404 / 0.333 | 18 |
| one v7 detector, 40-50 epochs (three seeds) | 0.104, 0.119, 0.124 | 0.464 to 0.489 / 0.466 to 0.493 | 0.447 to 0.459 / 0.370 to 0.382 | 40 to 56 |
| one detector, 120 epochs | 0.134, 0.135 | 0.479 / 0.491 | 0.467 / 0.387 | 49 |
| four detectors averaged | 0.147 | 0.494 / 0.496 | 0.468 / 0.390 | 46 |
| + verifier (final) | **0.163** | 0.508 / 0.513 | 0.470 / 0.390 | 34 |

Readings:

- The detector alone is saturated. Three seeds, size-adaptive targets, wider networks, a larger batch and different density mixes all land within about 0.02 of each other. Longer training and averaging help a little.
- The verifier buys precision (dense 0.51 to 0.55) and costs small crowns (46 to 34). The detector-only ensemble is therefore kept as a declared secondary operating point.
- Per collection at full density (single detector): CEDAR_CYPRESS 0.31 to 0.45 point PQ, FGI_EMIT 0.35 to 0.40, CULS and SCION slightly up, WILDFOREST3D flat. WILDFOREST3D has wide, diffuse crowns; only half of its wide trees get a confident peak within 1 m.

## 6. Held-out results

Each suite was evaluated once per declared candidate, with frozen checkpoints and configurations (`test_plan_v7.json`). Primary: ensemble + verifier. Secondary (declared in advance): the same ensemble without the verifier.

**24 native test plots, full density** (`output_32_...`, `output_35_...`)

| Model | Point SB-PQ | Crown SB-PQ | Annotated-area point SB-PQ | Point / crown precision | Point / crown recall | Small crowns <= 10 m2 | <= 4 m2 | Large-tree recall | Over-split GT / merged predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model20 | 0.418 | 0.422 | 0.507 | 0.448 / 0.547 | 0.521 / 0.528 | 21 / 241 | 0 / 65 | 0.622 | 140 / 150 |
| two-pass v5 (earlier session) | 0.462 | 0.449 | | 0.469 / 0.547 | 0.602 / 0.599 | 46 / 241 | 1 / 65 | 0.716 | 163 / 117 |
| **two-pass v7 (primary)** | **0.512** | **0.486** | **0.611** | 0.564 / 0.639 | 0.669 / 0.665 | 88 / 241 | 5 / 65 | 0.788 | 72 / 106 |
| two-pass v7 without verifier | 0.496 | 0.473 | 0.603 | 0.519 / 0.584 | 0.688 / 0.685 | 95 / 241 | 10 / 65 | 0.802 | 88 / 100 |
| classical CHM pipeline | 0.292 | 0.270 | 0.370 | 0.286 / 0.339 | 0.332 / 0.281 | 6 / 241 | | 0.391 | 256 / 176 |

**The same plots thinned to 10 voxels/m2** (`output_33_...`, `output_36_...`)

| Model | Point SB-PQ | Crown SB-PQ | Annotated-area point SB-PQ | Point / crown precision | Point / crown recall | Small crowns <= 10 m2 | Large-tree recall | Over-split GT / merged predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Model20 | 0.389 | 0.322 | 0.466 | 0.381 / 0.432 | 0.476 / 0.443 | 24 / 241 | 0.500 | 181 / 141 |
| two-pass v6 (earlier session) | 0.394 | 0.328 | | 0.395 / 0.432 | | 20 / 241 | 0.469 | 218 / |
| **two-pass v7 (primary)** | **0.500** | **0.374** | **0.578** | 0.596 / 0.558 | 0.605 / 0.504 | 41 / 241 | 0.625 | 84 / 117 |
| two-pass v7 without verifier | 0.491 | 0.356 | 0.579 | 0.545 / 0.497 | 0.637 / 0.516 | 47 / 241 | 0.719 | 99 / 106 |

**7 HELIOS test plots** (`output_34_...`, `output_37_...`; never read before)

| Model | Point SB-PQ | Crown SB-PQ | Point / crown precision | Point / crown recall | Small crowns <= 10 m2 | Large-tree recall |
|---|---:|---:|---:|---:|---:|---:|
| Model20 | 0.612 | 0.576 | 0.872 / 0.880 | 0.685 / 0.691 | 7 / 39 | 1.000 |
| **two-pass v7 (primary)** | **0.719** | **0.638** | 0.927 / 0.919 | 0.772 / 0.765 | 14 / 39 | 1.000 |
| two-pass v7 without verifier | 0.716 | 0.622 | 0.913 / 0.882 | 0.779 / 0.752 | 15 / 39 | 1.000 |

Per collection, primary candidate (Model20 -> v7):

| Collection | Plots | Point PQ, dense | Crown PQ, dense | Small crowns, dense | Point PQ, thinned | Crown PQ, thinned |
|---|---:|---:|---:|---:|---:|---:|
| CEDAR_CYPRESS | 5 | 0.424 -> 0.532 | 0.328 -> 0.454 | 6 -> 27 of 33 | 0.333 -> 0.480 | 0.321 -> 0.383 |
| CULS | 1 | 0.744 -> 0.799 | 0.779 -> 0.774 | none | 0.816 -> 0.848 | 0.675 -> 0.658 |
| FGI_EMIT | 3 | 0.472 -> 0.582 | 0.467 -> 0.522 | 4 -> 19 of 53 | 0.407 -> 0.572 | 0.329 -> 0.453 |
| NIBIO | 10 | 0.383 -> 0.527 | 0.468 -> 0.582 | 7 -> 36 of 89 | 0.376 -> 0.543 | 0.330 -> 0.302 |
| SCION | 1 | 0.343 -> 0.409 | 0.316 -> 0.408 | 4 -> 5 of 23 | 0.320 -> 0.399 | 0.203 -> 0.307 |
| WILDFOREST3D | 4 | 0.142 -> 0.222 | 0.174 -> 0.176 | 0 -> 1 of 43 | 0.085 -> 0.160 | 0.076 -> 0.141 |

Readings:

- The validation gains hold on held-out plots in all three suites: +0.094 / +0.064 point / crown SB-PQ at full density, +0.111 / +0.052 thinned, +0.107 / +0.062 on HELIOS. v6 had shown no benefit on the thinned test.
- Small crowns found rise from 21 to 88 of 241 at full density, large-tree recall from 0.62 to 0.79, and precision rises too. Over-split reference trees halve (140 to 72), where v5 had increased them.
- Point PQ rises in every collection at both densities. Crown PQ is flat on CULS and on WILDFOREST3D at full density, and **falls on NIBIO when thinned** (0.330 to 0.302) although its point PQ rises.
- Crowns of 4 m2 or less remain almost undetected (5 of 65).
- WILDFOREST3D stays the weak collection (point PQ 0.22).
- The verifier trades small crowns and recall for precision and PQ, on test as on validation.

### Reference scene (no ground truth)

`output_38_two_pass_v7_scene` (primary) and `output_39_two_pass_v7_scene_fallback` (primary plus coverage fallback). Same exporters and field names as `output_20`; `legacy_tree_id` holds Model20's ID of the same point.

| | Model20 | v7 primary | v7 primary + coverage fallback | classical pipeline |
|---|---:|---:|---:|---:|
| Trees in the four core tiles | 9,644 | 4,143 | 8,331 | 6,688 |
| Voxels with an instance | 1.62 M | 1.16 M | 1.62 M | |
| Crowns <= 10 m2 | 1,948 | 1,023 | 2,506 | |
| Median crown area | 15.5 m2 | 20.1 m2 | 14.0 m2 | 16.6 to 29.0 m2 per tile |

**On the real scene stage 2 is much quieter than on the proxies.** It keeps 4,892 centres where Model20 had 9,961 instances, and a third of the predicted tree voxels end up without an instance (10 % for Model20). On thinned validation the same model left 17 % unlabelled. Thinned UAV-laser plots and HELIOS simulations therefore do not fully represent this scan, and no accuracy can be measured here.

The coverage fallback (`--fallback 0.7`) keeps a Model20 tree wherever stage 2 claimed less than 30 % of its voxels; 4,188 trees come back that way (`assignment_source` 10). On validation it costs about 0.01 PQ and keeps every gate passed (thinned: 0.460 / 0.378 with coverage 0.95, against 0.470 / 0.390 with coverage 0.83). It was not run on the test splits. `output_39` is the one to look at for a complete map; `output_38` shows what stage 2 alone is confident about.

## 7. Classical CHM pipeline on the same plots

The production R pipeline (`skrypt/pcopw_chunks_500m_Segmentacja.R`: 3x3 median, variable-window local maxima, marker-controlled watershed) was run with its production parameters on the plot CHMs (0.5 m) and scored with the same code path. Script `scripts/evaluate_classical_chm_baseline.py`, output `outputs/classical_chm_baseline_v1/`. The runner reproduces the production script's polygons exactly on all 38 plots.

| Metric | Classical, validation | Classical, test | Model20, validation | Model20, test |
|---|---:|---:|---:|---:|
| Point SB-PQ | 0.313 | 0.292 | 0.405 | 0.418 |
| Crown SB-PQ, same hull path as the deep models | 0.315 | 0.270 | 0.444 | 0.422 |
| Crown SB-PQ, native watershed polygons | 0.286 | 0.262 | | |
| Annotated-area point SB-PQ | 0.407 | 0.370 | 0.518 | |
| Small crowns | 5 / 161 | 6 / 241 | 6 / 161 | 21 / 241 |
| Large-tree recall | 0.367 | 0.391 | 0.565 | 0.622 |

Caveats: the plot CHMs are raw maximum-height rasters with gaps, not the production CHM product, and plots are too small for the 25 m buffer. A variant tuned on IDEAS-ALS plots scores 0.315 / 0.278 on test but was tuned on plots that overlap this test split.

## 8. Cost

| | Stage-1 forward | Stage-1 consensus | Vote ensemble (4 rotations) | Stage 2 (4 detectors x 4 flips, verifier, assignment) |
|---|---:|---:|---:|---:|
| 24 test plots, full density, laptop GPU | 46 s | 27 s | 113 s | 97 s |
| 24 test plots thinned, laptop GPU | 14 s | 2 s | 46 s | 60 s |
| Reference scene, 78 ha, 7.7 M voxels | 228 s (reused) | 17 s | 963 s (laptop GPU) | 360 s (one A100; not run on the laptop) |

One detector trains in about 20 minutes (50 epochs) to 50 minutes (120 epochs) on an A100, four times longer on the 6 GB laptop GPU. The verifier needs about 20 minutes, mostly detector inference on the training plots. On the test plots the complete pipeline takes about 4 times (full density) to 8 times (thinned) as long as Model20 alone. One 120-epoch detector instead of four costs about 0.013 of the summed validation gain (0.134 against 0.147) and a quarter of the stage-2 time. Dropping the flip averaging was not measured.

## 9. Stage 3: learned voxel-to-centre assignment (exploratory)

With the real v7 candidates above score 0.2, giving every annotated voxel to the candidate of its own tree would reach 0.93 point SB-PQ on dense validation, 0.91 thinned and 0.97 on HELIOS. Nearest-centre assignment of the same candidates reaches about 0.5. The votes of boundary voxels are imprecise and unmatched candidates take voxels.

`pointcloud/candidate_assignment.py` and `pointcloud/assign_net.py` implement this as a choice among the six nearest candidates per voxel ("slots") plus "none", made by a sparse 3-D U-Net over the voxel grid with a shared slot encoder and scorer (`scripts/train_assign_net.py`).

Result so far (five runs on Athena, candidates of one v7 detector, 30 to 40 epochs): **parity, not a gain.**

| Assignment of the same candidates (score >= 0.5) | Dense point / crown | Dense annotated point | Thinned-10 point / crown | HELIOS point / crown | Summed gain |
|---|---:|---:|---:|---:|---:|
| nearest centre within its predicted reach (v7) | 0.464 / 0.466 | 0.584 | 0.459 / 0.375 | 0.670 / 0.576 | 0.104 |
| stage-3 network (best run, width 32) | 0.474 / 0.456 | 0.592 | 0.466 / 0.346 | 0.684 / 0.579 | 0.091 |
| ceiling: every voxel to its own tree's candidate | 0.873 / 0.531 | 0.873 | 0.761 / 0.403 | 0.863 / 0.712 | |

The network picks the right slot for 79 % of annotated voxels on validation (82 to 84 % on training crops) and stops improving there. It finds slightly more trees at full density and gives smaller crowns at low density. Admitting weaker candidates (score 0.1 to 0.4) lowers precision faster than it adds trees. Reading: votes plus voxel occupancy do not carry enough information to beat the geometric rule; the information has to come from stage 1 (sharper votes or its backbone features as input). The code, tests and Athena runs are kept; the stage is **not** part of the candidate.

## 10. Stage 1 at scale (exploratory)

Model20 was trained with a laptop budget: 40 epochs of 160 crops, about 18 minutes, offset loss on XY only. A retraining with a much larger budget on Athena, judged by vote error and the true-centre oracle, was started in this session.

Six fine-tuning variants were started on Athena (jobs `s1_r1a_ctrl`, `s1_r1b_full`, `s1_r1c_z`, `s1_r1d_thin`, `s1_r1e_combo`, `s1_r1f_combo_assign`; scripts `scripts/train_stage1_scaled.py`, output `outputs/dualcrown3d_stage1_scaled_v1/`): the Model20 recipe with a far larger budget, the whole backbone unfrozen, a 3-D offset loss, strong density thinning, and combinations. Each epoch covers 8,000 crops and takes about 20 minutes, so the runs last several hours. **They were still running when this report was written; no result is claimed.** They are judged on validation only, by vote error and by the true-centre oracle (Model20 with ensembled votes: 0.748 / 0.683 dense, 0.732 / 0.535 thinned, 0.831 / 0.743 HELIOS). A better stage 1 means recomputing the vote caches and retraining stage 2 on them.

## 11. Negative and neutral results of this session

| Idea | Outcome |
|---|---|
| Model20's 5-D instance embedding as an extra cue | separates neighbours about as well as the votes, but adding it to the assignment lowers the true-centre oracle (0.761 to 0.699 at the smallest weight) |
| Size-adaptive heat targets (v8) | no gain over fixed targets on the selection measure; better large-tree recall on HELIOS only |
| Extent-aware suppression of duplicate peaks | helps single detectors slightly, not the ensemble |
| Growing crowns into unlabelled neighbouring voxels | raises coverage, lowers PQ |
| Radius, probability and size-filter variants of the assignment | all within noise of the selected rule |
| Corrective mode (keep stage-1 centres) | worse than replacing them once the detector is good, clearly so at low density |

## 12. Limitations

- **Validation was reused heavily**, including for the choice of thresholds and of the ensemble members. The gate margins are large this time (0.10 point PQ on dense, 0.08 on thinned), but the selected numbers are still optimistic.
- **The dense and thinned real test plots had been read before** (v5, v6 and earlier sessions). The HELIOS test plots had not.
- **Thinning is a crude proxy** for a sparse scan; HELIOS is a simulation. No labelled real ALS at 10 voxels/m2 was available.
- **Half of each real plot is unannotated.** The protocol metric penalises trees found there; crown SB-PQ on real plots cannot exceed about 0.53 even with perfect labels of the annotated trees.
- **The verifier and stage 3 were trained on candidates of detectors that had seen the same plots.** Detector ranking quality is only slightly better on training plots (0.92 against 0.90 average precision), but out-of-fold candidates were not produced.
- **Coverage and the scene domain.** In replace mode a tree the detector misses is left unlabelled instead of being merged into a neighbour: on thinned validation 83 % of annotated tree voxels get an instance, against 93 % for Model20, and on the real reference scene only about two thirds of the predicted tree voxels do (section 6). The coverage fallback addresses the symptom, not the cause.
- **Cost.** Stage 1 runs four more times for the rotation ensemble and stage 2 averages four detectors over four flips.
- Crown polygons are still convex hulls, as in Model20's exporter.

## 13. Files and commands

| Path | Content |
|---|---|
| `pointcloud/vote_ensemble.py` | rotation-ensembled Model20 votes |
| `pointcloud/vote_field.py` | vote volume, `VoteFieldNet`, targets, loss, peak decoding, selection, assignment, `second_pass` |
| `pointcloud/centre_verifier.py` | candidate set, features, targets, score rules |
| `pointcloud/candidate_assignment.py`, `pointcloud/assign_net.py` | stage 3 (slots, sparse U-Net, `third_pass`) |
| `pointcloud/paths.py` | `DUALCROWN_PATH_MAP` for running from another machine |
| `scripts/cache_stage1_ensemble.py` | caches for dense, thinned and HELIOS variants |
| `scripts/freeze_two_pass_v7_protocol.py` | baselines and gates |
| `scripts/train_vote_field.py`, `validate_vote_field.py`, `build_vote_field_ensemble.py`, `summarize_vote_field_runs.py` | detector training, full sweep and selection, ensembles, run comparison |
| `scripts/train_centre_verifier.py` | verifier training, sweep and selection |
| `scripts/train_assign_net.py`, `train_assignment_gbm.py`, `build_two_pass_pipeline.py` | stage 3 |
| `scripts/evaluate_vote_field_test.py`, `predict_vote_field_scene.py` | held-out suites and reference scene |
| `scripts/evaluate_classical_chm_baseline.py` | classical pipeline on the labelled plots |
| `slurm/` | Athena job scripts (`env.sh`, `run.sbatch`, `train_vote_field.sbatch`) |
| `tests/test_vote_field.py`, `test_centre_verifier.py`, `test_candidate_assignment.py` | 11 + 3 + 3 tests; the whole suite has 113 |
| `outputs/dualcrown3d_two_pass_v1/protocol_v7.json`, `test_plan_v7.json` | frozen gates and test plan |
| `outputs/dualcrown3d_two_pass_v1/runs/ens4_b/` | final candidate (`weights/best.pt`, `selected.json`, validation JSON, verifier) |
| `outputs/dualcrown3d_two_pass_v1/runs/ens4_b_detector/` | the same ensemble without the verifier |

```bash
cd /home/maciej.wielgosz/Projects/segm/DL_model_version
.venv-gpu/bin/python -m unittest discover -s tests
.venv-gpu/bin/python scripts/cache_stage1_ensemble.py
.venv-gpu/bin/python scripts/train_vote_field.py --run <name> --epochs 120 --samples 2000 --val-every 60
.venv-gpu/bin/python scripts/validate_vote_field.py --run <name> --weights last.pt --select
.venv-gpu/bin/python scripts/build_vote_field_ensemble.py --run <ens> --members <run>/last.pt <run2>/last.pt
.venv-gpu/bin/python scripts/train_centre_verifier.py --run <ens> --weights last.pt --select
.venv-gpu/bin/python scripts/evaluate_vote_field_test.py --run <ens> --suite dense --output <new folder>
.venv-gpu/bin/python scripts/predict_vote_field_scene.py --run <ens> --output-dir <new folder>
# on Athena
sbatch --job-name=vf_<name> slurm/train_vote_field.sbatch <name> --epochs 120 --val-every 60
```

## 14. Next steps

1. **Label a few hectares of the real scene domain.** The benchmark gains are large and held out, but the reference scene behaves differently and nothing can be measured there. Even 2 to 3 ha of annotated sparse ALS would allow thresholds, the fallback and stage 2 itself to be judged where they will be used. This is the most valuable next step.
2. **Finish and evaluate the scaled stage-1 retraining** (section 10). Sharper votes help detection and assignment at once and are what the stage-3 ceiling needs.
3. **Give stage 3 better inputs**: the LitePT features of stage 1 per voxel, or the retrained votes. With the present inputs it only matches the geometric rule.
4. **Decide on promotion.** The primary candidate improves every headline metric on three held-out suites. Open points before promoting: the scene domain (step 1), cost (section 8), and the choice between the primary and the detector-only operating point if small crowns matter more than precision.
5. **Crowns below 4 m2 and WILDFOREST3D-type diffuse crowns** remain unsolved; both need better votes more than a better detector.
6. **Cost.** Distil the four detectors into one, drop flip averaging, rasterise on the GPU.
7. **Out-of-fold candidates** for the verifier and stage 3 (train detectors on folds of the training plots) would remove the in-sample caveat cheaply on Athena.
