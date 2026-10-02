# TLS-to-ALS visibility pilot (2026-10-01)

## Decision

The small pilot does **not** justify mass production of synthetic ALS or promotion of a new DualCrown3D checkpoint. A simple overhead-visibility simulator can create label-independent, reproducible ALS-like samples and match several coarse HELIOS statistics, but none of five matched short fine-tuning controls improved both point-instance and crown-instance validation quality. The retained checkpoint remains selected in every run.

This is a screening result, not proof that controlled TLS-to-ALS simulation cannot help. The pilot used six TreeScanPL10k training parents, one seed, 12 short epochs per run, and 14 native real-ALS validation plots. The real test split was not used for selection.

## What was tested

`scripts/simulate_tls_als_pilot.py` constructs two virtual airborne strips per 30 × 30 m TLS plot (0° and 18°). It bins source geometry into beam footprints and 0.3 m height layers, tests upper layers first with a saturating occupancy/opacity approximation, allows up to four returns and ground transmission, and copies source tree IDs to retained points only after geometry is selected. A matched random-thinning control and the pre-existing HELIOS output use the same six parent scenes, full TLS-derived crown polygons, 0.25 m model voxelisation, and label-independent voxel representative selection. Intensity is drawn from an unconditional calibrated distribution; this is **not** a full waveform, scanner, or radiometric simulator.

The selected parents are Gorlice, Herby, Katrynka, Milicz, Piensk, and Suprasl. All were in the TreeScan training split. The synthetic manifests retain parent IDs for split auditing. Every original tree in these plots had at least one return (169/169), though that alone does not validate the simulated crown shape or visibility.

The model experiment starts from the existing retained `repeat_20260930` DualCrown3D checkpoint. It compares real-only, real + HELIOS, real + random thinning, real + initial visibility simulation, and real + profile-matched visibility simulation. Synthetic runs use 50% real and 50% synthetic crop draws; all runs use 80 draws/epoch, 12 epochs, the same optimizer/budget/seed, and evaluation at epochs 4, 8, and 12. Fine-tuning ran on CUDA. The archived baseline was recomputed with the current code on the same 14 real validation plots and matched exactly.

## Geometry check

Means across the same six parent plots are below. “Median height” is the mean of each plot's median height-above-ground, **not** the pooled median. The profile-matched variant used 0.49 m beam spacing, 0.25 opacity and 0.75 transmission; the first variant used 0.50 m, 0.40 and 0.55.

| Input | Points/m² | Returns/pulse | Tree-point fraction | Median height (m) |
| --- | ---: | ---: | ---: | ---: |
| HELIOS, same parents | 12.016 | 1.581 | 0.509 | 3.671 |
| Random thinning | 12.000 | 1.000 | 0.500 | 0.054 |
| Visibility, initial | 12.042 | 1.617 | 0.595 | 8.945 |
| Visibility, profile-matched | 12.624 | 1.612 | 0.523 | 3.601 |

The profile-matched simulator is much closer to HELIOS than the first variant on canopy fraction and height distribution. The real ALS reference tile used by the earlier HELIOS calibration had 13.971 points/m² and about 1.79 returns/pulse, but forest composition differs, so this comparison is only a coarse acquisition-profile check. No colocated real ALS–TLS pairs were available to validate occlusion, crown occupancy, or per-tree return distributions. Synthetic full crown polygons are derived from TLS, not independently checked manual ALS annotations.

## Segmentation check

The table reports the **best trained evaluation epoch** in each run, even though the saved selection in every case remains epoch 0 (the original checkpoint). SB-PQ and SB-F1 are source-balanced scores on the 14 real validation plots. “Small” is correctly matched crowns ≤10 m² out of 161; crowns ≤4 m² remained 0/39 in all runs.

| Variant | Best epoch | Point SB-PQ | Crown SB-PQ | Point SB-F1 | Crown SB-F1 | Small ≤10 m² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained checkpoint | 0 | **0.4051** | **0.4442** | **0.5165** | **0.6195** | 6/161 |
| Real-only | 12 | 0.3998 | 0.4371 | 0.5058 | 0.6090 | 6/161 |
| Real + HELIOS | 12 | 0.4021 | 0.4374 | 0.5092 | 0.6082 | 6/161 |
| Real + random thinning | 4 | 0.4051 | 0.4401 | 0.5138 | 0.6110 | 6/161 |
| Real + initial visibility | 4 | 0.3973 | 0.4379 | 0.5033 | 0.6073 | 6/161 |
| Real + profile-matched visibility | 4 | 0.3993 | 0.4420 | 0.5079 | 0.6151 | 5/161 |

The profile-matched run reached 7/161 small crowns at epochs 8 and 12, but its crown SB-PQ fell to about 0.4364 and point SB-PQ stayed near 0.399. Therefore a one-tree recall gain at those epochs is not a no-regression result. None passes a conservative gate requiring non-decreasing point/crown SB-PQ and SB-F1 plus better small-crown recall. The existing checkpoint was not replaced; no whole-area inference was launched for a rejected variant.

## What this means and next gate

Matching density and return counts is insufficient: the simulator may still place returns on the wrong parts of crowns or miss the real canopy/understory visibility pattern. The full TLS-derived crown targets can also disagree with what airborne LiDAR can actually observe. In addition, the retained model was already exposed to HELIOS data, and a six-parent/one-seed/short-run test has limited power.

Before a larger synthetic campaign, obtain several **paired, colocated real ALS–TLS plots** spanning canopy density and tree size. Compare per-height occupancy, pulse-level returns, canopy/ground fractions, and per-tree visibility on held-out pairs; calibrate the simulator on one set and validate it on another. Then repeat fine-tuning with multiple seeds and a fixed real validation/test protocol. Only scale up if it improves point and crown instance metrics without worsening small-crown performance. Superpoint architecture work can proceed independently, but this pilot gives no evidence that bulk synthetic generation should precede that validation.

## Reproducibility and artifacts

- Simulator: `scripts/simulate_tls_als_pilot.py`; dataset preparation and label-independent voxel sampling: `scripts/prepare_treescan_helios_dualcrown.py`.
- Training/evaluation: `scripts/train_tls_visibility_pilot.py`; consolidation: `scripts/report_tls_visibility_pilot.py`.
- Geometry manifests/QA: `outputs/tls_visibility_pilot_matched_v2/` and `outputs/tls_visibility_pilot_profile_matched_v1/`.
- Five fine-tuning runs with configurations, training logs, validation metrics and selected checkpoint records: `outputs/tls_visibility_finetune_pilot_v1/` and `outputs/tls_visibility_finetune_profile_pilot_v1/`.
- Consolidated machine-readable and Excel review: `outputs/tls_visibility_pilot_review_v2/review.json` and `review.xlsx`. The independently recomputed baseline is in `outputs/tls_visibility_pilot_review_v1/baseline_recomputed/metrics.json`.
- Synthetic geometry regression tests: `tests/test_tls_als_visibility_pilot.py`.
