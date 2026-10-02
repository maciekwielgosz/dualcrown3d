# Superpoint implementation: Stage 0–2 gate (2026-10-01)

## Decision

Stages 0, 1 and the Stage-2 affinity pilot are implemented and evaluated. **Do not advance this partition to the full graph decoder or a large HELIOS/TLS campaign yet.** The learned partition failed the plan's matched-compression boundary/instance gate, including two training-only revisions. The production DualCrown3D checkpoint and outputs are unchanged. No real-ALS test plots were used to select this approach.

The later ablations were motivated by the first validation result, so this is exploratory validation, not an untouched confirmation. A future promoted model will require a fresh site or genuinely independent held-out evidence.

The retained production validation metrics are point/crown source-balanced PQ **0.4051/0.4442** and F1 **0.5165/0.6195**. These are full prediction metrics. The values below are *GT-assisted partition oracles*, not predictions, and must not be compared numerically with production PQ.

## Stage 0: data contract

The audit checked 295 derived records: 147 native point-cloud records, 62 original HELIOS plots and 86 extra HELIOS training flights. All referenced NPZ and crown files existed; there were no detected parent-group split conflicts. The eligible native primary protocol remains 79 train, 14 validation and 24 historically exposed test plots. The v4 target convention is `tree_id=-1` unknown, `0` known background and positive values for tree instances. The source datasets are heterogeneous; the `native_als` inventory label follows the existing project naming, not a claim that every acquisition used the same airborne sensor.

Across 79 eligible training plots, the 10th/50th/90th percentiles of raw point density are approximately **127/943/5978 points/m²**; after 0.25 m voxelisation they are **71/106/386 voxels/m²**. The median unknown-label fraction is about **49.6%**. These differences matter for neighborhood construction and sampling. The synthetic TreeScan HELIOS clouds share parent scene IDs with TLS, but registered, temporally matched *real* ALS/TLS pairs with shared tree IDs have not been established. Synthetic matching must not be described as real cross-sensor registration.

## Stage 1: fixed-partition feasibility

The pilot deterministically chose a median-size eligible plot per split/source collection: 11 train and 5 validation plots. The oracle assigns each 3-D geometric cell the majority **known** reference ID and groups cells with the same ID. Unknown labels do not vote. It is a diagnostic only and uses GT IDs; no inference path uses this oracle.

| Cell size | Val compression | Val oracle point PQ | Val local boundary recall | Trees disconnected by 1 m, k=8 point graph |
| --- | ---: | ---: | ---: | ---: |
| 0.5 m | 4.17× | 0.971 | 0.714 | 93/167 |
| 1.0 m | 19.32× | 0.921 | 0.426 | 69/167 |
| 1.5 m | 49.94× | 0.854 | 0.288 | 53/167 |

Thus small fixed cells retain a useful oracle at meaningful compression, but both mixed boundaries and graph connectivity are bottlenecks. A separate label-free centroid candidate graph (0.5 m cells) reduced disconnected trees to **21/167** at 4 m/k=32 and **12/167** at 8 m/k=64, but the latter costs roughly 1–3 million candidate edges per plot and still leaves 10/40 trees disconnected in the representative WildForest3D plot. The true instance labels were used only to count these disconnections after graph construction.

## Stage 2: frozen-feature affinity pilot

A symmetric MLP predicts same-instance affinity from absolute/multiplicative LitePT feature pairs and relative geometry. Its 72-dimensional point features come from the **frozen retained encoder**, which previously saw HELIOS; this is an engineering pilot, not a rigorously ALS-only initialization. One deterministic, label-independent-anchor 20 m crop was cached for each of 79 training and 14 validation plots. The edge head used 12 CUDA epochs, balanced same-tree, cross-tree and tree/background edges, and a bounded agglomeration rule. The checkpoint is experimental and not a production crown model.

With a threshold calibrated on one training plot per source collection, validation compression was closely matched:

| Partition, 14 validation crops | Compression | Oracle point PQ | Cross-tree boundary recall | Mixed-group fraction |
| --- | ---: | ---: | ---: | ---: |
| Fixed 0.5 m | 1.638× | **0.9759** | **0.9456** | 0.0121 |
| Learned affinity | 1.643× | 0.9753 | 0.9040 | 0.0077 |

Although the learned method makes fewer mixed groups by count, it loses more *cross-tree edges* inside its groups. The initial non-matched threshold gave oracle PQ 0.9813, but only 1.471× compression and lower boundary recall 0.9313; that is not a fair win. Training-only sweeps of 0.6/0.8/1.0 m maximum extent and frozen center-vote coherence also failed to improve **both** oracle PQ and boundary recall within ±5% mean compression. Therefore no new validation setting was selected from those sweeps and Stage 2 did not pass.

The likely failure mode is single-link bridge merging: one confidently wrong local edge can join points from different crowns. This is an interpretation of the diagnostics and code, not proof that it is the only cause. The next architecture revision should require multi-edge/cluster-consistent evidence and explicitly test disconnected-tree recovery, rather than merely relaxing thresholds or increasing HELIOS volume. A fixed-cell graph baseline remains possible, but it needs its own Stage-3 pilot and comparison with the retained checkpoint before promotion.

## Reproduction and artifacts

Run from `DL_model_version` using `.venv-gpu/bin/python`:

```bash
python scripts/audit_superpoint_data.py
python scripts/audit_superpoint_feasibility.py
python scripts/audit_superpoint_connectivity.py
python scripts/train_superpoint_affinity.py
python scripts/calibrate_superpoint_affinity.py
python scripts/sweep_superpoint_partition.py
python scripts/sweep_superpoint_vote_coherence.py
python scripts/report_superpoint_pilot.py
```

Each completed output directory is protected against silent replacement; choose a new directory or version for a repeat. Inputs and completed geospatial data were not rewritten. The canonical machine-readable audit, per-plot metrics, configurations and experimental checkpoint are under `outputs/dualcrown3d_superpoint_v1/`. The consolidated `review_stage0_2/review.json` and `review.xlsx` contain run parameters, data inventory, source-balanced comparisons, connectivity, per-plot diagnostics and training history. The Stage-2 frozen-affinity checkpoint is `stage2_affinity_pilot/affinity_head.pt`; it does **not** export crown polygons or point IDs.

Stages 3–7 of the plan remain conditional and were not executed because the Stage-2 gate failed. In particular, there is no new LAZ/GPKG export, full-scene speed claim, TLS teacher, large simulation or new held-out test result.
