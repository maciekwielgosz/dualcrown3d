# Full-plot fragment verification and cross-window reconciliation

Status: experimental, **not promoted**. Model 20 and the existing output folders remain unchanged. The held-out test set was not used for selection or inference in this experiment.

## Implemented protocol

1. Cache every raw Q128 Model 22 mask from overlapping 20 m ALS windows (8 m overlap) on complete annotated training and validation plots, before tree IDs are assigned. Cache and model/checkpoint/manifest signatures are checked.
2. Build an overlap graph of the raw masks and fuse related proposals *before* final IDs. The default graph uses point-support overlap >=0.55, point IoU >=0.20, XY centroid distance <=4 m and treetop-height difference <=3 m. An optional mask-size-similarity condition was added for diagnostics without changing the default protocol.
3. Train a 14-feature, 64-32-1 MLP proposal verifier on the fused masks. Positives are GT point IoU >=0.5; same-tree fragments with best IoU in [0.1, 0.5) receive extra negative weight. Unknown-dominated proposals are excluded. Training ran for 50 epochs on the NVIDIA RTX PRO 500 Blackwell GPU; the selected checkpoint is epoch 17 (validation proposal average precision 0.5429).
4. At assignment time, an accepted proposal supplies its complete mask. A heavily claimed mask is rejected rather than turning its unclaimed tail into an automatic new tree ID.

The 79 training plots supplied 17,992 graph proposals: 1,021 positives, 3,435 weighted hard negatives and 9,535 ignored proposals. The 14 validation plots supplied 3,020 graph proposals, including 164 point-IoU positives. The checkpoint and validation sweep are in `outputs/dualcrown3d_fullplot_verifier_v1/`.

## Full-plot validation

| Variant | Source-balanced point PQ | Source-balanced crown PQ | Small crowns (area <=10 m², IoU >=0.5) |
|---|---:|---:|---:|
| Existing Model 22 | 0.1243 | 0.1967 | 20/161 |
| Full-plot verifier, probability >=0.30, claimed limit 0.15 | 0.1995 | 0.2491 | 1/161 |
| Full-plot verifier, probability >=0.15, claimed limit 0.15 | 0.1960 | 0.2480 | 3/161 |

The preregistered validation gate required both PQ values to exceed Model 22 **and at least 16/161 small-tree hits** (80% of Model 22). No threshold passed. The apparent PQ gain therefore does not qualify as a better crown model for the user's goal.

## Why the gate failed

An audit of all 14 validation plots compared GT small-crown polygons with *full* raw-mask footprints before point competition. At polygon IoU >=0.5, there were only 5 possible small-crown matches after the default graph, 13 with a 0.85 minimum mask-size ratio, and 15 with no mask fusion at all. These figures are candidate potential, **not** deployable model detections. Model 22 nevertheless detected 20 small crowns after its greedy point assignment. In FGI_EMIT plot 1018 alone, Model 22 hit 12/114 small crowns, while only 5/114 are recoverable as full-mask footprints before competition.

The observed discrepancy means that at least some useful small crowns emerge from the point support left after competing masks claim their points. A rule that categorically forbids residual instances removes these as well as false fragments. Merely changing graph thresholds cannot meet the current small-crown gate. The next model would need a supervised *residual/fragment completeness* decision, or a decoder that predicts small crowns as distinct complete masks, with explicit training examples for both true small crowns and fragments of large crowns. It must still be selected on full-plot validation before any new held-out test output is created.

## Reproduction and files

- Raw masks: `scripts/cache_fullplot_proposals.py`, `outputs/dualcrown3d_fullplot_verifier_v1/raw/`.
- Graph, verifier, geometry: `pointcloud/superpoints/fullplot_graph.py`, `fullplot_verifier.py`, `graph_geometry.py`.
- Training/validation: `scripts/train_fullplot_mask_verifier.py`; checkpoint `outputs/dualcrown3d_fullplot_verifier_v1/weights/best.pt`; sweep `validation_selection.json`.
- Candidate-potential audit: `scripts/audit_fullplot_mask_recall.py`; per-plot results `outputs/dualcrown3d_fullplot_verifier_v1/validation_full_mask_audit.json`.
- Synthetic tests: `tests/test_fullplot_verifier.py`; all three pass with `.venv-gpu/bin/python -m unittest discover -s tests -p test_fullplot_verifier.py -v`.

To rebuild from the already cached raw masks, run `.venv-gpu/bin/python scripts/train_fullplot_mask_verifier.py --phase all` (the script checks and reuses completed matching caches). The audit can be rerun with `.venv-gpu/bin/python scripts/audit_fullplot_mask_recall.py`.
