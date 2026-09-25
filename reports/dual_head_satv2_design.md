# LitePT-S with two output branches

Implementation reference: [SegmentAnyTreeV2](https://arxiv.org/html/2606.08206v2), sections 3.5–3.9. This is an adaptation, not an exact reproduction of that model or its reported results.

One frozen LitePT-S encoder/decoder processes XYZ (height above terrain) and intensity. Its 72-dimensional voxel features feed two branches:

1. The existing semantic and centroid-offset heads, followed by vote clustering and filled crown polygons.
2. A trainable point-instance branch: multiscale context (0.75, 2, 6 m), two residual MLP blocks (128 channels), foreground refinement, ISA embedding (5 dimensions), and a mask decoder (up to 96 seeded queries, 3 layers, 4 attention heads, 128 channels).

The mask decoder adapts the paper's tree-only attention, distinct attention/mask feature projections, embedding-space farthest-point sampling, seed-based one-to-many supervision, auxiliary mask losses, and mask-IoU scoring. We use Dice/focal mask losses, IoU-score regression and discriminative embedding supervision. The first three epochs use spatial seeds while embeddings mature. A centroid-vote spatial prior and multiscale pooling are additional adaptations. The model retains binary foreground supervision because this merged dataset does not provide a uniform wood/leaf taxonomy.

To fit the 6 GB GPU, attention uses at most 1024 memory tokens, while mask prediction retains the full crop's voxel resolution. Windows remain 20 x 20 m with 8 m overlap and at most 40000 sampled voxels per forward. Evaluation never receives GT labels as model inputs. Training may use GT foreground for the decoder as in the paper.

The legacy backbone and heads are frozen, including BatchNorm statistics. Training verifies exact equality of their tensors against the original checkpoint. The new branch is selected using validation source-balanced point-instance PQ@0.50. Crown PQ/F1 and native-point-only metrics are reported separately. Native and polygon-projected point supervision must not be presented as equally strong annotation. The test split is not used for selecting this implementation.

Scene-level masks originally used confidence-ranked asymmetric point-overlap suppression, followed by [support-preserving fusion](support_fusion_fix.md). Current inference adds [dual-head consensus](dual_consensus_fix.md): stronger centre-vote instances act as anchors, masks complete shared support or recover independent missing trees, and a one-pass spatial/centre-consistent stage recovers residual points. Network architecture and weights are unchanged. A point has at most one final instance ID. IDs are global across exported tiles. Full-resolution LAZ uses an exact voxel-to-original-point inverse; original XYZ, intensity and source classification are retained. `tree_id`, `legacy_tree_id`, `tree_confidence`, `pred_semantic`, `height_agl`, `assignment_source`, and `segmentation_status` support CloudCompare inspection. RGB displays instance colours. Ground class 2 receives instance ID 0.

The two branches need not yield identical trees. `Segmentation3` is the legacy polygon output; `PointHead/Segmentation3` contains polygons from new point masks. New `tree_id` matches `PointHead` polygon `treeID`. `legacy_tree_id_map.csv` maps the globally numbered legacy labels to the old per-tile polygon IDs.

Run from the workspace root:

```bash
DL_model_version/.venv-gpu/bin/python -u DL_model_version/scripts/train_dual_head.py --output-dir DL_model_version/outputs/dual_head_satv2_litept_v3
DL_model_version/.venv-gpu/bin/python -u DL_model_version/scripts/calibrate_dual_merge.py
DL_model_version/.venv-gpu/bin/python -u DL_model_version/scripts/audit_dual_head.py
DL_model_version/.venv-gpu/bin/python -u DL_model_version/scripts/predict_dual_head.py
DL_model_version/.venv-gpu/bin/python -m unittest discover -s DL_model_version/tests -p test_dual_head.py -v
```

Training writes `configuration.json`, `training_log.csv`, `experiments.xlsx`, full validation reports, `weights/best.pt`, `weights/last.pt` and the frozen checkpoint/configuration in `selected.json`. Existing model checkpoints and source geospatial data are not overwritten.

Post-training merge calibration compares core-window mask ownership with pooling all overlap masks, using asymmetric suppression thresholds 0.15 and 0.5. This is validation-only selection with a fixed trained checkpoint. The authoritative final settings are `selected.json`; `selection_before_merge_calibration.json` preserves the prior selection. `point_comparison.json` compares both heads on the same validation point labels, and its figures are also added to the Excel workbook.
