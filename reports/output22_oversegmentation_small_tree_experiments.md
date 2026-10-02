# Model 22 oversegmentation and small-crown recovery experiments

Status: **no candidate promoted**. All new thresholds were selected or rejected on 14 labelled validation plots; the 24 held-out test plots were not reused for threshold selection or for scoring any rejected variant. The original Model 20 and Model 22 outputs remain unchanged.

## Diagnosis

- On the labelled CULS test plot, Model 20 produced 42 instances and Model 22 produced 94 for 21 reference crowns. Seventeen of those reference crowns contain at least two Model 22 labels with >=100 voxels each (up to six fragments).
- The Model 22 full-scene merger accepts a partly overlapping proposal if less than 60% of it is already claimed, then turns its residual points into a new instance. Its later support fusion can add points to an anchor but cannot merge two accepted fragments. Overlapping 20 m windows and Q128 proposals make this failure mode frequent.
- On validation, 87/161 reference crowns <=10 m2 had >=80% of their points already assigned by Model 20. A method restricted to previously unassigned points cannot recover most small trees.
- Among Model 22 validation proposals, 22 polygon candidates matched a small reference crown at IoU >=0.5; their median combined instance confidence (object score times mask confidence) was 0.104, versus 0.144 for other candidates. Raising the object threshold therefore preferentially removes true small crowns. This is not a measurement of raw score-head logits alone.

## Validation experiments (14 full plots)

Scores are source-balanced PQ at IoU >=0.5; small-tree hits refer to reference crown area <=10 m2 (161 total).

| Variant | Point PQ | Crown PQ | Small hits | Result |
| --- | ---: | ---: | ---: | --- |
| Model 20 anchor | 0.4051 | 0.4442 | 6 | retained baseline |
| Model 22 direct | 0.1243 | 0.1967 | 20 | severe oversegmentation |
| Protected residual addition (output 24) | 0.4051 | 0.4442 | 6 | no small-tree gain |
| Bounded internal split, best small gain (output 25) | 0.4005 | 0.4386 | 8 | precision and crown PQ below gate |
| Geometry/low-score split, best small gain (output 26) | 0.3905 | 0.4227 | 11 | substantial quality loss |

The quality gate allowed at most -0.005 in point PQ, -0.005 in crown PQ and -0.01 in pooled point precision relative to Model 20. The bounded split's best small-gain setting missed the crown PQ gate and reduced precision from 0.4170 to 0.4013. None of the geometry-first settings passed.

Two GPU fine-tuning experiments then tested the existing `object_logits` score head:

1. Full decoder fine-tuning with duplicate-aware object supervision raised 20 m validation-crop score from 0.2436 to 0.3096, but sparse-tree recall dropped from 0.1467 to 0.0267. The gated selection therefore remained at epoch 0.
2. Training only `decoder.score` and `decoder.wide.score`, with all mask and backbone parameters frozen, likewise failed the sparse-tree recall gate after every trained epoch. Its initial epoch remains the selected checkpoint.

These crop metrics are not directly comparable with the full-plot PQ table above. Neither fine-tuned checkpoint was promoted to full-plot test inference, because both failed the validation condition that small-tree detection be preserved.

## Interpretation and next experiment

The problem is not simply a missing score head: Model 22 already has two score layers. Post-hoc thresholding cannot reliably distinguish true small trees from fragments. A stronger follow-up is a new training campaign with explicit *same-tree fragment negatives* and *small-crown positive examples* sampled from complete plots, plus a cross-window reconciliation step that merges fragments before final point assignment. It should be selected on full plots with both small-crown recall and PQ/precision gates before touching the test set again. Until that succeeds, Model 20 remains the trustworthy output; Model 22 is experimental.

Artifacts: `output_24_guarded_small_crown_fusion/selection.json`, `output_25_guarded_internal_small_splits/selection.json`, `output_26_low_confidence_small_split_verification/selection.json`, `outputs/dualcrown3d_small_tree_quality_v1/`, and `outputs/dualcrown3d_score_head_small_trees_v1/`. They contain validation and training results, not improved test LAS files.
