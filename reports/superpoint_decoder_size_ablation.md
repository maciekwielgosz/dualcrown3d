# Superpoint decoder size ablation (GPU pilot)

Date: 2026-10-01. This experiment tested whether a larger graph, more instance
queries and a wider mask decoder improve the previous EZ-SP downstream pilot.
All arms use the same frozen LitePT72 features, 79 native-ALS training crops,
14 validation crops of 20 m, 12 epochs and seed 20261001. The held-out test
split was not used. The encoder is not trained in this pilot.

## Architecture and result

The previous arm has three graph layers of width 96, a 128-D masked instance
transformer with 96 queries and 1024 memory tokens. The larger arms use four
graph layers of width 192 and two additional width-192 mask-attention layers.
The q96 arm retains 96 queries and 1024 memory tokens; the q128 arm has 128
queries and 1536 memory tokens. Both retain the same three 128-D pretrained
masked-transformer layers and the point-specific residual. The new layers are
zero-initialized at their outputs. This is a combined architecture ablation,
not an isolated test of each component.

Source-balanced instance PQ and F1 at IoU 0.5 on the 14 validation crops:

| Arm | Epoch | Point PQ | Point F1 | Crown PQ | Crown F1 | Joint PQ score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small, common start | 0 | 0.1862 | 0.2533 | 0.2256 | 0.3442 | 0.2050 |
| Small, 96 queries | 12 | 0.2259 | 0.3034 | 0.1828 | 0.2910 | 0.2032 |
| Wider, 96 queries | 12 | 0.2196 | 0.3038 | 0.1639 | 0.2579 | 0.1897 |
| Wider, 128 queries | 12 | **0.2608** | **0.3552** | **0.2275** | **0.3622** | **0.2436** |

The q128 arm improves over the equally trained small arm by +0.0349 point PQ
and +0.0447 crown PQ. Exploratory paired within-source crop-bootstrap intervals
are [0.0156, 0.0697] and [0.0318, 0.0644], respectively. It also exceeds the
common initial arm on both metrics, although the crown-PQ margin there is only
+0.0019. Its best checkpoint is epoch 12. In contrast, the wider q96 arm is
below the small arm after 12 epochs, and its best checkpoint remains epoch 0.
Thus width alone did not help in this run; the beneficial q128 treatment also
changes the query count and memory budget, so the gain cannot be attributed to
one of them separately. Small-tree recall remains low: 0.1467 for q128 versus
0.1333 for the small trained arm.

The decoder has 3,056,279 parameters in either wider arm versus 1,210,965 in
the small arm. Peak allocated training VRAM was approximately 2,715 MiB for
q128 and 2,681 MiB for q96, versus 1,235 MiB for the small arm. Measured
cached-decoder forward time averaged 0.0765 s/crop for q128 versus 0.0437
s/crop for the small trained arm. These timings exclude encoder, superpoint
preparation, whole-scene merging and exports; they are **not** seconds per
hectare or end-to-end inference speed.

## Interpretation and limits

This is a promising **crop-level validation** result, not evidence that the
larger model beats the retained whole-scene dual-head production pipeline.
Those published local production-validation PQ numbers use a different
evaluation scope and are not directly comparable. The validation crops have
already been used in prior exploratory work, nine threshold pairs were tried
per arm on this same split, and only one seed was run. CULS and SCION each
contribute just one validation crop, so the bootstrap intervals understate
between-site uncertainty. Training does not include direct full-crown polygon
loss, nor additional crop augmentation. A clean held-out test and whole-scene
runtime/quality check are needed before considering promotion.

## Artifacts and reproduction

- [Excel experiment log](../outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot/experiments.xlsx): architecture/configuration, every epoch, metrics, paired intervals and export checks.
- [Comparison plot](../outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot/size_ablation.png).
- [q128 JSON report](../outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot/size_ablation_large_w192_q128.json) and [q96 JSON report](../outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot/size_ablation_w192_q96.json).
- [q128 checkpoint](../outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot/runs/ezsp_large_w192_q128/weights/best.pt) and [q128 14-crop LAZ/GPKG exports](../outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot/runs/ezsp_large_w192_q128/last_epoch_validation_exports/).
- [q96 final checkpoint](../outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot/runs/ezsp_w192_q96/weights/last.pt) and [q96 14-crop exports](../outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot/runs/ezsp_w192_q96/last_epoch_validation_exports/). The q96 `best.pt` is epoch zero, not the trained final model.

The 14 validation exports per arm were checked: every nonzero `tree_id` in
LAZ exists in both crown and treetop GeoPackages, and crown geometries are
valid. The input crop coordinates and any unknown CRS were preserved; no EPSG
code was guessed. Three decoder/integration tests passed. The retained previous
results and original geospatial data were not overwritten.

To reproduce in a **new** output directory after preparation of the same
native-ALS crop cache, run:

```bash
.venv-gpu/bin/python scripts/train_superpoint_decoder_pilot.py \
  --output NEW_RUN --method ezsp --run-name ezsp_large_w192_q128 \
  --queries 128 --memory-tokens 1536 --graph-width 192 --graph-layers 4 \
  --wide-dim 192 --wide-layers 2 --epochs 12 --eval-every 4
.venv-gpu/bin/python scripts/train_superpoint_decoder_pilot.py \
  --output NEW_RUN --method ezsp --run-name ezsp_w192_q96 \
  --queries 96 --memory-tokens 1024 --graph-width 192 --graph-layers 4 \
  --wide-dim 192 --wide-layers 2 --epochs 12 --eval-every 4
```

Use the same `prepare_superpoint_decoder_pilot.py --output NEW_RUN` protocol
documented in [the previous Stage-3 report](superpoint_stage3_decoder_result.md)
before these commands.
