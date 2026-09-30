# DualCrown3D joint fine-tuning campaign

The campaign is implemented in `scripts/train_dualcrown_campaign.py` and records
its configuration, per-epoch logs, validation metrics, checkpoints and an Excel
workbook under `outputs/dualcrown3d_joint_campaign_v1/`.

## Five experiments

1. Train both the original semantic/centre-vote heads and the transformer mask
   branch. Then train the deepest LitePT encoder stage (`enc4`) and all feature
   upsampling stages as well. Early encoder weights and batch-normalization
   buffers stay fixed. Both branches receive supervised losses.
2. Allow up to 40 epochs, with full real validation every four epochs and early
   stopping after 12 epochs without improvement (at least 16 epochs attempted).
   Keep epoch zero eligible. Heads use LR 5e-5, backbone 5e-6, cosine scheduling,
   AdamW, clipping at 2 and gradient accumulation over two crops. Each epoch has
   160 independent crop draws, capped at 16000 points for training. Production
   evaluation uses the existing 40000-point window limit.
3. Compare 50%/50% and 75%/25% real/synthetic sampling. Balance real collections;
   balance synthetic parent plots before distributing weight across variants.
4. Simulate two new HELIOS flights for every one of the 43 training scenes:
   `cross_sparse` (90 degrees, 650 m, 95 m/s, 25-degree half scan angle) and
   `diagonal_dense` (45 degrees, 400 m, 65 m/s, 18-degree half scan angle).
   Use the original 250 kHz sensor and scene objects. Retain exact object IDs
   and unchanged full TLS-derived crown polygons. A tree with no ALS returns
   remains in the full-crown reference; no synthetic points are invented.
   Hard crop sampling favours small, nearby and baseline-missed tree instances;
   missed-instance mining reads training data only.
5. Compare a larger decoder (5 layers, 128 queries, 1536 memory tokens) against
   the original decoder (3 layers, 96 queries, 1024 memory tokens). Width remains
   128, with four attention heads. Existing weights transfer strictly, and new
   layers initialize from the last pretrained attention layer. The enlarged
   decoder is trained from the same starting checkpoint as the other trials.

## Selection and reproducibility

All candidates start from the HELIOS fine-tuned epoch-14 checkpoint
`outputs/dualcrown3d_treescan_helios_finetune_v2/weights/best.pt`. Architecture
selection maximizes the geometric mean of source-balanced point and crown PQ
at IoU 0.50 on the 18 real validation plots. Postprocessing stays fixed to
`configs/dual_head_complete_consensus.json`. The winning setting is trained
with two additional seeds; mean and sample standard deviation are reported for
the three runs. The representative checkpoint is chosen using validation
before test evaluation.

The real 29-plot test and synthetic 7-plot test are scored after selection. The
real benchmark was exposed historically and is a paired regression benchmark,
not a pristine publication holdout. New simulations inherit train membership;
no new view of a validation/test source plot is added to training.

Internal GridPooling modules previously shuffled serialized orders even when
the top-level backbone's flag was false. This campaign disables all such flags,
checks exact repeated inference, and recomputes the baseline under the same
protocol. Old results should not be directly substituted for this baseline.

## Reproduction

From the repository root:

```bash
.venv-gpu/bin/python scripts/augment_helios_flights.py
.venv-gpu/bin/python scripts/train_dualcrown_campaign.py --smoke
.venv-gpu/bin/python scripts/train_dualcrown_campaign.py
.venv-gpu/bin/python scripts/benchmark_dualcrown_campaign.py
.venv-gpu/bin/python scripts/report_dualcrown_campaign.py
```

Completed runs are reused after interruption. An incomplete run raises an error
for investigation instead of silently overwriting checkpoints. Source scenes,
prior outputs and production defaults are preserved. New checkpoint model
arguments are supported by `scripts/predict_dual_head.py`.

The final workbook contains per-run configurations and checkpoint paths,
per-epoch logs, test metrics by collection, three-seed stability, and paired
plot-bootstrap intervals. The speed check uses three real validation plots,
three alternating-order repeats, and includes network inference plus CPU
merging (not file loading/export).

During training-only missed-instance mining, a floating-point upper-window
boundary issue was corrected. Final windows include the exact observed maximum
coordinate. Context membership was checked against the previous implementation
on all 66 real/synthetic validation and test plots and was unchanged, so the
correction does not change this campaign's held-out comparisons. A regression
test covers the triggering coordinate pattern.
