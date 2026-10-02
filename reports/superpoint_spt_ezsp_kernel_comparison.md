# SPT Cut Pursuit and EZ-SP partition pilot (2026-10-01)

## Decision

Neither tested superpoint partition kernel passes the Stage-2 gate. At nearly
identical compression, SPT Cut Pursuit and EZ-SP contour-prior merging both
lose considerably more boundaries between neighboring tree instances than the
0.5 m fixed-cell partition. **Do not replace the current partition or promote
either kernel into the production DualCrown3D inference path.** The current
model, production checkpoints, LAZ and GPKG outputs were not modified.

This pilot tests the **official partition algorithms**, not the complete
published SPT or EZ-SP neural networks. Both kernels receive the same 20 m
cached LitePT crops and 0.75 m/k=12 point graph. The second series uses an
instance-boundary embedding head trained on the existing frozen LitePT features;
it is an EZ-SP-style adaptation, **not** the paper's sparse CNN. The retained
LitePT encoder had previously seen HELIOS, so this is an engineering comparison,
not an ALS-only-from-scratch experiment. The kernel packages came from the
official SPT/EZ-SP project: `pycut-pursuit==0.1.4`,
`torch-graph-components==0.1.1`, with `pygrid-graph==0.0.4`. An experimental
native-PyTorch scatter shim replaces unavailable `torch-scatter` binaries for
the current Torch/CUDA build; small deterministic tests cover its operations.

## Protocol and results

There were 79 eligible training crops for the 16-D instance-boundary embedding
(12 GPU epochs; 78 plots contributed sampled valid edges per epoch) and 14
eligible native validation crops. No test split was read. Reference IDs are
never used to form partitions or choose regularization. For each validation
crop, regularization was selected by a label-free search to match the number
of 0.5 m fixed-cell groups. This is a *matched-budget diagnostic*, not a
deployment-ready fixed hyperparameter; its repeated search cost is excluded
from the one-shot timing below. Unknown labels (`tree_id=-1`) are ignored in
quality metrics.

Each value is a mean across plots, except the one-shot partition times.
Boundary recall uses 13 plots because one crop has no annotated adjacent
cross-tree point pairs. `Oracle PQ` assigns each superpoint the majority known
reference tree ID, then evaluates the resulting point instance masks; it is an
optimistic partition diagnostic, **not model-predicted PQ**.

| Features / edge weight | Partition | Compression | Oracle PQ | Boundary recall | One-shot partition time/crop |
| --- | --- | ---: | ---: | ---: | ---: |
| Any | Fixed 0.5 m | 1.638× | 0.9759 | **0.9456** | ~0.007–0.009 s |
| Frozen LitePT / prior affinity | SPT Cut Pursuit | 1.640× | 0.9741 | 0.8586 | 0.631 s |
| Frozen LitePT / prior affinity | EZ-SP contour prior | 1.634× | 0.9725 | 0.8493 | 0.042 s |
| Trained instance embedding / unit | SPT Cut Pursuit | 1.640× | 0.9770 | 0.8632 | 0.692 s |
| Trained instance embedding / unit | EZ-SP contour prior | 1.640× | 0.9755 | 0.8540 | 0.042 s |
| Trained instance embedding / embedding affinity | SPT Cut Pursuit | 1.637× | **0.9774** | 0.8736 | 0.679 s |
| Trained instance embedding / embedding affinity | EZ-SP contour prior | 1.638× | 0.9768 | 0.8689 | 0.031 s |

The strongest Cut Pursuit variant gained only +0.00153 oracle PQ versus fixed
cells (paired plot bootstrap 95% interval -0.00370 to +0.00675) and lost
0.0720 boundary recall (-0.1166 to -0.0313). The strongest EZ-SP variant gained
+0.00088 oracle PQ (interval -0.00534 to +0.00733) and lost 0.0767 boundary
recall (-0.1239 to -0.0334). The intervals are descriptive for these crops,
not proof of out-of-sample performance. Maximum per-plot compression mismatch
in the strongest comparison was <0.5%; the first frozen-feature comparison
was within 2.5%.

On the four WildForest3D validation crops with cross-tree boundaries, the
strongest fixed/Cut Pursuit/EZ-SP mean recalls were respectively
0.954/0.814/0.798. Both advanced methods produced fewer mixed groups *by
count*, yet the remaining mixed groups crossed more true tree boundaries.
This is consistent with a few damaging bridge merges in touching crowns;
it is not a proof that this is the only error source. Oracle F1 was 1.0 for
all partitions because the diagnostic grants the true majority ID to each
group, and is not informative for the real model's tree-detection ability.

The times above cover only one partition call on cached ~12k-point crops;
they exclude the LitePT backbone, embedding/affinity inference, label-free
regularization search, large-scene window fusion and LAZ/GPKG export. They
must not be reported as seconds per hectare or full-inference speed.

## Interpretation and next experiment

Replacing greedy single-link merging with a respected published partition
kernel did not by itself preserve individual crown boundaries. The issue is
not merely the choice of optimizer: the local representation and its supervision
still fail to distinguish many touching crowns. A defensible next pilot is an
explicit **cluster-consistent boundary veto** or uncertainty-aware split of
mixed groups, trained on more than one crop per parent plot and checked at the
same compression. A new untouched validation site is needed before promoting
a model after these exploratory comparisons. The Stage-3 graph instance model
and full production export remain conditional on passing the partition gate.

## Artifacts and reproduction

- Consolidated [Excel/JSON review](../outputs/dualcrown3d_superpoint_v1/review_methods/review.xlsx)
  and `review.json`: configurations, 14-plot metrics, paired intervals and
  the embedding training history.
- Three complete result folders under `outputs/dualcrown3d_superpoint_v1/`:
  `stage2_spt_ezsp_partition_pilot_v2`, `stage2_instance_partition_unit`,
  and `stage2_instance_partition_embed_weight`.
- Experimental checkpoint:
  `outputs/dualcrown3d_superpoint_v1/stage2_instance_embedding/embedding.pt`.
  This is only a point embedding head, not a crown segmentation checkpoint.
- Scripts: `scripts/train_superpoint_instance_embedding.py`,
  `scripts/benchmark_superpoint_algorithms.py`,
  `scripts/benchmark_superpoint_instance_embedding.py`, and
  `scripts/report_superpoint_algorithms.py`.

The optional packages were installed with `pip --target` in
`/tmp/dualcrown3d_superpoint_deps_20261001`, without administrator rights or
changes to `.venv-gpu`. The NumPy-1.x pycut wheel required isolated
`numpy==1.26.4` and `scipy==1.15.3`; the main GPU environment retains its own
NumPy 2.x. Recreate that target and set `PYTHONPATH` to it before running the
benchmark scripts. Relevant compatibility/partition tests: 10 passed.
