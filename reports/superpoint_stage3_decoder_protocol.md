# Stage-3 revisable decoder pilot — 2026-10-01

The user approved continuing to a small downstream experiment despite the
Stage-2 partition gate not passing. For this experiment only, final predicted
point/crown quality is the decision criterion. Passing both oracle diagnostics
is not required. This does not grant deployment approval or establish a gain.

## Controlled comparison

- Retained: original pretrained mask decoder evaluated on the same crops with
  the same point assignment and polygon exporter. This is a mask-head reference,
  **not** a re-evaluation of the complete deployed dual-consensus pipeline.
- Control: pretrained decoder with hybrid spatial/embedding queries and all
  canopy seed candidates, fine-tuned without graph context.
- Fixed: the control plus 0.5 m geometric grouping and a residual graph network.
- EZ-SP: identical network/training to Fixed, using the official contour-prior
  partition kernel on the previously trained 16-D instance embedding.

The graph network has three message-passing layers of width 96, with a 3 m/k16
centroid neighborhood. It pools 72-D frozen LitePT features, centroid, spread
and count. Graph context is broadcast to the points and combined with each
point's original features and relative position through a learned residual.
The last residual projection starts at zero. Group membership does not force
identical point masks. The existing decoder uses 128-D hidden features,
96 object queries, three masked-attention layers and 1024 memory representatives.

Training updates the complete mask decoder and graph adapter, with the LitePT
encoder and legacy semantic/offset heads frozen. The existing annotation-aware
point, offset, embedding and Hungarian mask losses supervise predictions.
Unknown IDs are ignored. There is no ground-truth-assisted partition or
inference assignment. No semantic probability veto excludes points from masks.

## Data, selection and cost

The prepared data reproduces the earlier 79 training and 14 validation crops
exactly (one deterministic 20 m crop, max 12k voxels, per eligible parent plot).
Coordinate and ID equality are asserted against the earlier feature cache.
Parent train/validation groups are checked for disjointness. The test split is
not used. Initialization had previously seen HELIOS; this is an engineering
comparison rather than ALS-only pretraining.

All three trainable arms use the same initial decoder weights, seed, crop order,
12 epochs, AdamW, decoder LR 1e-4 and graph LR 3e-4 where applicable. No new
augmentation is applied to frozen feature caches. Validation is at epochs
0/4/8/12, with the same nine object/mask threshold pairs for each arm. Selection
maximizes the geometric mean of source-balanced point and crown PQ.

EZ-SP regularization is selected **without labels**, per crop, to match fixed
group count. The repeated search overhead is logged and must be included in
any deployment-cost estimate. Decoder timings exclude LitePT extraction,
partition preparation and disk export; they are not full seconds per hectare.

Point PQ/F1 use IoU 0.5 and ignore unknown point labels. Crown PQ/F1 compare
predicted footprints with complete native polygons clipped to the 20 m crop,
with the dataset's annotation-ignore mask. Sparse-tree recall means reference
instances with at most 100 retained voxels; it is not a physical height/area
definition of a small tree. Report coverage alongside precision and PQ.

The exporter is shared across all arms: 0.5 m occupancy, 3x3 closing, hole fill,
preservation of original support and all disconnected components under the same
instance ID. It does not draw a convex hull across remote components. Point
assignment uses pairwise duplicate suppression and confidence competition;
removing an undersized mask releases its points for reassignment.

## Outputs and interpretation

`outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot/` contains matched crops,
configuration, training histories, validation grids, best/last checkpoints,
Excel logs and small-crop LAZ/GeoPackage exports. Exported Z is height AGL,
not absolute elevation; LAZ files contain retained voxels rather than the full
original point cloud. Predictions and crown polygons use the same instance IDs.

This pilot measures learned segmentation, not oracle performance. Its small,
already-explored validation set and one seed do not establish generalization.
If promising, the next work is more training crops, independent real-ALS
validation and full-scene persistent-ID/overlap fusion, followed by full-pipeline
timing. TLS/simulation transfer and teacher distillation remain later controlled
experiments. The current pilot is not a completed whole-scene Stage-3 deployment.
