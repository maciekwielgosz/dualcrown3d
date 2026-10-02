# Held-out test: output 20 versus output 22

The two frozen models were run on the same 24 labelled, point-evaluation-eligible test plots from `combined_als_crowns_supervision_v4/manifest.csv`. No test labels were used to select checkpoints, thresholds, or instances. Five other test rows were excluded because the manifest marks them ineligible for point evaluation. Matching uses instance IoU >= 0.5. Source-balanced scores average the six source collections equally; pooled scores combine all eligible plots.

| Metric | Model 20: dualcrown3d joint fine-tune | Model 22: EZ-SP wide Q128 |
| --- | ---: | ---: |
| Point instance PQ, source-balanced | **0.415** | 0.143 |
| Point instance F1, source-balanced | **0.520** | 0.198 |
| Crown polygon PQ, source-balanced | **0.417** | 0.227 |
| Crown polygon F1, source-balanced | **0.584** | 0.333 |
| Point instance precision / recall, pooled | **0.450 / 0.522** | 0.100 / 0.321 |
| Crown polygon precision / recall, pooled | **0.547 / 0.528** | 0.182 / 0.500 |
| Matched crowns <= 4 m² | 0 / 65 | **5 / 65** |
| Matched crowns <= 10 m² | 19 / 241 | **40 / 241** |
| Predicted instances across test plots | 1,358 | 4,145 |

Model 20 is better on all four headline quality metrics. Model 22 catches more small crowns but produces about three times as many instances, with 2,561 point-instance false positives versus 566 for Model 20. This is oversegmentation, not a net improvement. It should remain experimental rather than replace Model 20.

Source-balanced crown PQ by collection:

| Collection | Model 20 | Model 22 |
| --- | ---: | ---: |
| FOR-instance CULS | 0.748 | 0.477 |
| FOR-instance NIBIO | 0.472 | 0.200 |
| FOR-instance SCION | 0.318 | 0.215 |
| ideas_als CEDAR_CYPRESS | 0.325 | 0.138 |
| ideas_als FGI_EMIT | 0.462 | 0.207 |
| ideas_als WILDFOREST3D | 0.178 | 0.125 |

The full report and visual inspection files are in `output_23_labeled_test_output20_vs_output22/`: `comparison.xlsx`, `comparison.json`, 24 LAS plus 24 LAZ clouds per model in `Model20/PointClouds` and `Model22/PointClouds`, and crown/treetop GeoPackages in each model's `Segmentation3` folder. The LAS/LAZ files contain voxelized input points, not every original full-resolution return. The `tree_id` extra dimension is the prediction; `reference_tree_id` is the test reference included only for visual inspection. XY retains the source coordinates, while Z is height above ground, not absolute elevation. The two models use their frozen production crown-generation methods, so crown PQ compares end-to-end output quality, not identical polygon postprocessing.
