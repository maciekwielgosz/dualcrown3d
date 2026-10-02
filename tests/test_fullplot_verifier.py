import unittest

import numpy as np

from pointcloud.superpoints.fullplot_graph import DEFAULT_GRAPH, reconcile, target_labels
from pointcloud.superpoints.fullplot_verifier import assign_complete_masks


class FullPlotVerifierTests(unittest.TestCase):
    def setUp(self):
        self.arrays = {'coord': np.column_stack((np.arange(10) * .1,
                                                np.zeros(10), np.full(10, 3.))).astype(np.float32)}
        self.raw = {'candidate_offset': np.asarray([0, 8, 14]),
                    'point_index': np.r_[np.arange(8), np.arange(6)].astype(np.int32),
                    'point_score': np.ones(14, np.float32),
                    'object_score': np.asarray([.9, .8], np.float32)}

    def test_size_guard_preserves_nested_small_proposal(self):
        base = {**DEFAULT_GRAPH, 'min_points': 4}
        self.assertEqual(len(reconcile(self.arrays, self.raw, base)['feature']), 1)
        guarded = reconcile(self.arrays, self.raw,
                            {**base, 'link_min_size_ratio': .9})
        self.assertEqual(len(guarded['feature']), 2)
        self.assertEqual(guarded['offset'][-1], 14)

    def test_rejected_overlap_never_becomes_residual_instance(self):
        graph = reconcile(self.arrays, self.raw,
                          {**DEFAULT_GRAPH, 'min_points': 4,
                           'link_min_size_ratio': .9})
        labels, _, accepted = assign_complete_masks(
            self.arrays, graph, np.asarray([.9, .8]),
            object_threshold=.5, claimed_limit=.2, min_points=4)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(set(np.unique(labels)), {0, 1})

    def test_truth_label_uses_full_point_iou(self):
        graph = reconcile(self.arrays, self.raw,
                          {**DEFAULT_GRAPH, 'min_points': 4})
        truth = np.r_[np.ones(6, np.int32), np.zeros(4, np.int32)]
        targets, iou, _ = target_labels(graph, truth)
        self.assertEqual(targets.tolist(), [1])
        self.assertAlmostEqual(float(iou[0]), .75)


if __name__ == '__main__':
    unittest.main()
