import unittest

import numpy as np
import torch

from pointcloud.superpoints.affinity import EdgeAffinityHead, bounded_partition


class AffinityTests(unittest.TestCase):
    def test_edge_head_is_symmetric(self):
        model = EdgeAffinityHead(feature_dim=3)
        feature = torch.randn(4, 3)
        coord = torch.randn(4, 3)
        forward = torch.tensor([[0, 1], [2, 3]])
        reverse = forward[:, [1, 0]]
        torch.testing.assert_close(model(feature, coord, forward),
                                   model(feature, coord, reverse))

    def test_bounded_merge_keeps_two_instances_apart(self):
        coord = np.asarray([[0., 0., 1.], [.2, 0., 1.],
                            [1.2, 0., 1.], [1.4, 0., 1.]])
        edge = np.asarray([[0, 1], [1, 2], [2, 3]])
        groups = bounded_partition(coord, edge, np.asarray([.99, .99, .99]),
                                   threshold=.5, max_extent_m=.6)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[2], groups[3])
        self.assertNotEqual(groups[1], groups[2])

    def test_low_affinity_prevents_merge(self):
        coord = np.asarray([[0., 0., 1.], [.1, 0., 1.]])
        edge = np.asarray([[0, 1]])
        groups = bounded_partition(coord, edge, np.asarray([.2]), threshold=.8)
        self.assertNotEqual(groups[0], groups[1])


if __name__ == "__main__":
    unittest.main()
