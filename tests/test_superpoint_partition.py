import unittest

import numpy as np

from pointcloud.superpoints.partition import (
    centroid_graph_connectivity, geometric_partition, neighbor_pairs, partition_diagnostics,
)


class SuperpointPartitionTests(unittest.TestCase):
    def test_partition_does_not_look_at_instance_ids(self):
        xyz = np.asarray([[0., 0., 1.], [.1, 0., 1.],
                          [1., 0., 1.], [1.1, 0., 1.]])
        a = geometric_partition(xyz, .5)
        b = geometric_partition(xyz.copy(), .5)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(np.unique(a)), 2)

    def test_oracle_penalizes_mixed_tree_group(self):
        xyz = np.asarray([[0., 0., 1.], [.1, 0., 1.],
                          [.2, 0., 1.], [.3, 0., 1.]])
        labels = np.asarray([1, 1, 2, 2])
        edges = neighbor_pairs(xyz, .3)
        mixed = partition_diagnostics(xyz, labels, np.zeros(4, np.int32), edges)
        separate = partition_diagnostics(xyz, labels,
                                         np.asarray([0, 0, 1, 1]), edges)
        self.assertEqual(mixed["mixed_group_fraction"], 1.)
        self.assertLess(mixed["oracle"]["pq"], separate["oracle"]["pq"])
        self.assertEqual(separate["oracle"]["pq"], 1.)
        self.assertEqual(separate["boundary_recall"], 1.)

    def test_unknown_cannot_override_known_group_label(self):
        xyz = np.asarray([[0., 0., 1.], [.1, 0., 1.], [.2, 0., 1.]])
        labels = np.asarray([-1, -1, 3])
        result = partition_diagnostics(xyz, labels, np.zeros(3, np.int32),
                                       neighbor_pairs(xyz, .3))
        self.assertEqual(result["oracle"]["pq"], 1.)
        self.assertEqual(result["known_point_purity"], 1.)

    def test_centroid_graph_connects_disjoint_superpoints(self):
        xyz = np.asarray([[0., 0., 1.], [.1, 0., 1.],
                          [.5, 0., 1.], [.6, 0., 1.]])
        ids = np.ones(4, np.int32)
        groups = np.asarray([0, 0, 1, 1])
        near = centroid_graph_connectivity(xyz, ids, groups, radius_m=.2, k=2)
        far = centroid_graph_connectivity(xyz, ids, groups, radius_m=1., k=2)
        self.assertEqual(near["graph_disconnected_trees"], 1)
        self.assertEqual(far["graph_disconnected_trees"], 0)


if __name__ == "__main__":
    unittest.main()
