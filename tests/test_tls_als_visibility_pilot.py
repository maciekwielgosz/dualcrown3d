import unittest

import numpy as np

from scripts.prepare_treescan_helios_dualcrown import voxelise
from scripts.simulate_tls_als_pilot import _beam_returns, simulate_visibility


class VisibilityPilotTests(unittest.TestCase):
    def test_voxel_representatives_do_not_depend_on_labels(self):
        xyz = np.asarray([[0., 0., 1.], [.01, 0., 1.],
                          [1., 0., 1.]], np.float64)
        intensity = np.arange(3, dtype=np.float32)
        labels_a = np.asarray([0, 4, 4])
        labels_b = np.asarray([9, 0, 0])
        a = voxelise(xyz, intensity, labels_a, .25, prefer_labelled=False)
        b = voxelise(xyz, intensity, labels_b, .25, prefer_labelled=False)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[0][0], xyz[0])

    def test_high_canopy_precedes_low_canopy_on_a_beam(self):
        terrain = dict(xs=np.asarray([-1., 0., 1.]), ys=np.asarray([-1., 0., 1.]),
                       z=np.zeros((3, 3)))
        xyz = np.asarray([[0., 0., 5.], [0., 0., 10.]])
        source, _, sequence = _beam_returns(xyz, xyz[:, 2], terrain,
            rng=np.random.default_rng(3), spacing=1., layer=.3,
            opacity=100., transmission=.5, ground_probability=0.,
            max_returns=1, angle_degrees=0.)
        self.assertEqual(source.tolist(), [1])
        self.assertEqual(sequence.tolist(), [[1, 1]])

    def test_visibility_geometry_is_independent_of_tree_ids(self):
        terrain = dict(xs=np.asarray([-1., 0., 1.]), ys=np.asarray([-1., 0., 1.]),
                       z=np.zeros((3, 3)))
        xyz = np.asarray([[0., 0., 5.], [0., 0., 10.]])
        first = simulate_visibility(xyz, np.asarray([1, 2]), xyz[:, 2],
                                    terrain, seed=1, spacing=.5, opacity=3.)
        second = simulate_visibility(xyz, np.asarray([8, 7]), xyz[:, 2],
                                     terrain, seed=1, spacing=.5, opacity=3.)
        np.testing.assert_array_equal(first["xyz"], second["xyz"])
        self.assertTrue(np.all(first["return_number"] <= first["number_of_returns"]))


if __name__ == "__main__":
    unittest.main()
