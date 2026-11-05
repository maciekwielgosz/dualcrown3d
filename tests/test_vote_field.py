import unittest
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pointcloud.vote_ensemble import SYMMETRIES, restore_offsets, transform_window
from pointcloud.vote_field import (ASSIGN, CHANNEL_NAMES, CHANNELS, GRID, SOURCE_ADDED, SOURCE_RELOCATED,
                                   SOURCE_STAGE1_CENTRE, VoteFieldNet, assign_field, cell_size, centre_limits,
                                   decode_peaks, field_labels, field_loss, field_targets, fine_shape, out_shape,
                                   predict_windows, rasterize_field, select_field, tree_vote_centres)


def cloud(centre, count, rng, spread=.3):
    """Voxels of one tree around ``centre`` whose votes land near the centre."""
    xyz = centre + rng.normal(0, 1.2, (count, 3)).astype(np.float32)
    xyz[:, 2] = np.clip(xyz[:, 2], 1., None)
    votes = centre + rng.normal(0, spread, (count, 3)).astype(np.float32)
    return xyz.astype(np.float32), votes.astype(np.float32)


class VoteFieldTests(unittest.TestCase):
    def test_rasterize_channels_and_ignore(self):
        rng = np.random.default_rng(0)
        xyz, votes = cloud(np.asarray([10., 10., 12.], np.float32), 400, rng)
        xyz[:, 0] -= 2.                                      # every vote points towards +x
        labels = np.ones(400, np.int64)
        known = np.ones(400, bool)
        volume, ignore = rasterize_field(xyz, votes, np.full(400, .9, np.float32), labels, np.full(400, .2, np.float32),
                                         np.zeros(2), known=known)
        self.assertEqual(volume.shape, (CHANNELS, *fine_shape()))
        self.assertEqual(ignore.shape, out_shape())
        self.assertFalse(ignore.any())
        occupied = volume[0] > 0
        self.assertGreater(volume[CHANNEL_NAMES.index('offset_x')][occupied].mean(), .2)
        self.assertEqual(volume[CHANNEL_NAMES.index('stage1_centre')].sum(), 1.)
        density = volume[CHANNEL_NAMES.index('density')]
        self.assertAlmostEqual(float(density.max()), float(density.min()))
        self.assertAlmostEqual(float(volume[CHANNEL_NAMES.index('vote_spread')][occupied].mean()), .2, places=4)
        _, ignore = rasterize_field(xyz, votes, np.full(400, .9, np.float32), labels, np.zeros(400, np.float32),
                                    np.zeros(2), known=~known)
        self.assertTrue(ignore.any())

    def test_targets_round_trip_through_decoder(self):
        centres = np.asarray([[6.3, 7.9, 11.4], [14.05, 3.6, 22.8]], np.float32)
        heat, offset, extent, positive = field_targets(centres, np.asarray([1.5, 3.], np.float32), np.zeros(2))
        self.assertEqual(int(positive.sum()), 2)
        self.assertEqual(float(heat.max()), 1.)
        position, score, reach = decode_peaks(torch.from_numpy(heat), torch.from_numpy(offset), torch.from_numpy(extent), .5)
        world = position * cell_size()
        order = np.argsort(world[:, 0])
        np.testing.assert_allclose(world[order], centres, atol=1e-4)
        np.testing.assert_allclose(reach[order], [1.5, 3.], rtol=1e-5)
        np.testing.assert_allclose(score, 1.)

    def test_adaptive_targets_widen_with_extent(self):
        centres = np.asarray([[5., 5., 10.], [14., 14., 20.]], np.float32)
        extents = np.asarray([1., 4.], np.float32)
        fixed, _, _, _ = field_targets(centres, extents, np.zeros(2))
        adaptive, _, _, positive = field_targets(centres, extents, np.zeros(2), adaptive=True)
        self.assertEqual(int(positive.sum()), 2)
        self.assertEqual(float(adaptive[positive].min()), 1.)
        small, wide = (10, 10, 10), (28, 28, 20)             # centre cells of the two trees
        self.assertLess(adaptive[small[0] + 2, small[1], small[2]], fixed[small[0] + 2, small[1], small[2]])   # 1 cell < 1.2
        self.assertGreater(adaptive[wide[0] + 3, wide[1], wide[2]], fixed[wide[0] + 3, wide[1], wide[2]] + .3)   # 2.4 cells

    def test_tree_vote_centres(self):
        rng = np.random.default_rng(1)
        xyz_a, votes_a = cloud(np.asarray([5., 5., 10.], np.float32), 300, rng, spread=.2)
        xyz_b, votes_b = cloud(np.asarray([12., 5., 4.], np.float32), 5, rng)
        ids, centres, extents, counts = tree_vote_centres(np.concatenate((xyz_a, xyz_b)),
                                                          np.concatenate((np.full(300, 7), np.full(5, 9))),
                                                          np.concatenate((votes_a, votes_b)))
        self.assertEqual(ids.tolist(), [7])                   # the five-voxel tree is below the minimum
        np.testing.assert_allclose(centres[0], [5., 5., 10.], atol=.1)
        self.assertTrue(.2 < extents[0] < .8)
        self.assertEqual(counts.tolist(), [300])

    def test_loss_decreases_and_ignores(self):
        torch.manual_seed(0)
        network = VoteFieldNet(width=8)
        heat, offset, extent, positive = (torch.from_numpy(v)[None] for v in field_targets(
            np.asarray([[10., 10., 12.]], np.float32), np.asarray([2.], np.float32), np.zeros(2)))
        rng = np.random.default_rng(2)
        xyz, votes = cloud(np.asarray([10., 10., 12.], np.float32), 500, rng)
        volume = torch.from_numpy(rasterize_field(xyz, votes, np.full(500, .9, np.float32), np.zeros(500, np.int64),
                                                  np.zeros(500, np.float32), np.zeros(2)))[None]
        ignore = torch.zeros_like(positive)
        optimizer = torch.optim.Adam(network.parameters(), lr=3e-3)
        first = None
        for _ in range(12):
            losses = field_loss(network(volume), heat, ignore, offset, extent, positive)
            first = first or float(losses['loss'].detach())
            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()
        self.assertLess(float(losses['loss']), first)
        everything = torch.ones_like(positive)
        ignored = field_loss(network(volume), heat, everything, offset, extent, positive)
        self.assertLess(float(ignored['focal']), float(field_loss(network(volume), heat, ignore, offset, extent, positive)['focal']) + 1e-6)

    def test_flip_average_is_consistent_for_an_equivariant_network(self):
        x_channel, y_channel = CHANNEL_NAMES.index('offset_x'), CHANNEL_NAMES.index('offset_y')

        class Equivariant(torch.nn.Module):
            def forward(self, volume):
                pooled = torch.nn.functional.avg_pool3d(volume, (2, 2, 1))
                return dict(heat=pooled[:, 0] * 4 - 2, extent=pooled[:, 2],
                            offset=torch.stack((pooled[:, x_channel], pooled[:, y_channel], pooled[:, 4]), 1))
        torch.manual_seed(1)
        volume = torch.rand(2, CHANNELS, 16, 16, 8)
        plain = predict_windows(Equivariant(), volume, flips=False)
        averaged = predict_windows(Equivariant(), volume, flips=True)
        for a, b in zip(plain, averaged):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_selection_modes(self):
        detection = dict(centres=np.asarray([[5., 5., 10.], [20., 20., 8.], [30., 30., 9.]], np.float32),
                         scores=np.asarray([.35, .45, .7], np.float32), extents=np.asarray([1., 2., 3.], np.float32),
                         query_scores=np.asarray([.3, .01], np.float32))
        anchors = np.asarray([[5.5, 5., 11.], [40., 40., 15.]], np.float32)
        centres, extents, kinds, info = select_field(detection, anchors, dict(mode='corrective', threshold=.3, add_threshold=.6))
        self.assertEqual(len(centres), 3)                    # relocated, one addition, one stage-1 centre kept
        self.assertEqual(kinds.tolist(), [SOURCE_RELOCATED, SOURCE_ADDED, SOURCE_STAGE1_CENTRE])
        self.assertTrue(np.isnan(extents[-1]))
        self.assertEqual((info['replacements'], info['additions'], info['stage1_kept']), (1, 1, 1))
        centres, _, kinds, info = select_field(detection, anchors, dict(mode='corrective', threshold=.3, add_threshold=.6,
                                                                        anchor_keep=.05))
        self.assertEqual(info['stage1_kept'], 0)             # the unsupported stage-1 centre is rejected
        centres, _, kinds, _ = select_field(detection, anchors, dict(mode='replace', threshold=.4))
        self.assertEqual(len(centres), 2)
        self.assertNotIn(SOURCE_STAGE1_CENTRE, kinds.tolist())

    def test_extent_aware_suppression(self):
        # a wide crown with a second, weaker peak 2 m away; an understory tree 12 m below; a far tree
        detection = dict(centres=np.asarray([[10., 10., 20.], [12., 10., 20.], [10.5, 10., 8.], [30., 30., 15.]], np.float32),
                         scores=np.asarray([.8, .4, .5, .6], np.float32), extents=np.asarray([3., 2., 1., 2.], np.float32),
                         query_scores=np.zeros(0, np.float32))
        none = np.zeros((0, 3), np.float32)
        centres, _, _, _ = select_field(detection, none, dict(mode='replace', threshold=.3))
        self.assertEqual(len(centres), 4)
        centres, _, _, _ = select_field(detection, none, dict(mode='replace', threshold=.3, suppress=.75))
        self.assertEqual(len(centres), 3)                    # the duplicate inside the wide crown is gone
        self.assertTrue(any(abs(c[2] - 8.) < 1e-6 for c in centres))   # the understory tree survives (3.96 scaled m away)

    def test_assignment_respects_each_centres_reach(self):
        rng = np.random.default_rng(3)
        small, big = np.asarray([10., 10., 3.], np.float32), np.asarray([11.5, 10., 3.], np.float32)
        xyz = rng.uniform([7., 7., 2.5], [15., 13., 3.5], (600, 3)).astype(np.float32)
        arrays = dict(coord=xyz, voxel_size=np.float32(.25))
        centres = np.stack((small, big))
        limits = np.asarray([.5, 4.], np.float32)
        labels, origin = assign_field(arrays, xyz, np.ones(600, np.float32), centres, limits)
        distance_small = np.linalg.norm((xyz - small) * [1, 1, .33], axis=1)
        distance_big = np.linalg.norm((xyz - big) * [1, 1, .33], axis=1)
        beyond = (distance_small > .5) & (distance_small < distance_big) & (distance_big < 3.9)
        self.assertTrue(beyond.any())
        big_label = int(np.flatnonzero(origin == 1)[0])
        self.assertTrue((labels[beyond] == big_label).all())  # not dropped: handed to the centre that reaches them
        limit = centre_limits(centres, np.asarray([1., np.nan], np.float32), {**ASSIGN, 'extent_scale': 1.5})
        np.testing.assert_allclose(limit, [1.5, 2.])          # learned extent, then the height rule for NaN
        _, _, source = field_labels(arrays, xyz, np.ones(600, np.float32), centres, np.asarray([np.nan, np.nan]),
                                    np.asarray([SOURCE_ADDED, SOURCE_STAGE1_CENTRE], np.uint8))
        self.assertTrue(set(np.unique(source)) <= {0, SOURCE_ADDED, SOURCE_STAGE1_CENTRE})

    def test_lattice_symmetries_round_trip(self):
        rng = np.random.default_rng(4)
        grid = rng.integers(0, 50, (200, 3)).astype(np.int32)
        coord = (grid * .25).astype(np.float32)
        offsets = rng.normal(0, 1, (200, 3)).astype(np.float32)
        for symmetry in SYMMETRIES:
            moved, moved_grid = transform_window(coord, grid, symmetry)
            # the transformed grid is still a 0.25 m lattice image of the transformed coordinates
            delta = moved[:, :2] - moved_grid[:, :2] * .25
            np.testing.assert_allclose(delta, np.broadcast_to(delta[0], delta.shape), atol=1e-4)
            forward = offsets.copy()
            forward[:, :2] = offsets[:, :2] @ symmetry.T.astype(np.float32)
            np.testing.assert_allclose(restore_offsets(forward, symmetry), offsets, atol=1e-5)

    def test_annotated_metrics_forgive_predictions_on_unannotated_voxels(self):
        from pointcloud.instance_output import point_instance_metrics
        from scripts.train_vote_field import annotated_point_metrics
        truth = np.r_[np.full(100, 1), np.full(100, 2), np.full(200, -1), np.zeros(20, np.int64)]
        prediction = np.r_[np.full(100, 1), np.full(60, 2), np.full(40, 3), np.full(200, 4), np.full(20, 5)]
        prediction[97:100] = 4                               # the tree in the unannotated zone touches three known voxels
        protocol = point_instance_metrics(truth, prediction)
        annotated = annotated_point_metrics(truth, prediction)
        self.assertEqual((protocol['tp'], protocol['fp'], protocol['fn']), (2, 3, 0))
        self.assertEqual((annotated['tp'], annotated['fp'], annotated['fn']), (2, 2, 0))
        self.assertEqual(annotated['fp_unannotated'], 1)
        self.assertEqual(annotated['fp_inside_one_tree'], 1)  # label 3 is a fragment of tree 2
        self.assertAlmostEqual(annotated['iou_sum'], protocol['iou_sum'])


if __name__ == '__main__':
    unittest.main()
