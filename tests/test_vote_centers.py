import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pointcloud.vote_centers import (CHANNELS, GRID, VoteCenterNet, assign_votes, centre_focal_loss, centre_heatmap,
                                     extract_peaks, grid_shape, merge_with_stage1, rasterize_votes,
                                     select_centres, stage1_centres, tree_centroids)

torch.set_num_threads(2)


class VoteCentreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        # A tall tree and an understory tree whose XY centres are only 1 m apart.
        tall = np.column_stack((10 + rng.normal(0, 1.5, 600), 10 + rng.normal(0, 1.5, 600), rng.uniform(12, 26, 600)))
        low = np.column_stack((11 + rng.normal(0, .6, 80), 10 + rng.normal(0, .6, 80), rng.uniform(1, 8, 80)))
        self.xyz = np.concatenate((tall, low)).astype(np.float32)
        self.truth = np.r_[np.ones(600, np.int64), np.full(80, 2)]
        self.centres = np.asarray([tall.mean(0), low.mean(0)], np.float32)
        self.votes = (self.centres[self.truth - 1] + rng.normal(0, .3, (680, 3))).astype(np.float32)
        self.arrays = dict(coord=self.xyz, voxel_size=np.float32(.25))

    def test_raster_and_heatmap(self):
        origin = np.zeros(2)
        volume, ignore = rasterize_votes(self.xyz, self.votes, np.full(680, .9, np.float32), np.ones(680, np.int64),
                                         origin, known=np.ones(680, bool))
        self.assertEqual(volume.shape, (CHANNELS, *grid_shape()))
        self.assertFalse(ignore.any())
        self.assertAlmostEqual(float(np.expm1(volume[0] * 3.).sum()), 680., delta=1.)
        self.assertEqual(float(volume[6].sum()), 1.)          # one stage-1 instance centre
        heat, cells = centre_heatmap(self.centres, origin)
        self.assertEqual(len(cells), 2)
        self.assertTrue(all(heat[tuple(c)] == 1. for c in cells))
        # The two centres share almost the same XY cell but differ in height.
        self.assertLessEqual(abs(cells[0, 0] - cells[1, 0]), 5)
        self.assertGreater(abs(cells[0, 2] - cells[1, 2]), 8)

    def test_unknown_voters_are_ignored(self):
        known = self.truth == 1
        _, ignore = rasterize_votes(self.xyz, self.votes, np.ones(680, np.float32), np.zeros(680, np.int64),
                                    np.zeros(2), known=known)
        _, cells = centre_heatmap(self.centres, np.zeros(2))
        self.assertTrue(ignore[tuple(cells[1])])
        self.assertFalse(ignore[tuple(cells[0])])

    def test_peaks_and_assignment_separate_understory(self):
        heat, cells = centre_heatmap(self.centres, np.zeros(2))
        logits = torch.logit(torch.from_numpy(heat).clamp(1e-4, 1 - 1e-4))
        positions, scores = extract_peaks(logits, threshold=.5)
        self.assertEqual(len(positions), 2)
        world = np.column_stack((positions[:, 0] * GRID['resolution'], positions[:, 1] * GRID['resolution'],
                                 positions[:, 2] * GRID['z_resolution']))
        labels = assign_votes(self.arrays, self.votes, np.ones(680, np.float32), world)
        self.assertEqual(len(np.unique(labels)), 2)
        for identifier in (1, 2):
            members = labels[self.truth == identifier]
            self.assertGreater(np.bincount(members).max() / len(members), .95)
        adaptive = assign_votes(self.arrays, self.votes, np.ones(680, np.float32), world,
                                dict(radius=3., radius_min=2., radius_base=1.5, radius_per_metre=.1))
        self.assertEqual(len(np.unique(adaptive[adaptive > 0])), 2)
        far = self.votes.copy()
        far[600:, 0] += 2.5          # understory votes 2.5 m off: inside 3 m, outside its 2 m adaptive reach
        self.assertGreater((assign_votes(self.arrays, far, np.ones(680, np.float32), world)[600:] > 0).mean(), .5)
        limited = assign_votes(self.arrays, far, np.ones(680, np.float32), world,
                               dict(radius=3., radius_min=2., radius_base=1.5, radius_per_metre=.1))
        low_label = np.bincount(adaptive[600:]).argmax()
        self.assertLess((limited[600:] == low_label).mean(), .2)
        # One 2-D centre cannot separate them.
        merged = assign_votes(self.arrays, self.votes, np.ones(680, np.float32), world[:1])
        self.assertEqual(len(np.unique(merged[merged > 0])), 1)

    def test_corrective_merge_keeps_stage1_where_detector_is_silent(self):
        anchors = stage1_centres(self.votes, np.r_[np.ones(600, np.int64), np.full(80, 0)])
        self.assertEqual(anchors.shape, (1, 3))
        far = np.asarray([[30., 30., 10.]], np.float32)
        merged, kept = merge_with_stage1(far, anchors)
        self.assertTrue(kept.all())
        self.assertEqual(len(merged), 2)
        merged, kept = merge_with_stage1(self.centres, anchors)     # tall tree re-detected, understory added
        self.assertFalse(kept.any())
        self.assertEqual(len(merged), 2)
        merged, kept = merge_with_stage1(self.centres[1:], anchors)  # only the understory detected
        self.assertTrue(kept.all())
        self.assertEqual(len(merged), 2)
        merged, kept = merge_with_stage1(np.zeros((0, 3), np.float32), anchors)
        self.assertTrue(kept.all())
        self.assertEqual(len(merged), 1)

    def test_asymmetric_selection(self):
        anchors = self.centres[:1]                                 # stage 1 knows only the tall tree
        detected = np.concatenate((self.centres, [[30., 30., 5.]])).astype(np.float32)
        scores = np.asarray([.45, .65, .65], np.float32)            # relocation, understory addition, far addition
        merged, info = select_centres(detected, scores, anchors, replace_threshold=.4, add_threshold=.6)
        self.assertEqual((info['replacements'], info['additions'], info['stage1_kept']), (1, 2, 0))
        self.assertEqual(info['kinds'].tolist(), [8, 9, 9])
        labels, origin = assign_votes(self.arrays, self.votes, np.ones(680, np.float32), merged, return_centre_index=True)
        self.assertEqual(sorted(origin[1:].tolist()), [0, 1])       # the far centre has no voters
        merged, info = select_centres(detected, scores, anchors, replace_threshold=.4, add_threshold=.7)
        self.assertEqual((info['replacements'], info['additions'], info['stage1_kept']), (1, 0, 0))
        self.assertEqual(len(merged), 1)
        merged, info = select_centres(detected, scores, anchors, replace_threshold=.5, add_threshold=.7)
        self.assertEqual((info['replacements'], info['additions'], info['stage1_kept']), (0, 0, 1))
        np.testing.assert_allclose(merged, anchors)

    def test_network_loss_and_centroids(self):
        torch.manual_seed(0)
        network = VoteCenterNet(width=8)
        volume = torch.rand(2, CHANNELS, 16, 16, 8)
        heat = torch.zeros(2, 16, 16, 8)
        heat[0, 4, 4, 2] = 1.
        ignore = torch.zeros(2, 16, 16, 8, dtype=torch.bool)
        ignore[1] = True
        logits = network(volume)
        self.assertEqual(logits.shape, heat.shape)
        loss = centre_focal_loss(logits, heat, ignore)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        all_ignored = centre_focal_loss(logits.detach()[1:], heat[1:], ignore[1:])
        self.assertEqual(float(all_ignored), 0.)
        ids, centres, counts = tree_centroids(self.xyz, self.truth)
        np.testing.assert_allclose(centres, self.centres, atol=1e-3)
        self.assertEqual(counts.tolist(), [600, 80])


if __name__ == '__main__':
    unittest.main()
