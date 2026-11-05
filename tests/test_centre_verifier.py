import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pointcloud.centre_verifier import (FEATURE_NAMES, build_candidates, candidate_features, candidate_targets,
                                        verified_detection)


class CentreVerifierTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.true = np.asarray([[10., 10., 15.], [20., 10., 6.]], np.float32)
        xyz, votes, tree = [], [], []
        for number, (centre, count) in enumerate(zip(self.true, (500, 120)), 1):
            xyz.append(centre + rng.normal(0, 1.5, (count, 3)))
            votes.append(centre + rng.normal(0, .35, (count, 3)))
            tree.append(np.full(count, number))
        unknown = np.asarray([40., 40., 10.]) + rng.normal(0, 1., (200, 3))     # an unannotated tree
        xyz.append(unknown)
        votes.append(np.asarray([40., 40., 10.]) + rng.normal(0, .3, (200, 3)))
        tree.append(np.full(200, -1))
        self.xyz = np.concatenate(xyz).astype(np.float32)
        self.xyz[:, 2] = np.clip(self.xyz[:, 2], 1., None)
        self.votes = np.concatenate(votes).astype(np.float32)
        self.arrays = dict(coord=self.xyz, tree_id=np.concatenate(tree), voxel_size=np.float32(.25))
        self.probability = np.full(len(self.xyz), .9, np.float32)
        self.spread = np.full(len(self.xyz), .2, np.float32)
        self.stage1 = np.where(self.arrays['tree_id'] == 1, 1, 0).astype(np.int64)    # stage 1 found only the big tree
        # detections: both trees, a duplicate on the flank of the big tree, the unannotated tree
        self.detection = dict(centres=np.asarray([[10., 10., 15.], [20.1, 10., 6.], [11.1, 10., 15.], [40., 40., 10.]], np.float32),
                              scores=np.asarray([.9, .6, .35, .8], np.float32), extents=np.asarray([1.2, 1., 1.2, 1.], np.float32),
                              query_scores=np.asarray([.9, .02], np.float32))
        self.anchors = np.asarray([[10.1, 10., 15.2], [30., 30., 12.]], np.float32)    # one on a detection, one alone

    def test_candidates_add_only_unsupported_stage1_centres(self):
        candidates = build_candidates(self.detection, self.anchors)
        self.assertEqual(len(candidates['centres']), 5)
        self.assertEqual(candidates['is_anchor'].tolist(), [False] * 4 + [True])
        self.assertAlmostEqual(float(candidates['scores'][-1]), .02, places=5)      # heat sampled at the lonely centre
        self.assertTrue(np.isfinite(candidates['extents']).all())

    def test_features_separate_centre_from_flank_duplicate(self):
        candidates = build_candidates(self.detection, self.anchors)
        features = candidate_features(self.arrays, self.votes, self.probability, self.spread, self.stage1, candidates, self.anchors)
        self.assertEqual(features.shape, (5, len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())
        column = {name: i for i, name in enumerate(FEATURE_NAMES)}
        centre, duplicate, lonely = features[0], features[2], features[4]
        self.assertGreater(centre[column['votes_05']], duplicate[column['votes_05']])
        self.assertGreater(centre[column['exclusive_votes']], duplicate[column['exclusive_votes']])
        self.assertGreaterEqual(duplicate[column['covered_by_stronger']], 1)
        self.assertEqual(centre[column['covered_by_stronger']], 0)
        self.assertEqual(lonely[column['is_stage1_centre']], 1)
        self.assertEqual(lonely[column['votes_extent']], 0)                         # nothing votes for it
        self.assertGreater(centre[column['stage1_capture']], .8)

    def test_targets_one_per_tree_and_unannotated_ignored(self):
        candidates = build_candidates(self.detection, self.anchors)
        target, weight = candidate_targets(candidates, self.true, self.arrays, self.votes)
        self.assertEqual(target.tolist(), [1., 1., 0., 0., 0.])
        self.assertEqual(weight.tolist(), [1., 1., 1., 0., 0.])   # unannotated tree and the vote-less centre carry no label
        scored = verified_detection(candidates, [.9, .8, .1, .7, .05])
        self.assertEqual(len(scored['scores']), 5)


if __name__ == '__main__':
    unittest.main()
