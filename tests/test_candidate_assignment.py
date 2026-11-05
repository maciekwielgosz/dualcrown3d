import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pointcloud.candidate_assignment import (K, PAIR_EXTRA, SLOT_FEATURES, VOXEL_FEATURES, labels_from_choice,
                                             match_candidates, pair_table, slot_inputs, slot_targets)


class CandidateAssignmentTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.true = np.asarray([[10., 10., 15.], [14., 10., 15.]], np.float32)       # two neighbouring trees
        xyz, votes, tree = [], [], []
        for number, centre in enumerate(self.true, 1):
            xyz.append(centre + rng.normal(0, 1.2, (300, 3)))
            votes.append(centre + rng.normal(0, .8, (300, 3)))
            tree.append(np.full(300, number))
        xyz.append(np.asarray([30., 30., 8.]) + rng.normal(0, 1., (100, 3)))          # unannotated
        votes.append(np.asarray([30., 30., 8.]) + rng.normal(0, .3, (100, 3)))
        tree.append(np.full(100, -1))
        self.arrays = dict(coord=np.concatenate(xyz).astype(np.float32), tree_id=np.concatenate(tree), voxel_size=np.float32(.25))
        self.votes = np.concatenate(votes).astype(np.float32)
        self.probability = np.full(len(self.votes), .9, np.float32)
        self.spread = np.full(len(self.votes), .2, np.float32)
        # candidates: both trees, a weak duplicate between them, the unannotated tree
        self.candidates = dict(centres=np.asarray([[10., 10., 15.], [14., 10., 15.], [12., 10., 15.], [30., 30., 8.]], np.float32),
                               scores=np.asarray([.9, .8, .3, .7], np.float32), extents=np.asarray([1.5, 1.5, 1.5, 1.], np.float32),
                               is_anchor=np.zeros(4, bool))

    def test_matching_one_candidate_per_tree(self):
        owner = match_candidates(self.candidates, np.asarray([1, 2]), self.true)
        self.assertEqual(owner.tolist(), [1, 2, 0, 0])          # the duplicate and the unannotated tree own nothing

    def test_slots_targets_and_oracle_labels(self):
        owner = match_candidates(self.candidates, np.asarray([1, 2]), self.true)
        index, slots, voxel, slot = slot_inputs(self.arrays, self.votes, self.probability, self.spread, self.candidates)
        self.assertEqual(slots.shape, (len(index), K))
        self.assertEqual(voxel.shape[1], len(VOXEL_FEATURES))
        self.assertEqual(slot.shape[2], len(SLOT_FEATURES))
        self.assertTrue((slots[:, 4:] == -1).all())             # only four candidates exist
        target = slot_targets(self.arrays['tree_id'][index], slots, owner)
        self.assertTrue((target[self.arrays['tree_id'][index] < 0] == -1).all())
        known = target >= 0
        self.assertTrue((target[known] < K).all())               # every annotated voxel has its tree among the slots
        chosen = slots[np.arange(len(index))[known], target[known]]
        np.testing.assert_array_equal(owner[chosen], self.arrays['tree_id'][index][known])
        labels, origin = labels_from_choice(self.arrays, index, slots, np.where(known, target, -1))
        self.assertEqual(int(labels.max()), 2)                   # the duplicate got no voxel and vanished
        for tree in (1, 2):
            values = labels[self.arrays['tree_id'] == tree]
            self.assertEqual(len(np.unique(values)), 1)
        nearest, _ = labels_from_choice(self.arrays, index, slots, np.zeros(len(index), np.int64))
        self.assertGreater(int(nearest.max()), 2)                # nearest-centre lets the duplicate steal voxels

    def test_pair_table(self):
        index, slots, voxel, slot = slot_inputs(self.arrays, self.votes, self.probability, self.spread, self.candidates)
        table, valid = pair_table(voxel, slot)
        self.assertEqual(table.shape, (len(index) * K, len(VOXEL_FEATURES) + len(SLOT_FEATURES) + len(PAIR_EXTRA)))
        self.assertTrue(np.isfinite(table).all())
        self.assertEqual(int(valid.sum()), int((slots >= 0).sum()))
        margin = table[:, -1].reshape(len(index), K)
        self.assertTrue((margin[:, 0] >= -1e-6).all())           # slot 0 is the nearest: no other slot is closer


if __name__ == '__main__':
    unittest.main()
