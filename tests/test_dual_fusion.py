import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pointcloud.dual_fusion import fuse_heads, vote_instances


class ConsensusTests(unittest.TestCase):
    def fixture(self):
        xyz = np.array([[x*.5, y*.5, 5.] for x in range(24) for y in range(4)], np.float32)
        arrays = dict(coord=xyz, source_origin=np.zeros(3), voxel_size=.25)
        votes = xyz.copy()
        votes[:32, :2] = [1., 1.]
        votes[32:64, :2] = [5., 1.]
        votes[64:, :2] = [9., 1.]
        raw = dict(shifted_center=votes, tree_probability=np.ones(len(xyz)))
        labels = np.zeros(len(xyz), np.uint32)
        labels[:16] = 1
        auxiliary = np.repeat(np.arange(1, 4, dtype=np.uint32), 32)
        cfg = dict(minimum_voxels=4, dual_max_distance_m=2., dual_vote_distance_m=1., dual_growth_distance_m=0.)
        return arrays, raw, labels, auxiliary, cfg

    def test_completion_and_new_instances_preserve_anchors_and_do_not_use_gt(self):
        arrays, raw, labels, auxiliary, cfg = self.fixture()
        output, scores, instances, source = fuse_heads(arrays, raw, labels, (labels>0)*.8, cfg, auxiliary)
        np.testing.assert_array_equal(output[:16], labels[:16])
        self.assertTrue((output[:32] == 1).all())
        self.assertTrue((source[16:32] == 2).all())
        self.assertTrue((source[32:] == 3).all())
        self.assertEqual(len(instances), 3)
        self.assertTrue(all(not p['geometry'].interiors for p in instances))
        arrays['tree_id'] = np.full(len(labels), -1)
        repeated = fuse_heads(arrays, raw, labels, (labels>0)*.8, cfg, auxiliary)[0]
        np.testing.assert_array_equal(output, repeated)

    def test_ambiguous_bridge_is_not_completed_or_added(self):
        arrays, raw, labels, auxiliary, cfg = self.fixture()
        labels[16:24] = 2
        labels[:8] = 0
        output, _, _, source = fuse_heads(arrays, raw, labels, (labels>0)*.8, cfg, auxiliary)
        self.assertTrue((output[:8] == 0).all())
        self.assertTrue((output[24:32] == 0).all())

    def test_semantic_distance_and_existing_center_guard(self):
        arrays, raw, labels, auxiliary, cfg = self.fixture()
        raw['tree_probability'][16:20] = .1
        raw['shifted_center'][64:, :2] = [1., 1.]
        output, _, _, source = fuse_heads(arrays, raw, labels, (labels>0)*.8, cfg, auxiliary)
        self.assertTrue((output[16:20] == 0).all())
        self.assertTrue((output[64:] == 0).all())

    def test_growth_has_no_transitive_expansion(self):
        arrays, raw, labels, auxiliary, cfg = self.fixture()
        raw['shifted_center'][:, :2] = [1., 1.]
        cfg.update(dual_add_instances=False, dual_growth_distance_m=.75)
        output, _, _, source = fuse_heads(arrays, raw, labels, (labels>0)*.8, cfg, np.zeros_like(auxiliary))
        self.assertTrue((output[16:20] == 1).all())
        self.assertTrue((output[20:] == 0).all())
        self.assertTrue((source[16:20] == 4).all())

    def test_no_mask_instances_can_still_recover_new_trees(self):
        arrays, raw, labels, auxiliary, cfg = self.fixture()
        output, _, instances, _ = fuse_heads(arrays, raw, labels*0, labels*.0, cfg, auxiliary)
        self.assertEqual(len(instances), 3)
        self.assertTrue((output > 0).all())

    def test_world_aligned_vote_clustering_is_origin_invariant(self):
        arrays, raw, labels, auxiliary, cfg = self.fixture()
        cluster = dict(probability=.3, vote_smoothing=.5, peak_separation=1., peak_threshold_fraction=.02,
                       assignment_radius=2., min_voxels=4)
        first = vote_instances(arrays, raw, cluster)
        offset = np.array([99., 199., 0.], np.float32)
        translated = {**arrays, 'coord': arrays['coord']-offset, 'source_origin': offset}
        second = vote_instances(translated, {**raw, 'shifted_center': raw['shifted_center']-offset}, cluster)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first[first>0])), 3)

    def test_reverse_consensus_preserves_vote_anchors_and_reports_source(self):
        arrays, raw, labels, auxiliary, cfg = self.fixture()
        output, _, instances, source = fuse_heads(arrays, raw, labels, (labels>0)*.8,
                                                 {**cfg, 'dual_anchor': 'vote'}, auxiliary)
        np.testing.assert_array_equal(output, auxiliary)
        self.assertTrue((source == 3).all())
        self.assertEqual(len(instances), 3)

    def test_invalid_parameters_and_mismatched_auxiliary_are_rejected(self):
        arrays, raw, labels, auxiliary, cfg = self.fixture()
        with self.assertRaises(ValueError):
            fuse_heads(arrays, raw, labels, labels*.0, {**cfg, 'dual_max_distance_m': -1}, auxiliary)
        with self.assertRaises(ValueError):
            fuse_heads(arrays, raw, labels, labels*.0, cfg, auxiliary[:-1])

    def test_empty_scene(self):
        arrays = dict(coord=np.empty((0,3)), source_origin=np.zeros(3), voxel_size=.25)
        raw = dict(shifted_center=np.empty((0,3)), tree_probability=np.empty(0))
        result = fuse_heads(arrays, raw, np.empty(0, np.uint32), np.empty(0), {}, np.empty(0, np.uint32))
        self.assertEqual(len(result[0]), 0)
        self.assertEqual(result[2], [])

    def test_growth_rejects_ambiguous_center_and_respects_mask_support(self):
        arrays, raw, labels, auxiliary, cfg = self.fixture()
        labels[:] = 0
        labels[:8], labels[24:32] = 1, 2
        raw['shifted_center'][:8, :2] = [1., 1.]
        raw['shifted_center'][24:32, :2] = [3., 1.]
        raw['shifted_center'][8:24, :2] = [2., 1.]
        cfg.update(dual_add_instances=False, dual_growth_distance_m=3., dual_growth_vote_distance_m=3.)
        output, _, _, _ = fuse_heads(arrays, raw, labels, labels*.0, cfg, np.zeros_like(auxiliary))
        self.assertTrue((output[8:24] == 0).all())
        raw['shifted_center'][8:16, :2] = [1., 1.]
        cfg['dual_growth_support_only'] = True
        output, _, _, _ = fuse_heads(arrays, raw, labels, labels*.0, cfg, np.zeros_like(auxiliary))
        self.assertTrue((output[8:24] == 0).all())

    def test_vote_peaks_at_context_boundary_have_one_global_instance(self):
        arrays, raw, _, _, _ = self.fixture()
        arrays['coord'][:, 0] += 96.
        raw['shifted_center'][:, :2] = [100.125, 1.125]
        cluster = dict(probability=.3, vote_smoothing=.5, peak_separation=1., peak_threshold_fraction=.02,
                       assignment_radius=2., min_voxels=4)
        result = vote_instances(arrays, raw, cluster)
        self.assertEqual(set(result), {1})

    def test_public_merger_supports_consensus_and_historical_signature(self):
        from pointcloud.instance_output import merge_masks, merge_masks_with_sources
        arrays, raw, _, _, _ = self.fixture()
        raw.update(object_score=np.array([.9]), candidate_offset=np.array([0,16]),
                   point_index=np.arange(16), point_score=np.full(16,.9))
        cfg = dict(merge_strategy='dual_consensus_v3', object_threshold=.1, mask_threshold=.5,
                   minimum_voxels=4, merge_overlap=.15, dual_anchor='vote',
                   vote_cluster_config=dict(probability=.3, vote_smoothing=.5, peak_separation=1.,
                       peak_threshold_fraction=.02, assignment_radius=2., min_voxels=4))
        labels, _, instances, source = merge_masks_with_sources(arrays, raw, cfg)
        historical_api = merge_masks(arrays, raw, cfg)
        self.assertEqual(len(historical_api), 3)
        np.testing.assert_array_equal(labels, historical_api[0])
        self.assertEqual(len(instances), 3)
        self.assertTrue((source > 0).all())

    def test_mask_fallback_does_not_require_legacy_foreground_agreement(self):
        arrays, raw, labels, auxiliary, cfg = self.fixture()
        raw['tree_probability'][:32] = .05
        raw['point_probability'] = np.full(len(labels), .4)
        auxiliary[:32] = 0
        cfg.update(dual_anchor='vote', dual_add_instances=True, dual_complement_own_semantic=True,
                   dual_complement_min_probability=.3, dual_new_use_vote_guard=False)
        output, confidence, _, source = fuse_heads(arrays, raw, labels, (labels>0)*.7, cfg, auxiliary)
        self.assertTrue((output[:16] > 0).all())
        self.assertTrue((source[:16] == 5).all())
        np.testing.assert_allclose(confidence[:16], .7)
        raw['point_probability'][:16] = .1
        output = fuse_heads(arrays, raw, labels, (labels>0)*.7, cfg, auxiliary)[0]
        self.assertTrue((output[:16] == 0).all())


if __name__ == '__main__':
    unittest.main()
