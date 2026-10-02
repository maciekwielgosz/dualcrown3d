import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pointcloud.two_pass import (CONDITION_DIM, RefinementDecoder, column_structure, corrupt_stage1,
                                 crop_with_extras, match_instances, need_target, per_point_matched_label,
                                 refinement_losses, seed_refinement_losses, stage1_conditioning)
from pointcloud.two_pass_reconcile import reconcile, segmentation_diagnostics

torch.set_num_threads(2)


def blob(rng, center, radius, height, count):
    xy = center + rng.normal(0., radius, (count, 2))
    z = rng.uniform(.5, height, count)
    return np.column_stack((xy, z)).astype(np.float32)


class ConditioningTests(unittest.TestCase):
    def test_rotation_invariance_and_shape(self):
        rng = np.random.default_rng(1)
        xyz = np.concatenate([blob(rng, (0., 0.), 2., 20., 300), blob(rng, (6., 0.), 1., 6., 60)])
        labels = np.r_[np.ones(300, np.int64), np.full(60, 2, np.int64)]
        vote = np.zeros((360, 2), np.float32)
        vote[:300] = -xyz[:300, :2]
        vote[300:] = np.array([6., 0.]) - xyz[300:, :2]
        args = (np.full(360, .8, np.float32), np.r_[np.ones(300), np.full(60, 3)].astype(np.uint8),
                np.full(360, .9, np.float32), np.full(360, .7, np.float32),
                np.ones(360, np.float32), np.zeros(360, np.float32))
        base = stage1_conditioning(xyz, vote, labels, *args)
        angle = .7
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32)
        rotated = xyz.copy()
        rotated[:, :2] = xyz[:, :2] @ rot.T + 100.
        turned = stage1_conditioning(rotated, vote @ rot.T, labels, *args)
        self.assertEqual(base.shape, (360, CONDITION_DIM))
        # Local peaks depend on raster cells, everything else must be invariant.
        keep = [i for i in range(CONDITION_DIM) if i not in (16, 20, 21, 22)]
        np.testing.assert_allclose(base[:, keep], turned[:, keep], atol=2e-2)
        self.assertLess(float(base[300:, 1].max()), float(base[:300, 1].min()))


class CropTests(unittest.TestCase):
    def test_vectors_follow_rotation(self):
        rng = np.random.default_rng(3)
        coord = np.column_stack((rng.uniform(0, 30, 2000), rng.uniform(0, 30, 2000), rng.uniform(0, 10, 2000))).astype(np.float32)
        target = np.array([15., 15.], np.float32)
        arrays = dict(coord=coord, intensity=rng.random(2000).astype(np.float32),
                      tree_id=np.ones(2000, np.int64), instance_offset=np.zeros((2000, 3), np.float32),
                      voxel_size=np.float32(.25))
        extras = dict(labels=np.arange(2000))
        vectors = dict(vote_xy=target - coord[:, :2])
        crop = crop_with_extras(arrays, extras, vectors, rng, 20., 5000, True, True, anchor_index=0)
        votes = crop['coord'][:, :2] + crop['vote_xy']
        self.assertLess(float(votes.std(0).max()), 1e-3)
        self.assertEqual(len(crop['labels']), len(crop['coord']))
        self.assertTrue(len(torch.unique(crop['labels'])) == len(crop['labels']))


class TargetTests(unittest.TestCase):
    def test_match_and_need(self):
        truth = np.r_[np.ones(50), np.full(20, 2), np.zeros(10), np.full(5, -1)].astype(np.int64)
        labels = np.r_[np.ones(50), np.ones(20), np.zeros(15)].astype(np.int64)
        ids, matched, _ = match_instances(truth, labels)
        self.assertEqual(matched.tolist(), [1, 0])
        per_point = per_point_matched_label(truth, labels)
        need = need_target(truth, labels, per_point)
        self.assertEqual(need[:50].sum(), 0.)
        self.assertEqual(need[50:70].sum(), 20.)
        self.assertTrue((need[80:] == -1).all())

    def test_corruption_absorbs(self):
        rng = np.random.default_rng(0)
        xyz = np.concatenate([blob(rng, (0., 0.), 2., 20., 300), blob(rng, (6., 0.), 1., 6., 60)])
        labels = np.r_[np.ones(300), np.full(60, 2)].astype(np.int64)
        for _ in range(20):
            out, _, src = corrupt_stage1(xyz, labels, np.ones(360, np.float32), np.ones(360, np.uint8), rng,
                                         absorb_probability=1., drop_probability=0., max_voxels=100)
            self.assertEqual(len(np.unique(out[out > 0])), 1)
            self.assertTrue((src[300:] == 2).all())


class DecoderTests(unittest.TestCase):
    def test_forward_loss_backward(self):
        torch.manual_seed(0)
        n = 400
        xyz = torch.rand(n, 3) * torch.tensor([20., 20., 15.])
        tree_id = torch.where(xyz[:, 0] < 10, 1, 2)
        tree_id[:20] = -1
        data = dict(coord=xyz, feat=torch.cat((xyz, torch.ones(n, 1)), 1), offset=torch.tensor([n]))
        refiner = RefinementDecoder(queries=8, layers=1, memory_tokens=32, center_shift=True)
        refiner.train()
        refiner.teacher_probability = 1.
        matched = torch.where(tree_id == 1, 7, 0)
        need = torch.where(tree_id == 2, 1., 0.)
        need[tree_id < 0] = -1.
        batch = dict(tree_id=tree_id, need_target=need, gt_matched_label=matched)
        stage1 = torch.where(tree_id == 1, 7, 0)
        out = refiner(torch.rand(n, 72), torch.rand(n, 128), data, torch.rand(n, CONDITION_DIM),
                      torch.zeros(n, 2), torch.zeros(n, 3), teacher_need=need, stage1_labels=stage1)
        self.assertEqual(out['mask_logits'].shape[1], n)
        self.assertEqual(out['decision_logits'].shape, (len(out['seed_index']), 3))
        self.assertTrue((xyz[out['seed_index'], 2] >= .5).all())
        losses = refinement_losses(out, batch)
        self.assertTrue(torch.isfinite(losses['loss']))
        losses['loss'].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in refiner.decision.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in refiner.need.parameters()))
        self.assertGreaterEqual(float(losses['matched_missed']), 1.)
        refiner.zero_grad()
        seeded = seed_refinement_losses(out2 := refiner(torch.rand(n, 72), torch.rand(n, 128), data,
                                        torch.rand(n, CONDITION_DIM), torch.zeros(n, 2), torch.zeros(n, 3),
                                        teacher_need=need, stage1_labels=stage1), batch)
        seeded['loss'].backward()
        self.assertTrue(torch.isfinite(seeded['loss']))
        self.assertGreater(float(refiner.same.weight.grad.abs().sum()) + float(refiner.lid.weight.grad.abs().sum()), 0.)
        expected = torch.where(tree_id[out2['seed_index']] == 2, 1, 2)[tree_id[out2['seed_index']] > 0]
        self.assertTrue(set(expected.tolist()) <= {1, 2})

    def test_legacy_decoder_without_priors(self):
        torch.manual_seed(0)
        n = 200
        xyz = torch.rand(n, 3) * 10
        data = dict(coord=xyz, feat=torch.cat((xyz, torch.ones(n, 1)), 1), offset=torch.tensor([n]))
        refiner = RefinementDecoder(queries=8, layers=1, memory_tokens=32, context_priors=False, condition_dim=20)
        out = refiner.eval()(torch.rand(n, 72), torch.rand(n, 128), data, torch.rand(n, CONDITION_DIM),
                             torch.zeros(n, 2), torch.zeros(n, 3))
        self.assertEqual(out['mask_logits'].shape[1], n)


class WarmStartTests(unittest.TestCase):
    def test_initial_masks_equal_model20_decoder_form(self):
        from pointcloud.dual_head import TreeMaskDecoder
        torch.manual_seed(1)
        n = 300
        xyz = torch.rand(n, 3) * torch.tensor([20., 20., 15.]) + torch.tensor([0., 0., .6])
        data = dict(coord=xyz, feat=torch.cat((xyz, torch.ones(n, 1)), 1), offset=torch.tensor([n]))
        source = TreeMaskDecoder(queries=8, layers=2, memory_tokens=64)
        with torch.no_grad():
            source.radius.bias.fill_(.3)
        refiner = RefinementDecoder(queries=8, layers=2, memory_tokens=64, warm_start=True, center_shift=True).eval()
        refiner.load_model20_decoder(source)
        hidden, offset = torch.rand(n, 128), torch.randn(n, 3)
        out = refiner(torch.rand(n, 72), hidden, data, torch.rand(n, CONDITION_DIM), torch.zeros(n, 2), offset,
                      stage1_labels=torch.zeros(n, dtype=torch.long))
        seed, memory_index = out['seed_index'], out['memory_index']
        # Reference: Model20's decoding with the same seeds and memory tokens.
        position = source.position(xyz / xyz.new_tensor([10., 10., 30.]))
        memory = source.attention_memory(hidden[memory_index]) + position[memory_index]
        features = torch.nn.functional.normalize(source.mask_memory(hidden) + position, dim=1)
        query = hidden[seed] + position[seed]
        centers = xyz[seed, :2] + offset[seed, :2]
        distance2 = (xyz[:, :2][None] - centers[:, None]).square().sum(2)

        def decode(q):
            normalized = source.query_norm(q)
            masks = 8. * (torch.nn.functional.normalize(source.mask_query(normalized), dim=1) @ features.T)
            radius = 1. + 5. * source.radius(normalized).sigmoid()
            return masks + (2.5 - .5 * distance2 / radius.square()).clamp(min=-30.)
        masks = decode(query)
        for layer in source.layers:
            blocked = masks[:, memory_index] < 0
            blocked[blocked.all(1)] = False
            query = layer(query, memory, blocked)
            masks = decode(query)
        torch.testing.assert_close(out['mask_logits'], masks, atol=1e-4, rtol=1e-4)


class StructureTests(unittest.TestCase):
    def test_understory_top_is_sub_canopy_peak(self):
        rng = np.random.default_rng(2)
        canopy = np.column_stack((rng.uniform(0, 6, 800), rng.uniform(0, 6, 800), rng.uniform(18, 22, 800)))
        understory = np.column_stack((3 + rng.normal(0, .4, 80), 3 + rng.normal(0, .4, 80), rng.uniform(2, 7, 80)))
        xyz = np.concatenate((canopy, understory)).astype(np.float32)
        cover, gap, peak = column_structure(xyz)
        top = 800 + int(np.argmax(understory[:, 2]))
        self.assertTrue(peak[top])
        self.assertGreater(cover[top], 10.)
        self.assertGreater(gap[top], 2.)
        self.assertFalse(peak[:800].any())


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        big = blob(rng, (0., 0.), 3., 25., 400)
        small = blob(rng, (4., 0.), .6, 5., 40)
        self.xyz = np.concatenate([big, small])
        self.arrays = dict(coord=self.xyz, voxel_size=np.float32(.25))
        self.labels = np.ones(440, np.uint32)   # small tree absorbed by the big instance
        self.confidence = np.full(440, .8, np.float32)
        self.source = np.ones(440, np.uint8)
        self.top = int(np.argmax(self.xyz[:, 2]))

    def proposals(self, masks, p_new, quality=.9):
        offsets = [0]
        index, score, decision = [], [], []
        for members, p in zip(masks, p_new):
            index.append(np.asarray(members, np.int32))
            score.append(np.ones(len(members), np.float16))
            decision.append([1. - p, p, 0.])
            offsets.append(offsets[-1] + len(members))
        return dict(candidate_offset=np.asarray(offsets), point_index=np.concatenate(index),
                    point_score=np.concatenate(score), decision=np.asarray(decision, np.float32),
                    quality=np.full(len(masks), quality, np.float32))

    def test_absorbed_small_tree_recovered(self):
        proposals = self.proposals([np.arange(400, 440)], [.9])
        labels, _, source, records = reconcile(self.arrays, self.labels, self.confidence, self.source, proposals)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['op'], 'new')
        self.assertTrue((labels[400:] == 2).all())
        self.assertEqual(int((labels == 1).sum()), 400)
        self.assertTrue((source[400:] == 6).all())

    def test_big_tree_fragment_and_top_rejected(self):
        fragment = np.arange(0, 250)              # 62% of the big instance
        with_top = np.r_[np.arange(400, 440), self.top]
        proposals = self.proposals([fragment, with_top], [.95, .9])
        labels, _, _, records = reconcile(self.arrays, self.labels, self.confidence, self.source, proposals)
        self.assertEqual(records, [])
        self.assertTrue((labels == 1).all())

    def test_duplicates_across_windows_and_low_probability(self):
        members = np.arange(400, 440)
        proposals = self.proposals([members, members[:36], np.arange(380, 400)], [.9, .8, .3])
        labels, _, _, records = reconcile(self.arrays, self.labels, self.confidence, self.source, proposals)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(np.unique(labels)), 2)

    def test_merge_only_when_allowed(self):
        split = self.labels.copy()
        split[200:400] = 2
        proposals = self.proposals([np.arange(0, 400)], [.9])
        labels, _, _, records = reconcile(self.arrays, split, self.confidence, self.source, proposals)
        self.assertEqual(records, [])
        labels, _, _, records = reconcile(self.arrays, split, self.confidence, self.source, proposals,
                                          dict(allow_merge=True))
        self.assertEqual(records[0]['op'], 'merge')
        self.assertEqual(len(np.unique(labels[:400])), 1)

    def test_diagnostics(self):
        truth = np.r_[np.ones(400), np.full(40, 2)].astype(np.int64)
        split = np.r_[np.ones(200), np.full(200, 3), np.ones(40)].astype(np.int64)
        result = segmentation_diagnostics(truth, split, large_voxels=300, small_voxels=50)
        self.assertEqual(result['oversplit_gt'], 1)
        self.assertEqual((result['large_gt'], result['small_gt'], result['small_tp']), (1, 1, 0))


if __name__ == '__main__':
    unittest.main()
