import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pointcloud.dual_head import DualHeadLitePT, dense_losses
from pointcloud.stem_seeded import (SHIFT_M, StemSeededDecoder, assign_points, corrupt_stems, stem_mask_losses,
                                    stem_targets, stems_for_crop, stems_to_local)

torch.set_num_threads(2)


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 72), nn.BatchNorm1d(72))

    def forward(self, data):
        return SimpleNamespace(feat=self.net(data['feat']))


class FakeLegacy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()
        self.semantic_head = nn.Linear(72, 2)
        self.offset_head = nn.Linear(72, 3)
        self.offset_scale_m = 10.


def two_trees(n=200, seed=3):
    """Two crowns 6 m apart over 2-14 m height, plus a few ground points."""
    g = torch.Generator().manual_seed(seed)
    xyz = torch.rand(n, 3, generator=g) * torch.tensor([4., 4., 12.]) + torch.tensor([0., 0., 2.])
    tree = torch.where(torch.arange(n) < n // 2, 1, 2)
    xyz[tree == 2, 0] += 6.
    ground = torch.rand(20, 3, generator=g) * torch.tensor([10., 4., .2])
    xyz = torch.cat((xyz, ground)); tree = torch.cat((tree, torch.zeros(20, dtype=torch.long)))
    data = dict(coord=xyz, feat=torch.cat((xyz, torch.ones(len(xyz), 1)), 1), offset=torch.tensor([len(xyz)]),
                tree_id=tree, instance_offset=torch.zeros(len(xyz), 3))
    stems = torch.tensor([[2., 2., 14.], [8., 2., 14.]])
    return data, stems


class DecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.decoder = StemSeededDecoder(hidden_dim=32, queries=8, layers=2, memory_tokens=64)
        self.data, self.stems = two_trees()
        self.features = torch.randn(len(self.data['coord']), 72)
        self.logits = torch.randn(len(self.data['coord']), 2)
        self.offset = torch.zeros(len(self.data['coord']), 3)

    def test_one_mask_per_stem_and_centre_near_foot(self):
        data = {**self.data, 'stems': self.stems, 'stem_tree_id': torch.tensor([1, 2])}
        self.decoder.train()
        out = self.decoder(self.features, data, self.logits, self.offset)
        self.assertEqual(tuple(out['mask_logits'].shape), (2, len(self.data['coord'])))
        self.assertEqual(tuple(out['object_logits'].shape), (2,))
        self.assertEqual(out['mask_supervision'], 'stem_seeded')
        self.assertEqual(len(out['aux_outputs']), 1)
        self.assertTrue(((out['stem_center_xy'] - self.stems[:, :2]).norm(dim=1) <= SHIFT_M + 1e-5).all())
        self.assertTrue(torch.equal(out['stem_tree_id'], torch.tensor([1, 2])))

    def test_no_stems_gives_no_masks(self):
        out = self.decoder(self.features, {**self.data, 'stems': torch.zeros(0, 3)}, self.logits, self.offset)
        self.assertEqual(tuple(out['mask_logits'].shape), (0, len(self.data['coord'])))
        self.assertEqual(len(out['object_logits']), 0)

    def test_unknown_height_and_score_are_accepted(self):
        stems = self.stems.clone(); stems[0, 2] = 0.
        data = {**self.data, 'stems': stems, 'stem_score': torch.tensor([.3, 1.])}
        out = self.decoder(self.features, data, self.logits, self.offset)
        self.assertTrue(torch.isfinite(out['mask_logits']).all())

    def test_losses_supervise_real_false_and_ignored_feet(self):
        stems = torch.cat((self.stems, torch.tensor([[5., 2., 10.], [2., 2., 14.]])))  # false foot, duplicate of tree 1
        data = {**self.data, 'stems': stems, 'stem_tree_id': torch.tensor([1, 2, 0, -1])}
        self.decoder.train()
        out = self.decoder(self.features, data, self.logits, self.offset)
        target, real, false, owner = stem_targets(data['tree_id'], data['stem_tree_id'])
        self.assertEqual(real.tolist(), [True, True, False, False])
        self.assertEqual(false.tolist(), [False, False, True, False])
        self.assertEqual(int(target[2].sum()), 0)
        self.assertTrue((owner[data['tree_id'] == 1] == 0).all() and (owner[data['tree_id'] == 2] == 1).all())
        self.assertTrue((owner[data['tree_id'] == 0] == -1).all())
        losses = stem_mask_losses(out, data['tree_id'])
        for key in ('mask_loss', 'assignment_loss', 'embedding_loss'):
            self.assertTrue(torch.isfinite(losses[key]), key)
        losses['mask_loss'].backward()
        grads = {n: p.grad for n, p in self.decoder.named_parameters() if n.startswith('stem_')}
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads.values()))

    def test_objectness_loss_separates_real_and_false_feet(self):
        stems = torch.cat((self.stems, torch.tensor([[5., 2., 10.]])))
        data = {**self.data, 'stems': stems, 'stem_tree_id': torch.tensor([1, 2, 0])}
        self.decoder.objectness = True
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=3e-3)
        features = torch.cat((self.data['coord'], torch.ones(len(self.data['coord']), 69)), 1)
        self.decoder.train()
        for _ in range(40):
            optimizer.zero_grad()
            out = self.decoder(features, data, self.logits, self.offset)
            self.assertTrue(out['objectness'])
            stem_mask_losses(out, data['tree_id'])['mask_loss'].backward(); optimizer.step()
        self.decoder.eval()
        score = self.decoder(features, data, self.logits, self.offset)['object_logits'].sigmoid()
        self.assertGreater(float(score[:2].min()), .5)
        self.assertLess(float(score[2]), .5)

    def test_training_separates_two_trees(self):
        """A few steps on a trivial scene: each stem's mask should prefer its own tree."""
        data = {**self.data, 'stems': self.stems, 'stem_tree_id': torch.tensor([1, 2])}
        features = torch.cat((self.data['coord'], torch.ones(len(self.data['coord']), 69)), 1)
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=3e-3)
        self.decoder.train()
        for _ in range(40):
            optimizer.zero_grad()
            out = self.decoder(features, data, self.logits, self.offset)
            losses = stem_mask_losses(out, data['tree_id'])
            (losses['mask_loss']).backward(); optimizer.step()
        self.decoder.eval()
        out = self.decoder(features, data, self.logits, self.offset)
        labels = assign_points(out['mask_logits'].detach(), out['object_logits'].detach())
        tree = data['tree_id'].numpy()
        agreement = np.mean(labels[tree > 0] == tree[tree > 0])
        self.assertGreater(agreement, .9, agreement)


class HelperTests(unittest.TestCase):
    def test_assign_points_threshold_and_minimum(self):
        logits = torch.tensor([[3., 3., 3., 3., -3.], [-3., -3., -3., 4., 3.]])
        labels = assign_points(logits, minimum_points=2)
        self.assertEqual(labels.tolist(), [1, 1, 1, 2, 2])
        labels = assign_points(logits, minimum_points=3)
        self.assertEqual(labels.tolist(), [1, 1, 1, 0, 0])
        labels = assign_points(logits, object_logits=torch.tensor([5., -5.]), object_threshold=.5, minimum_points=1)
        self.assertEqual(labels.tolist(), [1, 1, 1, 1, 0])
        self.assertEqual(len(assign_points(torch.zeros(0, 5))), 5)

    def test_corrupt_stems_marks_false_feet(self):
        rng = np.random.default_rng(0)
        stems = np.column_stack((rng.uniform(0, 20, (30, 2)), rng.uniform(5, 30, 30)))
        ids = np.arange(1, 31)
        out, out_ids, score = corrupt_stems(stems, ids, rng, xy_canopy=rng.uniform(0, 20, (100, 2)))
        self.assertEqual(len(out), len(out_ids)); self.assertEqual(len(out), len(score))
        self.assertTrue(set(out_ids[out_ids > 0]) <= set(ids))
        self.assertTrue((out_ids == 0).any())
        self.assertLessEqual(len(set(out_ids[out_ids > 0])), 30)
        self.assertTrue(((score > 0) & (score <= 1)).all())

    def test_fragment_feet_are_false_and_offset(self):
        rng = np.random.default_rng(1)
        stems = np.column_stack((np.arange(10) * 20., np.zeros(10), np.full(10, 20.)))
        out, ids, _ = corrupt_stems(stems, np.arange(1, 11), rng, drop=(0., 0.), jitter=(1e-6,), false_rate=0.,
                                    duplicate=0., fragment=1., unknown_height=0.)
        self.assertEqual(int((ids == 0).sum()), 10)
        true_xy = out[ids > 0, :2]
        gaps = np.linalg.norm(out[ids == 0, None, :2] - true_xy[None], axis=2).min(1)
        self.assertTrue(((gaps >= 1.5 - 1e-6) & (gaps <= 4 + 1e-6)).all(), gaps)
        self.assertTrue((out[ids == 0, 2] <= 20. * 1.15 + 1e-6).all())  # true heights are scaled by up to 1.15 first

    def test_stems_to_local_inverts_prepare_crop_transform(self):
        angle, scale = .7, 1.05
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        inverse = R.T / scale
        local_to_world = np.eye(3); local_to_world[:2, :2] = inverse; local_to_world[:2, 2] = [500., 700.]
        local = np.array([[1., 2., 15.], [-3., 4., 20.]])
        world = np.column_stack(((local[:, :2] @ inverse.T) + [500., 700.], local[:, 2]))
        back = stems_to_local(world, local_to_world)
        np.testing.assert_allclose(back, local, atol=1e-9)
        coord = np.array([[-5., -5., 0.], [5., 5., 10.]])
        kept, ids = stems_for_crop(np.vstack((world, [[900., 900., 1.]])), np.array([1, 2, 3]), local_to_world, coord)
        self.assertEqual(ids.tolist(), [1, 2])


class DualHeadPolicyTests(unittest.TestCase):
    def test_policy_dispatch_and_frozen_legacy(self):
        torch.manual_seed(2)
        model = DualHeadLitePT(legacy=FakeLegacy(), hidden_dim=32, decoder_policy='stem_seeded', memory_tokens=64)
        self.assertIsInstance(model.point_decoder, StemSeededDecoder)
        data, stems = two_trees()
        batch = {**data, 'stems': stems, 'stem_tree_id': torch.tensor([1, 2]), 'stem_score': torch.ones(2)}
        original = {k: v.clone() for k, v in model.legacy.state_dict().items()}
        model.train()
        result = model(batch)
        losses = dense_losses(result, batch)
        self.assertIn('assignment_loss', losses)
        losses['loss'].backward()
        self.assertTrue(torch.isfinite(losses['loss']))
        self.assertTrue(all(p.grad is None for p in model.legacy.parameters()))
        self.assertTrue(all(torch.equal(v, original[k]) for k, v in model.legacy.state_dict().items()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for n, p in model.point_decoder.named_parameters() if n.startswith('stem_')))

    def test_model20_weights_load_into_new_policy(self):
        base = DualHeadLitePT(legacy=FakeLegacy(), hidden_dim=32, memory_tokens=64)
        new = DualHeadLitePT(legacy=FakeLegacy(), hidden_dim=32, decoder_policy='stem_seeded', memory_tokens=64)
        incompatible = new.load_state_dict(base.state_dict(), strict=False)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(all(k.startswith('point_decoder.stem_') for k in incompatible.missing_keys))


if __name__ == '__main__':
    unittest.main()
