import sys
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pointcloud.dual_head import DualHeadLitePT, TreeMaskDecoder, dense_losses, seed_mask_losses
from pointcloud.instance_output import point_instance_metrics, merge_masks

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


class DualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.xyz = torch.rand(160, 3) * 10
        self.data = dict(coord=self.xyz, feat=torch.cat((self.xyz, torch.ones(160, 1)), 1),
                         offset=torch.tensor([160]), tree_id=torch.where(self.xyz[:, 0] < 5, 1, 2),
                         instance_offset=torch.zeros(160, 3))

    def test_frozen_old_branch_and_new_gradients(self):
        model = DualHeadLitePT(legacy=FakeLegacy())
        model.point_decoder.queries = 8
        original = {k: v.clone() for k, v in model.legacy.state_dict().items()}
        model.train()
        result = model(self.data)
        torch.testing.assert_close(result['semantic_logits'], result['point_semantic_logits'], rtol=0, atol=0)
        loss = dense_losses(result, self.data)['loss']
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.point_decoder.layers.parameters()))
        self.assertTrue(all(p.grad is None for p in model.legacy.parameters()))
        self.assertTrue(all(torch.equal(v, original[k]) for k, v in model.legacy.state_dict().items()))

    def test_tree_filter_and_one_to_many(self):
        decoder = TreeMaskDecoder(queries=8, layers=1, memory_tokens=32)
        data = dict(self.data)
        data['tree_id'] = self.data['tree_id'].clone()
        data['tree_id'][:40] = 0
        out = decoder(torch.rand(160, 72), data, torch.zeros(160, 2), torch.zeros(160, 3))
        self.assertTrue((out['mask_logits'][:, :40] == -100).all())
        self.assertTrue((data['tree_id'][out['seed_index']] > 0).all())
        self.assertLess(len(torch.unique(data['tree_id'][out['seed_index']])), len(out['seed_index']))
        loss1 = seed_mask_losses(out, data['tree_id'])['mask_loss']
        loss2 = seed_mask_losses(out, data['tree_id'] * 100)['mask_loss']
        torch.testing.assert_close(loss1, loss2)

    def test_empty_tree_pool(self):
        decoder = TreeMaskDecoder(queries=8, layers=1)
        data = {**self.data, 'tree_id': torch.zeros(160, dtype=torch.long)}
        out = decoder(torch.rand(160, 72), data, torch.zeros(160, 2), torch.zeros(160, 3))
        self.assertEqual(out['mask_logits'].shape, (0, 160))
        self.assertTrue(torch.isfinite(seed_mask_losses(out, data['tree_id'])['mask_loss']))

    def test_fps_unique_when_embeddings_collapsed(self):
        ids = TreeMaskDecoder.fps(torch.zeros(20, 5), 8)
        self.assertEqual(len(torch.unique(ids)), 8)

    def test_evaluation_never_consumes_reference_ids(self):
        model = DualHeadLitePT(legacy=FakeLegacy()).eval()
        model.point_decoder.queries = 8
        with torch.no_grad():
            a = model(self.data)['instance_masks']['mask_logits']
            b = model({**self.data, 'tree_id': torch.zeros(160, dtype=torch.long)})['instance_masks']['mask_logits']
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_windows_cover_upper_edges_and_dense_overflow(self):
        from scripts.predict_dual_head import predict
        class Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))
            def forward(self, data):
                n = len(data['coord'])
                return dict(semantic_logits=torch.zeros(n, 2), offset_m=torch.zeros(n, 3),
                            point_semantic_logits=torch.zeros(n, 2),
                            instance_masks=dict(mask_logits=torch.empty(0, n), object_logits=torch.empty(0)))
        xyz = np.array([[0., 0., 1.], [0., 0., 2.], [20., 20., 3.], [40., 40., 4.]], np.float32)
        arrays = dict(coord=xyz, intensity=np.ones(4, np.float32), grid_coord=np.floor(xyz/.25).astype(np.int32))
        result = predict(Stub(), arrays, max_points=1)
        np.testing.assert_array_equal(result['shifted_center'], xyz)
        np.testing.assert_array_equal(result['tree_probability'], np.full(4, .5))

    def test_point_matching_id_invariant(self):
        gt = np.array([0, 1, 1, 2, 2])
        result = point_instance_metrics(gt, np.array([0, 4, 4, 9, 9]))
        self.assertEqual((result['tp'], result['fp'], result['fn']), (2, 0, 0))
        self.assertEqual(result['iou_sum'], 2.)
        self.assertEqual(point_instance_metrics(gt, np.zeros(5))['fn'], 2)

    def test_mask_merge_deduplicates_shared_points(self):
        xyz = np.array([[x, y, 5.] for x in range(4) for y in range(4)], np.float32)
        arrays = dict(coord=xyz, source_origin=np.zeros(3), voxel_size=.25)
        raw = dict(object_score=np.array([.9, .8]), candidate_offset=np.array([0, 16, 32]),
                   point_index=np.tile(np.arange(16), 2), point_score=np.ones(32))
        cfg = dict(object_threshold=.1, mask_threshold=.5, minimum_voxels=4, merge_overlap=.5)
        ids, score, instances = merge_masks(arrays, raw, cfg)
        self.assertEqual(len(instances), 1)
        self.assertTrue((ids == 1).all())
        self.assertEqual(len(instances[0]['geometry'].interiors), 0)
        raw['candidate_owner'] = np.array([False, True])
        _, owned_score, _ = merge_masks(arrays, raw, {**cfg, 'owner_only': True})
        _, all_score, _ = merge_masks(arrays, raw, {**cfg, 'owner_only': False})
        np.testing.assert_allclose(owned_score, .8)
        np.testing.assert_allclose(all_score, .9)

    def test_laz_roundtrip_original_xyz_and_shared_instance_ids(self):
        import laspy
        from scripts.predict_dual_head import export_laz
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cloud = laspy.create(point_format=3, file_version='1.2')
            cloud.header.scales = [.01, .01, .01]
            cloud.x = [100.1, 100.1, 100.6, 100.6, 101.1, 101.1, 101.6, 101.6]
            cloud.y = [200.1] * 8
            cloud.z = [50, 55, 50, 56, 50, 57, 50, 58]
            cloud.classification = [2, 5] * 4
            cloud.intensity = np.arange(8, dtype=np.uint16)
            source = root / 'source.laz'
            cloud.write(source)
            xyz = np.column_stack((cloud.x, cloud.y, cloud.z))
            xyz[:, 2] -= 50
            origin = np.array([100.1, 200.1, 0.])
            arrays = dict(voxel_origin=origin, voxel_size=np.float32(.25),
                          grid_coord=np.floor((xyz-origin)/.25).astype(np.int32))
            metadata = dict(processing_bounds=[100., 200., 102., 201.], als_files=[str(source)],
                            ground_grid_size_m=.5, chm_crs='EPSG:2180',
                            tiles=[dict(tile_id='a', bounds=[100., 200., 101., 201.]),
                                   dict(tile_id='b', bounds=[101., 200., 102., 201.])])
            identifiers = np.array([0, 7] * 4, np.uint32)
            provenance = np.array([0, 2, 0, 3, 0, 4, 0, 5], dtype=np.uint8)
            report = export_laz(root, arrays, metadata, identifiers, identifiers, np.ones(8),
                                (identifiers > 0).astype(np.uint8), provenance)
            self.assertEqual(sum(r['points'] for r in report), 8)
            for tile in ('a', 'b'):
                restored = laspy.read(root / f'PointClouds/trees_{tile}.laz')
                self.assertEqual(set(np.unique(restored.tree_id)), {0, 7})
                self.assertTrue((np.asarray(restored.tree_id)[np.asarray(restored.classification) == 2] == 0).all())
                np.testing.assert_array_equal(restored.assignment_source, provenance[:4] if tile == 'a' else provenance[4:])
                np.testing.assert_array_equal(restored.segmentation_status, [0, 1, 0, 1])

    def fusion_fixture(self, masks, quality=None):
        xyz = np.array([[x*.5, y*.5, 5.] for x in range(12) for y in range(4)], np.float32)
        arrays = dict(coord=xyz, source_origin=np.zeros(3), voxel_size=.25)
        raw = dict(object_score=np.asarray(quality or [.95, .8, .7][:len(masks)]),
                   candidate_offset=np.cumsum([0, *map(len, masks)]),
                   point_index=np.concatenate(masks), point_score=np.full(sum(map(len, masks)), .9))
        config = dict(object_threshold=.1, mask_threshold=.5, minimum_voxels=4, merge_overlap=.15,
                      merge_strategy='support_fusion_v2', fusion_min_iou=.2,
                      fusion_dominance=.8, fusion_max_distance_m=1.5)
        return arrays, raw, config

    def test_fusion_recovers_complementary_support_and_rebuilds_polygon(self):
        arrays, raw, cfg = self.fusion_fixture([np.arange(12), np.arange(20)])
        old_ids, _, old_instances = merge_masks(arrays, raw, {**cfg, 'merge_strategy': 'legacy'})
        ids, confidence, instances = merge_masks(arrays, raw, cfg)
        self.assertEqual(np.count_nonzero(old_ids), 12)
        self.assertEqual(np.count_nonzero(ids), 20)
        self.assertEqual(len(instances), 1)
        np.testing.assert_array_equal(ids[:12], old_ids[:12])
        self.assertGreater(instances[0]['geometry'].area, old_instances[0]['geometry'].area)
        self.assertEqual(instances[0]['points'], 20)
        np.testing.assert_allclose(confidence[12:20], .8*.9)

    def test_fusion_does_not_grow_transitively(self):
        arrays, raw, cfg = self.fusion_fixture([np.arange(12), np.arange(20), np.arange(8, 28)])
        ids, _, instances = merge_masks(arrays, raw, cfg)
        self.assertEqual(len(instances), 1)
        self.assertTrue((ids[12:20] == 1).all())
        self.assertTrue((ids[20:28] == 0).all())

    def test_fusion_rejects_bridge_between_neighbouring_trees(self):
        arrays, raw, cfg = self.fusion_fixture([np.arange(12), np.arange(28, 40), np.arange(40)])
        ids, _, instances = merge_masks(arrays, raw, cfg)
        self.assertEqual(len(instances), 2)
        self.assertTrue((ids[:12] == 1).all())
        self.assertTrue((ids[28:40] == 2).all())
        self.assertTrue((ids[12:28] == 0).all())

    def test_fusion_limits_distance_and_point_probability(self):
        arrays, raw, cfg = self.fusion_fixture([np.arange(12), np.r_[np.arange(20), 47]])
        raw['point_score'][12+12:12+16] = .55
        ids, _, _ = merge_masks(arrays, raw, {**cfg, 'fusion_min_probability': .6})
        self.assertTrue((ids[12:16] == 0).all())
        self.assertTrue((ids[16:20] == 1).all())
        self.assertEqual(ids[47], 0)

    def test_fusion_competing_support_uses_confidence_without_relabelling_anchors(self):
        arrays, raw, cfg = self.fusion_fixture([np.arange(12), np.arange(20, 32),
                                               np.r_[np.arange(12), 16], np.r_[np.arange(20, 32), 16]],
                                              [.99, .98, .8, .9])
        ids, confidence, instances = merge_masks(arrays, raw, cfg)
        self.assertEqual(len(instances), 2)
        self.assertEqual(ids[16], 2)
        self.assertAlmostEqual(float(confidence[16]), .9*.9, places=6)
        self.assertTrue((ids[:12] == 1).all())
        self.assertTrue((ids[20:32] == 2).all())

    def test_fusion_can_fill_interior_without_expanding_crown_boundary(self):
        boundary = np.setdiff1d(np.arange(16), [5, 6, 9, 10])
        arrays, raw, cfg = self.fusion_fixture([boundary, np.arange(20)])
        _, _, old = merge_masks(arrays, raw, {**cfg, 'merge_strategy': 'legacy'})
        ids, _, instances = merge_masks(arrays, raw, {**cfg, 'fusion_inside_hull': True})
        self.assertTrue((ids[:16] == 1).all())
        self.assertTrue((ids[16:20] == 0).all())
        self.assertTrue(instances[0]['geometry'].equals(old[0]['geometry']))

    def test_fusion_vote_agreement_rejects_other_tree_center(self):
        arrays, raw, cfg = self.fusion_fixture([np.arange(12), np.arange(20)])
        raw['shifted_center'] = np.zeros_like(arrays['coord'])
        raw['shifted_center'][16:20, 0] = 10.
        ids, _, _ = merge_masks(arrays, raw, {**cfg, 'fusion_max_vote_distance_m': 1.})
        self.assertTrue((ids[12:16] == 1).all())
        self.assertTrue((ids[16:20] == 0).all())
        del raw['shifted_center']
        with self.assertRaisesRegex(ValueError, 'shifted_center'):
            merge_masks(arrays, raw, {**cfg, 'fusion_max_vote_distance_m': 1.})


if __name__ == '__main__':
    unittest.main()
