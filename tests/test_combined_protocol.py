"""Small regression checks; no training or writes to the source datasets."""
import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import box

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.mask_decoder import LitePTMaskDecoder, mask_decoder_losses
from scripts.evaluate_combined_full_crowns import metrics, totals
from scripts.evaluate_pointcloud_litept import model_input
from scripts.evaluate_pointcloud_mask_decoder import filter_instances, polygon_iou


class ProtocolTests(unittest.TestCase):
    def test_perfect_and_duplicate(self):
        truth = [box(0, 0, 2, 2), box(3, 0, 5, 2)]
        self.assertEqual(totals([metrics(truth, truth)])["pq"], 1.)
        duplicate = totals([metrics(truth, truth + [truth[0]])])
        self.assertEqual(duplicate["fp"], 1)
        self.assertAlmostEqual(duplicate["f1"], .8)

    def test_empty(self):
        self.assertEqual(metrics([box(0, 0, 1, 1)], [])["fn"], 1)
        self.assertEqual(metrics([], [box(0, 0, 1, 1)])["fp"], 1)

    def test_annotation_ignore_keeps_matches_and_duplicates(self):
        crown, outside = box(0, 0, 2, 2), box(5, 5, 7, 7)
        result = metrics([crown], [crown, crown, outside], crown.union(outside))
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["ignored_predictions"], 1)

    def test_full_overlapping_crowns_remain_separate(self):
        truth = [box(0, 0, 3, 3), box(1, 0, 4, 3)]
        self.assertEqual(metrics(truth, truth)["tp"], 2)

    def test_height_contract(self):
        xyz = np.array([[0., 0., 5.], [1., 1., 7.]], dtype=np.float32)
        batch = model_input(xyz, np.array([[0,0,0], [4,4,8]]), np.ones(2), torch.device("cpu"), preserve_height=True)
        np.testing.assert_array_equal(batch["coord"][:, 2].numpy(), [5., 7.])

    def test_sparse_grid_collisions_are_rejected(self):
        xyz = np.array([[0., 0., 5.], [0.1, 0.1, 5.1]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Duplicate sparse-grid"):
            model_input(xyz, np.zeros((2,3), dtype=np.int32), np.ones(2), torch.device("cpu"))

    def test_masked_decoder_gradients(self):
        torch.manual_seed(5)
        model = LitePTMaskDecoder(backbone_type="point_mlp", queries=8, hidden_dim=32, decoder_layers=2, memory_tokens=64, spatial_prior=True, masked_attention=True, auxiliary_losses=True, semantic_head=True)
        xyz = torch.rand(256, 3)
        prediction = model({"coord": xyz, "feat": torch.cat([xyz, torch.ones(256, 1)], 1)})
        target = torch.where(xyz[:, 0] < .5, 1, 2)
        with torch.autocast("cpu", enabled=False):
            loss = mask_decoder_losses(prediction, target, False)["loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_frozen_groups(self):
        root = PROJECT.parent / "combined_als_crowns_v1"
        if not (root / "manifest.csv").exists():
            self.skipTest("Prepared dataset not available")
        with (root / "manifest.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        mapping = {r["dataset_id"]: r["model_split"] for r in rows}
        self.assertEqual(len(mapping), len(rows))
        with (root / "spatial_links.csv").open() as stream:
            for link in csv.DictReader(stream):
                self.assertEqual(mapping[link["a"]], mapping[link["b"]])
        for r in rows:
            self.assertTrue(Path(r["source_las"]).is_file())
            self.assertFalse(Path(r["source_las"]).is_symlink())
            self.assertTrue(Path(r["output"]).is_file())

    def test_height_peak_proposals(self):
        model = LitePTMaskDecoder(backbone_type="point_mlp", anchor_mode="height_peaks", queries=8, hidden_dim=32, decoder_layers=2, memory_tokens=64, spatial_prior=True, masked_attention=True)
        xyz = torch.rand(200, 3) * torch.tensor([10., 10., 20.])
        peaks = model._height_peak_indices(xyz)
        self.assertLessEqual(len(peaks), 8)
        self.assertTrue((xyz[peaks, 2] >= 2).all())
        prediction = model({"coord": xyz, "feat": torch.cat([xyz, torch.ones(200, 1)], 1)})
        self.assertEqual(prediction["mask_logits"].shape, (8, 200))
        self.assertTrue(torch.isfinite(prediction["object_logits"]).all())

    def test_indexed_nms_matches_greedy(self):
        rng = np.random.default_rng(8)
        instances = [dict(geometry=box(x, y, x+2, y+2), confidence=float(i), object_score=.9) for i, (x,y) in enumerate(rng.uniform(0,10,(100,2)))]
        expected = []
        for item in sorted(instances, key=lambda p: p["confidence"], reverse=True):
            if all(polygon_iou(item["geometry"], other["geometry"]) < .5 for other in expected):
                expected.append(item)
        actual = filter_instances(instances, .2, .5)
        self.assertEqual([v["confidence"] for v in actual], [v["confidence"] for v in expected])


if __name__ == "__main__":
    unittest.main()
