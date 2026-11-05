import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pointcloud.joint_training import DrawDataset, joint_losses
from pointcloud.stage1_scaled import ScaledDrawDataset, scaled_losses, thin_window


def synthetic_plot(path, seed=0, trees=12, per_tree=900):
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, 30, (trees, 2))
    grid, ids = [], []
    for number, centre in enumerate(centres, 1):
        xyz = np.column_stack((centre + rng.normal(0, 1.2, (per_tree, 2)), rng.uniform(.5, 20, per_tree)))
        grid.append(np.floor(xyz / .25).astype(np.int32))
        ids.append(np.full(per_tree, number))
    ground = np.column_stack((rng.uniform(-5, 35, (4000, 2)), rng.uniform(0, .3, 4000)))
    grid.append(np.floor(ground / .25).astype(np.int32))
    ids.append(np.zeros(4000, int))
    grid, ids = np.concatenate(grid), np.concatenate(ids)
    grid, first = np.unique(grid, axis=0, return_index=True)
    ids = ids[first]
    grid -= grid.min(0)
    coord = ((grid + .5) * .25).astype(np.float32)
    offset = np.zeros_like(coord)
    for tree in np.unique(ids[ids > 0]):
        offset[ids == tree] = coord[ids == tree].mean(0) - coord[ids == tree]
    np.savez(path, coord=coord, grid_coord=grid, intensity=rng.random(len(coord)), tree_id=ids.astype(np.int32),
             instance_offset=offset, source_origin=np.zeros(3), voxel_size=np.float32(.25))
    return path


class Stage1ScaledTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        path = synthetic_plot(Path(self.folder.name) / 'plot.npz')
        self.rows = [dict(output=str(path), model_split='train', dataset_id='plot', source_dataset='real',
                          collection='A', height_normalization='ground')]

    def tearDown(self):
        self.folder.cleanup()

    def test_crops_equal_the_original_dataset_when_thinning_is_off(self):
        for hard in (False, True):
            old = DrawDataset(self.rows, 5, 6000, hard=hard)
            new = ScaledDrawDataset(self.rows, 5, 6000, hard=hard)
            for epoch, draw in ((1, 0), (1, 3), (7, 11)):
                old.epoch = epoch
                a, b = old[(0, draw)], new[(0, draw, epoch)]
                self.assertEqual(set(a), set(b))
                for key in a:
                    self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_thinning_reaches_the_target_density_and_keeps_centroid_targets(self):
        with np.load(self.rows[0]['output']) as archive:
            arrays = {k: archive[k] for k in archive.files}
        anchor = int(np.flatnonzero(arrays['tree_id'] > 0)[10])
        thinned, new_anchor = thin_window(arrays, anchor, 10., np.random.default_rng(0))
        self.assertTrue(np.array_equal(thinned['coord'][new_anchor], arrays['coord'][anchor]))
        cells = len(np.unique(np.floor(thinned['coord'][:, :2]).astype(int), axis=0))
        self.assertLess(len(thinned['coord']) / cells, 14.)
        # every kept voxel still points at the centroid of the complete tree
        full = {int(t): (arrays['coord'] + arrays['instance_offset'])[arrays['tree_id'] == t][0] for t in np.unique(arrays['tree_id']) if t > 0}
        target = thinned['coord'] + thinned['instance_offset']
        for index in np.flatnonzero(thinned['tree_id'] > 0)[::50]:
            np.testing.assert_allclose(target[index], full[int(thinned['tree_id'][index])], atol=1e-3)
        dataset = ScaledDrawDataset(self.rows, 5, 6000, hard=True, thin_probability=1., thin_range=(8., 12.))
        crop = dataset[(0, 2, 1)]
        self.assertLess(len(crop['coord']), 6000)
        self.assertEqual(len(torch.unique(crop['grid_coord'], dim=0)), len(crop['grid_coord']))

    def test_extra_losses_are_opt_in(self):
        torch.manual_seed(0)
        count = 60
        ids = torch.arange(count) % 3 + 1
        ids[:6] = 0
        coord = torch.rand(count, 3) * 10
        batch = dict(coord=coord, tree_id=ids, instance_offset=torch.randn(count, 3))
        masks = dict(embedding=torch.randn(count, 5), seed_index=torch.empty(0, dtype=torch.long))

        def prediction():
            torch.manual_seed(1)
            return dict(semantic_logits=torch.randn(count, 2), offset_m=torch.randn(count, 3, requires_grad=True),
                        point_semantic_logits=torch.randn(count, 2), point_offset_m=torch.randn(count, 3, requires_grad=True),
                        point_quality_logits=torch.randn(count), instance_masks=masks)
        base = joint_losses(prediction(), batch)
        same = scaled_losses(prediction(), batch)
        self.assertEqual(set(base), set(same))
        for key in base:
            self.assertTrue(torch.equal(base[key], same[key]), key)
        value = prediction()
        more = scaled_losses(value, batch, z_weight=.33, assign_weight=.5)
        self.assertGreater(float(more['loss'].detach()), float(base['loss'].detach()))
        more['loss'].backward()
        self.assertGreater(float(value['offset_m'].grad[ids > 0, 2].abs().sum()), 0.)
        self.assertEqual(float(value['offset_m'].grad[ids == 0].abs().sum()), 0.)
        # without the extra terms the vertical vote receives no gradient, as in Model20
        value = prediction()
        scaled_losses(value, batch)['loss'].backward()
        self.assertEqual(float(value['offset_m'].grad[:, 2].abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
