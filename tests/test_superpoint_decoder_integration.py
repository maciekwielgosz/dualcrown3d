"""Inference must not consume labels or fill the space between remote crowns."""
import numpy as np
import torch
from shapely.geometry import Point

from pointcloud.superpoints.decoder import SuperpointMaskDecoder


def test_masks_can_split_one_group_and_inference_ignores_reference_ids():
    torch.manual_seed(9)
    torch.set_num_threads(2)
    decoder = SuperpointMaskDecoder(queries=4, layers=1, memory_tokens=12).eval()
    features = torch.randn(12, 72)
    coord = torch.randn(12, 3)
    coord[:, 2] += 10.
    data = dict(coord=coord, feat=torch.cat((coord, torch.zeros(12, 1)), 1),
                groups=torch.zeros(12, dtype=torch.long), centers=coord.mean(0, keepdim=True),
                group_edges=torch.empty((0, 2), dtype=torch.long), tree_id=torch.arange(12))
    with torch.no_grad():
        first = decoder(features, data, torch.zeros(12, 2), torch.zeros(12, 3))
        data['tree_id'] = torch.full((12,), -1)
        second = decoder(features, data, torch.zeros(12, 2), torch.zeros(12, 3))
    torch.testing.assert_close(first['mask_logits'], second['mask_logits'])
    # All points share one superpoint, yet every query can vary within it.
    assert first['mask_logits'].std(dim=1).min() > 1e-3


def test_crown_footprint_preserves_support_without_a_remote_convex_bridge():
    from scripts.train_superpoint_decoder_pilot import crown_records
    arrays = dict(world_xy=np.array([[0., 0.], [.25, .25], [20., 0.], [20.25, .25]]),
                  coord=np.array([[0., 0., 10.], [.25, .25, 11.], [20., 0., 12.], [20.25, .25, 13.]]))
    result = crown_records(arrays, np.ones(4, np.int32), np.ones(4, np.float32))
    assert len(result) == 1
    geometry = result[0]['geometry']
    assert geometry.geom_type == 'MultiPolygon'
    assert not geometry.covers(Point(10., .25))
    assert all(geometry.covers(Point(*xy)) for xy in arrays['world_xy'])
