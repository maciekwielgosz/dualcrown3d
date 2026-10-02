"""Warm-start and gradient checks for the optional wide decoder stage."""
import torch

from pointcloud.superpoints.wide_decoder import WideMaskRefiner


def test_wide_refiner_keeps_inherited_masks_exact_at_initialization():
    torch.manual_seed(17)
    refiner = WideMaskRefiner(width=192, layers=2, memory_tokens=8)
    result = dict(query_features=torch.randn(4, 128),
                  point_features=torch.randn(12, 128),
                  mask_logits=torch.randn(4, 12), object_logits=torch.randn(4))
    original_masks = result['mask_logits'].clone()
    original_scores = result['object_logits'].clone()
    refined = refiner(result)
    torch.testing.assert_close(refined['mask_logits'], original_masks, rtol=0, atol=0)
    torch.testing.assert_close(refined['object_logits'], original_scores, rtol=0, atol=0)
    refined['mask_logits'].sum().backward(retain_graph=True)
    assert refiner.mask_gain.grad.abs() > 0
    refined['object_logits'].sum().backward()
    assert refiner.score.weight.grad.abs().sum() > 0
