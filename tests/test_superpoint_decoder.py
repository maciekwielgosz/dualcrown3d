"""Irreversibility, graph invariance and point-assignment regression tests."""
import numpy as np
import torch

from pointcloud.superpoints.decoder import GraphContext, assign_masks


def test_mixed_group_retains_distinct_point_features_and_gradients():
    torch.manual_seed(7)
    graph = GraphContext(feature_dim=4, width=8, layers=2)
    feature = torch.randn(5, 4, requires_grad=True)
    coord = torch.randn(5, 3)
    group = torch.zeros(5, dtype=torch.long)
    result = graph(feature, coord, group, coord.mean(0, keepdim=True), torch.empty((0, 2), dtype=torch.long))
    # Zero-initialized context is an exact identity, including mixed groups.
    torch.testing.assert_close(result, feature)
    result.square().sum().backward()
    assert feature.grad.abs().sum() > 0
    assert graph.point_update[-1].weight.grad.abs().sum() > 0


def test_graph_relabeling_does_not_change_point_output():
    torch.manual_seed(3)
    graph = GraphContext(feature_dim=4, width=8, layers=2)
    torch.nn.init.normal_(graph.point_update[-1].weight, std=.1)
    feature = torch.randn(6, 4)
    coord = torch.randn(6, 3)
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    centers = coord.reshape(3, 2, 3).mean(1)
    edges = torch.tensor([[0, 1], [1, 2]])
    first = graph(feature, coord, groups, centers, edges)
    mapping = torch.tensor([2, 0, 1])
    reordered = centers[torch.argsort(mapping)]
    second = graph(feature, coord, mapping[groups], reordered, mapping[edges])
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)


def test_overlapping_distinct_queries_survive_union_overlap():
    p = np.array([[.95, .95, .7, .1, .1, .1], [.1, .1, .75, .95, .95, .1]], np.float32)
    labels, confidence, queries = assign_masks(p, np.ones(2), minimum_points=2)
    assert len(queries) == 2
    assert labels[0] == labels[1] > 0
    assert labels[2] == labels[3] == labels[4] > 0
    assert labels[0] != labels[2]
    assert labels[5] == 0
    assert confidence[-1] == 0


def test_duplicate_suppression_and_empty_predictions():
    p = np.array([[.9, .9, .1], [.8, .8, .1]], np.float32)
    labels, _, queries = assign_masks(p, np.ones(2), minimum_points=2)
    assert len(queries) == 1
    assert labels.tolist() == [1, 1, 0]
    labels, _, _ = assign_masks(p, np.zeros(2), minimum_points=2)
    assert not labels.any()
