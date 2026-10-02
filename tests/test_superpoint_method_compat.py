"""Checks for the isolated EZ-SP partition compatibility operations."""
import torch

from pointcloud.superpoints.instance_embedding import (
    InstanceBoundaryEmbedding, edge_affinity,
)
from pointcloud.superpoints.torch_scatter_compat import (
    scatter_max, scatter_min, scatter_sum,
)


def test_scatter_sum_1d_and_2d():
    index = torch.tensor([2, 0, 2, 1])
    assert scatter_sum(torch.tensor([1., 2., 3., 4.]), index, dim_size=4).tolist() == [2., 4., 4., 0.]
    source = torch.tensor([[1., 10.], [2., 20.], [3., 30.], [4., 40.]])
    assert scatter_sum(source, index, dim_size=4).tolist() == [
        [2., 20.], [4., 40.], [4., 40.], [0., 0.]]


def test_scatter_extrema_values_and_arg_indices():
    source = torch.tensor([5., 2., 2., 4., 9.])
    index = torch.tensor([1, 0, 1, 0, 1])
    smallest, argmin = scatter_min(source, index, dim_size=3)
    largest, argmax = scatter_max(source, index, dim_size=3)
    assert smallest[:2].tolist() == [2., 2.]
    assert argmin.tolist() == [1, 2, 5]
    assert largest[:2].tolist() == [4., 9.]
    assert argmax.tolist() == [3, 4, 5]


def test_embedding_affinity_is_symmetric_and_identity_is_one():
    model = InstanceBoundaryEmbedding()
    features = torch.randn(4, 72)
    embedding = model(features)
    edges = torch.tensor([[0, 1], [1, 0], [0, 0]])
    affinity = edge_affinity(embedding, edges)
    assert torch.allclose(affinity[0], affinity[1])
    assert torch.allclose(affinity[2], torch.tensor(1.))
