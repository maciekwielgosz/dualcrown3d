"""Small native-PyTorch scatter shim for the EZ-SP partition pilot.

The upstream ``torch-graph-components`` partition requires torch-scatter,
which has no binary wheel for this project's PyTorch/CUDA build.  The three
operations used by its partition and weak-components routines are expressed
with PyTorch's native reductions here.  This module is injected only by the
experimental benchmark; it does not modify the production environment.
"""
from __future__ import annotations

import sys
from types import ModuleType

import torch


def _count(index: torch.Tensor, dim_size: int | None) -> int:
    return int(dim_size) if dim_size is not None else (
        int(index.max().item()) + 1 if index.numel() else 0)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: int | None = None) -> torch.Tensor:
    if dim != 0 or index.ndim != 1 or src.shape[0] != index.numel():
        raise ValueError("The pilot shim supports dim=0 and a 1-D index only")
    out = src.new_zeros((_count(index, dim_size), *src.shape[1:]))
    return out.index_add_(0, index.long(), src)


def _scatter_extreme(src: torch.Tensor, index: torch.Tensor, *, dim: int = 0,
                     dim_size: int | None = None, mode: str):
    if dim != 0 or src.ndim != 1 or index.ndim != 1 or len(src) != len(index):
        raise ValueError("The pilot shim supports 1-D dim=0 extrema only")
    count = _count(index, dim_size)
    fill = float("inf") if mode == "amin" else float("-inf")
    if not src.is_floating_point():
        fill = torch.iinfo(src.dtype).max if mode == "amin" else torch.iinfo(src.dtype).min
    out = torch.full((count,), fill, dtype=src.dtype, device=src.device)
    out.scatter_reduce_(0, index.long(), src, reduce=mode, include_self=True)
    arg = torch.full((count,), len(src), dtype=torch.long, device=src.device)
    if len(src):
        candidate = torch.arange(len(src), device=src.device)
        candidate = torch.where(src == out[index], candidate, len(src))
        arg.scatter_reduce_(0, index.long(), candidate, reduce="amin", include_self=True)
    return out, arg


def scatter_min(src, index, dim=0, dim_size=None):
    return _scatter_extreme(src, index, dim=dim, dim_size=dim_size, mode="amin")


def scatter_max(src, index, dim=0, dim_size=None):
    return _scatter_extreme(src, index, dim=dim, dim_size=dim_size, mode="amax")


def install() -> None:
    """Expose only the subset required by the upstream graph partition."""
    if "torch_scatter" in sys.modules:
        raise RuntimeError("Refusing to override an imported torch_scatter module")
    shim = ModuleType("torch_scatter")
    shim.scatter_sum = scatter_sum
    shim.scatter_min = scatter_min
    shim.scatter_max = scatter_max
    sys.modules["torch_scatter"] = shim
