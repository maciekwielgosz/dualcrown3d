"""Data loading and crop augmentation for direct point-cloud training."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_manifest(
    path: Path, split: str, source_dataset: str | None = None
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [
            row
            for row in csv.DictReader(stream)
            if row["model_split"] == split
            and (source_dataset is None or row["source_dataset"] == source_dataset)
        ]


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        result = {
            "coord": data["coord"].astype(np.float32),
            "grid_coord": data["grid_coord"].astype(np.int32),
            "intensity": data["intensity"].astype(np.float32),
            "tree_id": data["tree_id"].astype(np.int64),
            "instance_offset": data["instance_offset"].astype(np.float32),
            "source_origin": data["source_origin"].astype(np.float64),
            "voxel_size": np.asarray(data["voxel_size"]).astype(np.float32),
        }
        if "semantic_target" in data:
            result["semantic_target"] = data["semantic_target"].astype(np.int64)
        return result


def unique_voxels(coord: np.ndarray, voxel_size: float) -> np.ndarray:
    grid = np.floor((coord - coord.min(axis=0)) / voxel_size).astype(np.int32)
    extent = grid.max(axis=0).astype(np.int64) + 1
    key = grid[:, 0].astype(np.int64)
    key += extent[0] * grid[:, 1].astype(np.int64)
    key += extent[0] * extent[1] * grid[:, 2].astype(np.int64)
    _, index = np.unique(key, return_index=True)
    return np.sort(index)


def prepare_crop(
    arrays: dict[str, np.ndarray],
    rng: np.random.Generator,
    crop_size_m: float,
    max_points: int,
    augment: bool,
    density_keep_fractions: tuple[float, ...] | None = None,
    preserve_height: bool = False,
    anchor_index: int | None = None,
    return_transform: bool = False,
) -> dict[str, torch.Tensor]:
    coord = arrays["coord"]
    tree_id = arrays["tree_id"]
    positive = np.flatnonzero(tree_id > 0)
    if anchor_index is None:
        anchor_index = int(rng.choice(positive if len(positive) else len(coord)))
    anchor = coord[anchor_index, :2]
    half = crop_size_m / 2.0
    inside = (
        (coord[:, 0] >= anchor[0] - half)
        & (coord[:, 0] < anchor[0] + half)
        & (coord[:, 1] >= anchor[1] - half)
        & (coord[:, 1] < anchor[1] + half)
    )
    selected = np.flatnonzero(inside)
    if len(selected) > max_points:
        selected = np.sort(rng.choice(selected, size=max_points, replace=False))
    coord = coord[selected].copy()
    intensity = arrays["intensity"][selected].copy()
    tree_id = tree_id[selected].copy()
    semantic_target = arrays.get("semantic_target", np.where(arrays["tree_id"] < 0, -1,
                                  (arrays["tree_id"] > 0).astype(np.int64)))[selected].copy()
    instance_offset = arrays["instance_offset"][selected].copy()

    if augment and density_keep_fractions:
        keep_fraction = float(rng.choice(density_keep_fractions))
        if keep_fraction < 1.0 and len(coord) > 256:
            keep_count = max(256, int(round(len(coord) * keep_fraction)))
            keep_count = min(keep_count, len(coord))
            retained_density = np.sort(
                rng.choice(len(coord), size=keep_count, replace=False)
            )
            semantic_target = semantic_target[retained_density]
            coord, intensity, tree_id, instance_offset = (
                coord[retained_density],
                intensity[retained_density],
                tree_id[retained_density],
                instance_offset[retained_density],
            )

    rotation = np.eye(3, dtype=np.float32)
    scale = 1.0
    center = np.mean(coord, axis=0, keepdims=True)
    if augment:
        angle = float(rng.uniform(-np.pi, np.pi))
        scale = float(rng.uniform(0.9, 1.1))
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        coord = (coord - center) @ rotation.T * scale + center
        instance_offset = instance_offset @ rotation.T * scale
        if rng.random() < 0.3:
            drop = rng.random(len(coord)) >= rng.uniform(0.0, 0.15)
            semantic_target = semantic_target[drop]
            coord, intensity, tree_id, instance_offset = (
                coord[drop],
                intensity[drop],
                tree_id[drop],
                instance_offset[drop],
            )
        intensity = np.clip(
            intensity * rng.uniform(0.9, 1.1) + rng.normal(0.0, 0.01, len(intensity)),
            0.0,
            1.0,
        ).astype(np.float32)

    voxel_size = float(arrays["voxel_size"])
    retained = unique_voxels(coord, voxel_size)
    semantic_target = semantic_target[retained]
    coord, intensity, tree_id, instance_offset = (
        coord[retained],
        intensity[retained],
        tree_id[retained],
        instance_offset[retained],
    )
    xy_mean = np.mean(coord[:, :2], axis=0)
    coord[:, :2] -= xy_mean
    if return_transform:
        inverse = rotation[:2, :2].T.astype(np.float64) / scale
        local_to_world = np.eye(3, dtype=np.float64)
        local_to_world[:2, :2] = inverse
        local_to_world[:2, 2] = (arrays['source_origin'][:2] + center[0, :2] +
                                  inverse @ (xy_mean - center[0, :2]))
    if not preserve_height:
        coord[:, 2] -= np.min(coord[:, 2])
    grid_coord = np.floor((coord - coord.min(axis=0)) / voxel_size).astype(np.int32)
    # Floating-point centering can move coordinates lying exactly on a voxel
    # edge by one ULP. Enforce the uniqueness required by spconv on the final
    # integer grid rather than relying on the pre-centering grid alone.
    _, final_index = np.unique(grid_coord, axis=0, return_index=True)
    final_index.sort()
    semantic_target = semantic_target[final_index]
    coord, intensity, tree_id, instance_offset, grid_coord = (
        coord[final_index],
        intensity[final_index],
        tree_id[final_index],
        instance_offset[final_index],
        grid_coord[final_index],
    )
    features = np.column_stack((coord, intensity)).astype(np.float32)
    semantic = (tree_id > 0).astype(np.int64)
    count = len(coord)
    result = {
        "coord": torch.from_numpy(coord),
        "grid_coord": torch.from_numpy(grid_coord),
        "feat": torch.from_numpy(features),
        "offset": torch.tensor([count], dtype=torch.long),
        "semantic": torch.from_numpy(semantic),
        "instance_offset": torch.from_numpy(instance_offset),
        "tree_id": torch.from_numpy(tree_id),
        "semantic_target": torch.from_numpy(semantic_target.astype(np.int64)),
    }
    if return_transform:
        result['local_to_world'] = torch.from_numpy(local_to_world)
    return result


class PointCloudCropDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        split: str,
        crop_size_m: float = 20.0,
        max_points: int = 30_000,
        repeats: int = 1,
        augment: bool = False,
        seed: int = 20260924,
        source_dataset: str | None = None,
        density_keep_fractions: tuple[float, ...] | None = None,
    ):
        self.rows = read_manifest(manifest, split, source_dataset=source_dataset)
        if not self.rows:
            raise ValueError(f"No rows for split={split} in {manifest}")
        self.crop_size_m = crop_size_m
        self.max_points = max_points
        self.repeats = repeats
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.density_keep_fractions = density_keep_fractions

    def __len__(self) -> int:
        return len(self.rows) * self.repeats

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row_index = index % len(self.rows)
        cycle = index // len(self.rows)
        row = self.rows[row_index]
        seed = self.seed + self.epoch * 1_000_003 + cycle * 10_007 + row_index
        rng = np.random.default_rng(seed)
        arrays = load_npz(row["output"])
        return prepare_crop(
            arrays,
            rng,
            crop_size_m=self.crop_size_m,
            max_points=self.max_points,
            augment=self.augment,
            density_keep_fractions=self.density_keep_fractions,
            preserve_height=bool(row.get("height_normalization")),
        )


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
