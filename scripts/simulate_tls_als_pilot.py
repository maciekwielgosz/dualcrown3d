#!/usr/bin/env python3
"""Small, reproducible TLS-to-ALS visibility pilot on training parent plots.

This is a controlled geometric augmentation, not a waveform or sensor simulator.
The full TLS-derived plot is projected onto virtual airborne beam footprints.
Sampling never reads tree IDs; IDs are attached to retained points afterward.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys

import laspy
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT))
from scripts.prepare_treescan_helios_dualcrown import convert_plot, sha256, write_csv

SOURCE = WORKSPACE / "TreeScanPL10k_HELIOS_ALS_v1"
DEFAULT_OUTPUT = PROJECT / "outputs/tls_visibility_pilot_v1"
PLOTS = (
    "Rem_Gorlice_2015_1302103",
    "Rem_Herby_2016_3301806",
    "Rem_Katrynka_2016_3201705",
    "Rem_Milicz_2015_2805201",
    "Rem_Piensk_2016_1701502",
    "Rem_Suprasl_2015_2002504",
)


def terrain_height(x: np.ndarray, y: np.ndarray, terrain: dict) -> np.ndarray:
    xs, ys, elevation = terrain["xs"], terrain["ys"], terrain["z"]
    xi = np.clip(np.rint((x - xs[0]) / (xs[1] - xs[0])).astype(int), 0, len(xs)-1)
    yi = np.clip(np.rint((y - ys[0]) / (ys[1] - ys[0])).astype(int), 0, len(ys)-1)
    return elevation[yi, xi]


def load_scene(plot: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    folder = SOURCE / "assets" / plot
    metadata = json.loads((folder / "scene_metadata.json").read_text())
    if metadata["split"] != "train":
        raise ValueError(f"Pilot plot must be a training parent: {plot}")
    with np.load(folder / "ground_grid.npz") as archive:
        terrain = {name: archive[name] for name in archive.files}
    origin = np.array([metadata["center_x"], metadata["center_y"],
                       metadata["z_reference"]], np.float64)
    points, labels = [], []
    for part in metadata["tree_parts"]:
        local = np.loadtxt(folder / part["xyz_file"], dtype=np.float32, ndmin=2)
        points.append(local.astype(np.float64) + origin)
        labels.append(np.full(len(local), part["tree_id"], np.int32))
    xyz = np.concatenate(points)
    tree_id = np.concatenate(labels)
    hag = np.maximum(0., xyz[:, 2] - terrain_height(xyz[:, 0], xyz[:, 1], terrain))
    return xyz, tree_id, hag, terrain, metadata


def _beam_returns(
    xyz: np.ndarray, hag: np.ndarray, terrain: dict, *, rng: np.random.Generator,
    spacing: float, layer: float, opacity: float, transmission: float,
    ground_probability: float, max_returns: int, angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source-point indices (-1 for ground), XY and beam-local order.

    Label-independent occupied height layers approximate partial canopy cover.
    The source point count within a layer is clipped to limit TLS sampling bias.
    """
    if spacing <= 0 or layer <= 0 or opacity <= 0 or not 0 < transmission < 1:
        raise ValueError("Invalid beam, layer or opacity settings")
    if not 0 <= ground_probability <= 1 or max_returns < 1:
        raise ValueError("Invalid return settings")
    xmin, xmax = float(terrain["xs"][0]), float(terrain["xs"][-1])
    ymin, ymax = float(terrain["ys"][0]), float(terrain["ys"][-1])
    shift = float(np.max(hag) * abs(math.tan(math.radians(angle_degrees))))
    x0 = math.floor((xmin - spacing) / spacing) * spacing
    y0 = math.floor((ymin - shift - spacing) / spacing) * spacing
    nx = int(math.ceil((xmax - x0 + spacing) / spacing))
    ny = int(math.ceil((ymax + shift - y0 + spacing) / spacing))
    # Scanner cross-track is the y axis for this pilot.  A nonzero angle
    # changes which overhead beam can see each source point.
    projected_y = xyz[:, 1] + hag * math.tan(math.radians(angle_degrees))
    ix = np.floor((xyz[:, 0] - x0) / spacing).astype(np.int32)
    iy = np.floor((projected_y - y0) / spacing).astype(np.int32)
    within = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    source_index = np.flatnonzero(within)
    beam_key = ix[within].astype(np.int64) + nx * iy[within].astype(np.int64)
    height_bin = np.floor(hag[within] / layer).astype(np.int32)
    radix = int(height_bin.max()) + 2
    combined = beam_key * radix + height_bin
    # Representative height within a layer comes from geometry alone.
    sort = np.lexsort((-hag[source_index], combined))
    _, first, occupancy = np.unique(combined[sort], return_index=True,
                                    return_counts=True)
    representatives = source_index[sort[first]]
    keys = beam_key[sort[first]]
    order = np.lexsort((-hag[representatives], keys))
    representatives, keys, occupancy = (value[order] for value in
                                          (representatives, keys, occupancy))
    begin = np.searchsorted(keys, np.arange(nx * ny), side="left")
    end = np.searchsorted(keys, np.arange(nx * ny), side="right")
    kept, beam_xy, sequence = [], [], []
    for beam in range(nx * ny):
        bx = x0 + (beam % nx + .5) * spacing
        by = y0 + (beam // nx + .5) * spacing
        inside_ground = xmin <= bx <= xmax and ymin <= by <= ymax
        if begin[beam] == end[beam] and not inside_ground:
            continue
        hits = []
        energy = 1.
        for j in range(begin[beam], end[beam]):
            # Occupied geometry, rather than the number of raw TLS rays,
            # determines interception. Count saturation is an approximation.
            cover = min(1., float(occupancy[j]) / 8.)
            probability = (1. - math.exp(-opacity * cover)) * energy
            if rng.random() < probability:
                hits.append(int(representatives[j]))
                energy *= transmission
                if len(hits) == max_returns:
                    break
        if inside_ground and len(hits) < max_returns:
            if rng.random() < ground_probability * energy:
                hits.append(-1)
        for position, point in enumerate(hits, 1):
            kept.append(point)
            beam_xy.append((bx, by))
            sequence.append((position, len(hits)))
    return (np.asarray(kept, np.int64), np.asarray(beam_xy, np.float64).reshape(-1, 2),
            np.asarray(sequence, np.uint8).reshape(-1, 2))


def simulate_visibility(xyz: np.ndarray, labels: np.ndarray, hag: np.ndarray,
                        terrain: dict, *, seed: int, spacing: float,
                        opacity: float, transmission: float = .55,
                        ground_probability: float = .85) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    chunks = []
    for strip, angle in enumerate((0., 18.)):
        source, beam_xy, sequence = _beam_returns(
            xyz, hag, terrain, rng=rng, spacing=spacing, layer=.3,
            opacity=opacity, transmission=transmission,
            ground_probability=ground_probability,
            max_returns=4, angle_degrees=angle)
        if not len(source):
            continue
        points = np.empty((len(source), 3), np.float64)
        positive = source >= 0
        points[positive] = xyz[source[positive]]
        points[~positive, :2] = beam_xy[~positive]
        points[~positive, 2] = terrain_height(
            points[~positive, 0], points[~positive, 1], terrain)
        chunks.append(dict(xyz=points, tree_id=np.where(positive, labels[np.maximum(source, 0)], 0),
                           hag=np.where(positive, hag[np.maximum(source, 0)], 0.),
                           return_number=sequence[:, 0], number_of_returns=sequence[:, 1],
                           strip=np.full(len(source), 117 + strip, np.uint16),
                           angle=np.full(len(source), angle, np.int8)))
    if not chunks:
        raise ValueError("No simulated returns")
    return {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}


def simulate_thinning(xyz: np.ndarray, labels: np.ndarray, hag: np.ndarray,
                      terrain: dict, *, seed: int, density: float = 12.) -> dict[str, np.ndarray]:
    """Equal-density but visibility-unaware control, including sampled terrain."""
    rng = np.random.default_rng(seed)
    area = (terrain["xs"][-1] - terrain["xs"][0]) * (
        terrain["ys"][-1] - terrain["ys"][0])
    total = int(round(area * density))
    n_tree = total // 2
    selected = rng.choice(len(xyz), n_tree, replace=False)
    n_ground = total - n_tree
    ground_xy = np.column_stack((rng.uniform(terrain["xs"][0], terrain["xs"][-1], n_ground),
                                 rng.uniform(terrain["ys"][0], terrain["ys"][-1], n_ground)))
    ground_z = terrain_height(ground_xy[:, 0], ground_xy[:, 1], terrain)
    return dict(xyz=np.concatenate((xyz[selected], np.column_stack((ground_xy, ground_z)))),
                tree_id=np.concatenate((labels[selected], np.zeros(n_ground, np.int32))),
                hag=np.concatenate((hag[selected], np.zeros(n_ground))),
                return_number=np.ones(total, np.uint8),
                number_of_returns=np.ones(total, np.uint8),
                strip=np.full(total, 117, np.uint16), angle=np.zeros(total, np.int8))


def write_laz(path: Path, result: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cfg = json.loads((SOURCE / "config/calibration.json").read_text())["target"]
    level = np.interp(rng.random(len(result["xyz"])),
                      cfg["intensity_probabilities"], cfg["intensity_quantiles"])
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [.001, .001, .001]
    for name, dtype in (("treeID", np.int32), ("height_agl", np.float32),
                        ("completelyInside", np.uint8)):
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = result["xyz"].T
    cloud.treeID = result["tree_id"].astype(np.int32)
    cloud.height_agl = result["hag"].astype(np.float32)
    cloud.completelyInside = (result["tree_id"] > 0).astype(np.uint8)
    cloud.classification = np.where(result["tree_id"] > 0, 5, 2).astype(np.uint8)
    cloud.intensity = np.rint(level).astype(np.uint16)
    cloud.return_number = result["return_number"]
    cloud.number_of_returns = result["number_of_returns"]
    cloud.point_source_id = result["strip"]
    cloud.scan_angle_rank = result["angle"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.laz")
    cloud.write(temporary)
    os.replace(temporary, path)
    counts = np.bincount(result["return_number"].astype(int), minlength=5)
    return dict(points=len(cloud.points), density_points_m2=len(cloud.points) / 900.,
                returns_per_pulse=len(cloud.points) / max(int(counts[1]), 1),
                tree_point_fraction=float(np.mean(result["tree_id"] > 0)),
                ground_point_fraction=float(np.mean(result["tree_id"] == 0)),
                trees_with_returns=int(len(np.unique(result["tree_id"][result["tree_id"] > 0]))),
                return_number_counts={str(i): int(n) for i, n in enumerate(counts) if i and n},
                height_quantiles_m=np.quantile(result["hag"], [.1, .5, .9]).tolist(),
                laz_sha256=sha256(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plots", nargs="+", default=PLOTS)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--spacing", type=float, default=.5)
    parser.add_argument("--opacity", type=float, default=.4)
    parser.add_argument("--transmission", type=float, default=.55)
    parser.add_argument("--ground-probability", type=float, default=.85)
    parser.add_argument("--thinning-density", type=float, default=12.)
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / "READY.json").exists():
        raise FileExistsError(f"Completed pilot is protected: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows, reports = [], []
    for plot_index, plot in enumerate(args.plots):
        xyz, labels, hag, terrain, meta = load_scene(plot)
        for mode in ("visibility", "thinning", "helios"):
            prepared = output / "prepared" / mode / plot
            if mode == "helios":
                prepared = SOURCE / "prepared_dataset/train" / plot
                result = None
            else:
                if mode == "visibility":
                    result = simulate_visibility(xyz, labels, hag, terrain,
                        seed=args.seed + plot_index, spacing=args.spacing,
                        opacity=args.opacity, transmission=args.transmission,
                        ground_probability=args.ground_probability)
                else:
                    result = simulate_thinning(xyz, labels, hag, terrain,
                        seed=args.seed + 1000 + plot_index,
                        density=args.thinning_density)
                laz = prepared / f"{plot}_ALS.laz"
                report = write_laz(laz, result, args.seed + plot_index + 2000)
                shutil.copy2(SOURCE / "assets" / plot / "crowns_full.gpkg",
                             prepared / "crowns_full.gpkg")
            target = output / "model" / mode / plot
            if (target / "metadata.json").exists():
                raise FileExistsError(f"Incomplete pilot target protected: {target}")
            row = convert_plot(prepared, target, "train", .25,
                allow_unobserved_training_crowns=True,
                prefer_labelled_voxels=False)
            row.update(dataset_id=f"treescan_helios__{plot}__pilot_{mode}",
                       parent_plot=plot, pilot_mode=mode,
                       annotation_method=("physics_simulation_exact_points_tls_full_crowns"
                                          if mode == "helios" else
                                          "tls_voxel_sampled_points_with_derived_full_crowns"),
                       simulation_method="solid_HELIOS_XYZ" if mode == "helios"
                       else "TLS_random_thinning" if mode == "thinning"
                       else "TLS_probabilistic_overhead_visibility",
                       label_independent_visibility=True,
                       source_plot_split=meta["split"])
            if mode == "helios":
                cloud = laspy.read(prepared / f"{plot}_ALS.laz")
                n = len(cloud.points)
                one = int(np.sum(np.asarray(cloud.return_number) == 1))
                report = dict(points=n, density_points_m2=n / 900.,
                    returns_per_pulse=n / max(one, 1),
                    tree_point_fraction=float(np.mean(np.asarray(cloud.treeID) > 0)),
                    ground_point_fraction=float(np.mean(np.asarray(cloud.treeID) == 0)),
                    trees_with_returns=int(len(np.unique(np.asarray(cloud.treeID)[np.asarray(cloud.treeID)>0]))),
                    height_quantiles_m=np.quantile(np.asarray(cloud.height_agl), [.1, .5, .9]).tolist(),
                    laz_sha256=sha256(prepared / f"{plot}_ALS.laz"))
            reports.append(dict(plot=plot, mode=mode, source_trees=meta["tree_count"], **report))
            rows.append(row)
            write_csv(output / "manifest_progress.csv", rows)
            print(f"{plot} {mode}: {report['points']} points; "
                  f"{report['trees_with_returns']}/{meta['tree_count']} trees", flush=True)
    write_csv(output / "manifest.csv", rows)
    write_csv(output / "qa.csv", reports)
    (output / "READY.json").write_text(json.dumps(dict(plots=list(args.plots),
        split="train", modes=["visibility", "thinning", "helios"],
        seed=args.seed, spacing_m=args.spacing, opacity=args.opacity,
        transmission=args.transmission,
        ground_probability=args.ground_probability,
        thinning_density_points_m2=args.thinning_density,
        manifest_sha256=sha256(output / "manifest.csv"),
        qa_sha256=sha256(output / "qa.csv")), indent=2) + "\n")


if __name__ == "__main__":
    main()
