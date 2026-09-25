#!/usr/bin/env python3
"""Spatial before/after diagnostic of unassigned non-ground returns >= 2 m."""
import argparse
from pathlib import Path
import laspy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--tile', default='764000_197000')
    args = parser.parse_args()
    name = f'trees_{args.tile}.laz'
    clouds = [laspy.read(folder/'PointClouds'/name) for folder in (args.before, args.after)]
    old, new = clouds
    for field in ('X','Y','Z','classification','height_agl'):
        if not np.array_equal(np.asarray(old[field]), np.asarray(new[field])):
            raise ValueError(f'Cannot compare different input points: {field}')
    canopy = (np.asarray(new.classification) != 2) & (np.asarray(new.height_agl) >= 2)
    xy = np.column_stack((new.x, new.y))[canopy]
    origin = np.floor(xy.min(0))
    xy -= origin
    extent = np.ceil(xy.max(0))+1
    # Two-metre cells; common denominator, coordinates, and colour scale.
    edges = [np.arange(0, length+2, 2) for length in extent]
    counts, _, _ = np.histogram2d(*xy.T, bins=edges)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), layout='constrained', sharex=True, sharey=True)
    for ax, cloud, title in zip(axes, clouds, ['Before: output_17', 'After: dual-head consensus']):
        unassigned = np.asarray(cloud.tree_id)[canopy] == 0
        missing, _, _ = np.histogram2d(*xy[unassigned].T, bins=edges)
        fraction = np.divide(missing, counts, out=np.full_like(counts, np.nan), where=counts>0)
        chart = ax.pcolormesh(edges[0], edges[1], fraction.T*100, cmap='YlOrRd', vmin=0, vmax=100, rasterized=True)
        ax.set_title(f'{title}\nUnassigned canopy points: {100*unassigned.mean():.1f}%')
        ax.set_aspect('equal')
        ax.set_xlabel(f'Easting relative to {origin[0]:.0f} m')
    axes[0].set_ylabel(f'Northing relative to {origin[1]:.0f} m')
    fig.colorbar(chart, ax=axes, label='Unassigned points in each 2 m cell (%)', shrink=.75)
    fig.suptitle(f'{args.tile}: non-ground returns at least 2 m above terrain\nCoverage diagnostic, not instance accuracy')
    target = args.after/f'coverage_map_{args.tile}.png'
    fig.savefig(target, dpi=160)
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    main()
