#!/usr/bin/env python3
"""Figures for the DualCrown3D two-pass technical paper.

Every number is read from the held-out comparison files (output_32 to output_34), the
validation caches, the scene exports (output_38, output_39) and the classical production
polygons, or copied from the report named next to it. Writes PDF (vector) and PNG files to
``figures/``. Run from ``DL_model_version``:

    .venv-gpu/bin/python reports/technical_paper/make_figures.py
"""
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
OUT = Path(__file__).resolve().parent / 'figures'
sys.path.insert(0, str(PROJECT))

# Reference palette (dataviz skill, light mode): colour follows the model in every figure.
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#ffffff'
COLOURS = {'Model20': '#2a78d6', 'Two-pass v7': '#eb6834', 'v7 + fallback': '#1baf7a',
           'Classical CHM': '#a8a69f'}
UNANNOTATED, NO_INSTANCE = '#d9d8d2', '#6f6e69'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.titlesize': 9, 'axes.titleweight': 'bold',
    'axes.titlecolor': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.axisbelow': True, 'grid.color': GRID, 'grid.linewidth': .6,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'legend.frameon': False, 'legend.fontsize': 8, 'pdf.fonttype': 42, 'savefig.dpi': 300,
    'lines.linewidth': 2, 'lines.solid_capstyle': 'round'})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{suffix}', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', name)


def comparison(folder):
    return json.loads((PROJECT / folder / 'comparison.json').read_text())


def plot_pq(metrics):
    return metrics['iou_sum'] / max(metrics['tp'] + .5 * metrics['fp'] + .5 * metrics['fn'], 1e-9)


# ---------------------------------------------------------------- instance maps
def instance_colours(ids, seed=7):
    """Distinct, mid-lightness colours per instance id (stable for a given id)."""
    ids = np.asarray(ids, np.int64)
    golden = .61803398875
    hue = (ids * golden + seed * .13) % 1.
    sat = .55 + .35 * ((ids * 7919) % 5) / 4
    val = .72 + .23 * ((ids * 104729) % 3) / 2
    return matplotlib.colors.hsv_to_rgb(np.stack([hue, sat, val], 1))


def top_down(x, y, z, x0, y0, size, cell):
    """Index of the highest point in every raster cell (-1 where empty)."""
    n = int(np.ceil(size / cell))
    col = np.floor((x - x0) / cell).astype(np.int64)
    row = np.floor((y0 + size - y) / cell).astype(np.int64)
    inside = (col >= 0) & (col < n) & (row >= 0) & (row < n)
    order = np.flatnonzero(inside)[np.argsort(z[inside])]
    image = np.full(n * n, -1, np.int64)
    image[row[order] * n + col[order]] = order  # later (higher) points overwrite lower ones
    return image.reshape(n, n)


def render(index, labels, colour_of, background=SURFACE):
    """RGB image: instances in their colours, -1 unannotated, 0 no instance."""
    rgb = np.ones(index.shape + (3,)) * matplotlib.colors.to_rgb(background)
    hit = index >= 0
    lab = labels[index[hit]]
    pix = np.zeros((hit.sum(), 3))
    pix[lab == -1] = matplotlib.colors.to_rgb(UNANNOTATED)
    pix[lab == 0] = matplotlib.colors.to_rgb(NO_INSTANCE)
    pos = lab > 0
    if pos.any():
        pix[pos] = colour_of(lab[pos])
    rgb[hit] = pix
    return rgb


def matched_colours(reference, predicted):
    """Colour a prediction like the reference tree it matches at IoU >= 0.5."""
    ref_ids = np.unique(reference[reference > 0])
    pred_ids = np.unique(predicted[predicted > 0])
    mapping = {}
    both = (reference > 0) & (predicted > 0)
    pairs, counts = np.unique(np.stack([reference[both], predicted[both]], 1), axis=0, return_counts=True)
    ref_size = dict(zip(*np.unique(reference[reference > 0], return_counts=True)))
    pred_size = dict(zip(*np.unique(predicted[(predicted > 0) & (reference >= 0)], return_counts=True)))
    for (r, p), c in zip(pairs, counts):
        iou = c / (ref_size[r] + pred_size.get(p, 0) - c)
        if iou >= .5:
            mapping[p] = r
    offset = int(max(ref_ids.max(initial=0), 1)) + 1000

    def colour_of(lab):
        mapped = np.array([mapping.get(v, v + offset) for v in lab])
        return instance_colours(mapped)
    return colour_of, len(mapping), len(pred_ids)


def legend_handles(items):
    return [Patch(facecolor=c, edgecolor='none', label=t) for t, c in items]


# ---------------------------------------------------------------- figures 1 and 3: schematics
# A side view of four made-up trees. Only illustrates the mechanism; every measured
# number in the report comes from the result files.
TREE_COLOURS = {'A': '#008300', 'B': '#6250d6', 'C': '#eda100', 'D': '#e87ba4', 'A2': '#e34948'}
SCHEMATIC_TREES = [  # centre x, crown bottom and top, half width (m), points, vote spread (m)
    dict(name='A', x=7.0, bottom=11., top=25., half=5.5, n=75, spread=1.1),
    dict(name='C', x=13.6, bottom=2.5, top=7.5, half=1.6, n=16, spread=.35),
    dict(name='B', x=19.2, bottom=9., top=19., half=3.2, n=44, spread=.6),
    dict(name='D', x=26.4, bottom=7., top=15., half=2.6, n=30, spread=.8)]


def schematic_scene(seed=4):
    rng = np.random.default_rng(seed)
    xs, zs, vx, vz, tree = [], [], [], [], []
    for t in SCHEMATIC_TREES:
        mid, half_h = (t['bottom'] + t['top']) / 2, (t['top'] - t['bottom']) / 2
        got = 0
        while got < t['n']:  # ALS sees mostly the upper surface of a crown
            u = rng.uniform(-1, 1)
            depth = rng.exponential(.3 * half_h)
            z = mid + half_h * np.sqrt(1 - u * u) - depth
            if z < mid - half_h * np.sqrt(1 - u * u):
                continue
            xs.append(t['x'] + u * t['half']), zs.append(z)
            vx.append(t['x'] + rng.normal(0, t['spread'])), vz.append(mid + rng.normal(0, t['spread']))
            tree.append(t['name'])
            got += 1
    return dict(x=np.array(xs), z=np.array(zs), vx=np.array(vx), vz=np.array(vz), tree=np.array(tree))


def schematic_axes(ax, title, subtitle=None):
    ax.set_xlim(-0.5, 30.5), ax.set_ylim(-1.5, 27.5), ax.set_aspect('equal')
    ax.axis('off')
    ax.plot([-0.5, 30.5], [0, 0], color=AXIS, lw=1.2, solid_capstyle='butt')
    for t in SCHEMATIC_TREES:  # faint stems, for orientation only (ALS rarely hits them)
        ax.plot([t['x'], t['x']], [0, t['bottom'] + 1], color=GRID, lw=1.6, zorder=0)
    ax.text(0, 1.13, title, transform=ax.transAxes, fontsize=8.4, weight='bold', color=INK, va='bottom')
    if subtitle:
        ax.text(0, 1.11, subtitle, transform=ax.transAxes, fontsize=7, color=INK2, va='top')


def fig_how_it_works():
    s = schematic_scene()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.55))
    titles = [('1  Points', 'what the scanner sees'), ('2  Votes', "each point points to\nits tree's centre"),
              ('3  Centres', 'where many votes meet'), ('4  Trees', 'each point joins\nits centre')]
    for ax, (t, sub) in zip(axes, titles):
        schematic_axes(ax, t, sub)
    axes[0].scatter(s['x'], s['z'], s=7, color=INK2, linewidths=0)
    ax = axes[1]
    for x, z, vx, vz in zip(s['x'], s['z'], s['vx'], s['vz']):
        ax.plot([x, vx], [z, vz], color=AXIS, lw=.45, zorder=1)
    ax.scatter(s['x'], s['z'], s=5, color='#bdbcb5', linewidths=0, zorder=2)
    ax.scatter(s['vx'], s['vz'], s=5, color=INK, linewidths=0, zorder=3)
    ax = axes[2]
    gx, gz = np.meshgrid(np.linspace(-.5, 30.5, 310), np.linspace(-1.5, 27.5, 290))
    heat = np.zeros_like(gx)
    for vx, vz in zip(s['vx'], s['vz']):
        heat += np.exp(-((gx - vx) ** 2 + (gz - vz) ** 2) / (2 * .8 ** 2))
    ax.scatter(s['x'], s['z'], s=5, color='#dddcd6', linewidths=0, zorder=1)
    ax.contourf(gx, gz, heat, levels=np.linspace(.12, 1, 6) * heat.max(), cmap='Blues', alpha=.9, zorder=2)
    for t in SCHEMATIC_TREES:
        mid = (t['bottom'] + t['top']) / 2
        ax.plot(t['x'], mid, marker='+', color=INK, markersize=7, mew=1.4, zorder=4)
        reach = min(max(2.2 * t['spread'], 1.), 4.)
        ax.add_patch(plt.Circle((t['x'], mid), reach, fill=False, ls=(0, (2, 1.5)), lw=.8, color=INK2, zorder=3))
    ax.text(15, -3.2, 'the step the two-pass\nmodel learns', ha='center', va='top', fontsize=7, color=COLOURS['Two-pass v7'],
            weight='bold')
    axes[3].scatter(s['x'], s['z'], s=7, color=[TREE_COLOURS[t] for t in s['tree']], linewidths=0)
    fig.subplots_adjust(wspace=.08)
    save(fig, 'fig01_how_it_works')


def fig_three_variants():
    s = schematic_scene()
    a_tree = SCHEMATIC_TREES[0]
    left = (s['tree'] == 'A') & (s['x'] < a_tree['x'] + .6)
    model20 = np.where(left, 'A', np.where(np.isin(s['tree'], ['A', 'C']), 'A2', s['tree']))
    v7 = np.where(s['tree'] == 'D', 'none', s['tree'])
    variants = [('Model20', model20, [('big crown\ncut in two', (16.2, 21.5)), ('small tree\nswallowed', (13.6, 10.4))]),
                ('Two-pass v7', v7, [('unsure:\nno tree', (26.4, 18.6))]),
                ('Two-pass v7 + fallback', s['tree'], [('taken from\nModel20', (26.4, 18.6))])]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
    for ax, (title, labels, notes) in zip(axes, variants):
        schematic_axes(ax, title)
        colours = [NO_INSTANCE if l == 'none' else TREE_COLOURS[l] for l in labels]
        ax.scatter(s['x'], s['z'], s=9, color=colours, linewidths=0)
        for text, (x, z) in notes:
            ax.text(x, z, text, ha='center', va='bottom', fontsize=6.8, color=INK2, linespacing=1.1)
    axes[0].annotate('', xy=(13.6, 7.8), xytext=(13.6, 10.2), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=.8))
    fig.legend(handles=[Line2D([], [], marker='o', ls='', color='#6250d6', markersize=4, label='one colour = one tree'),
                        Line2D([], [], marker='o', ls='', color=NO_INSTANCE, markersize=4, label='grey = no tree')],
               loc='lower center', ncol=2, bbox_to_anchor=(.5, -.04), fontsize=7)
    fig.subplots_adjust(wspace=.1)
    save(fig, 'fig03_three_variants')


# ---------------------------------------------------------------- figure 2: votes on a validation plot
def fig_votes():
    from pointcloud.data import load_npz
    name = 'ideas__CEDAR_CYPRESS__plot_Kokonoe_17_annotated'
    arrays = load_npz(ROOT / 'combined_als_crowns_supervision_v4/val' / name / 'points_0p25m.npz')
    cache = np.load(PROJECT / 'outputs/dualcrown3d_two_pass_v1/stage1_ensemble/dense/val' / f'{name}.npz')
    coord, truth = arrays['coord'], arrays['tree_id']
    votes, prob, stage1 = cache['ensemble_center'], cache['ensemble_probability'], cache['labels']
    tall = coord[:, 2] > 2.
    x0, y0 = coord[:, 0].min(), coord[:, 1].min()
    size = max(np.ptp(coord[:, 0]), np.ptp(coord[:, 1]))
    keep = tall
    annotated = keep & (truth > 0)  # unannotated vegetation is left out: one colour per known tree
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.45))
    idx = top_down(coord[annotated, 0], coord[annotated, 1], coord[annotated, 2], x0, y0, size, .25)
    axes[0].imshow(render(idx, truth[annotated], instance_colours), extent=(0, size, 0, size), interpolation='nearest')
    axes[0].set_title('(a) The trees\n(seen from above)', fontsize=8)
    order = np.argsort(coord[annotated, 2])
    vx, vy, lab = votes[annotated, 0][order] - x0, votes[annotated, 1][order] - y0, truth[annotated][order]
    axes[1].scatter(vx, vy, s=.25, c=instance_colours(lab), linewidths=0, rasterized=True)
    axes[1].set_xlim(0, size), axes[1].set_ylim(0, size), axes[1].set_aspect('equal')
    axes[1].set_title('(b) Every point moved\nto where it votes', fontsize=8)
    bins = np.arange(0, size + .5, .5)
    heat, _, _ = np.histogram2d(votes[annotated, 0] - x0, votes[annotated, 1] - y0, bins=[bins, bins])
    axes[2].imshow(np.log1p(heat.T)[::-1], extent=(0, size, 0, size), cmap='Blues', interpolation='bilinear')
    ids = np.unique(truth[truth > 0])
    centres = np.array([np.median(votes[truth == t, :2], 0) for t in ids]) - [x0, y0]
    axes[2].scatter(centres[:, 0], centres[:, 1], s=16, marker='+', c=INK, linewidths=.9)
    axes[2].set_title('(c) How many votes land\nwhere (+ = true centre)', fontsize=8)
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].plot([2, 12], [-1.6, -1.6], color=INK, lw=1.5, clip_on=False)
    axes[0].text(7, -2.3, '10 m', ha='center', va='top', fontsize=7, color=INK2)
    fig.subplots_adjust(wspace=.06)
    save(fig, 'fig02_votes_real_plot')
    return name


# ---------------------------------------------------------------- figure 3: qualitative test plots
def median_gain_plot(per_plot, collection):
    rows = [(m['dataset_id'], plot_pq(v['point_metrics']) - plot_pq(m['point_metrics']))
            for m, v in zip(per_plot['Model20'], per_plot['TwoPassV7']) if m['collection'] == collection]
    gains = np.array([g for _, g in rows])
    return rows[int(np.argmin(np.abs(gains - np.median(gains))))][0]


def read_plot(folder, model, name):
    import laspy
    las = laspy.read(PROJECT / folder / model / 'PointClouds' / f'trees_{name}.las')
    return (np.asarray(las.x), np.asarray(las.y), np.asarray(las.z),
            np.asarray(las.tree_id).astype(np.int64), np.asarray(las.reference_tree_id).astype(np.int64))


def fig_test_plots():
    data = comparison('output_32_two_pass_v7_test_dense')
    chosen = [median_gain_plot(data['per_plot'], c) for c in ('NIBIO', 'CEDAR_CYPRESS', 'FGI_EMIT', 'WILDFOREST3D')]
    pq = {m: {r['dataset_id']: plot_pq(r['point_metrics']) for r in data['per_plot'][m]} for m in data['per_plot']}
    fig, axes = plt.subplots(len(chosen), 3, figsize=(5.9, 1.95 * len(chosen)))
    for row, name in enumerate(chosen):
        x, y, z, m20, ref = read_plot('output_32_two_pass_v7_test_dense', 'Model20', name)
        _, _, _, v7, ref2 = read_plot('output_32_two_pass_v7_test_dense', 'TwoPassV7', name)
        assert np.array_equal(ref, ref2)
        tall = z > 2.
        x0, y0 = x.min(), y.min()
        size = max(np.ptp(x), np.ptp(y))
        idx = top_down(x[tall], y[tall], z[tall], x0, y0, size, .25)
        panels = [(ref[tall], instance_colours, 'Reference')]
        for lab, label, key in ((m20, 'Model20', 'Model20'), (v7, 'Two-pass v7', 'TwoPassV7')):
            colour_of, matched, total = matched_colours(ref[tall], lab[tall])
            panels.append((lab[tall], colour_of, f'{label}: PQ {pq[key][name]:.2f}'))
        for col, (lab, colour_of, title) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(render(idx, lab, colour_of), extent=(0, size, 0, size), interpolation='nearest')
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(title, fontsize=8, weight='normal' if col else 'bold')
        collection, plot = name.split('__')[1], name.split('__')[2].replace('_annotated', '')
        axes[row, 0].set_ylabel(f'{collection}\n{plot}', fontsize=7.5, color=INK2, rotation=90)
        axes[row, 0].plot([1, 11], [-1.2, -1.2], color=INK, lw=1.5, clip_on=False)
        axes[row, 0].text(6, -1.8, '10 m', ha='center', va='top', fontsize=6.5, color=INK2)
    fig.legend(handles=legend_handles([('unannotated vegetation', UNANNOTATED), ('tree voxel without instance', NO_INSTANCE)]),
               loc='lower center', ncol=2, bbox_to_anchor=(.5, .045), fontsize=7)
    fig.subplots_adjust(wspace=.04, hspace=.22)
    save(fig, 'fig06_test_plots')
    return chosen


# ---------------------------------------------------------------- figure 4: held-out PQ
CLASSICAL_TEST = dict(point=.292, crown=.270, small=6, large=.391, oversplit=256, precision=.286)  # report sec. 6-7


def bar(ax, x, h, colour, width, label=None, text=True, fmt='{:.2f}'):
    ax.bar(x, h, width, color=colour, edgecolor='white', linewidth=1.2, label=label, zorder=3)
    if text:
        ax.text(x, h + .012 * ax.get_ylim()[1], fmt.format(h), ha='center', va='bottom', fontsize=6.6, color=INK2)


def fig_heldout():
    suites = [('output_32_two_pass_v7_test_dense', 'Real plots, native density\n(24 plots)'),
              ('output_33_two_pass_v7_test_thin10', 'Real plots thinned\nto 10 voxels/m²'),
              ('output_34_two_pass_v7_test_helios', 'HELIOS ALS simulation\n(7 plots, ~14 pts/m²)')]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for ax, (folder, title) in zip(axes, suites):
        s = comparison(folder)['summaries']
        ax.set_ylim(0, .82)
        models = [('Model20', s['Model20']), ('Two-pass v7', s['TwoPassV7'])]
        if 'output_32' in folder:
            models.insert(0, ('Classical CHM', dict(point_sb_pq=CLASSICAL_TEST['point'], crown_sb_pq=CLASSICAL_TEST['crown'])))
        w = .8 / 3
        for k, (label, v) in enumerate(models):
            off = (k - (len(models) - 1) / 2) * w
            bar(ax, 0 + off, v['point_sb_pq'], COLOURS[label], w * .9, label)
            bar(ax, 1 + off, v['crown_sb_pq'], COLOURS[label], w * .9)
        ax.set_xticks([0, 1], ['Point PQ', 'Crown PQ'])
        ax.set_title(title, fontsize=8)
        ax.grid(axis='x', visible=False)
        ax.tick_params(axis='x', length=0)
    axes[0].set_ylabel('Source-balanced PQ (higher is better)')
    handles = [Patch(facecolor=COLOURS[k], label=k) for k in ('Classical CHM', 'Model20', 'Two-pass v7')]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(.5, 1.1))
    fig.subplots_adjust(wspace=.08)
    save(fig, 'fig04_heldout_pq')


# ---------------------------------------------------------------- figure 5: error profile
def fig_errors():
    s = comparison('output_32_two_pass_v7_test_dense')['summaries']
    rows = {'Classical CHM': CLASSICAL_TEST,
            'Model20': dict(small=s['Model20']['small_10_tp'], large=s['Model20']['large_recall'],
                            oversplit=s['Model20']['oversplit_gt'], precision=s['Model20']['point_precision']),
            'Two-pass v7': dict(small=s['TwoPassV7']['small_10_tp'], large=s['TwoPassV7']['large_recall'],
                                oversplit=s['TwoPassV7']['oversplit_gt'], precision=s['TwoPassV7']['point_precision'])}
    panels = [('small', 'Small crowns found\n(of 241, ≤ 10 m²) ↑', '{:.0f}', 100),
              ('large', 'Large-tree recall ↑', '{:.2f}', 1.0),
              ('oversplit', 'Reference trees split\ninto 2+ pieces ↓', '{:.0f}', 290),
              ('precision', 'Point precision ↑', '{:.2f}', 1.0)]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1))
    for ax, (key, title, fmt, top) in zip(axes, panels):
        ax.set_ylim(0, top)
        for k, (label, v) in enumerate(rows.items()):
            bar(ax, k, v[key], COLOURS[label], .62, label, fmt=fmt)
        ax.set_xticks([]), ax.set_title(title, fontsize=7.8)
        ax.grid(axis='x', visible=False)
    handles = [Patch(facecolor=COLOURS[k], label=k) for k in rows]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(.5, 1.13))
    fig.subplots_adjust(wspace=.4)
    save(fig, 'fig05_error_profile')


# ---------------------------------------------------------------- figure 6: per collection
def fig_collections():
    suites = [('output_32_two_pass_v7_test_dense', 'Native density'), ('output_33_two_pass_v7_test_thin10', 'Thinned to 10 voxels/m²')]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    order = None
    for ax, (folder, title) in zip(axes, suites):
        per = comparison(folder)['per_plot']
        stats = {}
        for model in ('Model20', 'TwoPassV7'):
            groups = {}
            for r in per[model]:
                groups.setdefault(r['collection'], []).append(r['point_metrics'])
            for c, ms in groups.items():
                tot = {k: sum(m[k] for m in ms) for k in ('tp', 'fp', 'fn', 'iou_sum')}
                stats.setdefault(c, {})[model] = plot_pq(tot)
                stats[c]['n'] = len(ms)
        if order is None:
            order = sorted(stats, key=lambda c: stats[c]['Model20'])
        for i, c in enumerate(order):
            a, b = stats[c]['Model20'], stats[c]['TwoPassV7']
            ax.plot([a, b], [i, i], color=GRID, lw=2.5, zorder=1, solid_capstyle='round')
            ax.scatter([a], [i], s=36, color=COLOURS['Model20'], edgecolor='white', linewidth=1.2, zorder=3)
            ax.scatter([b], [i], s=36, color=COLOURS['Two-pass v7'], edgecolor='white', linewidth=1.2, zorder=3)
            ax.text(max(a, b) + .02, i, f'{b - a:+.2f}', va='center', fontsize=6.8, color=INK2)
        ax.set_yticks(range(len(order)), [f"{c} ({stats[c]['n']})" for c in order])
        ax.set_xlim(0, 1), ax.set_title(title, fontsize=8.5)
        ax.set_xticks([0, .2, .4, .6, .8, 1], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
        ax.set_xlabel('Point PQ of the collection (pooled over its plots)')
        ax.grid(axis='y', visible=False)
    handles = [Line2D([], [], marker='o', ls='', markersize=6, color=COLOURS[k], label=k) for k in ('Model20', 'Two-pass v7')]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(.55, 1.08))
    fig.subplots_adjust(wspace=.14)
    save(fig, 'fig07_per_collection')


# ---------------------------------------------------------------- figure 7: per plot
def fig_per_plot():
    suites = [('output_32_two_pass_v7_test_dense', 'Native density'), ('output_33_two_pass_v7_test_thin10', 'Thinned to 10 voxels/m²'),
              ('output_34_two_pass_v7_test_helios', 'HELIOS simulation')]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45), sharey=True)
    counts = {}
    for ax, (folder, title) in zip(axes, suites):
        per = comparison(folder)['per_plot']
        a = np.array([plot_pq(r['point_metrics']) for r in per['Model20']])
        b = np.array([plot_pq(r['point_metrics']) for r in per['TwoPassV7']])
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1, zorder=1)
        ax.scatter(a, b, s=26, color=COLOURS['Two-pass v7'], edgecolor='white', linewidth=1, zorder=3)
        better = int((b > a).sum())
        counts[folder] = (better, len(a))
        ax.text(.04, .95, f'v7 better on {better} of {len(a)} plots', fontsize=7, color=INK2, va='top')
        ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_aspect('equal')
        ax.set_title(title, fontsize=8.5), ax.set_xlabel('Model20 point PQ')
    axes[0].set_ylabel('Two-pass v7 point PQ')
    for ax in axes:
        ax.set_xticks([0, .25, .5, .75, 1], ['0', '0.25', '0.5', '0.75', '1'])
    fig.subplots_adjust(wspace=.22)
    save(fig, 'fig08_per_plot')
    return counts


# ---------------------------------------------------------------- figure 8: density (validation)
VALIDATION = {  # reports/two_pass_v7_vote_field.md, section 5 (14 validation plots)
    'Model20': {'point': (.394, .410, .405), 'crown': (.314, .388, .444)},
    'Two-pass v7': {'point': (.470, .495, .508), 'crown': (.390, .441, .513)}}


def fig_density():
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.2), sharey=True)
    xs = [0, 1, 2]
    for ax, key, title in zip(axes, ('point', 'crown'), ('Point PQ', 'Crown PQ')):
        for model, v in VALIDATION.items():
            ax.plot(xs, v[key], color=COLOURS[model], marker='o', markersize=5, markeredgecolor='white', label=model)
            ax.text(2.08, v[key][-1], f'{v[key][-1]:.2f}', va='center', fontsize=6.8, color=INK2)
            ax.text(-.08, v[key][0], f'{v[key][0]:.2f}', va='center', ha='right', fontsize=6.8, color=INK2)
        ax.set_xticks(xs, ['10', '25', 'native'])
        ax.set_xlim(-.4, 2.4), ax.set_ylim(.25, .6)
        ax.set_xlabel('Density, voxels per m²'), ax.set_title(title, fontsize=8.5)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('Validation, source-balanced PQ')
    axes[0].legend(loc='lower right')
    fig.subplots_adjust(wspace=.12)
    save(fig, 'fig09_density_validation')


# ---------------------------------------------------------------- figure 9: real ALS (Weiser)
WEISER = {  # trunk_reconstruction/reports/SEGMENTATION.md, step 1 (12 scans, 14 pts/m², 1,476 trees)
    'Model20': dict(kept=58, split=39, swallowed=24, unlabelled=2, matched=70),
    'Two-pass v7': dict(kept=58, split=8, swallowed=25, unlabelled=42, matched=41),
    'v7 + fallback': dict(kept=71, split=25, swallowed=25, unlabelled=5, matched=67)}


def fig_weiser():
    panels = [('kept', "Tree's points in its\nbest instance ↑"), ('matched', 'Trees with DBH\nmatched ↑'),
              ('split', 'Trees split\ninto 2+ ↓'), ('swallowed', 'Trees swallowed\nby a neighbour ↓'),
              ('unlabelled', 'Trees mostly\nunlabelled ↓')]
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 2.05), sharey=True)
    for ax, (key, title) in zip(axes, panels):
        ax.set_ylim(0, 100)
        for k, (label, v) in enumerate(WEISER.items()):
            bar(ax, k, v[key], COLOURS[label], .62, label, fmt='{:.0f} %')
        ax.set_xticks([]), ax.set_title(title, fontsize=7.6)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('Percent')
    handles = [Patch(facecolor=COLOURS[k], label=k) for k in WEISER]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(.5, 1.14))
    fig.subplots_adjust(wspace=.18)
    save(fig, 'fig10_weiser_real_als')


# ---------------------------------------------------------------- figure 10: reference scene
def fig_scene(x0=764280., y0=197260., size=120., cell=.5):
    import laspy
    import geopandas as gpd
    tile = 'trees_764000_197000.laz'
    fb = laspy.read(PROJECT / 'output_39_two_pass_v7_scene_fallback/PointClouds' / tile)
    pr = laspy.read(PROJECT / 'output_38_two_pass_v7_scene/PointClouds' / tile)
    x, y, h = np.asarray(fb.x), np.asarray(fb.y), np.asarray(fb.height_agl)
    assert np.array_equal(x, np.asarray(pr.x)) and np.array_equal(y, np.asarray(pr.y))
    inside = (x >= x0) & (x < x0 + size) & (y >= y0) & (y < y0 + size)
    tall = inside & (h > 2.)
    sem = np.asarray(fb.pred_semantic)
    idx = top_down(x[tall], y[tall], h[tall], x0, y0, size, cell)

    def labels_of(arr):
        lab = np.asarray(arr).astype(np.int64)[tall]
        lab[(lab == 0) & (sem[tall] == 0)] = -2  # predicted non-tree vegetation: drawn as empty
        return lab

    def draw(ax, lab, colour_of, title):
        img = render(idx, np.where(lab == -2, -1, lab), colour_of)
        img[(idx >= 0) & (np.where(idx >= 0, lab[np.maximum(idx, 0)], 0) == -2)] = matplotlib.colors.to_rgb('#f1f0ec')
        ax.imshow(img, extent=(0, size, 0, size), interpolation='nearest')
        ax.set_title(title, fontsize=8)

    fig, axes = plt.subplots(2, 2, figsize=(5.6, 6.0))
    axes = axes.ravel()
    chm = np.full(idx.shape, np.nan)
    chm[idx >= 0] = h[tall][idx[idx >= 0]]
    axes[0].imshow(chm, extent=(0, size, 0, size), cmap='Greys', vmin=0, vmax=np.nanpercentile(chm, 99), alpha=.55, interpolation='nearest')
    crowns = gpd.read_file(ROOT / 'segmentatiion_benchmark/output_00_baseline/Segmentation3/crowns_764000_197000.gpkg')
    from shapely.geometry import box as shp_box
    crowns = crowns[crowns.intersects(shp_box(x0, y0, x0 + size, y0 + size))]
    crowns = crowns.translate(-x0, -y0)
    crowns.boundary.plot(ax=axes[0], color=INK, linewidth=.45)
    axes[0].set_xlim(0, size), axes[0].set_ylim(0, size)
    axes[0].set_title(f'(a) Classical CHM pipeline: {len(crowns)} crowns', fontsize=8)
    count = lambda lab: len(np.unique(lab[lab > 0]))
    m20 = labels_of(fb.legacy_tree_id)
    v7 = labels_of(pr.tree_id)
    v7f = labels_of(fb.tree_id)
    draw(axes[1], m20, lambda v: instance_colours(v, 1), f'(b) Model20: {count(m20)} trees')
    draw(axes[2], v7, lambda v: instance_colours(v, 2), f'(c) Two-pass v7: {count(v7)} trees')
    draw(axes[3], v7f, lambda v: instance_colours(v, 2), f'(d) v7 + fallback: {count(v7f)} trees')
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.legend(handles=legend_handles([('one colour = one tree', '#6250d6'), ('grey = tree points without a tree', NO_INSTANCE)]),
               loc='lower center', ncol=2, bbox_to_anchor=(.5, .04), fontsize=7)
    fig.subplots_adjust(wspace=.05, hspace=.14)
    save(fig, 'fig11_reference_scene')
    return dict(x0=x0, y0=y0, size=size, classical=len(crowns), model20=count(m20), v7=count(v7), v7_fallback=count(v7f))


def main():
    info = {}
    fig_how_it_works()
    fig_three_variants()
    info['votes_plot'] = fig_votes()
    info['test_plots'] = fig_test_plots()
    fig_heldout()
    fig_errors()
    fig_collections()
    info['per_plot_better'] = fig_per_plot()
    fig_density()
    fig_weiser()
    info['scene_crop'] = fig_scene()
    (OUT / 'figure_info.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
