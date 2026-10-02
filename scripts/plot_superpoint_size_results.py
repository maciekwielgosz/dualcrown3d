#!/usr/bin/env python3
"""Export a compact scientific chart of the matched decoder-size ablation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot'


def pick(report, variant):
    return next(row for row in report['comparison'] if row['variant'] == variant)


def main():
    first = json.loads((ROOT / 'size_ablation_large_w192_q128.json').read_text())
    second = json.loads((ROOT / 'size_ablation_w192_q96.json').read_text())
    records = [pick(first, 'small_initial'), pick(first, 'small_trained'),
               pick(second, 'large_trained'), pick(first, 'large_trained')]
    labels = ['Start', 'Original 96', 'Wide 96', 'Wide 128']
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(9.2, 4.7), constrained_layout=True)
    width = .34
    for offset, field, title, color in [(-width/2, 'point_pq', 'Point instances', '#19647e'),
                                        (width/2, 'crown_pq', 'Crown polygons', '#e88a4b')]:
        bars = axis.bar(x + offset, [row[field] for row in records], width, label=title, color=color)
        axis.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    axis.set_xticks(x, labels)
    axis.set_ylabel('Source-balanced PQ @ IoU 0.5')
    axis.set_ylim(0, .35)
    axis.set_title('Decoder size comparison · 14 native ALS validation crops')
    axis.legend(frameon=False)
    axis.grid(axis='y', alpha=.2)
    axis.set_axisbelow(True)
    figure.text(.5, .005, 'Same frozen encoder, partition, training crops and 12-epoch budget; exploratory validation',
                ha='center', va='bottom', fontsize=8)
    output = ROOT / 'size_ablation.png'
    if output.exists():
        raise FileExistsError(f'Protected figure: {output}')
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == '__main__':
    main()
