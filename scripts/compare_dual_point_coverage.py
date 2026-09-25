#!/usr/bin/env python3
"""Audit a postprocessing-only re-export against the previous labelled LAZ files."""
import argparse
import csv
import json
from pathlib import Path

import laspy
import numpy as np
from openpyxl import Workbook

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, default=PROJECT / 'output_16_dual_head_satv2_pointcloud')
    parser.add_argument('--after', type=Path, default=PROJECT / 'output_17_dual_head_support_fusion')
    args = parser.parse_args()
    rows = []
    before_paths = sorted((args.before / 'PointClouds').glob('trees_*.laz'))
    after_paths = sorted((args.after / 'PointClouds').glob('trees_*.laz'))
    if not before_paths or [p.name for p in before_paths] != [p.name for p in after_paths]:
        raise ValueError('Mismatched or missing point-cloud tiles')
    for path in before_paths:
        before, after = laspy.read(path), laspy.read(args.after / 'PointClouds' / path.name)
        for field in ('X', 'Y', 'Z', 'classification', 'intensity', 'height_agl', 'legacy_tree_id'):
            if not np.array_equal(np.asarray(before[field]), np.asarray(after[field])):
                raise AssertionError(f'Changed {field}: {path.name}')
        canopy = (np.asarray(after.height_agl) >= 2) & (np.asarray(after.classification) != 2)
        old, new = np.asarray(before.tree_id) > 0, np.asarray(after.tree_id) > 0
        rows.append(dict(tile=path.stem.removeprefix('trees_'), points=len(new), canopy_points=int(canopy.sum()),
                         old_labelled_canopy=int((old & canopy).sum()), new_labelled_canopy=int((new & canopy).sum()),
                         old_coverage=float(old[canopy].mean()), new_coverage=float(new[canopy].mean()),
                         recovered_points=int((~old & new).sum()), lost_labels=int((old & ~new).sum()),
                         source_fields_and_legacy_labels_unchanged=True))
    report = dict(before=str(args.before.resolve()), after=str(args.after.resolve()), per_tile=rows,
                  total_recovered_points=sum(r['recovered_points'] for r in rows),
                  total_lost_labels=sum(r['lost_labels'] for r in rows),
                  old_canopy_coverage=sum(r['old_labelled_canopy'] for r in rows)/sum(r['canopy_points'] for r in rows),
                  new_canopy_coverage=sum(r['new_labelled_canopy'] for r in rows)/sum(r['canopy_points'] for r in rows),
                  caveat='Coverage is label completeness, not accuracy; benchmark tiles have no reference tree IDs.')
    (args.after / 'coverage_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    with (args.after / 'coverage_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'Benchmark coverage'
    sheet.append(list(rows[0]))
    for row in rows:
        sheet.append(list(row.values()))
    sheet.freeze_panes = 'A2'
    sheet.auto_filter.ref = sheet.dimensions
    workbook.save(args.after / 'coverage_comparison.xlsx')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
