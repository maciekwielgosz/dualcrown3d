#!/usr/bin/env python3
"""Consolidate the small TLS visibility pilot into auditable JSON and Excel."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

from openpyxl import Workbook

PROJECT = Path(__file__).resolve().parents[1]
FIRST = PROJECT / "outputs/tls_visibility_finetune_pilot_v1"
PROFILE = PROJECT / "outputs/tls_visibility_finetune_profile_pilot_v1"
FIRST_DATA = PROJECT / "outputs/tls_visibility_pilot_matched_v2"
PROFILE_DATA = PROJECT / "outputs/tls_visibility_pilot_profile_matched_v1"
OUTPUT = PROJECT / "outputs/tls_visibility_pilot_review_v1"


def quality(item: dict) -> float:
    return math.sqrt(item["point_metrics"]["source_balanced_pq"] *
                     item["metrics"]["source_balanced_pq"])


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def append_sheet(book: Workbook, name: str, rows: list[dict]) -> None:
    if not rows:
        return
    sheet = book.create_sheet(name)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    sheet.append(fields)
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, str) and value.strip():
                try:
                    number = float(value)
                    if math.isfinite(number):
                        value = int(number) if number.is_integer() else number
                except ValueError:
                    pass
            cells.append(value)
        sheet.append(cells)
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Completed review protected: {output}")
    baseline = json.loads((PROJECT / "outputs/dualcrown3d_legacy_small_tree_calibration_extended/selected.json").read_text())["baseline"]
    runs = [("real_only", FIRST / "real_only"),
            ("helios", FIRST / "helios"),
            ("thinning", FIRST / "thinning"),
            ("visibility_initial", FIRST / "visibility"),
            ("visibility_profiled", PROFILE / "visibility")]
    summary = []
    source_rows = []
    epochs = []
    configurations = []
    base = dict(mode="retained_checkpoint", epoch=0,
                point_sb_pq=baseline["point_metrics"]["source_balanced_pq"],
                crown_sb_pq=baseline["metrics"]["source_balanced_pq"],
                point_sb_f1=baseline["point_metrics"]["source_balanced_f1"],
                crown_sb_f1=baseline["metrics"]["source_balanced_f1"],
                small_4_tp=baseline["small_crowns"]["up_to_4_m2"]["tp"],
                small_10_tp=baseline["small_crowns"]["up_to_10_m2"]["tp"],
                selected_epoch=0, reaches_gate=False,
                note="same archived native validation protocol")
    summary.append(base)
    for mode, run in runs:
        selection = json.loads((run / "selected.json").read_text())
        cfg = json.loads((run / "configuration.json").read_text())
        configurations.append(dict(mode=mode, **cfg))
        candidates = [(int(folder.name.split("_")[-1]),
                       json.loads((folder / "metrics.json").read_text()))
                      for folder in (run / "validation").glob("epoch_*")]
        epoch, result = max(candidates, key=lambda pair: quality(pair[1]))
        row = dict(mode=mode, epoch=epoch, selected_epoch=selection["best_epoch"],
                   point_sb_pq=result["point_metrics"]["source_balanced_pq"],
                   crown_sb_pq=result["metrics"]["source_balanced_pq"],
                   point_sb_f1=result["point_metrics"]["source_balanced_f1"],
                   crown_sb_f1=result["metrics"]["source_balanced_f1"],
                   small_4_tp=result["small_crowns"]["up_to_4_m2"]["tp"],
                   small_10_tp=result["small_crowns"]["up_to_10_m2"]["tp"],
                   point_tp=result["point_metrics"]["tp"],
                   point_fp=result["point_metrics"]["fp"],
                   crown_tp=result["metrics"]["tp"],
                   crown_fp=result["metrics"]["fp"],
                   reaches_gate=all(result[branch][metric] >= baseline[branch][metric]
                       for branch in ("point_metrics", "metrics")
                       for metric in ("source_balanced_pq", "source_balanced_f1"))
                       and result["small_crowns"]["up_to_10_m2"]["tp"] >
                       base["small_10_tp"],
                   checkpoint=(selection["checkpoint"] if
                               selection["best_epoch"] == epoch else ""),
                   note="best measured trained epoch; selected_epoch=0 means retained model wins")
        summary.append(row)
        for branch in ("point_metrics", "metrics"):
            for source, values in result[branch]["by_source"].items():
                source_rows.append(dict(mode=mode, branch=branch,
                                        source=source, **values))
        epochs.extend(dict(record, mode=mode)
                      for record in read_csv(run / "training_log.csv"))
    qa = []
    for variant, data in (("visibility_initial", FIRST_DATA),
                          ("visibility_profiled", PROFILE_DATA)):
        readiness = json.loads((data / "READY.json").read_text())
        qa.extend(dict(dataset=variant, spacing_m=readiness["spacing_m"],
                       opacity=readiness["opacity"],
                       transmission=readiness.get("transmission", .55),
                       **row) for row in read_csv(data / "qa.csv"))
    output.mkdir(parents=True)
    information = dict(summary=summary, qa=qa, source_metrics=source_rows,
                       runs=configurations, epochs=epochs,
                       split="six TreeScan training parents; real ALS validation 14 plots",
                       test_used_for_selection=False,
                       pilot_only=True)
    tmp = output / "review.tmp.json"
    tmp.write_text(json.dumps(information, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, output / "review.json")
    book = Workbook()
    book.remove(book.active)
    append_sheet(book, "validation", summary)
    append_sheet(book, "synthetic_qa", qa)
    append_sheet(book, "by_source", source_rows)
    append_sheet(book, "epochs", epochs)
    append_sheet(book, "configurations", configurations)
    temporary = output / "review.tmp.xlsx"
    book.save(temporary)
    os.replace(temporary, output / "review.xlsx")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
