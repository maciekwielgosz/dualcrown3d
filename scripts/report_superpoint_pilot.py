#!/usr/bin/env python3
"""Consolidate superpoint Stage 0-2 evidence into JSON and Excel."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from openpyxl import Workbook

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "outputs/dualcrown3d_superpoint_v1"
OUTPUT = ROOT / "review_stage0_2"
RETAINED = PROJECT / "outputs/dualcrown3d_joint_campaign_v1/runs/repeat_20260930/weights/best.pt"


def from_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def cell(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str) and value.strip():
        try:
            number = float(value)
            if math.isfinite(number):
                return int(number) if number.is_integer() else number
        except ValueError:
            pass
    return value


def sheet(book: Workbook, name: str, rows: list[dict]) -> None:
    if not rows:
        return
    page = book.create_sheet(name)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    page.append(columns)
    for row in rows:
        page.append([cell(row.get(key)) for key in columns])
    page.freeze_panes = "A2"
    page.auto_filter.ref = page.dimensions


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Completed review protected: {OUTPUT}")
    audit = json.loads((ROOT / "stage0/audit.json").read_text())
    inventory = from_csv(ROOT / "stage0/inventory.csv")
    feasibility = json.loads((ROOT / "stage1/summary.json").read_text())
    per_plot = from_csv(ROOT / "stage1/per_plot.csv")
    connectivity = from_csv(ROOT / "stage1_connectivity/per_plot.csv")
    trained = json.loads((ROOT / "stage2_affinity_pilot/result.json").read_text())
    configuration = json.loads((ROOT / "stage2_affinity_pilot/config.json").read_text())
    balanced = json.loads((ROOT / "stage2_balanced_calibration/result.json").read_text())
    sweep = json.loads((ROOT / "stage2_extent_sweep_v2/result.json").read_text())
    vote = json.loads((ROOT / "stage2_vote_coherence/result.json").read_text())
    runs = [dict(run_id="retained_dualcrown3d", role="external_production_control",
                 checkpoint=str(RETAINED),
                 checkpoint_sha256=hashlib.sha256(RETAINED.read_bytes()).hexdigest(),
                 initialization="HELIOS-exposed", point_sb_pq=.4051389650367029,
                 crown_sb_pq=.4441833039576545,
                 note="full-plot point/crown validation; not comparable to partition oracle"),
            dict(run_id="stage2_affinity_pilot", role="boundary/partition pilot",
                 checkpoint=configuration["checkpoint"],
                 affinity_head=configuration["checkpoint"],
                 affinity_head_sha256=configuration["checkpoint_sha256"],
                 train_plots=configuration["train_plots"],
                 val_plots=configuration["val_plots"],
                 epochs=configuration["epochs"],
                 selected_threshold=configuration["selected_threshold"],
                 backbone_peak_vram_mb=configuration["backbone_peak_vram_mb"],
                 head_peak_vram_mb=configuration["head_peak_vram_mb"],
                 code_provenance=configuration["git"])]
    comparison = [dict(calibration="first12_train", **row)
                  for row in trained["comparison"]]
    comparison += [dict(calibration="source_balanced_train", **row)
                   for row in balanced["comparison"]]
    candidates = [dict(extent_m=row["extent_m"], threshold=row["threshold"],
                       match_error=row["match_error"],
                       training_gate=row["training_gate"],
                       fixed=row["train_fixed"], learned=row["train_learned"])
                  for row in sweep["candidates"]]
    vote_candidates = [dict(extent_m=row["extent_m"], vote_sigma_m=row["vote_sigma_m"],
        threshold=row["threshold"], training_gate=row["training_gate"],
        fixed=row["train_fixed"], learned=row["train_candidate"])
        for row in vote["candidates"]]
    bundle = dict(stage0=audit, stage1=feasibility,
                  stage1_connectivity=connectivity,
                  stage2_training=configuration,
                  stage2_balanced_comparison=balanced["comparison"],
                  stage2_extent_candidates=candidates,
                  stage2_vote_candidates=vote_candidates,
                  selected_extent=sweep.get("selected"),
                  stage2_gate_passed=(sweep.get("selected") is not None and
                    sweep["comparison"][1]["mean_oracle_pq"] >
                    sweep["comparison"][0]["mean_oracle_pq"] and
                    sweep["comparison"][1]["mean_boundary_recall"] >
                    sweep["comparison"][0]["mean_boundary_recall"] and
                    abs(sweep["comparison"][1]["mean_compression"] /
                        sweep["comparison"][0]["mean_compression"] - 1.) <= .05)
                    or bool(vote.get("validation_gate_passed", False)),
                  test_used=False)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "review.json").write_text(json.dumps(bundle, indent=2,
                                                    allow_nan=False) + "\n")
    workbook = Workbook()
    workbook.remove(workbook.active)
    sheet(workbook, "runs", runs)
    sheet(workbook, "data_audit", inventory)
    sheet(workbook, "partition_summary", feasibility["summary"])
    sheet(workbook, "partition_per_plot", per_plot)
    sheet(workbook, "graph_connectivity", connectivity)
    sheet(workbook, "affinity_comparison", comparison)
    sheet(workbook, "affinity_per_plot", balanced["validation"])
    sheet(workbook, "extent_candidates", candidates)
    sheet(workbook, "vote_candidates", vote_candidates)
    sheet(workbook, "training_log", trained["train_history"])
    workbook.save(OUTPUT / "review.xlsx")
    print(json.dumps(dict(stage2_gate_passed=bundle["stage2_gate_passed"],
                          balanced=balanced["comparison"],
                          extent_candidates=candidates), indent=2), flush=True)


if __name__ == "__main__":
    main()
