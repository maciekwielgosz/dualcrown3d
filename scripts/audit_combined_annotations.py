#!/usr/bin/env python3
"""Post-freeze label-quality audit; never changes labels, predictions or selection."""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.experiment_log import ROOT, record
from scripts.evaluate_combined_full_crowns import aggregate


def main():
    dataset = PROJECT.parent / "combined_als_crowns_v1"
    with (dataset / "manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    counts = defaultdict(lambda: dict(crowns=0, axis_aligned_rectangles=0))
    for row in rows:
        gt = gpd.read_file(row["gt_vector"])
        value = counts[row["collection"]+":"+row["model_split"]]
        value["crowns"] += len(gt)
        value["axis_aligned_rectangles"] += sum(abs(g.area-g.envelope.area) <= max(g.area, 1.)*1e-6 for g in gt.geometry)
    ids = {r["dataset_id"] for r in rows if r["model_split"] == "test" and r["annotation_method"] == "point_native"}
    native = {}
    for method in ("model_test", "classical_test"):
        data = json.loads((ROOT / "selected" / method / "test_metrics.json").read_text())
        native[method] = aggregate([r for r in data["per_plot"] if r["dataset_id"] in ids])
    source = PROJECT.parent / "ideas_als/_raw/IDTReeS_2020/train/ITC/notes about corrected training data.txt"
    note = "All 1763 IDTREES GT polygons in this prepared dataset are axis-aligned rectangles (1292 train, 233 val, 238 test). Source documentation describes bounding-box conversion. Treat as weak box supervision, not accurate crown masks. This post-freeze audit did not change model selection or metrics."
    result = dict(geometry_by_collection_split=dict(counts), source_document=str(source),
                  source_document_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  native_only_test_plots=len(ids), native_only_test=native,
                  caveat=note, subgroup_status="Descriptive post-freeze analysis, not a replacement primary endpoint")
    path = ROOT / "selected/annotation_quality_audit.json"
    path.write_text(json.dumps(result, indent=2)+"\n")
    (ROOT / "selected/IDTREES_source_notes.txt").write_bytes(source.read_bytes())
    frozen = json.loads((ROOT / "selected/frozen_selection.json").read_text())
    record(frozen["model"]["experiment_id"], annotation_audit=str(path), annotation_caveat=note,
           descriptive_native_only_test_f1=native["model_test"]["f1"],
           descriptive_native_only_classical_f1=native["classical_test"]["f1"])
    report_path = ROOT / "selected/REPORT.md"
    if report_path.exists() and "annotation_quality_audit.json" not in report_path.read_text():
        addition = f"""\n## Audyt jakosci anotacji (po zamrozeniu)

Wszystkie 1763 uzyte anotacje IDTREES sa prostokatami, nie dokladnymi
maskami koron. Konwersje do bounding boxes potwierdza dokumentacja zrodla
(`IDTREES_source_notes.txt`). Nalezy traktowac je jako slaby nadzor.
Audyt nie zmienia GT, predykcji, wyboru modelu ani glownej tabeli metryk.

Opisowo, na {len(ids)} plikach z natywnymi anotacjami punktowymi: F1
{native['model_test']['f1']:.4f} dla modelu oraz
{native['classical_test']['f1']:.4f} dla baseline. To dodatkowa analiza,
nie nowy glowny wynik ani kryterium strojenia. Szczegoly:
`annotation_quality_audit.json`. Dalsze prace powinny oddzielic slabe etykiety
prostokatne od dokladnych masek i zapewnic niezalezny anotowany test.
"""
        report_path.write_text(report_path.read_text()+addition)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
