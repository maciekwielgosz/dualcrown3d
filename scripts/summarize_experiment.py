#!/usr/bin/env python3
"""Build machine-readable and human-readable comparisons of completed runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_DIR / "reports"
EVALUATION_DIR = PROJECT_DIR / "outputs" / "evaluation"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def yolo_row(run_name: str, label: str, input_name: str) -> dict | None:
    report_path = EVALUATION_DIR / run_name / "evaluation_report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report["test"]["custom_instance_metrics"]["0.5"]["overall"]
    validation = report["validation"]["overall_iou50"]
    coco = report["test"]["ultralytics_coco_metrics"]
    return {
        "model": label,
        "input": input_name,
        "comparison_status": "fair; no treeID in model input",
        "validation_f1_iou50": validation["f1"],
        "validation_panoptic_quality": validation["panoptic_quality"],
        "gt_count": metrics["gt_count"],
        "prediction_count": metrics["prediction_count"],
        "true_positives": metrics["true_positives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "precision_iou50": metrics["precision"],
        "recall_iou50": metrics["recall"],
        "f1_iou50": metrics["f1"],
        "segmentation_quality": metrics["segmentation_quality"],
        "panoptic_quality": metrics["panoptic_quality"],
        "mean_matched_dice": metrics["mean_dice_matched"],
        "area_iou": metrics["area_iou"],
        "mask_map50": coco["metrics/mAP50(M)"],
        "mask_map50_95": coco["metrics/mAP50-95(M)"],
        "confidence": report["protocol"]["selected_confidence"],
        "inference_fps": report["test"]["inference_fps"],
        "end_to_end_images_per_second": report["test"]["images_per_second_end_to_end"],
        "device": report["hardware"]["device"],
        "source": str(report_path.relative_to(PROJECT_DIR)),
    }


def baseline_row(run_name: str, label: str, status: str) -> dict | None:
    metrics_path = EVALUATION_DIR / run_name / "overall_metrics.csv"
    if not metrics_path.is_file():
        return None
    metrics = next(row for row in read_csv(metrics_path) if float(row["threshold"]) == 0.5)
    return {
        "model": label,
        "input": "CHM",
        "comparison_status": status,
        "validation_f1_iou50": None,
        "validation_panoptic_quality": None,
        "gt_count": int(metrics["gt_count"]),
        "prediction_count": int(metrics["prediction_count"]),
        "true_positives": int(metrics["true_positives"]),
        "false_positives": int(metrics["false_positives"]),
        "false_negatives": int(metrics["false_negatives"]),
        "precision_iou50": float(metrics["precision"]),
        "recall_iou50": float(metrics["recall"]),
        "f1_iou50": float(metrics["f1"]),
        "segmentation_quality": float(metrics["segmentation_quality"]),
        "panoptic_quality": float(metrics["panoptic_quality"]),
        "mean_matched_dice": float(metrics["mean_dice"]),
        "area_iou": float(metrics["area_iou"]),
        "mask_map50": None,
        "mask_map50_95": None,
        "confidence": None,
        "inference_fps": None,
        "end_to_end_images_per_second": None,
        "device": None,
        "source": str(metrics_path.relative_to(PROJECT_DIR)),
    }


def training_row(run_name: str) -> dict | None:
    path = PROJECT_DIR / "outputs" / "training" / run_name / "results.csv"
    if not path.is_file():
        return None
    rows = read_csv(path)
    if not rows:
        return None
    # In the pinned Ultralytics version, segmentation fitness is the sum of
    # box mAP50-95 and mask mAP50-95. Reproduce that criterion so the reported
    # epoch corresponds to best.pt rather than to one arbitrarily chosen metric.
    fitness = [
        float(row["metrics/mAP50-95(B)"])
        + float(row["metrics/mAP50-95(M)"])
        for row in rows
    ]
    best_index = max(range(len(rows)), key=fitness.__getitem__)
    best = rows[best_index]
    return {
        "run": run_name,
        "epochs_completed": len(rows),
        "best_epoch": int(float(best["epoch"])),
        "training_seconds": float(rows[-1]["time"]),
        "checkpoint_validation_mask_map50": float(best["metrics/mAP50(M)"]),
        "checkpoint_validation_mask_map50_95": float(best["metrics/mAP50-95(M)"]),
        "best_validation_fitness": fitness[best_index],
    }


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [
        baseline_row(
            "watershed_lmf_leakage_free",
            "Watershed/LMF (leakage-free)",
            "fair input; legacy parameters transferred to new CHM",
        ),
        baseline_row(
            "watershed_lmf_for_instance_v2",
            "Watershed/LMF (historical)",
            "diagnostic only; CHM was built from annotated tree points",
        ),
        yolo_row(
            "yolo11s_physical",
            "YOLO11s-seg physical",
            "CHM + point density + mean intensity",
        ),
        yolo_row(
            "yolo11s_paper_fusion",
            "YOLO11s-seg paper fusion",
            "RGB fusion of CHM, point density and mean intensity",
        ),
        yolo_row(
            "yolo11s_chm_only_gpu",
            "YOLO11s-seg CHM-only",
            "Normalized CHM repeated in three technical channels",
        ),
        yolo_row(
            "yolo11s_chm_only_ideas_finetuned_gpu",
            "YOLO11s-seg CHM-only + ideas_als",
            "Normalized CHM repeated in three technical channels",
        ),
    ]
    comparisons = [row for row in candidates if row is not None]
    if not comparisons:
        raise SystemExit("No completed evaluation results found")
    training = [
        row
        for name in (
            "yolo11s_physical",
            "yolo11s_paper_fusion",
            "yolo11s_chm_only_gpu",
            "yolo11s_chm_only_ideas_combined_gpu",
            "yolo11s_chm_only_ideas_finetuned_gpu",
        )
        if (row := training_row(name)) is not None
    ]
    write_csv(REPORTS_DIR / "model_comparison.csv", comparisons)
    payload = {
        "evaluation_protocol": {
            "split": "official FOR-instance test subset available locally",
            "test_plots": 11,
            "test_ground_truth_instances": 278,
            "instance_match": "Hungarian one-to-one at mask IoU >= 0.5",
            "confidence": "selected on validation only for YOLO models",
        },
        "models": comparisons,
        "training": training,
        "limitations": [
            "NIBIO2 is listed by FOR-instance metadata but absent from the local dataset and archive.",
            "paper_fusion is a documented approximation; the paper does not publish exact color maps or CCD-YOLO code.",
            "Speed was measured on CPU and is not directly comparable with the paper's GPU FPS.",
            "The historical watershed result is diagnostic because its CHM construction used annotated-tree points.",
        ],
    }
    (REPORTS_DIR / "model_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    yolo_candidates = [row for row in comparisons if row["validation_f1_iou50"] is not None]
    best = max(yolo_candidates, key=lambda row: row["validation_f1_iou50"])
    table_rows = "\n".join(
        "| "
        + " | ".join(
            [
                row["model"],
                fmt(row["precision_iou50"]),
                fmt(row["recall_iou50"]),
                fmt(row["f1_iou50"]),
                fmt(row["segmentation_quality"]),
                fmt(row["panoptic_quality"]),
                fmt(row["mask_map50"]),
                fmt(row["mask_map50_95"]),
            ]
        )
        + " |"
        for row in comparisons
    )
    training_lines = "\n".join(
        f"- `{row['run']}`: {row['epochs_completed']} epok, najlepsza epoka "
        f"{row['best_epoch']}, czas {row['training_seconds'] / 60:.1f} min, "
        f"val mask mAP50-95={row['checkpoint_validation_mask_map50_95']:.3f}."
        for row in training
    )
    report = f"""# Raport końcowy: FOR-instance + YOLO11s-seg

## Wynik

Na podstawie walidacyjnego F1 wybrano **{best['model']}**
(F1={best['validation_f1_iou50']:.3f}). Po zamrożeniu wyboru model osiągnął na
oficjalnym teście F1={best['f1_iou50']:.3f}, SQ={best['segmentation_quality']:.3f}
i PQ={best['panoptic_quality']:.3f}. Próg pewności również został wybrany wyłącznie
na walidacji; test nie uczestniczył w doborze wariantu ani progu.

## Porównanie na teście

Dopasowanie instancji: algorytm węgierski, IoU maski >= 0,5. Test obejmuje 11
powierzchni i 278 koron.

| Model | Precision | Recall | F1/RQ | SQ | PQ | mask mAP50 | mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
{table_rows}

`Watershed/LMF (historical)` nie jest uczciwym baseline'em publikacyjnym: wejściowy
CHM utworzono tylko z punktów mających adnotacje drzew. Właściwym baseline'em jest
wariant `leakage-free`, choć odziedziczył parametry dostrojone historycznie.

## Trening

{training_lines}

Urządzenie użyte dla każdego przebiegu zapisano w kolumnie `device`; wariant
CHM-only wytrenowano i oceniono na GPU, a starsze warianty na CPU. Pole
`inference_fps` w `model_comparison.csv` obejmuje samą inferencję kafli i nie
obejmuje rasteryzacji LAS, eksportu GeoPackage ani GeoTIFF.

## Zakres reprodukcji

- Wejście fizyczne: CHM, liczba punktów korony w pikselu i średnia intensywność.
- Wejście CHM-only: jeden znormalizowany CHM powielony do trzech kanałów
  technicznych wymaganych przez standardową architekturę YOLO.
- Wejście nie korzysta z `treeID`; identyfikator służy tylko do utworzenia GT.
- `paper_fusion` odtwarza opisaną ideę map kolorów, ale nie jest dokładnym
  CCD-YOLO, ponieważ autorzy nie udostępnili kodu ani pełnej definicji fuzji.
- Predykcje zachowują wszystkie instancje w GeoPackage. Raster `tree_id` rozwiązuje
  nakładanie masek na korzyść instancji o wyższej pewności.
- Lokalna kopia FOR-instance nie zawiera kolekcji NIBIO2 wymienionej w metadanych.

Szczegółowe liczby znajdują się w `model_comparison.csv` i
`model_comparison.json`, a surowe raporty w `outputs/evaluation/`.
"""
    (REPORTS_DIR / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(
        f"Wrote {len(comparisons)} model rows; validation-selected model: "
        f"{best['model']} (val F1={best['validation_f1_iou50']:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
