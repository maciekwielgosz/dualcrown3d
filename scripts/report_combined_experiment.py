#!/usr/bin/env python3
"""Build a concise final report and reproducible, unselected example overlays."""
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import plotting_extent

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pointcloud.experiment_log import ROOT, record

DATA_ROOT = Path(os.environ.get("SEGMENTATION_DATA_ROOT", PROJECT.parent / "combined_als_crowns_v1")).resolve()


def main():
    selected = ROOT / "selected"
    comparison = json.loads((selected / "comparison.json").read_text())
    frozen = json.loads((selected / "frozen_selection.json").read_text())
    with (DATA_ROOT / "manifest_test.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    # Fixed lexicographic examples by source, not hand-picked by model score.
    available = {r["collection"] for r in rows}
    preferred = ("CULS", "CEDAR_CYPRESS", "FGI_EMIT", "WILDFOREST3D", "NIBIO", "ECODSE", "SCION")
    examples = [sorted([r for r in rows if r["collection"] == c], key=lambda r:r["dataset_id"])[0]
                for c in preferred if c in available][:4]
    fig, axes = plt.subplots(len(examples), 2, figsize=(12, 19), constrained_layout=True)
    for i, row in enumerate(examples):
        gt = gpd.read_file(row["gt_vector"])
        with rasterio.open(row["chm"]) as source:
            chm = source.read(1, masked=True)
            extent = plotting_extent(source)
        for j, method in enumerate(("classical_test", "model_test")):
            ax = axes[i, j]
            pred = gpd.read_file(selected / method / "Segmentation3" / f"crowns_{row['dataset_id']}.gpkg")
            ax.imshow(chm, extent=extent, cmap="Greys", vmin=0, vmax=40)
            if len(pred):
                pred.plot(ax=ax, facecolor="#F39C12" if j == 0 else "#12B8A6", edgecolor="#9E5500" if j == 0 else "#006658", alpha=.40, linewidth=.8)
            gt.boundary.plot(ax=ax, color="#E00045", linewidth=1.0)
            ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:]); ax.set_aspect("equal")
            ax.set_title(f"{row['collection']} | {'classical' if j == 0 else 'DL'} | GT={len(gt)}, predictions={len(pred)}", fontsize=10)
            ax.ticklabel_format(useOffset=False, style="plain")
            ax.tick_params(labelsize=7)
    fig.suptitle("Full crowns: red = reference; orange = classical; teal = DL\nFixed first test plot per source; no selection by performance", fontsize=14)
    fig.savefig(selected / "crown_comparison.png", dpi=170)
    fig.savefig(selected / "crown_comparison.svg")
    plt.close(fig)
    m, b = comparison["model"], comparison["classical"]
    ci = comparison["uncertainty"]["delta_source_balanced_pq_95ci"]
    result = "TAK (wartosci punktowe)" if comparison["target_met_on_test"] else "NIE"
    with (DATA_ROOT / "manifest.csv").open() as stream:
        all_rows = list(csv.DictReader(stream))
    split_counts = Counter(r["model_split"] for r in all_rows)
    crown_counts = {split: sum(int(r["instances"]) for r in all_rows if r["model_split"] == split) for split in ("train", "val", "test")}
    excluded = json.loads((DATA_ROOT / "exclusion_report.json").read_text()) if (DATA_ROOT / "exclusion_report.json").exists() else None
    exclusion_text = (f"Kolekcja **{excluded['removed_collection']}** ({excluded['removed_plots']} plikow) zostala calkowicie wykluczona, "
                      "poniewaz jej anotacje sa prostokatami, a nie dokladnymi maskami koron." if excluded else "Nie zastosowano filtrowania calej kolekcji po typie anotacji.")
    validation_note = ("Wyniki walidacyjne tej kampanii korzystaja z poprawionej siatki sparse i tej samej reguly out-points. "
                       "Do porownania checkpointow uzywaj `final_val_*`; `val_*` zachowuje przebieg treningu."
                       if excluded else
                       "Wczesne wyniki walidacyjne z epok poprzedzaly korekte indeksow siatki sparse-convolution i protokolu ignorowania out-points. "
                       "Uzywaj kolumn `final_val_*` oraz `test_*` w Excelu; stare wyniki zachowano dla audytu.")
    text = f"""# Wspolny eksperyment pelnych koron

Wybrany model: **{comparison['experiment_id']}**. Przewaga nad przeliczonym
baseline na zamrozonym tescie: **{result}**.

| Metryka testowa | Najlepszy klasyczny (wybor na val) | Model DL |
|---|---:|---:|
| PQ, srednia po zrodlach | {b['source_balanced_pq']:.4f} | {m['source_balanced_pq']:.4f} |
| F1, lacznie | {b['f1']:.4f} | {m['f1']:.4f} |
| Precision | {b['precision']:.4f} | {m['precision']:.4f} |
| Recall | {b['recall']:.4f} | {m['recall']:.4f} |
| PQ, lacznie | {b['pq']:.4f} | {m['pq']:.4f} |
| SQ | {b['sq']:.4f} | {m['sq']:.4f} |

Dopasowanie: pelne, takze nakladajace sie poligony; IoU >= 0.50;
jednoznaczne dopasowanie Hungarian. Przedzial bootstrap 95% dla roznicy
PQ po zrodlach: [{ci[0]:.4f}, {ci[1]:.4f}], losowanie parami calych grup
przestrzennych. Nie nalezy utozsamiac przewagi wartosci punktowej z dowodem
istotnosci statystycznej, zwlaszcza przy malych kolekcjach.

## Dane i uczciwosc porownania

{split_counts['train']} train / {split_counts['val']} val / {split_counts['test']} test;
{crown_counts['train']} / {crown_counts['val']} / {crown_counts['test']} referencyjnych koron.
Fizyczna kopia: `{DATA_ROOT.name}`. Oryginaly niezmienione.
{exclusion_text}
Zachowano grupy nakladajacych sie obszarow; lokalnych ukladow bez
georeferencji nie mozna w pelni sprawdzic przestrzennie.

Stare wyniki `run_r` (inne podzialy, topmost GT i czesciowo CHM budowany
z oznaczonych drzew) **nie sa bezposrednio porownywalne** z ta tabela.
Tutaj oba algorytmy maja identyczny zbior i pelne GT; CHM nie uzywa treeID.
Nie deklarujemy, ze ten wynik liczbowo przekracza kazdy historyczny F1.
Nowy test nie jest dziewiczym holdoutem publikacyjnym: czesc obszarow
ogladano we wczesniejszych eksperymentach. Testu nie uzyto do wyboru
architektury, checkpointu ani progow w tej kampanii.

Native point labels sa rzutowane na poligony, nie stanowia niezaleznie
zdigitalizowanych obrysow. Crown-overlay ma inne zrodlo i gestosc etykiet.
Metryki opisuja zgodnosc z dostepnymi adnotacjami; nie dowodza, ze kazda
niepasujaca predykcja jest fizycznie nieistniejacym drzewem. Szczegoly
per zrodlo sa w Excelu i JSON.

Regula obszarow bez anotacji: po dopasowaniu GT pomijamy niepasujace
predykcje pokryte w >=50% projekcja jawnej klasy 3 (out-points), bez
innych niegruntowych punktow w tej samej komorce 0.5 m. Duplikaty GT
pozostaja FP. Pelnych koron nie przycinamy do tej maski. Ta sama regula
obowiazuje baseline i DL; maska nie jest wejsciem modelu. W treningu
klasa 3 byla traktowana jako tlo, co pozostaje ograniczeniem nadzoru.

{validation_note}

## Predykcje i checkpoint

- `best.pt` — zamrozony checkpoint, SHA-256: `{frozen['model']['checkpoint_sha256']}`.
- `frozen_selection.json` — konfiguracja wybrana przed dostepem do testu.
- `model_test/Segmentation3` — pary `crowns_*.gpkg` i `ttops_*.gpkg` modelu.
- `classical_test/Segmentation3` — te same obszary dla baseline.
- `crown_comparison.png` — stale przyklady (pierwszy plik danej kolekcji).
- `comparison.json` — metryki i bootstrap.
- `../experiments.xlsx` — parametry, metryki, epoki, zrodla, checkpointy.

Korony modelu sa pelnymi wypelnionymi poligonami. Zachowany jest CRS
zrodla; lokalnym wspolrzednym nie przypisano fikcyjnego EPSG:2180.
Pole dbh w wierzcholkach jest empiryczna estymacja z wysokosci,
zachowana dla zgodnosci formatu, a nie pomiarem referencyjnym.

## Szybkosc

Inferencja modelu na przygotowanych punktach: {comparison['model_point_inference_seconds']:.2f} s
na {split_counts['test']} obszarow. Segmentacja klasyczna z gotowego CHM:
{comparison['classical_chm_segmentation_seconds']:.2f} s. To rozne przygotowane
reprezentacje wejscia; czasy nie obejmuja pelnego procesu od LAS ani nie
sa czystym czasem GPU. Nie nalezy podawac ich jako end-to-end FPS.

Model jest dokladniejszy w tym protokole, lecz wolniejszy od klasycznej
segmentacji gotowego CHM. Kontrola artefaktow sprawdza pary GeoPackage,
geometrie, treeID i CRS w `artifact_validation.json`.
"""
    (selected / "REPORT.md").write_text(text)
    record(comparison["experiment_id"], report=str(selected / "REPORT.md"), comparison_image=str(selected / "crown_comparison.png"))
    print(selected / "REPORT.md")


if __name__ == "__main__":
    main()
