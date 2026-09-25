# Segmentacja koron drzew: FOR-instance + YOLO11-seg

## Zawartosc repozytorium

Git przechowuje kod, konfiguracje, testy, male manifesty i raporty tekstowe.
Dane LAS/LAZ, rastry, GeoPackage, srodowiska Pythona, checkpointy oraz katalogi
`output_*` i `outputs/` sa celowo pominiete. Ich oczekiwane lokalizacje i sumy
kontrolne wybranych wag zapisano w `CHECKSUMS.sha256`. Minimalna, lokalnie
dostosowana czesc LitePT wraz z informacja o rewizji znajduje sie w `vendor/`.

## Corrected point-mask merging

The dual-head inference runner now uses `configs/dual_head_inference.json` and
writes to `output_17_dual_head_support_fusion`. Overlapping masks can recover
previously unassigned points when mask overlap, spatial proximity and predicted
tree centres agree. Paired validation improved point PQ from 0.1437 to 0.1505
and crown PQ from 0.1881 to 0.2029, with unchanged network weights.
See [the correction report and reproduction commands](reports/support_fusion_fix.md).

For CloudCompare open `PointClouds/trees_*.laz` and choose RGB colours. Restore
the full displayed range when viewing the `tree_id` scalar field.

## Kampania v2 bez prostokątnych anotacji IDTREES

Kolekcję IDTREES usunięto w całości z nowego zbioru
`../combined_als_crowns_no_rectangles_v2`; v1 i źródła pozostają zachowane.
V2 zawiera 100 train / 18 val / 29 test oraz 5071 / 822 / 1028 koron.
LitePT-S trenowany od wag NuScenes przez 80 epok osiągnął na zamrożonym
teście source-balanced PQ **0,3191** i pooled F1 **0,2895**. Przeliczony na
tych samych danych LMF/watershed uzyskał **0,2564** i **0,2001**. Recall:
**0,4426 DL / 0,2928 baseline**.

- Raport: `outputs/combined_full_crowns_no_rectangles_v2/selected/REPORT.md`
- Predykcje: `outputs/combined_full_crowns_no_rectangles_v2/selected/model_test/Segmentation3`
- Excel: `outputs/combined_full_crowns_no_rectangles_v2/experiments.xlsx`
- Checkpoint: `outputs/combined_full_crowns_no_rectangles_v2/selected/best.pt`

Model jest dokładniejszy, lecz wolniejszy od klasycznej segmentacji gotowego
CHM. Test jest historycznie oglądany, a nie dziewiczym holdoutem publikacyjnym.

## Nowa kampania punktowa: połączone dane i pełne korony (2026-09-25)

Wynik zamrożonego testu (99 plików): source-balanced PQ **0,2712 DL / 0,2289
klasyczny**, pooled F1 **0,2114 / 0,1555**, recall **0,3942 / 0,2717**.
Wybrano uśredniony LitePT-S z głowami semantyczną i offsetową. Model jest
dokładniejszy w tym protokole, ale wolniejszy: 20,0 s inferencji z gotowych
punktów wobec 4,5 s klasycznej segmentacji z gotowego CHM (nie end-to-end).
Ważne: anotacje IDTREES są prostokątami, nie dokładnymi maskami koron.
Raport oraz audyt jakości anotacji znajdują się w `outputs/combined_full_crowns_v1/selected`.

Poniższa sekcja dotyczy nowego eksperymentu; starsze sekcje YOLO zachowano
jako dokumentację poprzednich modeli, z innym podziałem i protokołem.

- Dane: `../combined_als_crowns_v1` — fizyczne kopie 584 LAS, podział
  **406 train / 79 val / 99 test** po grupach przestrzennych; szczegóły i
  ograniczenia w README zbioru. Oryginały pozostają bez zmian.
- Konfiguracja: `configs/combined_full_crowns_campaign.json`.
- Rejestr: `outputs/combined_full_crowns_v1/experiments.xlsx`, atomowe rekordy
  JSON, logi epok i checkpointy w katalogach poszczególnych eksperymentów.
- Architektury: LitePT-S z lekkim decoderem maskowym inspirowanym Mask3D;
  PointMLP z decoderem maskowym; LitePT-S z głową semantyczną i offsetami;
  wariant XY oraz uśrednianie wag. Nie jest to oficjalna reprodukcja Mask3D.
- Końcowe metryki: `final_val_*` i `test_*`. Wcześniejsze metryki epok są
  robocze i poprzedzają korektę siatki/obszarów bez anotacji.
- Końcowy wybór i raport, po ewaluacji: `outputs/combined_full_crowns_v1/selected/`.
  Poligony są pełne, dopuszczają nakładanie się koron. Wyniki historyczne
  `run_r` nie są bezpośrednio porównywalne liczbowo; baseline przeliczamy
  na identycznym podziale i pełnych koronach.

Uruchamiaj z głównego katalogu `segm`, w `.venv-gpu` tego projektu:

```bash
DL_model_version/.venv-gpu/bin/python DL_model_version/scripts/build_combined_als.py
DL_model_version/.venv-gpu/bin/python DL_model_version/scripts/validate_combined_als.py
DL_model_version/.venv-gpu/bin/python DL_model_version/scripts/evaluate_combined_full_crowns.py baseline
DL_model_version/.venv-gpu/bin/python DL_model_version/scripts/finalize_combined_experiment.py assess
DL_model_version/.venv-gpu/bin/python DL_model_version/scripts/calibrate_combined_crown_scale.py
DL_model_version/.venv-gpu/bin/python DL_model_version/scripts/finalize_combined_experiment.py freeze
DL_model_version/.venv-gpu/bin/python DL_model_version/scripts/finalize_combined_experiment.py test
DL_model_version/.venv-gpu/bin/python DL_model_version/scripts/report_combined_experiment.py
DL_model_version/.venv-gpu/bin/python DL_model_version/scripts/audit_combined_annotations.py
DL_model_version/.venv-gpu/bin/python DL_model_version/scripts/audit_combined_experiment.py
```

Trening odtwarza się z parametrów w rekordach JSON i checkpointach
każdego wariantu. `freeze` wymaga przewagi na walidacji i zakończenia
treningów; `test` odmawia ponownego wyboru modelu po obejrzeniu testu.
Zależności zainstalowano lokalnie, bez uprawnień administratora.

## Historyczny eksperyment rastrowy

Reprodukowalna adaptacja metody opisanej przez Liu, Zhang i Gao (2025),
*From Crown Detection to Boundary Segmentation*, do zbioru FOR-instance i
standardowego modelu YOLO11s-seg.

Pipeline:

```text
FOR-instance LAS
  -> CHM + gestosc punktow + intensywnosc
  -> obraz 3-kanalowy
  -> YOLO11s-seg
  -> maski instancji i predykcyjne tree_id
  -> metryki detekcji i segmentacji
```

## Zasada braku przecieku

Kanaly wejsciowe nie korzystaja z `treeID`. Pole `treeID` jest czytane tylko
przez historyczny proces budowy etykiet referencyjnych. Skopiowane historyczne
CHM byly wykonane w trybie `annotated-tree`, dlatego sa przechowywane jako
material referencyjny, a nie jako wejscie treningowe nowego modelu.

## Dane odzyskane z istniejacego workflow

- `reused/for_instance_chm_gt_0p5m/` - 32 zestawy CHM i GT (21 dev, 11 test),
  rozdzielczosc 0,5 m; bez zmian wzgledem `run_r/data_input_from_laz_for_instance`.
- `reused/watershed_lmf_for_instance_v2/` - predykcje najlepszego istniejacego
  wariantu LMF/watershed i jego parametry.
- `reused/reference_scripts/` - skrypty pochodzenia danych, segmentacji i oceny.

Oryginalne LAS pozostaja w `../FOR-instance` i nie sa duplikowane.

## Podzial danych

Oficjalny `test` FOR-instance pozostaje testem. Powierzchnie `dev` sa dzielone
na `train` i `val` calymi plikami, nie losowymi fragmentami rastra. Lista
walidacyjna znajduje sie w `configs/default.json`.

## Uruchomienie

Zaleznosci sa rozdzielone na czesc geoprzestrzenna i DL:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-geospatial.txt
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.14.0+cpu torchvision==0.29.0+cpu
.venv/bin/pip install -r requirements-dl.txt
```

W wykonanym eksperymencie preprocessing i eksport korzystaly z istniejacego
srodowiska `treescan`, a trening z lokalnego `.venv`. Dokladny zapis srodowiska
treningowego znajduje sie w `requirements-lock-cpu.txt`.

Przygotowanie danych (dziala w istniejacym srodowisku `treescan`):

```bash
../.tools/miniforge3/envs/treescan/bin/python scripts/prepare_dataset.py
../.tools/miniforge3/envs/treescan/bin/python scripts/create_paper_fusion_dataset.py
../.tools/miniforge3/envs/treescan/bin/python scripts/create_chm_only_dataset.py
```

Pierwsza komenda tworzy bezposrednia reprezentacje fizyczna. Druga tworzy
kontrolowany wariant `paper_fusion`. Poniewaz artykul nie podaje dokladnych
map kolorow ani kodu, wzor fuzji jest jawnie zapisany w
`reports/paper_fusion_method.json` i nie jest przedstawiany jako identyczna
reprodukcja CCD-YOLO. Trzecia komenda tworzy uczciwa ablacje CHM-only: CHM
znormalizowany do zakresu 0--45 m jest powielany do trzech identycznych kanalow
technicznych wymaganych przez standardowe, wstepnie wytrenowane YOLO11.

Trening, po zainstalowaniu zaleznosci DL:

```bash
.venv/bin/python scripts/train_yolo.py --name yolo11s_physical
.venv/bin/python scripts/train_yolo.py \
  --data dataset_paper_fusion/dataset.yaml \
  --name yolo11s_paper_fusion

.venv-gpu/bin/python scripts/train_yolo.py \
  --data dataset_chm_only/dataset.yaml \
  --name yolo11s_chm_only_gpu --device 0
```

Eksperyment z dodatkowym zbiorem `../ideas_als` ma dwa etapy. Do treningu
wlaczono 193 powierzchnie `dev` z `ideas_als` i 16 powierzchni treningowych
FOR-instance; 359 powierzchni oficjalnego testu `ideas_als` pozostawiono poza
treningiem. Najpierw model uczono na zbiorze polaczonym, a nastepnie dostrojono
na FOR-instance z `lr0=0.0001`. Dane i ich pochodzenie sa zapisane w
`reports/ideas_chm_only_dataset.json` oraz
`manifests/dataset_manifest_chm_only_ideas_combined.csv`.

Srodowisko wykorzystane w tym uruchomieniu jest zamrozone w
`requirements-lock-cpu.txt`. PyTorch CPU instalowano z oficjalnego indeksu
`https://download.pytorch.org/whl/cpu`.

Domyslne `imgsz=320` jest celowe: oryginalne rastry maja 56--173 piksele na
bok, wiec wiekszy rozmiar jedynie interpoluje te same dane. Limit treningu to
300 epok z early stopping (`patience=50`).

Ocena zamrozonego modelu na oficjalnym tescie:

```bash
.venv/bin/python scripts/evaluate_yolo.py \
  --weights outputs/training/yolo11s_physical/weights/best.pt

.venv/bin/python scripts/evaluate_yolo.py \
  --weights outputs/training/yolo11s_paper_fusion/weights/best.pt \
  --manifest manifests/dataset_manifest_paper_fusion.csv \
  --data dataset_paper_fusion/dataset.yaml \
  --output-dir outputs/evaluation/yolo11s_paper_fusion

.venv-gpu/bin/python scripts/evaluate_yolo.py \
  --weights outputs/training/yolo11s_chm_only_gpu/weights/best.pt \
  --manifest manifests/dataset_manifest_chm_only.csv \
  --data dataset_chm_only/dataset.yaml \
  --output-dir outputs/evaluation/yolo11s_chm_only_gpu --device 0
```

Predykcje sa zapisywane jako nakladajace sie poligony GeoPackage oraz raster
`tree_id`. W rastrze, ktory nie moze przechowywac dwoch identyfikatorow w jednym
pikselu, wygrywa instancja o wyzszej pewnosci; GeoPackage zachowuje wszystkie
maski.

Szczegoly kazdego uruchomienia sa zapisywane w `reports/` i `outputs/`.

Eksport wynikow przestrzennych i odswiezenie raportu zbiorczego:

```bash
../.tools/miniforge3/envs/treescan/bin/python \
  scripts/export_geospatial_predictions.py \
  --evaluation-dir outputs/evaluation/yolo11s_physical --overwrite
.venv/bin/python scripts/summarize_experiment.py
```

## Wynik wykonanego eksperymentu

Wybor miedzy reprezentacjami wykonano na walidacji: wariant fizyczny uzyskal
F1=0,785, a `paper_fusion` F1=0,776. Zamrozony wariant fizyczny osiagnal na
oficjalnym tescie (11 powierzchni, 278 koron) F1=0,747, SQ=0,805 i PQ=0,601.
Odpowiadajacy wynik `paper_fusion` to F1=0,725 i PQ=0,576, a uczciwy baseline
watershed/LMF: F1=0,519 i PQ=0,390. Pelne wyniki i zastrzezenia sa w
`reports/FINAL_REPORT.md`. Wariant CHM-only wytrenowany na GPU osiagnal na
walidacji F1=0,781, a na tescie F1=0,736, SQ=0,803 i PQ=0,591. Jego najlepszy
checkpoint pochodzi z epoki 214; early stopping zakonczyl trening po 264
epokach (okolo 2,8 min na RTX PRO 500).

Dodanie `ideas_als` nie poprawilo wyniku na zamrozonym tescie FOR-instance.
Model po dostrojeniu uzyskal precision=0,834, recall=0,633, F1=0,720,
SQ=0,802, PQ=0,577, mask mAP50=0,727 i mask mAP50-95=0,384 przy progu 0,43.
Dla porownania pierwotny CHM-only osiagnal F1=0,736 i recall=0,701. Wynik ten
jest raportowany jako negatywny, ale poprawnie wykonany eksperyment, bez
wybierania modelu na podstawie testu.

## Eksperyment bezposrednio na chmurze punktow (LitePT-S)

Dodano eksperymentalny pipeline, ktory nie rasteruje danych przed inferencja.
Wejsciem sieci sa wylacznie `XYZ + znormalizowana intensywnosc`; `treeID` jest
uzywany tylko jako cel uczenia. Backbone to LitePT-S (12,7 mln parametrow) z
glowica klasyfikacji drzewo/tlo oraz regresja przesuniecia XY/Z do centrum
instancji. Glosy na centra sa zamieniane na korony przez mape gestosci,
lokalne maksima i przypisanie punktow. Parametry postprocessingu wybrano tylko
na pieciu powierzchniach walidacyjnych FOR-instance.

Przygotowanie, trening, ocena i eksport benchmarku:

```bash
.venv-gpu/bin/python scripts/prepare_pointcloud_dataset.py

PYTHONPATH=. .venv-gpu/bin/python scripts/train_pointcloud_litept.py \
  --device cuda:0 --epochs 80 --patience 15 --workers 2 \
  --train-repeats 2 --output-dir outputs/training/litept_tree_instance

PYTHONPATH=. .venv-gpu/bin/python scripts/evaluate_pointcloud_litept.py all \
  --weights outputs/training/litept_tree_instance/weights/best.pt \
  --output-dir outputs/evaluation/litept_tree_instance_combined --device cuda:0

PYTHONPATH=. .venv-gpu/bin/python scripts/predict_benchmark_pointcloud.py prepare
PYTHONPATH=. .venv-gpu/bin/python scripts/predict_benchmark_pointcloud.py predict \
  --tile-size 24 --overlap 4 --device cuda:0
PYTHONPATH=. .venv-gpu/bin/python scripts/predict_benchmark_pointcloud.py export \
  --tile-size 24 --overlap 4
```

Oficjalny test FOR-instance (11 powierzchni, 278 koron) dal F1=0,382,
precision=0,325, recall=0,464, SQ=0,633 i PQ=0,242. Sama siec przetworzyla
3,19 mln wokseli z szybkoscia okolo 633 tys. wokseli/s. Jest to wynik wyraznie
slabszy od CHM-only YOLO11s-seg (F1=0,736), dlatego kryterium poprawy jakosci
nie zostalo spelnione i LitePT pozostaje prototypem badawczym, a nie modelem
zalecanym do wdrozenia. Szczegoly sa w `reports/pointcloud_experiment.json`.

Eksperymentalny eksport polskiego obszaru znajduje sie w
`output_14_litept_pointcloud/Segmentation3`. Zawiera 1278 koron i wierzcholkow
w tym samym schemacie GeoPackage co pozostale wyniki. Niska liczba detekcji
wskazuje na przesuniecie domeny i znacznie nizsza gestosc tego ALS; do analiz
produkcyjnych nadal nalezy uzywac `output_10_yolo11s_chm_only`.

### LitePT-S + lekki dekoder Mask3D + density augmentation

Sprawdzono tez dekoder instancji z 64 zapytaniami, trzema warstwami
transformera, dopasowaniem Hungarian oraz stratami object/BCE/Dice. Pierwszy
wariant tworzyl maski globalne i uzyskal tylko okolo F1=0,064. W poprawionym
wariancie kazde zapytanie dostaje kotwice FPS, uczy sie srodka i promienia
korony, a jawny prior przestrzenny ogranicza maske do lokalnego sasiedztwa.
Trening losowo zachowuje 2,5%, 5%, 10%, 25%, 50% albo 100% punktow, aby
symulowac rozne gestosci ALS.

```bash
PYTHONPATH=. .venv-gpu/bin/python scripts/train_pointcloud_mask_decoder.py \
  --device cuda:0 --spatial-prior --epochs 80 --patience 20 \
  --freeze-backbone-epochs 3 --workers 2 --train-repeats 2 \
  --val-repeats 6 --max-points 30000 \
  --output-dir outputs/training/litept_mask_decoder_spatial_prior

PYTHONPATH=. .venv-gpu/bin/python scripts/evaluate_pointcloud_mask_decoder.py all \
  --weights outputs/training/litept_mask_decoder_spatial_prior/weights/best.pt \
  --output-dir outputs/evaluation/litept_mask_decoder_spatial_prior \
  --device cuda:0
```

Najlepszy checkpoint pochodzi z epoki 25: walidacyjne IoU dopasowanych masek
na cropach wynioslo 0,460. Ukonczono 33 pelne epoki; kolejna trafila na
niestabilny batch AMP, ale nie naruszyla `best.pt`. Na pelnych powierzchniach
FOR-instance model osiagnal walidacyjne F1=0,161, a na zamrozonym tescie
F1=0,135, precision=0,091, recall=0,259, SQ=0,578 i PQ=0,078. Inferencja 3,19
mln wokseli z 11 powierzchni zajela 8,76 s czasu GPU (okolo 364 tys.
wokseli/s). Wszystkie testowe GPKG sa w
`outputs/evaluation/litept_mask_decoder_spatial_prior/geospatial`.

Wynik jest lepszy od nieudanego dekodera bez skutecznej lokalnosci, lecz
slabszy od bezposredniej glowicy semantic+offset LitePT (F1=0,382) i CHM-only
YOLO11s-seg (F1=0,736). Dlatego nie nadpisano `output_14` ani rekomendowanego
`output_10`. Pelny raport znajduje sie w
`reports/litept_mask_decoder_experiment.json`.

## Predykcja czterech kafli benchmarkowych

Wynik dla `../run_r/data_input` znajduje sie w
`output_09_yolo11s_physical/Segmentation3`. Ma ten sam schemat nazw, warstw i
atrybutow co `../segmentatiion_benchmark`. Odtworzenie wymaga trzech etapow,
poniewaz zaleznosci GIS/LAZ i PyTorch sa w oddzielnych srodowiskach:

```bash
../.tools/miniforge3/envs/treescan/bin/python \
  scripts/predict_benchmark_tiles.py prepare
.venv/bin/python scripts/predict_benchmark_tiles.py infer
../.tools/miniforge3/envs/treescan/bin/python \
  scripts/predict_benchmark_tiles.py export
```

Runner automatycznie wykrywa pokrywajacy obszar plik LAZ, buduje kanaly gestosci
i intensywnosci oraz wykonuje inferencje w nakladajacych sie oknach 32 m. Wynik
tego uruchomienia zawiera 4268 sparowanych koron i wierzcholkow.

Wynik niezaleznego wariantu CHM-only znajduje sie w
`output_10_yolo11s_chm_only/Segmentation3`. Nie czyta on ALS ani DTM. Komendy:

```bash
../.tools/miniforge3/envs/treescan/bin/python \
  scripts/predict_benchmark_tiles.py prepare --input-mode chm_only \
  --output-dir output_10_yolo11s_chm_only
.venv-gpu/bin/python scripts/predict_benchmark_tiles.py infer \
  --input-mode chm_only \
  --weights outputs/training/yolo11s_chm_only_gpu/weights/best.pt \
  --confidence 0.31 --device 0 --output-dir output_10_yolo11s_chm_only
../.tools/miniforge3/envs/treescan/bin/python \
  scripts/predict_benchmark_tiles.py export --input-mode chm_only \
  --output-dir output_10_yolo11s_chm_only
```

Eksport CHM-only zawiera 3964 sparowane korony i wierzcholki. Suma powierzchni
unikalnych koron odpowiada 64--75% pikseli CHM o wysokosci co najmniej 2 m,
zamiast 12--19% dla poprzedniego wariantu fizycznego. To potwierdza, ze
usuniecie niedopasowanych kanalow ALS wyraznie poprawilo transfer przestrzenny,
choc dla tych czterech kafli nadal nie ma koron referencyjnych pozwalajacych
policzyc F1 lub PQ.

Wyniki modeli trenowanych z `ideas_als` znajduja sie w:

- `output_11_yolo11s_chm_only_ideas_finetuned` — prog 0,43 maksymalizujacy F1
  walidacyjne; 2571 koron, wariant zachowawczy,
- `output_12_yolo11s_chm_only_ideas_recall` — ten sam model, prog 0,20 wybrany
  na walidacji jako kompromis ukierunkowany na recall; 4912 koron,
- `output_13_yolo11s_chm_only_ideas_combined_recall` — checkpoint przed
  dostrojeniem, prog 0,10; 11042 bardzo malych koron i widoczna nadsegmentacja,
  dlatego jest tylko wynikiem diagnostycznym.

Do porownania w QGIS najbardziej uzyteczny z nowych wynikow jest `output_12`.
Ma wiecej malych koron niz pierwotny model, ale pokrywa 60--69% powierzchni
CHM >=2 m, wobec 64--75% dla `output_10`. Zielony watershed nie jest prawda
referencyjna; bez recznych koron GT dla tego obszaru nie mozna uznac jego
dodatkowych poligonow automatycznie za pominiete drzewa.

## Dwie glowice LitePT i eksport do CloudCompare

Wariant `dual_head_satv2_litept_v3` zachowuje dotychczasowa galaz koron i dodaje
dekoder masek punktowych inspirowany SegmentAnyTreeV2 (ISA, maskowana
cross-attention, nadzor one-to-many). [Opis architektury i komendy](reports/dual_head_satv2_design.md)
oraz [wyniki pierwszego treningu](outputs/dual_head_satv2_litept_v3/EXPERIMENT_SUMMARY.md).
Nowa galaz pozostaje eksperymentalna: po 32 epokach uzyskala nizsze walidacyjne
PQ niz stara galaz. Metryki, parametry i checkpointy sa w
`outputs/dual_head_satv2_litept_v3/experiments.xlsx`.

Runner `scripts/predict_dual_head.py` zapisuje do
`output_16_dual_head_satv2_pointcloud`: stare korony w `Segmentation3`, nowe
korony w `PointHead/Segmentation3` i oryginalne punkty z etykietami w
`PointClouds/trees_*.laz`. W CloudCompare wybierz RGB lub pole `tree_id`;
`legacy_tree_id` pozwala porownac przypisania starej galezi. Walidacje zapisu
wykonuje `scripts/validate_dual_outputs.py`.

## Cytowanie wzorca

Liu, Y.; Zhang, A.; Gao, P. (2025). From Crown Detection to Boundary
Segmentation: Advancing Forest Analytics with Enhanced YOLO Model and Airborne
LiDAR Point Clouds. *Forests*, 16(2), 248.
https://doi.org/10.3390/f16020248
