# DualCrown3D — przekazanie sesji i plan dalszych prac

Stan na **2026-10-02**, strefa Europe/Warsaw. Dokument przygotowany na prośbę użytkownika: „zapisz całą sesję i wszystkie pomysły w pliku markdown dla innego agenta”.

To uporządkowana rekonstrukcja kontekstu rozmowy, decyzji, istniejącego kodu i raportów, a nie stenogram wszystkich wypowiedzi. Starsze etapy podsumowano na podstawie dostępnej historii i lokalnych artefaktów. Rozróżniono **wykonane eksperymenty**, **interpretacje wyników** i **propozycje jeszcze niewdrożone**. Nie należy traktować samego istnienia pliku z planem jako dowodu wykonania planu.

## 1. Najważniejsze informacje na start

- Cel użytkownika: segmentacja pojedynczych drzew bezpośrednio z LiDAR, dobra detekcja małych drzew **bez dzielenia dużych koron na liczne fałszywe drzewa**, kompletne obrysy koron, szybka inferencja na GPU.
- Wyniki muszą być dostępne jednocześnie jako chmura LAS/LAZ z `tree_id` do CloudCompare oraz korony/treetopy GeoPackage do QGIS i porównania z klasycznym pipeline’em R.
- **Model20 pozostaje zachowanym modelem produkcyjnym i punktem odniesienia jakościowego.** Model22 lepiej wykrywa małe korony, ale daje dużo nadsegmentacji. Żaden z późniejszych eksperymentów opisanych poniżej nie zastąpił Model20.
- Ostatnio wdrożono i wytrenowano na GPU weryfikator propozycji z pełnych plotów, z negatywnymi przykładami fragmentów i grafem łączenia masek. Poprawił PQ względem Model22, ale prawie usunął detekcje małych koron; **nie przeszedł bramki jakości**.
- Najnowszy pomysł użytkownika: **model pracujący etapami**, który po pierwszej segmentacji ponownie analizuje pozostałe/niepewne obszary. Zaproponowano dwa przebiegi ze wspólnym backbone’em i uczonym drugim dekoderem. **Ten model etapowy nie został jeszcze zaimplementowany ani wytrenowany.**
- Aktualna prośba dotyczyła zapisania przekazania. W ramach tworzenia tego dokumentu nie uruchamiano nowego treningu, inferencji ani push do GitHub.
- Repozytorium jest mocno zmienione lokalnie: wiele plików jest modified/untracked. Nie resetować worktree. Samo sklonowanie GitHub nie odtworzy obecnego stanu eksperymentów.

## 2. Użytkownik, preferencje i zakres projektu

Użytkownik pracuje po polsku, ogląda wyniki w QGIS i CloudCompare, oczekuje prostych wyjaśnień oraz konkretnego działania po zatwierdzeniu eksperymentu. Zależy mu na jakości całych instancji, a nie na samym procentowym pokryciu punktów kolorami. Wielokrotnie podkreślał kompletność koron, wykrywanie małych drzew i brak nadsegmentacji.

Wcześniejsze wymagania:

1. Korzystać z istniejących danych i przetworzonych CHM, nie powielać niepotrzebnie przygotowania.
2. Trenować na GPU; użytkownik nie ma uprawnień administratora. Pakiety instalować w środowisku użytkownika/venv, jeśli potrzebne.
3. Łączyć FOR-instance i ideas_als przy zachowaniu train/val/test i pochodzenia danych.
4. Wykluczyć wadliwe prostokątne etykiety koron. Nie przywracać ich przypadkiem przez starszy manifest.
5. Zachować format i nazewnictwo wyników kompatybilne z benchmarkiem R.
6. Każdy eksperyment/inferencja w oddzielnym katalogu, z parametrami, metrykami i checkpointami; preferowane również Excel/JSON.
7. Długoterminowy cel: przekroczyć najlepszy klasyczny wynik z `run_r`, ale porównanie musi używać tego samego zbioru i protokołu. Nie wykazano tego przez prostą konfrontację liczb pochodzących z różnych raportów.
8. Użytkownik prosił wcześniej o commit/push; repo istnieje. Ostatnie zadanie to dokument przekazania, nie kolejny commit/push.

Główne instrukcje projektu są w `/home/maciej.wielgosz/Projects/segm/AGENTS.md`. Dotyczą pierwotnego workflow R: chronić rastry/VRT/XML/GPKG, zachować nazwy kafli, uważać na zapisy zastępujące GeoPackage. Zawarte tam stwierdzenie o braku testów odnosi się do R — podprojekt DL ma już testy Python.

## 3. Ścieżki i stan repozytorium

W dalszej części `ROOT` oznacza `/home/maciej.wielgosz/Projects/segm`, a ścieżki względne bez `../` odnoszą się do `ROOT/DL_model_version`.

| Element | Lokalizacja |
|---|---|
| Projekt DL | `/home/maciej.wielgosz/Projects/segm/DL_model_version` |
| Repo GitHub | `git@github.com:maciekwielgosz/dualcrown3d.git` |
| Ostatni lokalny commit przy przekazaniu | `fca1d72 feat(pointcloud): add joint training and supervision v4 workflows` |
| Surowy FOR-instance | `ROOT/FOR-instance` |
| Surowe ideas_als | `ROOT/ideas_als` |
| Aktualne przygotowane dane z poprawioną superwizją | `ROOT/combined_als_crowns_supervision_v4/manifest.csv` |
| Starsze wersje danych | `ROOT/combined_als_crowns_v1`, `combined_als_crowns_no_rectangles_v2`, `combined_als_crowns_augmented_v3`, `combined_als_instance_v1` |
| Klasyczny workflow i optymalizacja | `ROOT/run_r` |
| Wejście referencyjnej sceny użytkownika | `ROOT/run_r/data_input` |
| Klasyczny benchmark wizualny | `ROOT/segmentatiion_benchmark/output_00_baseline/Segmentation3` — pisownia `segmentatiion` jest rzeczywista |
| Przygotowana scena ALS używana przez nowe inferencje | `output_15_litept_v2_no_rectangles_pointcloud/work/benchmark_pointcloud.npz` i `preparation.json` |
| Pierwotny dokument pomysłu ALS/TLS | `/home/maciej.wielgosz/Downloads/ALS_TLS_superpointy_HELIOS_plan.md` |
| Plan implementacyjny w repo | `reports/als_tls_superpoint_implementation_plan.md` |

Stan Git sprawdzony 2026-10-02: zmienione m.in. `README.md`, `pointcloud/data.py`, `decoder_v4.py`, `dual_fusion.py`, `dual_head.py`, `instance_output.py`, `joint_training.py`, `scripts/predict_dual_head.py` i `prepare_treescan_helios_dualcrown.py`. Niezacommitowane są m.in. moduły shared-v5, `pointcloud/superpoints/`, skrypty nowych eksperymentów, raporty i testy. Przed jakimikolwiek zmianami odczytać `git status --short` i zachować istniejące prace. Nie dodawać do Git danych, wag, venv ani wygenerowanych chmur.

## 4. Środowisko wykonawcze

Zweryfikowane przez odczyt środowiska podczas przygotowania dokumentu:

| Element | Wartość |
|---|---|
| Interpreter | `DL_model_version/.venv-gpu/bin/python` |
| Python | 3.12.3 |
| PyTorch | 2.14.0+cu130 |
| CUDA w buildzie PyTorch | 13.0 |
| `torch.cuda.is_available()` | `True` |
| GPU | NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU |
| VRAM zgłaszany przez PyTorch | około 5.54 GiB, karta klasy 6 GB |
| NumPy / SciPy | 2.5.2 / 1.18.1 |
| laspy / rasterio / geopandas / openpyxl | 2.7.0 / 1.5.1 / 1.1.4 / 3.1.5 |

Historyczne problemy ze sterownikiem i brakiem `torch` w systemowym Pythonie nie są aktualną blokadą. Używać `.venv-gpu`, nie zakładać, że polecenie `python` wskazuje ten interpreter.

Pakiety partycjonowania SPT/EZ-SP są osobno w `/tmp/dualcrown3d_superpoint_deps_20261001` (katalog istniał przy przekazaniu). Używano `pycut-pursuit==0.1.4`, `torch-graph-components==0.1.1`, `pygrid-graph==0.0.4`, izolowanych `numpy==1.26.4` i `scipy==1.15.3` oraz shim dla scatter. Nie zamieniać bez analizy NumPy w głównym venv. `/tmp` jest nietrwałe — może wymagać odtworzenia. Szczegóły: `reports/superpoint_spt_ezsp_kernel_comparison.md`, `scripts/benchmark_superpoint_algorithms.py`.

Podczas importów pojawia się komunikat, że natywne CUDA `pointrope` jest niedostępne; działa fallback PyTorch. Nie oznacza to automatycznie treningu na CPU. Uwzględniać to przy pomiarach czasu.

W tej sesji odczyt przez standardowy sandbox narzędzia kończył się błędem `error building bubblewrap command: mountinfo path is not absolute`. Polecenia wykonywano przez mechanizm podwyższonego dostępu narzędzia — nie przez `sudo`. `apply_patch` działał dla nowych plików, a aktualizacja istniejących czasami wymagała fallbacku `patch` po udokumentowanym błędzie narzędzia. Preferować `apply_patch`, gdy działa; nie zmieniać globalnych zabezpieczeń.

W odczycie procesów nie znaleziono aktywnego treningu/inferencji z tego zadania. To stan chwilowy, sprawdzić ponownie przed zajęciem GPU.

## 5. Dane i protokół — nie pomylić wersji

### Aktualny manifest realnych danych

`../combined_als_crowns_supervision_v4/manifest.csv` zawiera 147 rekordów. Kolumna splitu nazywa się **`model_split`**, nie `split`.

| Zakres | Train | Validation | Test |
|---|---:|---:|---:|
| Wszystkie rekordy manifestu | 100 | 18 | 29 |
| `point_eval_eligible == 'true'` | **79** | **14** | **24** |

Pozostałe 30 rekordów ECODSE mają słabą referencję 2D, nie są główną superwizją/ewaluacją 3D. Nie traktować projekcji poligonów jako równoważnych natywnym etykietom punktów.

Kolekcje w manifestach: FOR-instance — CULS, NIBIO, RMIT, SCION, TUWIEN; ideas_als — CEDAR_CYPRESS, ECODSE, FGI_EMIT, LA_PALMA, SEPILOK, WILDFOREST3D, WYTHAM. Dane są heterogeniczne pod względem sensora, lasu, gęstości i zakresu anotacji.

Konwencja v4: `tree_id = -1` oznacza nieznane/nieanotowane, `0` znane tło, wartości dodatnie instancje drzew. Nie zamieniać `-1` na tło. Używać m.in. `train_eligible`, `point_eval_eligible`, `ignore_vector`, `group_id`, `source_dataset`, `collection`, `annotation_protocol`. W starszych preparacjach ignorowane punkty były usuwane i nie było wartości ujemnych — nie mieszać tych konwencji.

Audyt 79 treningowych plotów: percentyle 10/50/90 surowej gęstości około 127/943/5978 pkt/m²; po wokselizacji 0.25 m około 71/106/386 wokseli/m². Mediana udziału nieznanych etykiet około 49.6%. To duża różnica względem docelowej sceny około 14 pkt/m². Nie nazywać całej puli jednorodnym zbiorem skanów z jednego sensora ALS.

### Pobranie zbioru nie oznacza jego użycia w ostatnim treningu

`../external_lidar_datasets/README.md` opisuje pobrane i przygotowywane dane:

- DALES-2: 29 oficjalnych kafli train; 11 test pozostawionych osobno. Instancje drzew z natywnego pola `instance`, klasa semantyczna 5.
- SyntheticForest: trzy symulowane chmury UAV, raportować oddzielnie jako syntetyczne.
- TreeScanPL10k: 28 paczek, źródłowo 272 polskie ploty i 10 417 instancji; TLS, nie natywny ALS. W dawnym przygotowaniu brano drzewa `completelyInside == 1`.
- TLSBenchmark: Litchfield, Ofental, Robson Creek, oficjalne pliki treningowe; Wytham nie dublowano.
- AvocadoTrees: ręczne skany sadownicze i powtarzane daty, inna domena.
- LaPalma pobrana, ale duplikat kolekcji już występującej w ideas_als został wykluczony z manifestu rozszerzonego.

Przygotowane rozszerzenie aerial: `../combined_als_crowns_augmented_v3/external_train/`; osobna pula TLS: `../external_tls_tree_instances_v1/`. Aktualny eksperyment full-plot v1 korzystał z 79/14 natywnych rekordów v4, nie z całej puli pobranych zbiorów. Nie twierdzić, że ostatni model trenowano na każdym pobranym zbiorze.

### HELIOS i własna symulacja TLS → ALS

- Sceny/HELIOS++: `../TreeScanPL10k_HELIOS_ALS_v1`, runtime 2.2.2, 62 przygotowane ploty.
- Dane modelowe: `../dualcrown3d_treescan_helios_v1`: 43 train, 12 val, 7 test.
- Dodatkowe loty: `../TreeScanPL10k_HELIOS_ALS_flight_augmentation_v2`: 86 nowych widoków dla 43 rodziców train; łącznie 129 treningowych wariantów lotu.
- Wszystkie widoki jednego plotu muszą pozostawać w tym samym splicie. Dotyczy też TLS-teachera.
- Kalibracja HELIOS celowała w około 13.97 pkt/m² i 1.79 odbicia/impuls, podobnie do wskazanej sceny użytkownika. To nie identyfikuje rzeczywistego skanera i nie dowodzi zgodności penetracji koron.
- Źródłowa scena HELIOS używała pełnych drzew i nieprzezroczystych wokseli XYZ 0.10 m; problem zasłaniania, usuniętych niekompletnych drzew i jakości geometrii pozostaje istotny.
- Nie wykazano dostępności zarejestrowanych, czasowo zgodnych **rzeczywistych** par TLS/ALS ze wspólnymi tree ID. Para TLS/syntetyczny ALS nie jest takim dowodem.
- Własny pilot widoczności: `scripts/simulate_tls_als_pilot.py`; dwa pasy nadlotu, komórki wiązki/warstwy wysokości, przybliżenie nieprzezroczystości i transmisji, do czterech odbić. Nie jest pełnym symulatorem waveform/radiometrii.
- Sześć rodziców pilota: Gorlice, Herby, Katrynka, Milicz, Piensk, Suprasl. Real-only, HELIOS, thinning i dwa warianty symulacji porównano przy tym samym krótkim budżecie. Żaden nie poprawił jednocześnie wszystkich wymaganych metryk; zachowano checkpoint początkowy.
- Raport: `reports/tls_visibility_pilot.md`, wyniki/Excel: `outputs/tls_visibility_pilot_review_v2/`.

Pełne korony z TLS to referencje geometryczne, nie automatycznie idealne ręczne granice widoczne w ALS. Drzewo bez żadnego odbicia ALS nie powinno być zwykłym dodatnim targetem segmentacji punktów. Dobór reprezentantów wokseli nie może zależeć od GT etykiety; w starszym konwerterze HELIOS znaleziono taką preferencję i nowy pilot stosował wybór niezależny od labeli. Sprawdzać wersję przygotowania danych.

## 6. Historia rozwoju i mapowanie wyników

1. Początek: rasteryzacja CHM + gęstość + intensywność → YOLO11s-seg → maski `tree_id`, inspiracja publikacją MDPI. Rozważano watershed/LMF, YOLOv5 Straker, Mask R-CNN z CBAM/GCNet Liao, CCD-YOLO i YOLO11n/s-seg. Wstępne cytowane liczby z publikacji nie są wynikami tego repo i nie są tu ponownie zweryfikowane.
2. Transfer YOLO na CHM użytkownika dawał małe/niepełne obrysy. Użytkownik poprosił o CHM-only: jeden znormalizowany CHM powielony do trzech kanałów technicznych, trening GPU. Historyczny wynik: `output_10_yolo11s_chm_only/`.
3. Dołączono ideas_als i rozpoczęto modele działające na chmurach: rozważano PTv3, LitePT i Superpoint Transformer; kierunek wykonany to LitePT-S z lekkim dekoderem inspirowanym Mask3D oraz augmentacją gęstości.
4. Złączono dane i wydzielono splity, usunięto prostokątne korony z aktywnego przygotowania. Nie mylić oryginalnych danych z przygotowanymi wersjami bez prostokątów.
5. Dodano gałąź instancji punktowych obok wcześniejszej gałęzi koron/center votes, inspirując się SegmentAnyTreeV2. Powstał DualCrown3D i repo GitHub.
6. Użytkownik widział „ser”, niebieskie/szare łaty i brak przypisań. Rozdzielono problem zakresu wyświetlania scalar field w CloudCompare od realnych braków w maskach. Wdrożono support fusion i później consensus obu głowic oraz lokalne odzyskiwanie punktów.
7. Powstały wyniki 16/17/19, potem fine-tuning real + HELIOS, pięcioelementowa kampania treningowa i wynik 20.
8. Poprawiono protokół v4 (nieznane etykiety, rodzaje referencji, anotacje) i próbowano nowych lossów/decoderów. Nowy kod nie implikuje nowych lepszych wag: wiele selekcji pozostaje w epoce 0.
9. Wdrożono shared-v5: jedna instancja/query dla punktów i korony BEV; zgodność ID poprawna, skuteczność słaba. `output_21` to eksperyment, nie produkcyjny następca 20.
10. Kalibracja progów Model20 nie dała bezstratnej poprawy małych drzew. Użytkownik zaproponował większą zmianę z superpointami i TLS/ALS.
11. Wykonano audyt, proste grupowanie geometryczne, uczone affinity, testy SPT Cut Pursuit i EZ-SP. Partycje nie przeszły bramki granic, ale użytkownik następnie wyraźnie zgodził się na ograniczony eksperyment dekodera mimo tego wyniku.
12. Pilot dekodera i potem większy graph/decoder Q128 dały poprawę na cropach. Zrobiono inferencję 22 i porównanie testowe 23.
13. Użytkownik pokazał nadsegmentację, szczególnie prostego plotu CULS. Przetestowano ochronę kotwic, dodawanie/resplitting małych koron, score heads, a następnie full-plot verifier. Żaden nie spełnił łącznego celu.
14. Najnowsza rozmowa: propozycja iteracyjnego/dwuetapowego modelu; szczegółowy projekt dalej w dokumencie.

| Folder | Co zawiera / jak interpretować |
|---|---|
| `output_16_dual_head_satv2_pointcloud` | Historyczna dwugałęziowa inferencja, na której oglądano braki w CloudCompare |
| `output_17_dual_head_support_fusion` | Historyczna poprawka zachowania uzupełniającego wsparcia masek |
| `output_19_dual_head_complete_consensus` | Consensus obu gałęzi przed późniejszym wspólnym fine-tuningiem |
| `output_20_dualcrown3d_joint_finetune` | Zachowany model produkcyjny; cztery kafle referencyjnej sceny |
| `output_21_shared_instances_v5_experimental` | Wspólne queries point/BEV, funkcjonalny eksport; słabsza jakość; wczesna konfiguracja sprzed poprawki geometrii/kalibracji |
| `output_22_ezsp_wide_q128_experimental` | Większy dekoder EZ-SP Q128, wykrywa więcej małych drzew i silnie nadsegmentuje |
| `output_23_labeled_test_output20_vs_output22` | Te same 24 testowe ploty z labelami dla Model20 i Model22; XLSX/JSON, LAS/LAZ, GPKG |
| `output_24_guarded_small_crown_fusion` | Walidacja dodawania małych kandydatów do kotwic Model20; nie nowy lepszy LAS testowy |
| `output_25_guarded_internal_small_splits` | Walidacja ograniczonych podziałów wnętrza koron; niepromowana |
| `output_26_low_confidence_small_split_verification` | Geometria/niskie confidence i podziały; niepromowane |
| `outputs/dualcrown3d_fullplot_verifier_v1` | Ostatni wykonany trening weryfikatora i audyt; bez promocji |

## 7. Architektury — są trzy różne „ostatnie modele”

### A. Model20: zachowany DualCrown3D

- Wejście: geometria XYZ z wysokością nad gruntem i intensywność, woksele 0.25 m.
- Okna 20 × 20 m, overlap 8 m. Typowy krok to 12 m; na brzegach obowiązują reguły własności/przedziałów z kodu, nie proste ucinanie kafli.
- LitePT-S dostarcza 72-D cechy punktów.
- Gałąź semantyczna + offset do środka → center-vote clustering → korony.
- Gałąź masek: kontekst wieloskalowy 0.75/2/6 m, residual MLP width 128, embedding instancji, transformer width 128, 3 warstwy, 4 heads, do 96 queries, 1024 memory tokens. Inspiracja SATv2, nie dokładna reprodukcja.
- Consensus `dual_consensus_v3` używa kotwic vote, uzupełnień masek, niezależnych mask-only drzew i ograniczonego lokalnego odzyskiwania punktów.
- Około 13.78 mln parametrów całej sieci według README. To nie liczba parametrów samego większego eksperymentalnego dekodera.
- W początkowym dwugałęziowym treningu LitePT był zamrożony; w późniejszej kampanii joint trenowano również część backbone’u. Nie nazywać całej historii modelem z zawsze zamrożonym encoderem.

Oryginalny checkpoint Model20, z raportu inferencji:

```text
outputs/dualcrown3d_joint_campaign_v1/runs/repeat_20260930/weights/best.pt
SHA256: e0ffb4134a4661266699b38517a15908f6a5aadc432a5260fc7168906dab6688
selection: outputs/dualcrown3d_joint_campaign_v1/runs/repeat_20260930/selected.json
```

Wybrany checkpoint pochodzi z epoki 32. W v4 jest zgodny wagami artefakt epoki 0 z nowymi metadanymi:

```text
outputs/dualcrown3d_supervision_v4/runs/data_only/weights/best.pt
SHA256: add9230fc38a25e2c6f3ae05fe4425e4067f8052254cab8f088e5af9ec34588d
selection: outputs/dualcrown3d_supervision_v4/selected.json
```

Różne SHA całego pliku nie przeczą zgodności tensorów wag — metadane/serializacja mogą być różne. Weryfikować tensorowo, gdy to potrzebne. Historyczne domyślne argumenty skryptów mogą wskazywać starsze checkpointy: zawsze przekazywać jawnie checkpoint i selection.

### B. Model22: EZ-SP wide Q128

```text
LitePT-S (zamrożony, 72-D cechy)
 → embedding granic instancji + partycja EZ-SP
 → pooling superpointów
 → graph: 4 warstwy, width 192
 → residual do indywidualnych punktów
 → wcześniejszy decoder: 3 warstwy, width 128
 → 2 dodatkowe mask-attention layers, width 192
 → 128 queries, 1536 memory tokens
 → maski punktowe + score
 → łączenie okien i poligony z finalnych ID
```

Checkpoint: `outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot/runs/ezsp_large_w192_q128/weights/best.pt`, **epoka 12**.

Embedding: `outputs/dualcrown3d_superpoint_v1/stage2_instance_embedding/embedding.pt`. Oryginalny checkpoint backbone’u wskazuje pole `initial_checkpoint` w payloadzie modelu; sprawdzany jest hash. Główne pliki: `pointcloud/superpoints/model.py`, `decoder.py`, `scripts/train_superpoint_decoder_pilot.py`, `scripts/predict_superpoint_wide_scene.py`.

Dekoder ma około 3 056 279 parametrów. Pilot trenował dekoder na zamrożonych cechach; nie był nowym treningiem całego LitePT. Model22 ma już warstwy score (`decoder.score`, `decoder.wide.score`), więc problemem nie jest sam brak `score head`. Nie posiada niezależnej pełnej maski BEV z v5. Poligony powstają z końcowej segmentacji punktów. Loss pilota nie zawierał bezpośredniego lossu pełnej korony z poligonu.

### C. Ostatni full-plot verifier v1

To dodatkowy weryfikator nad **zamrożonym Model22**, nie nowo wytrenowany backbone ani nowy dekoder punktowy. Szczegóły i jego negatywny wynik w sekcji 10.

## 8. Metryki — zakres porównania jest częścią wyniku

SB-PQ/SB-F1 oznaczają agregację z równym udziałem kolekcji źródłowych, a nie prostą średnią po wszystkich drzewach. Oddzielać to od pooled precision/recall/F1. Matching instancji używa IoU >=0.5. Różne eksportery koron mogą zmienić crown PQ nawet przy tych samych wagach.

### Pełne 14 plotów validation v4

| Model / eksperyment | Point SB-PQ | Crown SB-PQ | Małe korony <=10 m² |
|---|---:|---:|---:|
| Model20 | **0.4051** | **0.4442** | 6/161 |
| Model22 | 0.1243 | 0.1967 | **20/161** |
| Shared-v5 | 0.1610 | 0.2066 | Nie wpisywać brakującego wyniku jako 0 |
| Output24: chronione dodawanie | 0.4051 | 0.4442 | 6/161 |
| Output25: ograniczone rozdzielanie | 0.4005 | 0.4386 | 8/161 |
| Output26: geometria + niski score | 0.3905 | 0.4227 | 11/161 |
| Full-plot verifier p=0.30, claimed=0.15 | 0.1995 | 0.2491 | 1/161 |
| Full-plot verifier p=0.15, claimed=0.15 | 0.1960 | 0.2480 | 3/161 |

Model20 na tym protokole ma point/crown SB-F1 0.5165/0.6195. Małych koron <=4 m² jest 39; w bazowym Model20 trafiono 0.

**Poprawa względem bardzo słabego Model22 nie oznacza pobicia Model20.**

### Dopasowane porównanie 24 plotów testowych — output23

| Metryka | Model20 | Model22 |
|---|---:|---:|
| Point SB-PQ | około 0.4147 | około 0.1429 |
| Crown SB-PQ | około 0.4170 | około 0.2268 |
| Point SB-F1 | 0.520 | 0.198 |
| Crown SB-F1 | 0.584 | 0.333 |
| Point pooled precision / recall | 0.450 / 0.522 | 0.100 / 0.321 |
| Crown pooled precision / recall | 0.547 / 0.528 | 0.182 / 0.500 |
| Korony <=4 m² | 0/65 | 5/65 |
| Korony <=10 m² | 19/241 | 40/241 |
| Liczba prognozowanych instancji | 1 358 | 4 145 |

Źródło: `reports/output20_output22_heldout_test.md`, `output_23_labeled_test_output20_vs_output22/comparison.json` i `.xlsx`. To ocena całych pipeline’ów, z właściwą im polygonizacją. README zawiera też starsze liczby 0.4180/0.4220 dla ponownej oceny zachowanych wag. Nie scalać tych liczb ani przedstawiać różnicy jako efektu treningu — używać jednego wskazanego raportu/protokołu.

Test był oglądany historycznie i nie jest nowym nietkniętym holdoutem publikacyjnym. Nie używać go teraz do strojenia. Późniejsze output24–26 i full-plot v1 oceniano/wybierano na validation, bez nowej testowej inferencji odrzuconych modeli.

### Wyniki cropów i oracles to inna skala

- Pilot Q128, 14 cropów 20 m: point PQ 0.2608, crown PQ 0.2275, joint score 0.2436, sparse-tree recall 0.1467. To nie full-plot validation 0.1243/0.1967.
- Po 12 epokach mały EZ-SP decoder miał crop point/crown PQ 0.2259/0.1828. Większy Q96 był gorszy; lepszy Q128 zmieniał jednocześnie szerokość, liczbę queries i memory tokens. Nie przypisywać całego zysku samej szerokości.
- Partition oracle używa GT do przypisania grup: np. fixed 0.5 m miał oracle PQ 0.9759 i boundary recall 0.9456 przy kompresji 1.638×. To diagnostyka możliwości partycji, nie predykcyjna jakość modelu.
- Najlepsze SPT/EZ-SP kernels przy podobnej kompresji miały oracle PQ około 0.9774/0.9768, ale gorszy boundary recall 0.8736/0.8689. Były to oficjalne algorytmy partycji, nie kompletne modele z publikacji.
- Użytkownik później zatwierdził Stage3 mimo nieprzejścia bramki partycji. Starsze raporty mówiące „nie przechodzić do Stage3” opisują ówczesny stan i nie są bieżącą blokadą.

## 9. Zdiagnozowane problemy i nieskuteczne próby

### Nadsegmentacja Model22

`pointcloud/instance_output.py` przy historycznym scalaniu akceptuje propozycję, jeśli nie jest zbyt mocno zajęta przez wcześniejsze maski, i nadaje **nowy ID jej pozostałym punktom**. W Model22 próg `merge_overlap` wynosi 0.6. Późniejsze `support_fusion_v2` dodaje punkty do istniejących kotwic, ale nie scala dwóch już zaakceptowanych fragmentów w jedno drzewo.

Na pokazanym testowym CULS: GT 21 koron, Model20 42 instancje, Model22 94. W 17 z 21 referencyjnych koron występowały co najmniej dwa ID Model22 z >=100 wokselami każde, miejscami do sześciu fragmentów. To potwierdzona nadsegmentacja, nie tylko kolory w CloudCompare.

### Małe drzewa nie są tylko w pustych miejscach

Na validation 87/161 małych koron miało już >=80% swoich punktów przypisanych przez Model20. Są więc często wchłonięte przez inne instancje. Drugi przebieg wyłącznie po `tree_id==0` ma istotne ograniczenie.

Wśród małych trafionych kandydatów Model22 medianowy złożony score (object × mask) wynosił 0.104, wobec 0.144 innych kandydatów. Zwiększanie progu score może wcześniej usunąć prawdziwe małe drzewa niż fałszywe fragmenty. Score nie jest skalibrowanym prawdopodobieństwem poprawności.

### Co już próbowano

- 15 ustawień progów Model20: minimalna liczba wokseli, wysokość, powierzchnia, mask/object, vote i peak. Jeden wariant miał minimalnie wyższe PQ 0.4080/0.4448, ale nadal 6/161 małych drzew; wariant z 7/161 pogorszył jakość. Brak promocji.
- Output24: dodawanie tylko prawie niezależnych nowych koron do zachowanych kotwic — brak dodatkowych małych trafień.
- Output25/26: dopuszczenie ograniczonych podziałów wnętrza kotwic i kandydatów z niskim score — więcej małych trafień, ale gorsza precyzja/PQ.
- Trening całego eksperymentalnego dekodera z duplicate-aware supervision: crop joint score 0.2436 → 0.3096, ale sparse recall 0.1467 → 0.0267. Selekcja została przy epoce 0.
- Trening samych `decoder.score` i `decoder.wide.score`: również nie przeszedł bramki sparse recall; nie uruchamiano go na teście jako lepszego modelu.
- Shared-v5: wspólne ID dla point/BEV udało się technicznie, ale false positives i słaba jakość pozostały.
- Dłuższy trening, szerszy decoder i więcej syntetycznych danych nie dały automatycznie poprawy obu rodzajów PQ; konieczna kontrola całych plotów i małych drzew.

Raport zbiorczy: `reports/output22_oversegmentation_small_tree_experiments.md`. Artefakty GPU: `outputs/dualcrown3d_small_tree_quality_v1/`, `outputs/dualcrown3d_score_head_small_trees_v1/`.

## 10. Ostatni wykonany krok: full-plot graph + learned verifier

Raport: `reports/fullplot_fragment_verifier_experiment.md`.

### Implementacja

| Plik | Rola |
|---|---|
| `scripts/cache_fullplot_proposals.py` | Zamrożony Q128, cache surowych propozycji pełnych plotów train/val na GPU, hashe wejścia/modelu |
| `pointcloud/superpoints/fullplot_graph.py` | Graf nakładania masek, union-find, cechy 14-D i targety IoU |
| `pointcloud/superpoints/fullplot_verifier.py` | MLP 14→64→32→1 i selekcja masek |
| `pointcloud/superpoints/graph_geometry.py` | Obrys z occupancy 0.5 m, zamykanie małych szczelin, wypełnianie dziur, dopasowanie do GT |
| `scripts/train_fullplot_mask_verifier.py` | Przygotowanie grafów, 50 epok CUDA, selekcja i 15 konfiguracji walidacyjnych |
| `scripts/audit_fullplot_mask_recall.py` | Audyt możliwości pełnych masek na 14 plotach val |
| `tests/test_fullplot_verifier.py` | Trzy testy: rozmiar masek, brak nowego ID z odrzuconej resztki, point-IoU target |

Cache pochodzi z `wide_predict_plot` w `scripts/evaluate_output20_output22_test.py` mimo nazwy tego modułu: w tym eksperymencie caller używał tylko train i val. Funkcja ma okna 20/8 m i limit 12 000 punktów w użytym wywołaniu. Zapisuje kandydatów po wstępnych progach object>=0.1, mask>=0.2 i filtrze własności środka w oknie; nie są to dosłownie wszystkie query każdego kontekstowego okna. Przy projektowaniu nowego stitchera sprawdzić ten filtr. Cache nie zawiera wszystkich informacji potrzebnych przyszłemu rekurencyjnemu dekoderowi.

Graf v1 tworzy węzły mask>=0.5, min 8 punktów, wysokość >=1.5 m. Łączy węzły przy overlap/min>=0.55, IoU>=0.20, dystansie centrów XY<=4 m i różnicy maksymalnej wysokości<=3 m. Składowa otrzymuje union punktów i maksymalne prawdopodobieństwo wsparcia. Union-find może tworzyć niepożądane łączenia przechodnie.

Cechy: log liczby punktów, log liczby komórek XY, max/std wysokości, mean/std mask score, max/mean object score, log liczby propozycji, compactness/aspect XY, offset wierzchołka, gęstość punktów, member agreement.

Target: dodatni przy GT point IoU>=0.5; kandydaci z większością punktów unknown są ignorowani; negatywy o IoU w [0.1,0.5) mają wagę ×2. Model uczy się oceniania propozycji, nie poprawia ich kształtu. Nie jest to jeszcze supervision decyzji „samodzielne małe drzewo vs fragment” w drugim przebiegu z kontekstem.

79 plotów train: 17 992 klastry, 1 021 pozytywnych, 7 436 negatywnych (w tym 3 435 hard negatives), 9 535 ignorowanych. 14 val: 3 020 klastrów, 164 pozytywne, 1 228 negatywnych, 1 628 ignorowanych. Po wyłączeniu unknown trening używał 8 457 kandydatów, walidacja 1 392.

Trening: 50 epok CUDA, AdamW lr=1e-3, weight_decay=.01, seed 20261001, source weighting i bilans klas. Najlepsza epoka 17, val proposal average precision 0.542919 — to AP klasyfikatora propozycji, nie instance AP/PQ całego lasu.

Checkpoint: `outputs/dualcrown3d_fullplot_verifier_v1/weights/best.pt`, SHA256 `a1d2943bcf8f5a5d4c541330f4f1ed35b1e9ae01c1a38aa4384cb184c122f15d`.

### Wynik i bramka

Wypróbowano 5 progów probability ×3 progi zajętości maski. Bramka v1: poprawić point/crown PQ względem Model22 i zachować co najmniej 16/161 małych trafień (80% jego 20). **Żaden wariant nie przeszedł, `validation_selection.json` ma `selected: null`.** Była to tylko wstępna bramka względem Model22, słabsza niż docelowe wymaganie niepogorszenia Model20.

W wybranej do opisania konfiguracji p=.30, claimed=.15: PQ wzrosło do 0.1995/0.2491, małe korony spadły do 1/161. P=.15 daje 3/161, nadal zdecydowanie za mało.

### Audyt pełnych masek

Na wszystkich 14 val, przy sprawdzaniu pełnego obrysu kandydata przed konkurencją punktów:

| Wariant | Małe GT mające pasującego kandydata z IoU>=0.5 |
|---|---:|
| Domyślny graf v1 | 5/161 |
| Graf z minimalnym stosunkiem rozmiarów 0.85 | 13/161 |
| Bez łączenia surowych masek | 15/161 |
| Finalna segmentacja historycznego Model22 | 20/161 rzeczywistych trafień |

To **potencjał pełnych masek**, a nie wynik predykcyjny ani uniwersalna granica możliwości dowolnego dekodera. Konkurencja/przycinanie punktów zmienia kształt instancji i może poprawić dopasowanie. Na FGI_EMIT plot1018 Model22 trafia 12/114 małych koron, podczas gdy pełne niescalone maski dają 5/114 potencjalnych trafień. Na Saiki3 graf domyślny ma 4/11, size-aware 8/11, niescalone 8/11, finalny Model22 6/11.

Wniosek: część prawdziwych małych koron wyłania się z resztek po konkurencji masek. Całkowity zakaz tworzenia instancji z resztki usuwa zarówno artefakty, jak i poprawne detekcje. Potrzebna jest uczona decyzja z kontekstem lub lepsza generacja pełnych masek małych drzew.

Dodano opcjonalne `config.get('link_min_size_ratio', 0.)` do grafu. Domyślnego `DEFAULT_GRAPH` nie zmieniono; stare wyniki v1 pozostają odtwarzalne. W audycie 1.01 oznacza brak możliwych krawędzi. Stosunek rozmiarów nie zapobiega całej przechodniości grafu. Nie trenowano nowego v2 z tym ograniczeniem.

### Stan artefaktów i reprodukcja

```text
outputs/dualcrown3d_fullplot_verifier_v1/
  configuration.json
  raw/train/*.npz                  # 79 plotów
  raw/val/*.npz                    # 14 plotów
  graph/train/*.npz
  graph/val/*.npz
  graph_protocol.json
  graph_summary.json
  weights/best.pt
  verifier_selected.json
  training_history.json
  validation_selection.json       # selected == null
  validation_full_mask_audit.json
```

```bash
cd /home/maciej.wielgosz/Projects/segm/DL_model_version
.venv-gpu/bin/python scripts/train_fullplot_mask_verifier.py --phase all
.venv-gpu/bin/python scripts/audit_fullplot_mask_recall.py
.venv-gpu/bin/python -m unittest discover -s tests -p test_fullplot_verifier.py -v
```

Pierwsza komenda wykorzystuje ukończone pasujące cache/selektor i nie oznacza automatycznie nowego treningu. Nowy wariant powinien mieć nowy katalog i sygnaturę. Zmiana parametrów grafu przy starym cache powoduje błąd signature mismatch — nie usuwać go bezmyślnie. Przy nowym eksperymencie można jawnie użyć tego samego zamrożonego raw cache i osobnego cache grafów. Trzy testy oraz kompilacja składniowa nowych modułów przeszły. Nie uruchomiono na tej podstawie nowej inferencji testowej/produkcyjnej.

## 11. Najnowszy pomysł: segmentacja etapowa / drugi przebieg

Ostatnia propozycja użytkownika: „może trzeba zrobić model który działa etapami? czyli rozpoznaje część, a później to co zostaje rozpoznaje ponownie w drugim pasie”.

Zaproponowana odpowiedź: **tak, dwa przebiegi mają sens, jeśli drugi jest specjalnie uczony na błędach/niepewności pierwszego i zachowuje kontekst**. Samo wielokrotne uruchamianie tej samej zamrożonej sieci na tych samych wejściach nie tworzy nowej informacji.

### Proponowany flow — jeszcze niewdrożony

```text
oryginalna chmura + HAG + cechy sensora
             |
       wspólny backbone
             |
    etap 1: korony i pewność
             |
 maski + cechy + niepewność + sporne obszary
             |
 etap 2: lokalny dekoder uzupełniania/korekty
             |
  nowa korona / korekta istniejącej / tło
             |
 wspólne uzgodnienie masek i nakładających się okien
             |
    finalne globalne tree_id
             |
     zgodne LAZ + pełne GPKG
```

1. **Etap 1:** wykrywa pewne korony. Rozsądny zewnętrzny kontrolny punkt startowy to Model20, bo Model22 ma dużo fałszywych instancji. Konkretny wybór inicjalizacji/gałęzi drugiego dekodera wymaga testów porównawczych, nie został zatwierdzony wynikiem.
2. **Wybór regionów:** punkty nieprzypisane, niepewne granice i obszary konfliktu między maskami. Uwzględnić drzewa już błędnie wchłonięte przez duże korony. Nie ograniczać puli wyłącznie do `tree_id=0` i nie stosować zbyt twardego progu semantyki, który zamknie drogę odzyskania pominiętych drzew.
3. **Kontekst:** zachować oryginalną geometrię sąsiednich koron i poprzednie przypisania jako cechy. Nie wycinać wcześniej znalezionych drzew z otoczenia; pozostawiona gałąź bez kontekstu może wyglądać jak samodzielna mała korona.
4. **Warunkowany decoder:** wejściem są cechy backbone’u, maski/prawdopodobieństwa etapu 1, lokalna geometria, niepewność i propozycje granic. Nowe query powinny być inicjowane w obszarach potrzebujących korekty. Zachować możliwość rewizji przypisań, zwłaszcza przy wchłoniętym małym drzewie.
5. **Trzy typy decyzji:** samodzielne nowe drzewo; fragment do dołączenia/korekty istniejącej instancji; tło/niepewne. Mask quality/completeness head powinien być uczony na tych przypadkach, nie tylko progowany po rozmiarze.
6. **Wspólna aktualizacja:** etap 2 może dodać drzewo, poprawić granicę albo zaproponować podział błędnie połączonej korony. Decyzja musi być uzgadniana z etapem 1, aby nie duplikować i nie niszczyć poprawnych dużych drzew. Finalne ID nadać po uzgodnieniu.
7. **Wydajność:** cache cech backbone’u i lokalne uruchomienia drugiego dekodera. Zacząć od maksymalnie dwóch przebiegów; trzeci dopiero po zmierzeniu przyrostu jakości/kosztu. Nie obiecywać konkretnego przyspieszenia bez pomiaru całego pipeline’u.
8. **Warunek stop:** brak nowych zaakceptowanych propozycji/poprawy pewności lub osiągnięcie limitu etapów. Nie iterować do pokolorowania wszystkich punktów; grunt, luki i tło są prawidłowymi wynikami.

### Superwizja i trening drugiego etapu

- Uczyć na rzeczywistych predykcjach etapu 1, nie wyłącznie na idealnych GT-maskach z usuniętymi koronami.
- Preferować predykcje out-of-fold w obrębie treningowych parent/site groups. Zamrożony model trenowany na tych samych plotach może dać zbyt łatwe resztki; obecne raw cache train v1 nie jest udokumentowaną predykcją out-of-fold.
- Wszystkie foldy/syntetyczne widoki jednego rodzica trzymać razem; nie używać validation/test do tworzenia targetów treningowych.
- Tworzyć przykłady: pominięte małe drzewa, małe drzewa połączone z dużymi, brzegi/gałęzie jednej dużej korony, duplikaty z okien, prawdziwe tło, unknown do ignorowania.
- Bilansować rozmiary drzew i źródła; nie wzmacniać dużych instancji tylko dlatego, że mają więcej punktów.
- Rozważyć mask Dice/focal/BCE, kompletność/IoU maski, relacje „to samo drzewo”, duplicate/fragment penalty, kontrolowaną zgodność pełnej korony 2D oraz ochronę poprawnych przypisań etapu 1. Są to kandydaci do kontrolowanych testów porównawczych, nie gotowa wybrana funkcja straty.
- Kształt pełnej korony i dopasowanie małych drzew muszą być sprawdzane po końcowym scalaniu i polygonizacji. Dobry score klasyfikatora nie wystarczy.

### Konkretna kolejność następnej implementacji

1. Zamrozić protokół porównania, checkpoint pierwszego etapu, manifest i hashe. Zapisać bazowe full-plot wyniki Model20/22 oraz koszt inferencji.
2. Przygotować treningowy cache: cechy, miękkie maski/przypisania, niepewność, trwałe ID punktów/wokseli i powiązania okien. Sprawdzić, czego brakuje w istniejącym raw cache.
3. Zbudować diagnostykę regionów drugiego etapu: ile małych GT jest kandydatami, ile jest wchłoniętych, ile fragmentów dużych koron jest fałszywymi kandydatami. GT służy wyłącznie targetom/metrykom w dopuszczonym splicie.
4. Wdrożyć mały warunkowany decoder z decyzją new/existing/background i jakością maski; rozpocząć od zamrożonego backbone’u na GPU.
5. Wdrożyć wspólne rozstrzyganie punktów oraz zapobieganie duplikatom między etapami i oknami. Sprawdzić, czy mała korona może odzyskać swoje punkty od zbyt dużej instancji.
6. Trenować na wielu cropach/regionach jednego rodzica, nie tylko jednym stałym cropie. Weryfikować pełne ploty co kilka epok. Epoka 0 pozostaje uczciwą kontrolą.
7. Porównać: etap1; naiwny drugi przebieg tylko po nieprzypisanych; uczony drugi etap z kontekstem; opcjonalnie bez/ze score/completeness. Rejestrować koszt, małe trafienia, błędne splity/merge i regresje dużych koron.
8. Tylko kandydat spełniający bramkę otrzymuje nową inferencję porównawczą w świeżym folderze. Numer/nazwę sprawdzić w bieżącym katalogu; nie zakładać, że kolejny numer jest wolny.

Nazwy nowych modułów/runów można dobrać przy implementacji, np. `residual_refinement` / `dualcrown3d_two_pass_v1`; to propozycje nazw, nie istniejące już artefakty.

### Zalecana bramka jakości dla tej pracy

Ustalić przed treningiem i zapisać w nowym protokole. Docelowo więcej małych trafień niż Model20 **przy zachowaniu jego point/crown PQ, F1 i precyzji**, z ambicją utrzymania korzyści małych drzew widocznej w Model22. Nie wystarczy pobić słabego Model22 w ogólnym PQ.

Na obecnej validation punktem odniesienia jest 0.4051/0.4442 SB-PQ i 6/161 małych trafień Model20; poziom Model22 to 20/161. Jeśli dopuszczona zostanie tolerancja statystyczna, ustalić ją przed porównaniem i raportować przedziały/odchylenia, nie dopasowywać jej do przegranego wyniku. Oddzielnie mierzyć liczbę koron GT rozbitych na kilka ID i degradację dużych drzew. FGI1018 zawiera 114 ze 161 małych referencji — wynik pooled small recall jest mocno zdominowany przez ten plot; potrzebne również wyniki per-source i per-plot.

Finalistów sprawdzić na kilku seedach, a do wiarygodnego twierdzenia publikacyjnego pozyskać niezależny holdout/site. Obecny validation był wielokrotnie używany eksploracyjnie.

## 12. Większy plan superpoint/TLS — co pozostaje otwarte

`reports/als_tls_superpoint_implementation_plan.md` opisuje szerszy program:

- learned instance affinities i granice;
- małe superpointy przy dopasowanym budżecie kompresji;
- graph context i grupowanie odporne na pojedyncze błędne krawędzie;
- punktowa rewizja mieszanych superpointów;
- wspólne cechy/krawędzie w overlapach przed końcowymi ID;
- test realizmu HELIOS i kontrola widoczności;
- porównanie A: real ALS; B: TLS+thinning; C: TLS+HELIOS; D: C+dense TLS teacher;
- distillation tylko tam, gdzie istnieją wiarygodne correspondences/widoczność;
- matched compute, density, parent splits, source-balanced metrics i pełny czas pipeline’u.

Wykonano audyt, piloty partycji, downstream crop decoder, jego powiększenie i eksperymentalną full-scene inferencję. Nie wykonano kompletnego, potwierdzonego jakościowo programu A/B/C/D ani produkcyjnego globalnego graph modelu/teachera. Własny mały symulator TLS→ALS przetestowano, ale nie uzasadnił masowego generowania danych.

Ważna pułapka transferu: zachowany backbone miał wcześniej kontakt z HELIOS. Pilot z jego zamrożonymi cechami może być dobrym eksperymentem inżynierskim, ale nie jest czystym ALS-only control do oceny wpływu TLS pretrainingu. Dla uczciwego A/B/C/D trzeba wspólnej audytowanej inicjalizacji sprzed transferu i równego budżetu.

Pomysły nadal sensowne do izolowanych testów: supervised boundary veto, cluster-consistent merging zamiast single-link bridges, poprawa jakości seed/query, jawny crown loss, wiele cropów per parent, gęstość i widoczność dopasowana do domeny, lokalne drobniejsze próbkowanie etapu 2. Żaden nie ma jeszcze gwarantowanego zysku na tym zbiorze.

## 13. Semantyka eksportów i oglądanie wyników

W Model20:

- `PointClouds/trees_<tile>.laz` zachowuje oryginalne XYZ, intensity i source classification. Z jest rzeczywistą wysokością źródłową, `height_agl` to osobne pole wysokości nad gruntem.
- `tree_id` odpowiada finalnej instancji consensus; `legacy_tree_id` odnosi się do starej gałęzi. Mapowanie lokalnych/starych ID: `legacy_tree_id_map.csv`.
- `PointHead/Segmentation3/crowns_<tile>.gpkg` odpowiada finalnym LAZ `tree_id`.
- `Segmentation3/` zawiera korony starej gałęzi. Te foldery mogą różnić się geometrią i liczbą drzew — nie traktować ich jako dwóch kopii tego samego wyniku.
- `assignment_source`: 0 unassigned, 1 mask/support anchor, 2 cross-head completion, 3 vote instance, 4 local recovery, 5 dodana mask-head instance.
- `segmentation_status`: 0 przewidziane tło/grunt, 1 przypisana instancja, 2 przewidziane drzewo bez instancji. To diagnostyka modelu, nie GT.
- `tree_id=0` obejmuje również poprawne tło, więc sam szary/niebieski kolor nie dowodzi pominięcia drzewa.

W output21/output22 oba foldery vector mają intencjonalnie zgodne korony/ID. Nie są dwiema niezależnymi głowicami; dla 22 poligony wynikają z tych samych punktów.

W output23 i crop exports zapisano **wokselizowane punkty wejściowe**, a Z to HAG, nie oryginalna wysokość bezwzględna. `reference_tree_id` jest tylko do oglądania etykiet. Nie interpretować braku każdego oryginalnego odbicia jako błędu eksportera. Część danych ma lokalny układ bez CRS; nie dopisywać EPSG:2180 tam, gdzie go nie ma.

Referencyjna scena użytkownika używa kafli `764000_197000`, `764000_197500`, `764500_197000`, `764500_197500`, EPSG:2180. Na tych kaflach nie mamy w tym porównaniu kompletnego ground truth. Bazeline R i drugi model nie są automatycznie GT.

CloudCompare: do kolorowania instancji używać RGB; scalar field `tree_id` jest identyfikatorem kategorialnym. Sprawdzić pełny zakres **displayed**, nie tylko saturation, oddzielnie dla każdej chmury. Dla dużych współrzędnych zaakceptować poprawny Global Shift. ID/kolor nie jest stabilny między wersjami modelu.

Korony powinny być pełnymi obrysami: obecny exporter punktowy używa rastra XY 0.5 m, małego closing i hole filling, zachowując rozdzielone komponenty. Jeden convex hull nad dalekimi fragmentami może stworzyć fałszywy obrys. Zgodność ID i poprawność geometrii sprawdzać automatycznie; pełne pokrycie nie dowodzi poprawnych granic.

## 14. Czas inferencji i rozmiar sceny

Model20 ma zapisane pięć pomiarów:

```text
output_20_dualcrown3d_joint_finetune/inference_timing_5runs.json
output_20_dualcrown3d_joint_finetune/raport_czasu_inferencji_i_chmur_punktowych.xlsx
```

JSON dotyczy **pełnej przygotowanej chmury czterech kafli**, obszar raportowany 78.027075 ha. Średnio 242.538 s, std 1.008 s, 3.1084 s/ha. Obejmuje przygotowanie okien, sieć i ekstrakcję surowych masek przy raz załadowanym modelu. **Nie obejmuje** ładowania plików, scalania masek, polygonizacji i LAZ export. Nie przedstawiać tej liczby jako całego end-to-end ani automatycznie jako pomiaru pojedynczego pliku `trees_764000_197000.laz`.

Model22 referencyjna scena: 8 069 167 oryginalnych punktów w czterech LAZ, 13 459 koron, 5 398 okien; około 1 221 s GPU prediction i 63.8 s dalszych czynności, około 21.4 min łącznie według raportu tego przebiegu. To pojedynczy run, nie kontrolowany benchmark wobec powyższych pięciu pomiarów. Sam kafel `764000_197000`: 1 987 560 punktów i 6 957 koron Model22.

Czas decoder-only na cached features, np. 0.0765 s/crop Q128, nie jest czasem całej inferencji ani sekundami/ha. Model etapowy musi rejestrować osobno backbone, wybór regionów, pass2, stitching, polygonizację i zapis, oraz pełny wall time i peak VRAM.

## 15. Klasyczny benchmark R i nierozwiązany cel porównania

Główne R wejście workspace: `ROOT/skrypt/pcopw_chunks_500m_Segmentacja.R`. Przed uruchomieniem sprawdzać Windows/Linux paths, input/output, 25 m bufor i core tile, kompletność par `crowns`/`ttops`. Istnienie crown może powodować skip; przechwycony błąd może dać `NULL`, więc zakończenie procesu nie dowodzi powodzenia wszystkich kafli.

W `ROOT/run_r` istnieje wiele nieidentycznych optymalizacji/protokołów, m.in.:

- `optimization_for_instance_dev_weighted_pq_v1/`
- `optimization_for_instance_dev_weighted_pq_v2_refined/`
- `optimization_ideas_als_dev_weighted_pq_v1/`
- `optimization_ideas_als_source_balanced_loso_v2/`
- `optimization_ideas_als_chm_flexible_lmf_v3/`
- raporty transferu `evaluation_for_instance_*`.

Przykładowy raport v3 klasycznego podejścia mówi o FOR-instance-optimized v2 F1≈0.516, PQ≈0.388, ale są to inne zbiory/agregacje niż 14-plot v4 validation DL. Nie dowodzić przewagi DL na podstawie 0.405>0.388. Aby zrealizować pierwotny cel użytkownika, zamrozić najlepszą właściwą konfigurację R i uruchomić ją na dokładnie tej samej referencji/metrykach co kandydat DL albo jasno oznaczyć brak takiego porównania.

## 16. Szybka mapa kodu i raportów do dalszej pracy

| Potrzeba | Pliki |
|---|---|
| Dane/manifest | `pointcloud/data.py`, `scripts/train_supervision_v4.py` (stała REAL) |
| Model zachowany | `pointcloud/dual_head.py`, `joint_training.py`, `scripts/predict_dual_head.py` |
| Consensus/scalanie | `pointcloud/instance_output.py`, `dual_fusion.py` |
| Shared query point/BEV | `pointcloud/shared_instance.py`, `shared_merge.py`, `scripts/train_shared_instances.py` |
| Superpoint + większy decoder | `pointcloud/superpoints/`, `scripts/train_superpoint_decoder_pilot.py`, `predict_superpoint_wide_scene.py` |
| Pełne ploty: inferencja/metryki | `scripts/evaluate_output20_output22_test.py`: `wide_predict_plot`, `score`, `summary` |
| Polygonizacja/punktowe diagnostyki | `scripts/train_superpoint_decoder_pilot.py`: `crown_records`, `point_diagnostics` |
| Małe korony | `scripts/calibrate_legacy_small_trees.py`: `small_hits` |
| Full-plot verifier | Pliki z sekcji 10 |
| Dane TLS/HELIOS | `scripts/prepare_treescan_helios_dualcrown.py`, `simulate_tls_als_pilot.py` |

Najważniejsze raporty do czytania w kolejności:

1. [Nadsegmentacja i próby ratowania małych drzew](output22_oversegmentation_small_tree_experiments.md).
2. [Ostatni full-plot verifier i dlaczego przegrał](fullplot_fragment_verifier_experiment.md).
3. [Dopasowany test Model20/22](output20_output22_heldout_test.md).
4. [Architektura dual-head](dual_head_satv2_design.md), [consensus](dual_consensus_fix.md), [v4](supervision_v4_protocol.md).
5. [Większy decoder Q128](superpoint_decoder_size_ablation.md), [inferencja 22](output22_ezsp_wide_q128_inference.md).
6. [Plan ALS/TLS/superpoints](als_tls_superpoint_implementation_plan.md), [pilot widoczności TLS](tls_visibility_pilot.md).
7. [Stage0–2](superpoint_stage0_stage2_result.md), [SPT/EZ-SP kernels](superpoint_spt_ezsp_kernel_comparison.md), [Stage3](superpoint_stage3_decoder_result.md).
8. [Kalibracja Model20](legacy_small_tree_calibration.md), [shared-v5](shared_v5_protocol.md), [kampania joint](joint_campaign_protocol.md).

Główne logi Excel: `outputs/dualcrown3d_joint_campaign_v1/experiments.xlsx`, `outputs/dualcrown3d_supervision_v4/experiments.xlsx`, `outputs/dualcrown3d_superpoint_v1/stage3_decoder_pilot/experiments.xlsx`, `output_23_labeled_test_output20_vs_output22/comparison.xlsx`. Ostatni full-plot v1 ma JSON-y i checkpoint; nie zakładać istnienia dodatkowego Excel, którego nie zapisano.

## 17. Lista kontrolna przed wznowieniem implementacji

1. Odczytać `AGENTS.md`, ten dokument, stan Git oraz najnowszą wiadomość użytkownika.
2. Sprawdzić GPU i brak kolizji z innymi treningami. Nie instalować sterowników bez administratora.
3. Sprawdzić checkpoint/config/hash i `model_split` + eligibility. Nie mieszać starych 100/18/29 z aktualnym 79/14/24.
4. Nie zmieniać danych źródłowych, splitów i starych eksportów dla poprawienia wyniku.
5. W nowym eksperymencie zapisać seed, source balance, architekturę, trainable parameters, loss, sampling, augmentation, checkpoint i hashe kodu/danych. Każdy rekord metryk musi mieć zakres: crop/full-plot, val/test, point/crown, pooled/source-balanced.
6. Testować punktowe wyrównanie/odwzorowanie wokseli, brak GT w inferencji, stabilność okien, dopuszczalność unknown, brak duplikatów ID i zgodność LAZ/GPKG. Dla dwóch etapów dodać test wchłoniętego małego drzewa i negatywny przykład fragmentu dużej korony.
7. Nie używać nowych testowych metryk do strojenia. Pełna validation musi obejmować wszystkie 14 plotów, a nie tylko łatwy CULS.
8. Promować dopiero po łącznej bramce jakości. W przypadku porażki wyraźnie zostawić `selected: null`/bazowy checkpoint i opisać wynik; nie nazywać modelu lepszym po samym wzroście coverage lub liczby koron.
9. Nowy folder inferencji tylko ze wskazanym checkpointem i konfiguracją. Raportować pliki do oglądania, zakres współrzędnych, znaczenie pól i czasy.

## 18. Literatura i inspiracje z rozmowy

To referencje idei; nie są dowodem, że ich opublikowane wyniki przenoszą się na lokalny las.

- Początkowy artykuł wskazany przez użytkownika: https://www.mdpi.com/1999-4907/16/2/248
- SegmentAnyTreeV2, inspiracja mask decoderem: https://arxiv.org/abs/2606.08206
- HELIOS: https://github.com/giscience/HELIOS
- Mask3D, iteracyjne queries/uwaga do cech punktów: https://arxiv.org/abs/2210.03105
- SoftGroup, etapowe grupowanie/refinement i tłumienie fałszywych instancji: https://arxiv.org/abs/2203.01509
- SuperCluster: https://arxiv.org/abs/2401.06704
- Learned oversegmentation: https://arxiv.org/abs/1904.02113
- Oficjalny projekt SPT/SuperCluster/EZ-SP: https://github.com/drprojects/superpoint_transformer

Najbliższy krok badawczy wynikający z tej sesji to **uczony drugi przebieg z kontekstem i rewizją przypisań**, oceniany względem zachowanego Model20, przy zachowaniu czułości na małe drzewa. To propozycja do realizacji, nie ukończony model.

## 19. Aktualizacja po sesji „two-pass” (2026-10-02, późniejsza sesja)

Zrealizowano sekcję 11. Pełny opis, protokół, wyniki i ograniczenia: [two_pass_refinement_design.md](two_pass_refinement_design.md). Najważniejsze punkty dla kolejnego agenta:

- **Wąskim gardłem jest detekcja środków drzew, nie maski ani cechy.** Zamrożone głosy 3D Model20 (`shifted_center`) z prawdziwymi centroidami dają na validation 523/673 drzew i point SB-PQ 0.670 (Model20: 299 i 0.405).
- Drugi przebieg generujący maski (trzy warianty `residual_v1`–`v3`) to **wynik negatywny**. Nie wracać do niego na zamrożonych cechach.
- Działający drugi przebieg: mały 3D U-Net wykrywający środki w przestrzeni głosów + korekcyjne przypisanie (`pointcloud/vote_centers.py`, `scripts/train_vote_centers.py`). Stage 1 = zamrożony Model20.
- **Pełna gęstość (ploty z etykietami):** kandydat `vote_centres_v5` przeszedł bramkę na validation i na teście (24 ploty) daje point/crown SB-PQ 0.462/0.449 wobec 0.418/0.422, małe korony 46/241 wobec 21/241, recall dużych drzew 0.716 wobec 0.622, bez spadku precyzji. Rośnie liczba rozbitych koron GT (140 → 163).
- **Gęstość sceny referencyjnej (~10 wokseli/m²): brak wykazanej korzyści.** Na przerzedzonym teście `vote_centres_v6` daje PQ 0.394/0.328 wobec 0.390/0.322, przy nieco gorszych małych koronach i recallu dużych drzew. Dla sceny użytkownika nadal obowiązuje `output_20`.
- Nowe foldery: `output_27`/`output_29` (test v5/v6), `output_30` (test przerzedzony), `output_28`/`output_31` (scena referencyjna v5/v6, tylko do oglądania; `STATUS.md` w środku).
- Model20 pozostaje modelem produkcyjnym. Nic nie zostało zacommitowane. Test został odczytany dwukrotnie (v5, v6) — nie używać go do dalszego strojenia.
- Następny krok: detektor środków działający na rzadkich chmurach (grubsza/wieloskalowa siatka głosów, trening na HELIOS ~14 pkt/m² i realnych rzadkich danych), oceniany na rzadkiej walidacji.
