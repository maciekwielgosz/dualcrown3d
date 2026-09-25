# Raport końcowy: FOR-instance + YOLO11s-seg

## Wynik

Na podstawie walidacyjnego F1 wybrano **YOLO11s-seg physical**
(F1=0.785). Po zamrożeniu wyboru model osiągnął na
oficjalnym teście F1=0.747, SQ=0.805
i PQ=0.601. Próg pewności również został wybrany wyłącznie
na walidacji; test nie uczestniczył w doborze wariantu ani progu.

## Porównanie na teście

Dopasowanie instancji: algorytm węgierski, IoU maski >= 0,5. Test obejmuje 11
powierzchni i 278 koron.

| Model | Precision | Recall | F1/RQ | SQ | PQ | mask mAP50 | mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Watershed/LMF (leakage-free) | 0.532 | 0.507 | 0.519 | 0.751 | 0.390 | – | – |
| Watershed/LMF (historical) | 0.609 | 0.514 | 0.558 | 0.753 | 0.420 | – | – |
| YOLO11s-seg physical | 0.780 | 0.716 | 0.747 | 0.805 | 0.601 | 0.765 | 0.424 |
| YOLO11s-seg paper fusion | 0.745 | 0.705 | 0.725 | 0.795 | 0.576 | 0.744 | 0.412 |
| YOLO11s-seg CHM-only | 0.774 | 0.701 | 0.736 | 0.803 | 0.591 | 0.738 | 0.411 |
| YOLO11s-seg CHM-only + ideas_als | 0.834 | 0.633 | 0.720 | 0.802 | 0.577 | 0.727 | 0.384 |

`Watershed/LMF (historical)` nie jest uczciwym baseline'em publikacyjnym: wejściowy
CHM utworzono tylko z punktów mających adnotacje drzew. Właściwym baseline'em jest
wariant `leakage-free`, choć odziedziczył parametry dostrojone historycznie.

## Trening

- `yolo11s_physical`: 296 epok, najlepsza epoka 246, czas 12.0 min, val mask mAP50-95=0.459.
- `yolo11s_paper_fusion`: 284 epok, najlepsza epoka 234, czas 11.4 min, val mask mAP50-95=0.441.
- `yolo11s_chm_only_gpu`: 264 epok, najlepsza epoka 214, czas 2.8 min, val mask mAP50-95=0.441.
- `yolo11s_chm_only_ideas_combined_gpu`: 118 epok, najlepsza epoka 68, czas 6.1 min, val mask mAP50-95=0.360.
- `yolo11s_chm_only_ideas_finetuned_gpu`: 200 epok, najlepsza epoka 194, czas 1.9 min, val mask mAP50-95=0.411.

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
