# Ponownie wykorzystane artefakty

Skopiowano bez modyfikacji z istniejacego workflow `run_r`.

## `for_instance_chm_gt_0p5m`

- 32 powierzchnie: 21 `dev`, 11 `test`.
- 807 nakladajacych sie koron GT i 661 widocznych koron topmost w `dev`.
- 323 nakladajace sie korony GT i 278 widocznych koron topmost w `test`.
- CRS jest zachowany osobno dla kazdej kolekcji.
- CHM i maski maja piksel 0,5 m.
- Historyczne CHM maja tag/pochodzenie `CHM_POINT_MODE=annotated-tree` i nie sa
  uzywane jako wejscie modelu DL z powodu ryzyka przecieku etykiet.
- `topmost_gt_*.gpkg` i `topmost_gt_*.tif` sa uzywane jako referencyjne maski
  widocznych koron.

## `watershed_lmf_for_instance_v2`

Predykcje dla wszystkich 32 dostepnych powierzchni pochodza z kandydata
`9681786f6dcb`. Parametry zostaly wybrane w istniejacym procesie optymalizacji
FOR-instance. Wyniki testowe sa traktowane jako historyczny baseline i nie beda
uzywane do doboru modelu YOLO.

## Brakujace dane

Oficjalny manifest FOR-instance wymienia kolekcje `NIBIO2`, ale nie wystepuje
ona w lokalnym archiwum. Projekt przetwarza 32 faktycznie dostepne pliki LAS.
