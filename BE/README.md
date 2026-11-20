# Wizualizacja: Backend

Backend odpowiedzialny za ekstrakcję cech wizualnych z kadrów gier wideo przy użyciu modelu CLIP w wariancie `openai/clip-vit-base-patch32`, redukcję wymiarowości do przestrzeni 3D algorytmem UMAP oraz udostępnianie danych i inferencji na żywo przez FastAPI.

## Struktura katalogu `BE/`

```text
BE/
├── src/
│   ├── be/                           # Aplikacja webowa FastAPI
│   │   ├── api/routes.py             # Endpointy REST API (/api/*)
│   │   ├── domain/schemas.py         # Modele walidacji danych Pydantic v2
│   │   ├── infrastructure/           # SQLite (user_samples.db, metadata.db + FTS5), indeks plików
│   │   ├── services/                 # MLService (CLIP + PCA/UMAP) oraz ImageStorageService
│   │   ├── config.py                 # Centralna konfiguracja ścieżek i zmiennych środowiskowych
│   │   └── main.py                   # Punkt wejścia aplikacji ASGI, GZip i CORS
│   ├── extract_features.py           # Etap 1: Ekstrakcja 512D embeddingów CLIP do plików Parquet
│   ├── merge_checkpoints.py          # Etap 1b: Scalanie checkpointów, deduplikacja i weryfikacja
│   ├── train_reducer.py              # Etap 2: PCA -> UMAP 3D, KMeans, metadata.db i lekki viz_data.arrow
│   └── generate_atlases.py           # Etap 3: Generowanie atlasów miniatur WebP
├── API.md                            # Pełna dokumentacja endpointów REST dla frontendu
├── .env.example                      # Szablon zmiennych środowiskowych
├── .gitignore                        # Wykluczenia artefaktów ML, cache oraz plików lokalnych
├── pyproject.toml                    # Zależności projektu i konfiguracja lintera Ruff
└── uv.lock                           # Zablokowane wersje pakietów
```

## Wymagania wstępne

* Python 3.13
* Menedżer pakietów `uv`: https://docs.astral.sh/uv/
* Opcjonalnie karta graficzna NVIDIA z obsługą CUDA 12.8 do szybkiej ekstrakcji pełnego zbioru. Sam serwer API oraz inferencja pojedynczych obrazów działają również na CPU oraz macOS.
* Opcjonalnie program 7-Zip, wymagany wyłącznie przy rozpakowywaniu pełnego, wieloczęściowego archiwum obrazów o rozmiarze około 62 GB w Etapie 1.

## 1. Instalacja środowiska

Należy przejść do katalogu backendu i zsynchronizować środowisko wirtualne:

```bash
cd BE
uv sync
```

## 2. Konfiguracja zmiennych środowiskowych

Wszystkie ustawienia zależne od maszyny, ścieżki dyskowe oraz parametry wydajnościowe są zarządzane centralnie przez moduł `be/config.py`. Konfiguracja odbywa się za pomocą pliku `.env` w katalogu `BE/`, utworzonego na podstawie szablonu `.env.example`. Plik `.env` jest ignorowany przez system kontroli wersji Git.

W systemie Windows plik `.env` musi być zapisany w kodowaniu UTF-8 bez znacznika BOM. Obecność nagłówka BOM na początku pliku powoduje błąd parsowania po stronie `uv` i zignorowanie wszystkich zmiennych.

### Dostępne zmienne w pliku `.env`

* `VIZ_DATA_DIR`: katalog główny na surowe obrazy, pliki pośrednie Parquet, wytrenowany model UMAP oraz wynikowy plik Arrow. Wartość domyślna to `./data`.
* `HF_HOME`: ścieżka pamięci podręcznej dla wag modelu CLIP oraz pobieranych archiwów z Hugging Face. Wartość domyślna to `./data/huggingface_cache`.
* `HF_TOKEN`: opcjonalny token dostępu do Hugging Face z uprawnieniami do odczytu. Zapobiega limitom zapytań HTTP 429 podczas pobierania danych.
* `DRY_RUN`: przełącznik trybu pracy potoku ML. Ustawienie `true` przełącza wszystkie skrypty oraz serwer API na lekki podzbiór ewaluacyjny oraz pliki z dopiskiem `_dry_run`. Ustawienie `false` oznacza pracę na pełnym zbiorze około 185 tysięcy kadrów. Wartość domyślna to `false`.
* `DRY_RUN_LIMIT`: maksymalna liczba obrazów przetwarzanych w trybie testowym. Wartość domyślna to `2048`.
* `MOCK_ML`: flaga pomocnicza dla środowiska deweloperskiego. Wartość domyślna to `false`.
* `BATCH_SIZE`: rozmiar paczki obrazów przetwarzanych jednocześnie przez model CLIP na GPU. Wartość domyślna to `512`. Na kartach graficznych z mniejszą ilością pamięci VRAM wartość tę należy zmniejszyć, na przykład do `128` lub `256`.
* `NUM_WORKERS`: liczba równoległych procesów CPU dekodujących obrazy podczas ekstrakcji cech. Wartość domyślna to `6`.
* `CORS_ORIGINS`: lista adresów frontendu uprawnionych do wykonywania zapytań do API, oddzielona przecinkami. Wartość domyślna to `http://localhost:5173,http://127.0.0.1:5173`.

### Przykładowy plik `.env`

```env
VIZ_DATA_DIR=./data
HF_HOME=./data/huggingface_cache
HF_TOKEN=
DRY_RUN=false
DRY_RUN_LIMIT=2048
MOCK_ML=false
BATCH_SIZE=512
NUM_WORKERS=6
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Aby polecenia uruchamiane przez `uv run` wczytywały lokalną konfigurację, należy dodawać flagę `--env-file .env`.

## 3. Uruchomienie serwera API

Uruchomienie samego serwera API bez ponownego przeliczania całego zbioru danych wymaga wygenerowanych artefaktów wewnątrz katalogu wskazanego przez `VIZ_DATA_DIR`:

* `viz_data.arrow`: lekka mapa punktów 3D (`x`, `y`, `z`, `cluster`, `atlas_id`, `tile_index`, ok. 2,4 MB).
* `metadata.db`: baza SQLite + FTS5 z pełnymi metadanymi punktów (pobierane leniwie przez `/api/samples/{idx}` i `/api/search`).
* `clusters.json`: definicje klastrów KMeans (kolory i liczebności).
* `models/umap_reducer.pkl`: zapisany potok `PCA → UMAP` rzutujący nowe obrazy na istniejącą przestrzeń.
* `atlases/atlas_*.webp`: atlasy tekstur WebP z miniaturami 64×64.

Start serwera lokalnego odbywa się poleceniem:

```bash
uv run --env-file .env uvicorn be.main:app --host 127.0.0.1 --port 8000 --reload
```

Po uruchomieniu dostępne są następujące zasoby (szczegółowe przykłady żądań i odpowiedzi znajdują się w [API.md](./API.md)):

* `http://localhost:8000/docs`: interaktywna dokumentacja OpenAPI Swagger UI.
* `GET /api/meta`: konfiguracja sceny, atlasów i zakresów współrzędnych 3D.
* `GET /api/viz-data`: lekki plik Apache Arrow z punktami 3D (`x`, `y`, `z`, `cluster`, `atlas_id`, `tile_index`).
* `GET /api/clusters`: lista klastrów z kolorami HEX i liczebnością.
* `GET /api/samples/{idx}`: pełne metadane pojedynczego punktu pobierane na żądanie.
* `GET /api/search?q=...`: wyszukiwanie pełnotekstowe FTS5 po opisach, kategoriach i nazwach plików.
* `GET /api/image?path=...`: podgląd pojedynczego kadru ze zbioru bazowego (WebP).
* `GET /api/atlases/{filename}`: statyczne atlasy miniatur WebP.
* `GET /api/user-samples` oraz `GET /api/uploads/{filename}`: próbki dodane przez użytkowników.
* `POST /api/infer`: upload obrazu (`multipart/form-data`), wyliczenie współrzędnych 3D i zapis próbki.

## 4. Odtworzenie potoku ML od zera

Pełny proces przygotowania danych składa się z trzech kolejnych skryptów.

### Wariant A: Test próbny na podzbiorze ewaluacyjnym

Pozwala zweryfikować poprawność konfiguracji oraz działanie GPU na zbiorze o rozmiarze około 670 MB. Wymaga ustawienia `DRY_RUN=true` w pliku `.env`, a następnie uruchomienia kolejno:

```bash
uv run --env-file .env python src/extract_features.py
uv run --env-file .env python src/merge_checkpoints.py
uv run --env-file .env python src/train_reducer.py
```

### Wariant B: Pełny zbiór treningowy

Wymaga ustawienia `DRY_RUN=false` w pliku `.env`.

1. Ekstrakcja cech CLIP:
   ```bash
   uv run --env-file .env python src/extract_features.py
   ```
   Przy pierwszym uruchomieniu skrypt pobierze wieloczęściowe archiwum `images.z01` do `images.z05` wraz z `images.zip` o łącznym rozmiarze około 61,8 GB i wyświetli dokładną komendę dla programu 7-Zip do jego jednorazowego rozpakowania. Po wypakowaniu ponowne uruchomienie skryptu rozpocznie obliczenia. Skrypt automatycznie deduplikuje powtarzające się ścieżki obrazów i wspiera wznawianie pracy od ostatniego zapisanego pliku Parquet.

2. Scalenie checkpointów i weryfikacja spójności:
   ```bash
   uv run --env-file .env python src/merge_checkpoints.py
   ```
   Skrypt łączy pliki cząstkowe w jedną macierz `raw_embeddings.parquet`, weryfikuje brak wartości pustych oraz wykonuje test podobieństwa cosinusowego.

3. Redukcja wymiarowości UMAP 3D:
   ```bash
   uv run --env-file .env python src/train_reducer.py
   ```
   Skrypt trenuje model UMAP na pełnej macierzy wektorów, zapisuje model w katalogu `models/` oraz eksportuje odchudzony plik `viz_data.arrow` pozbawiony surowych wektorów 512D.

## 5. Kontrola jakości kodu

Przed wysłaniem zmian do repozytorium należy zweryfikować styl oraz formatowanie kodu za pomocą narzędzia Ruff:

```bash
uv run ruff check .
uv run ruff format --check .
```