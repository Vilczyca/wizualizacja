# Dokumentacja API Backend (`BE`)

Bazowy URL lokalnie: `http://localhost:8000`
Interaktywny Swagger UI (OpenAPI): `http://localhost:8000/docs`

Wszystkie odpowiedzi JSON i Arrow obsługują kompresję `gzip` (`GZipMiddleware`).

---

## Przegląd endpointów

| Metoda | Ścieżka | Opis |
|---|---|---|
| `GET` | `/api/meta` | Konfiguracja sceny 3D, atlasów i zakresy współrzędnych |
| `GET` | `/api/viz-data` | Lekki plik Apache Arrow (~2,4 MB) z punktami 3D chmury |
| `GET` | `/api/clusters` | Lista klastrów KMeans z kolorami HEX i liczebnością |
| `GET` | `/api/samples/{idx}` | Pełne metadane pojedynczego punktu (leniwe ładowanie po kliknięciu) |
| `GET` | `/api/search` | Wyszukiwanie pełnotekstowe (SQLite FTS5) po opisach, kategoriach i nazwach plików |
| `GET` | `/api/image` | Podgląd pojedynczego kadru ze zbioru bazowego (WebP do 800 px) |
| `GET` | `/api/atlases/{filename}` | Statyczne atlasy tekstur WebP (np. `atlas_0000.webp`) |
| `GET` | `/api/user-samples` | Lista próbek dodanych przez użytkowników |
| `POST` | `/api/infer` | Upload nowego obrazu, inferencja `CLIP → PCA → UMAP 3D` i zapis w bazie |
| `GET` | `/api/uploads/{filename}` | Statyczne pliki podglądu i miniatur próbek użytkowników |

---

## Szczegóły endpointów

### 1. `GET /api/meta`
Zwraca globalne metadane zbioru, parametry siatki atlasów i zakres współrzędnych 3D (`bounds`) do ustawienia kamery.

**Odpowiedź `200 OK`:**
```json
{
  "total_points": 185259,
  "atlas_grid_size": 32,
  "atlas_tile_size": 64,
  "atlas_count": 181,
  "clusters_count": 40,
  "bounds": {
    "min": [-4.804, -14.696, -12.808],
    "max": [27.080, 16.541, 23.184]
  }
}
```
- `404 Not Found` — brak wygenerowanej bazy `metadata.db`.

---

### 2. `GET /api/viz-data`
Zwraca lekki binarny plik **Apache Arrow / Feather** (~2,4 MB dla 185k punktów) do bezpośredniego wczytania np. przez `apache-arrow` w JS.

**Nagłówki odpowiedzi:**
- `Content-Type: application/vnd.apache.arrow.file`
- `Cache-Control: public, max-age=3600`

**Schemat kolumn:**
| Kolumna | Typ Arrow | Opis |
|---|---|---|
| `x` | `float32` | Współrzędna X w przestrzeni UMAP 3D |
| `y` | `float32` | Współrzędna Y w przestrzeni UMAP 3D |
| `z` | `float32` | Współrzędna Z w przestrzeni UMAP 3D |
| `cluster` | `uint16` | ID klastra (odpowiada `id` z `/api/clusters`) |
| `atlas_id` | `int32` | Numer atlasu (`0..atlas_count-1`, `-1` gdy brak) |
| `tile_index` | `int32` | Pozycja kafelka w atlasie (`0..1023`, `-1` gdy brak) |

Indeks wiersza `i` w pliku Arrow odpowiada dokładnie parametrowi `{idx}` w `GET /api/samples/{idx}`.

---

### 3. `GET /api/clusters`
Zwraca listę klastrów KMeans do kolorowania chmury punktów i legendy.

**Odpowiedź `200 OK`:**
```json
[
  { "id": 0, "color": "#f23c3c", "count": 2538 },
  { "id": 1, "color": "#f2583c", "count": 4120 }
]
```

---

### 4. `GET /api/samples/{idx}`
Zwraca pełne metadane wybranego punktu z bazy SQLite (do wywołania dopiero po kliknięciu lub najechaniu punktu na scenie 3D).

**Parametry ścieżki:**
- `idx` (`int`, `>= 0`) — indeks wiersza z `viz_data.arrow`.

**Odpowiedź `200 OK`:**
```json
{
  "idx": 0,
  "sample_idx": "1e8aa6a6-18b5-43ed-a364-252c38bedc3f",
  "image_path": "./images/p0I7ccItmv8_frame_393482.jpg",
  "image_url": "/api/image?path=./images/p0I7ccItmv8_frame_393482.jpg",
  "video_id": "p0I7ccItmv8",
  "category": "",
  "caption": "The image shows a first-person perspective scene...",
  "x": 6.4137425,
  "y": 7.9083877,
  "z": 8.324497,
  "cluster": 25,
  "atlas_id": 0,
  "tile_index": 0
}
```
- `404 Not Found` — punkt o podanym `idx` nie istnieje lub brak `metadata.db`.

---

### 5. `GET /api/search`
Wyszukiwanie pełnotekstowe (SQLite **FTS5**, ranking **BM25**) po opisach (`caption`), kategoriach (`category`), `video_id` oraz nazwach plików (`image_path`).

**Parametry query:**
- `q` (`string`, wymagany, 1–200 znaków) — fraza wyszukiwania (np. `snowy cave`)
- `limit` (`int`, domyślnie `50`, zakres `1..200`) — liczba wyników
- `offset` (`int`, domyślnie `0`, `>= 0`) — paginacja
- `cluster` (`int`, opcjonalny, `>= 0`) — zawężenie wyników do jednego klastra

**Przykład:**
```http
GET /api/search?q=snowy+cave&limit=20&offset=0
```

**Odpowiedź `200 OK`:** tablica obiektów o tej samej strukturze co `GET /api/samples/{idx}` (każdy wynik zawiera `idx`, `x`, `y`, `z`, `image_url`, `caption` itd.).

---

### 6. `GET /api/image`
Zwraca pojedynczy zrzut ekranu ze zbioru bazowego przeskalowany do maksymalnie 800 px w formacie WebP.

**Parametry query:**
- `path` (`string`, wymagany) — wartość `image_path` zwrócona przez `/api/samples/{idx}` lub `/api/search`.

**Odpowiedź `200 OK`:**
- `Content-Type: image/webp`
- `Cache-Control: public, max-age=86400, immutable`
- `400` / `403` / `404` — błędna, niedozwolona lub nieistniejąca ścieżka.

---

### 7. `GET /api/atlases/{filename}`
Statyczne pliki tekstur z miniaturami kadrów (`atlas_0000.webp`, `atlas_0001.webp`, …).

- `Content-Type: image/webp`
- `Cache-Control: public, max-age=86400, immutable`
- Nazwa pliku dla punktu: `` `atlas_${String(atlas_id).padStart(4, "0")}.webp` ``
- Współrzędne kafelka w siatce `G = atlas_grid_size` (domyślnie `32`, rozmiar kafelka `64×64` px):
  - `col = tile_index % G`
  - `row = Math.floor(tile_index / G)`
  - `u0 = col / G`, `v0 = row / G`, rozmiar kafelka UV = `1 / G`

---

### 8. `GET /api/user-samples`
Zwraca listę wszystkich obrazów dodanych przez użytkowników, posortowaną od najnowszych.

**Odpowiedź `200 OK`:**
```json
[
  {
    "id": "a1b2c3d4e5f678901234567890abcdef",
    "x": 4.12,
    "y": -1.34,
    "z": 8.01,
    "category": "RPG",
    "caption": "Zrzut ekranu z walki z bossem",
    "image_url": "/api/uploads/a1b2c3d4e5f678901234567890abcdef.webp",
    "thumb_url": "/api/uploads/a1b2c3d4e5f678901234567890abcdef_thumb.webp",
    "created_at": "2026-11-18T18:30:00+00:00"
  }
]
```

---

### 9. `POST /api/infer`
Przyjmuje obraz w `multipart/form-data`, zapisuje podgląd i miniaturę WebP w `/api/uploads/`, wylicza współrzędne 3D (`CLIP → PCA → UMAP`) i zapisuje próbkę w bazie użytkowników.

**Body (`multipart/form-data`):**
- `file` (wymagany, `image/*`, max 10 MB) — plik graficzny
- `caption` (`string`, opcjonalny, max 500 znaków) — opis obrazu
- `category` (`string`, opcjonalny, max 100 znaków) — kategoria obrazu

**Odpowiedź `201 Created`:** pojedynczy obiekt w formacie jak w `GET /api/user-samples`.
- `400 Bad Request` — plik nie jest obrazem lub jest uszkodzony
- `413 Payload Too Large` — plik przekracza 10 MB
- `500 Internal Server Error` — błąd inferencji AI lub zapisu do bazy

---

### 10. `GET /api/uploads/{filename}`
Statyczne pliki podglądów (`{id}.webp`) oraz miniatur (`{id}_thumb.webp`) dodanych przez użytkowników.
- `Cache-Control: public, max-age=86400, immutable`

---

## Zalecany przepływ integracji na frontendzie

1. **Inicjalizacja**: pobierz `GET /api/meta`, `GET /api/clusters`, `GET /api/viz-data` (~2,4 MB) i `GET /api/user-samples`.
2. **Render 3D**: narysuj chmurę punktów z `viz_data.arrow`, kolorując punkty po `cluster` (kolory z `/api/clusters`), a przy zbliżeniu doładowuj atlasy z `/api/atlases/atlas_XXXX.webp`.
3. **Interakcja (kliknięcie punktu `i`)**: wywołaj `GET /api/samples/{i}` po opis/metadane oraz załaduj duży podgląd z zwróconego `image_url`.
4. **Wyszukiwanie**: wywołaj `GET /api/search?q=...` i podświetl na chmurze 3D zwrócone indeksy `idx`.
5. **Dodanie obrazu**: wyślij formularz na `POST /api/infer` i dodaj zwrócony punkt (`x, y, z`, `thumb_url`, `image_url`) do sceny.
