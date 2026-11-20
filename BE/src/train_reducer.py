import os
import time

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import feather
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from umap import UMAP

from be.config import settings
from be.infrastructure.metadata_db import (
    build_clusters_metadata,
    build_metadata_db,
    save_clusters_json,
)

INPUT_FILE = settings.RAW_EMBEDDINGS_PATH
UMAP_MODEL_PATH = settings.UMAP_MODEL_PATH
VIZ_DATA_PATH = settings.VIZ_DATA_PATH
METADATA_DB_PATH = settings.METADATA_DB_PATH
CLUSTERS_PATH = settings.CLUSTERS_PATH


def _load_embeddings_float32(col: pa.ChunkedArray) -> np.ndarray:
    arr = col.combine_chunks()
    n_rows = len(arr)
    if pa.types.is_fixed_size_list(arr.type):
        emb_dim = arr.type.list_size
    else:
        emb_dim = len(arr[0].as_py())
    flat = arr.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    return flat.reshape(n_rows, emb_dim).copy()


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Brak scalonego pliku embeddingów: {INPUT_FILE}. "
            "Uruchom najpierw src/merge_checkpoints.py."
        )

    UMAP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Wczytywanie scalonych wektorów z {INPUT_FILE}...")
    start_time = time.perf_counter()

    table = pq.read_table(INPUT_FILE)
    print(
        f"[INFO] Załadowano {table.num_rows} wektorów w {time.perf_counter() - start_time:.1f} s."
    )

    print("[INFO] Konwersja do macierzy Float32 i normalizacja L2...")
    embeddings = _load_embeddings_float32(table.column("embedding"))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.maximum(norms, 1e-8)

    # ---------------------------------------------------------
    # ZADANIE GŁÓWNE: PCA -> UMAP 3D (Pipeline dla FastAPI)
    # ---------------------------------------------------------
    n_samples, n_features = embeddings.shape
    pca_dim = settings.UMAP_PCA_DIM
    use_pca = pca_dim > 0 and pca_dim < min(n_samples, n_features)

    umap_model = UMAP(
        n_components=3,
        n_neighbors=settings.UMAP_N_NEIGHBORS,
        min_dist=settings.UMAP_MIN_DIST,
        metric="euclidean",
        low_memory=True,
        n_jobs=-1,
        compatibility_layout=True,
    )

    if use_pca:
        print(
            f"[INFO] Trening potoku PCA({pca_dim}) -> UMAP 3D "
            f"(n_neighbors={settings.UMAP_N_NEIGHBORS}, min_dist={settings.UMAP_MIN_DIST}, "
            "metric=euclidean, low_memory=True)..."
        )
        pca = PCA(n_components=pca_dim, random_state=42)
        t0 = time.perf_counter()
        reduced_pca = pca.fit_transform(embeddings)
        umap_3d = umap_model.fit_transform(reduced_pca).astype(np.float32, copy=False)
        reducer = Pipeline([("pca", pca), ("umap", umap_model)])
        cluster_input = reduced_pca
    else:
        print(
            f"[INFO] Trening modelu UMAP 3D bez PCA "
            f"(n_neighbors={settings.UMAP_N_NEIGHBORS}, min_dist={settings.UMAP_MIN_DIST}, "
            "metric=euclidean, low_memory=True)..."
        )
        t0 = time.perf_counter()
        umap_3d = umap_model.fit_transform(embeddings).astype(np.float32, copy=False)
        reducer = umap_model
        cluster_input = embeddings

    print(f"[SUKCES] Redukcja 3D zakończona w {time.perf_counter() - t0:.1f} s.")

    # ---------------------------------------------------------
    # KLASTERYZACJA (kolorowanie chmury punktów 3D)
    # ---------------------------------------------------------
    n_clusters = min(settings.N_CLUSTERS, max(1, n_samples))
    print(f"[INFO] Klasteryzacja MiniBatchKMeans (n_clusters={n_clusters})...")
    t_km = time.perf_counter()
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=4096,
        n_init=3,
    ).fit(cluster_input)
    clusters = km.labels_.astype(np.uint16)
    save_clusters_json(build_clusters_metadata(clusters, n_clusters), CLUSTERS_PATH)
    print(
        f"[SUKCES] Zapisano klastry ({CLUSTERS_PATH.name}) w {time.perf_counter() - t_km:.1f} s."
    )

    print("[INFO] Zapisywanie modelu na dysk (dla FastAPI)...")
    tmp_model_path = UMAP_MODEL_PATH.with_suffix(".pkl.tmp")
    joblib.dump(reducer, tmp_model_path)
    os.replace(tmp_model_path, UMAP_MODEL_PATH)

    # ---------------------------------------------------------
    # PEŁNE METADANE -> SQLite + FTS5 (pobierane leniwie po kliknięciu)
    # ---------------------------------------------------------
    print(
        f"[INFO] Budowanie bazy metadanych SQLite + FTS5 ({METADATA_DB_PATH.name})..."
    )
    t_db = time.perf_counter()
    atlas_ids = np.full(n_samples, -1, dtype=np.int32)
    tile_indices = np.full(n_samples, -1, dtype=np.int32)
    build_metadata_db(
        db_path=METADATA_DB_PATH,
        sample_ids=table.column("sample_idx").to_pylist(),
        image_paths=table.column("image_path").to_pylist(),
        categories=table.column("category").to_pylist(),
        captions=table.column("caption").to_pylist(),
        xs=umap_3d[:, 0],
        ys=umap_3d[:, 1],
        zs=umap_3d[:, 2],
        clusters=clusters,
        atlas_ids=atlas_ids,
        tile_indices=tile_indices,
    )
    db_mb = METADATA_DB_PATH.stat().st_size / (1024**2)
    print(
        f"[SUKCES] Zapisano {METADATA_DB_PATH.name} ({db_mb:.1f} MB) w {time.perf_counter() - t_db:.1f} s."
    )

    # ---------------------------------------------------------
    # LEKKI PLIK ARROW DLA FRONTENDU (tylko x, y, z, cluster, atlas_id, tile_index)
    # ---------------------------------------------------------
    print("[INFO] Zapisywanie lekkiego pliku Arrow dla frontendu...")
    viz_table = pa.table(
        {
            "x": pa.array(umap_3d[:, 0], type=pa.float32()),
            "y": pa.array(umap_3d[:, 1], type=pa.float32()),
            "z": pa.array(umap_3d[:, 2], type=pa.float32()),
            "cluster": pa.array(clusters, type=pa.uint16()),
            "atlas_id": pa.array(atlas_ids, type=pa.int32()),
            "tile_index": pa.array(tile_indices, type=pa.int32()),
        }
    )

    tmp_viz_path = VIZ_DATA_PATH.with_suffix(".arrow.tmp")
    feather.write_feather(viz_table, tmp_viz_path)
    os.replace(tmp_viz_path, VIZ_DATA_PATH)

    size_mb = VIZ_DATA_PATH.stat().st_size / (1024**2)
    print(
        f"[SUKCES] Zapisano lekki plik {VIZ_DATA_PATH.name} ({size_mb:.2f} MB). Backend jest gotowy do startu."
    )


if __name__ == "__main__":
    main()
