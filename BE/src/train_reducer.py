import time

import joblib
import numpy as np
import pandas as pd
from umap import UMAP

from be.config import settings

INPUT_FILE = settings.RAW_EMBEDDINGS_PATH
UMAP_MODEL_PATH = settings.UMAP_MODEL_PATH
VIZ_DATA_PATH = settings.VIZ_DATA_PATH


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Brak scalonego pliku embeddingów: {INPUT_FILE}. "
            "Uruchom najpierw src/merge_checkpoints.py."
        )

    UMAP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Wczytywanie scalonych wektorów z {INPUT_FILE}...")
    start_time = time.perf_counter()

    df = pd.read_parquet(INPUT_FILE, engine="pyarrow")
    print(
        f"[INFO] Załadowano {len(df)} wektorów w {time.perf_counter() - start_time:.1f} s."
    )

    print("[INFO] Konwersja do macierzy Float32...")
    embeddings = np.vstack(df["embedding"].to_numpy()).astype(np.float32)

    # ---------------------------------------------------------
    # ZADANIE GŁÓWNE: UMAP (Używany docelowo w aplikacji API)
    # ---------------------------------------------------------
    print("[INFO] Trening modelu UMAP wielowątkowo (to potrwa kilka minut)...")
    t0 = time.perf_counter()
    umap_model = UMAP(
        n_components=3,
        metric="cosine",
        n_jobs=-1,
        compatibility_layout=True,
    )
    umap_3d = umap_model.fit_transform(embeddings)
    print(f"[SUKCES] UMAP zakończony w {time.perf_counter() - t0:.1f} s.")

    print("[INFO] Zapisywanie modelu UMAP na dysk (dla FastAPI)...")
    joblib.dump(umap_model, UMAP_MODEL_PATH)

    df["x"] = umap_3d[:, 0]
    df["y"] = umap_3d[:, 1]
    df["z"] = umap_3d[:, 2]

    # ---------------------------------------------------------
    # PODZADANIA DLA ZESPOŁU (Opcjonalne, odkomentować w razie potrzeby)
    # ---------------------------------------------------------
    # print("[INFO] Obliczanie PCA do celów porównawczych...")
    # pca_3d = PCA(n_components=3).fit_transform(embeddings)
    # df["pca_x"], df["pca_y"], df["pca_z"] = pca_3d[:, 0], pca_3d[:, 1], pca_3d[:, 2]
    #
    # print("[INFO] Obliczanie t-SNE (UWAGA: bardzo długi czas wykonania na 185k próbek)...")
    # tsne_3d = TSNE(n_components=3, metric="cosine", n_jobs=-1).fit_transform(embeddings)
    # df["tsne_x"], df["tsne_y"], df["tsne_z"] = tsne_3d[:, 0], tsne_3d[:, 1], tsne_3d[:, 2]

    print("[INFO] Zapisywanie pliku Arrow dla frontendu...")
    df_viz = df.drop(columns=["embedding"])

    df_viz.to_feather(VIZ_DATA_PATH)

    size_mb = VIZ_DATA_PATH.stat().st_size / (1024**2)
    print(
        f"[SUKCES] Zapisano plik {VIZ_DATA_PATH.name} ({size_mb:.1f} MB). Backend jest gotowy do startu."
    )


if __name__ == "__main__":
    main()
