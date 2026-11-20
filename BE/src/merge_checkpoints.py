from pathlib import Path

import numpy as np
import pandas as pd

from be.config import settings

CHECKPOINTS_DIR = settings.CHECKPOINTS_DIR
OUTPUT_FILE = settings.RAW_EMBEDDINGS_PATH


def merge_parquet_checkpoints(checkpoints_dir: Path, output_file: Path) -> pd.DataFrame:
    ckpt_files = sorted(checkpoints_dir.glob("features_ckpt_*.parquet"))
    if not ckpt_files:
        raise FileNotFoundError(
            f"Nie znaleziono plików checkpointów w: {checkpoints_dir}"
        )

    print(f"[INFO] Znaleziono plików checkpointów: {len(ckpt_files)}")
    frames = [pd.read_parquet(path, engine="pyarrow") for path in ckpt_files]
    df = pd.concat(frames, ignore_index=True)

    initial_len = len(df)
    df = df.drop_duplicates(subset=["image_path"], keep="first").reset_index(drop=True)
    if len(df) != initial_len:
        print(f"[WARN] Usunięto zduplikowane ścieżki obrazów: {initial_len - len(df)}")

    if df["embedding"].isna().any():
        raise ValueError("Wykryto puste wartości (NaN) w kolumnie embedding!")

    emb_dim = len(df["embedding"].iloc[0])
    print(
        f"[INFO] Łączna liczba unikalnych próbek: {len(df)} | Wymiar wektora: {emb_dim}"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, engine="pyarrow", index=False)
    size_mb = output_file.stat().st_size / (1024**2)
    print(f"[SUKCES] Zapisano zbiorczy plik: {output_file} ({size_mb:.2f} MB)")
    return df


def run_cosine_similarity_sanity_check(
    df: pd.DataFrame, query_idx: int = 0, top_k: int = 3
) -> None:
    matrix = np.vstack(df["embedding"].to_numpy()).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)

    if np.isnan(normalized).any():
        raise ValueError("Macierz wektorów zawiera wartości NaN po normalizacji!")

    similarities = normalized @ normalized[query_idx]

    sorted_indices = np.argsort(similarities)[::-1]
    most_similar = [idx for idx in sorted_indices if idx != query_idx][:top_k]
    least_similar = sorted_indices[-top_k:][::-1]

    query_row = df.iloc[query_idx]
    print("\n" + "=" * 60)
    print(f"[SANITY CHECK] Obraz bazowy (idx={query_idx}): {query_row['image_path']}")
    print(f"Kategoria: {query_row['category']} | Opis: {query_row['caption'][:100]}...")
    print("-" * 60)
    print("Najbardziej podobne wizualnie (wysokie Cosine Similarity):")
    for idx in most_similar:
        row = df.iloc[idx]
        print(
            f"  score={similarities[idx]:.4f} | idx={idx} | {row['image_path']} | {row['caption'][:70]}"
        )

    print("-" * 60)
    print("Najmniej podobne wizualnie (niskie Cosine Similarity):")
    for idx in least_similar:
        row = df.iloc[idx]
        print(
            f"  score={similarities[idx]:.4f} | idx={idx} | {row['image_path']} | {row['caption'][:70]}"
        )
    print("=" * 60)


def main() -> None:
    df = merge_parquet_checkpoints(CHECKPOINTS_DIR, OUTPUT_FILE)
    run_cosine_similarity_sanity_check(df, query_idx=0, top_k=3)


if __name__ == "__main__":
    main()
