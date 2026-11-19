import multiprocessing
import os
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image, ImageOps, UnidentifiedImageError
from pyarrow import feather
from tqdm import tqdm

from be.config import settings
from be.infrastructure.image_index import build_file_index, resolve_image_path
from be.infrastructure.metadata_db import get_image_paths_ordered, update_atlas_mappings


def render_tile(
    args: tuple[int, str | None, int],
) -> tuple[int, bytes | None]:
    row_idx, img_path, tile_size = args
    if img_path is None:
        return row_idx, None

    try:
        with Image.open(img_path) as img:
            draft_dim = tile_size * 2
            img.draft("RGB", (draft_dim, draft_dim))
            rgb_img = ImageOps.exif_transpose(img).convert("RGB")
            tile = ImageOps.fit(
                rgb_img,
                (tile_size, tile_size),
                method=Image.Resampling.LANCZOS,
            )
            return row_idx, tile.tobytes()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        return row_idx, None


def _save_atlas_atomic(canvas: Image.Image, out_path: Path) -> None:
    tmp_path = out_path.with_suffix(".webp.tmp")
    canvas.save(tmp_path, format="WEBP", quality=80, method=4)
    os.replace(tmp_path, out_path)


def main() -> None:
    if not settings.VIZ_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Brak pliku wizualizacji: {settings.VIZ_DATA_PATH}. "
            "Należy najpierw uruchomić src/train_reducer.py."
        )

    if not settings.BASE_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Brak katalogu surowych obrazów: {settings.BASE_IMAGES_DIR}."
        )

    settings.ATLASES_DIR.mkdir(parents=True, exist_ok=True)
    for old_atlas in settings.ATLASES_DIR.glob("atlas_*.webp"):
        old_atlas.unlink()

    print(f"[INFO] Wczytywanie punktów z {settings.VIZ_DATA_PATH}...")
    viz_table = feather.read_table(settings.VIZ_DATA_PATH)
    total_rows = viz_table.num_rows

    if "image_path" in viz_table.column_names:
        image_paths = viz_table.column("image_path").to_pylist()
    elif settings.METADATA_DB_PATH.exists():
        image_paths = get_image_paths_ordered(settings.METADATA_DB_PATH)
    else:
        raise FileNotFoundError(
            f"Brak kolumny image_path w {settings.VIZ_DATA_PATH} oraz brak bazy {settings.METADATA_DB_PATH}."
        )

    if len(image_paths) != total_rows:
        raise ValueError(
            f"Niezgodna liczba wierszy: viz_data={total_rows}, metadata={len(image_paths)}"
        )

    print(f"[INFO] Indeksowanie plików obrazów w {settings.BASE_IMAGES_DIR}...")
    t0 = time.perf_counter()
    suffix_index, basename_index = build_file_index(settings.BASE_IMAGES_DIR)
    print(
        f"[INFO] Zaindeksowano {len(basename_index)} plików w {time.perf_counter() - t0:.2f} s."
    )

    grid_size = settings.ATLAS_GRID_SIZE
    tile_size = settings.ATLAS_TILE_SIZE
    tiles_per_atlas = grid_size * grid_size
    canvas_px = grid_size * tile_size

    resolved_paths: list[str | None] = [
        str(p)
        if (p := resolve_image_path(str(rel), suffix_index, basename_index)) is not None
        else None
        for rel in image_paths
    ]

    atlas_ids = np.full(total_rows, -1, dtype=np.int32)
    tile_indices = np.full(total_rows, -1, dtype=np.int32)

    current_atlas_id = 0
    current_tile_idx = 0
    packed_count = 0
    missing_count = 0
    canvas = Image.new("RGB", (canvas_px, canvas_px), color=(0, 0, 0))

    start_render = time.perf_counter()
    workers = max(1, settings.NUM_WORKERS)
    mp_ctx = multiprocessing.get_context("fork" if sys.platform != "win32" else "spawn")

    tasks = ((idx, resolved_paths[idx], tile_size) for idx in range(total_rows))
    pending_saves: list[Future[None]] = []

    with (
        ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as executor,
        ThreadPoolExecutor(max_workers=2) as save_executor,
    ):
        for row_idx, tile_bytes in tqdm(
            executor.map(render_tile, tasks, chunksize=64),
            total=total_rows,
            desc="Generowanie atlasów WebP",
        ):
            if tile_bytes is None:
                missing_count += 1
                continue

            tile = Image.frombytes("RGB", (tile_size, tile_size), tile_bytes)
            col = current_tile_idx % grid_size
            row = current_tile_idx // grid_size
            canvas.paste(tile, (col * tile_size, row * tile_size))

            atlas_ids[row_idx] = current_atlas_id
            tile_indices[row_idx] = current_tile_idx
            packed_count += 1
            current_tile_idx += 1

            if current_tile_idx == tiles_per_atlas:
                out_path = settings.ATLASES_DIR / f"atlas_{current_atlas_id:04d}.webp"
                pending_saves.append(
                    save_executor.submit(_save_atlas_atomic, canvas, out_path)
                )
                current_atlas_id += 1
                current_tile_idx = 0
                canvas = Image.new("RGB", (canvas_px, canvas_px), color=(0, 0, 0))

        if current_tile_idx > 0:
            out_path = settings.ATLASES_DIR / f"atlas_{current_atlas_id:04d}.webp"
            pending_saves.append(
                save_executor.submit(_save_atlas_atomic, canvas, out_path)
            )
            total_atlases = current_atlas_id + 1
        else:
            total_atlases = current_atlas_id

        for fut in pending_saves:
            fut.result()

    for col_name in ("atlas_id", "tile_index"):
        if col_name in viz_table.column_names:
            viz_table = viz_table.drop([col_name])

    viz_table = viz_table.append_column(
        "atlas_id", pa.array(atlas_ids, type=pa.int32())
    )
    viz_table = viz_table.append_column(
        "tile_index", pa.array(tile_indices, type=pa.int32())
    )

    tmp_viz_path = settings.VIZ_DATA_PATH.with_suffix(".arrow.tmp")
    feather.write_feather(viz_table, tmp_viz_path)
    os.replace(tmp_viz_path, settings.VIZ_DATA_PATH)

    if settings.METADATA_DB_PATH.exists():
        print(
            f"[INFO] Aktualizacja atlas_id/tile_index w {settings.METADATA_DB_PATH.name}..."
        )
        update_atlas_mappings(settings.METADATA_DB_PATH, atlas_ids, tile_indices)

    elapsed = time.perf_counter() - start_render
    total_size_mb = sum(
        p.stat().st_size for p in settings.ATLASES_DIR.glob("atlas_*.webp")
    ) / (1024**2)

    print("\n" + "=" * 55)
    print(f"[SUKCES] Wygenerowano atlasów: {total_atlases} w czasie {elapsed:.1f} s.")
    print(f"[METRYKI] Spakowane miniatury: {packed_count} | Pominięte: {missing_count}")
    print(f"[METRYKI] Łączny rozmiar atlasów: {total_size_mb:.2f} MB")
    print(f"[SUKCES] Zaktualizowano plik {settings.VIZ_DATA_PATH.name}.")
    print("=" * 55)


if __name__ == "__main__":
    main()
