import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

from be.config import settings


def build_file_index(images_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    suffix_index: dict[str, Path] = {}
    basename_index: dict[str, Path] = {}

    for root, _, files in os.walk(images_root):
        root_path = Path(root)
        for fname in files:
            full_path = root_path / fname
            rel_posix = full_path.relative_to(images_root).as_posix().lower()
            parts = rel_posix.split("/")
            suffix_index[rel_posix] = full_path
            if len(parts) >= 2:
                tail_two = "/".join(parts[-2:])
                suffix_index.setdefault(tail_two, full_path)
            basename_index.setdefault(fname.lower(), full_path)

    return suffix_index, basename_index


def resolve_image_path(
    rel_path: str,
    suffix_index: dict[str, Path],
    basename_index: dict[str, Path],
) -> Path | None:
    norm = rel_path.replace("\\", "/").lstrip("/").lower()
    if norm in suffix_index:
        return suffix_index[norm]

    parts = norm.split("/")
    if len(parts) >= 2:
        tail_two = "/".join(parts[-2:])
        if tail_two in suffix_index:
            return suffix_index[tail_two]

    return basename_index.get(parts[-1])


def render_tile(
    args: tuple[int, Path | None, int],
) -> tuple[int, Image.Image | None]:
    row_idx, img_path, tile_size = args
    if img_path is None:
        return row_idx, None

    try:
        with Image.open(img_path) as img:
            rgb_img = ImageOps.exif_transpose(img).convert("RGB")
            tile = ImageOps.fit(
                rgb_img,
                (tile_size, tile_size),
                method=Image.Resampling.LANCZOS,
            )
            return row_idx, tile
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        return row_idx, None


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

    print(f"[INFO] Wczytywanie metadanych z {settings.VIZ_DATA_PATH}...")
    df = pd.read_feather(settings.VIZ_DATA_PATH)
    total_rows = len(df)

    print(f"[INFO] Indeksowanie plików obrazów w {settings.BASE_IMAGES_DIR}...")
    t0 = time.perf_counter()
    suffix_index, basename_index = build_file_index(settings.BASE_IMAGES_DIR)
    print(
        f"[INFO] Zaindeksowano {len(basename_index)} plików w {time.perf_counter() - t0:.1f} s."
    )

    grid_size = settings.ATLAS_GRID_SIZE
    tile_size = settings.ATLAS_TILE_SIZE
    tiles_per_atlas = grid_size * grid_size
    canvas_px = grid_size * tile_size

    resolved_paths: list[Path | None] = [
        resolve_image_path(str(rel), suffix_index, basename_index)
        for rel in df["image_path"].tolist()
    ]

    atlas_ids = np.full(total_rows, -1, dtype=np.int32)
    tile_indices = np.full(total_rows, -1, dtype=np.int32)

    current_atlas_id = 0
    current_tile_idx = 0
    packed_count = 0
    missing_count = 0
    canvas = Image.new("RGB", (canvas_px, canvas_px), color=(0, 0, 0))

    start_render = time.perf_counter()
    batch_size = tiles_per_atlas

    with ThreadPoolExecutor(max_workers=settings.NUM_WORKERS) as executor:
        for batch_start in tqdm(
            range(0, total_rows, batch_size),
            desc="Generowanie atlasów WebP",
        ):
            batch_end = min(batch_start + batch_size, total_rows)
            tasks = [
                (idx, resolved_paths[idx], tile_size)
                for idx in range(batch_start, batch_end)
            ]

            for row_idx, tile in executor.map(render_tile, tasks):
                if tile is None:
                    missing_count += 1
                    continue

                col = current_tile_idx % grid_size
                row = current_tile_idx // grid_size
                canvas.paste(tile, (col * tile_size, row * tile_size))

                atlas_ids[row_idx] = current_atlas_id
                tile_indices[row_idx] = current_tile_idx
                packed_count += 1
                current_tile_idx += 1

                if current_tile_idx == tiles_per_atlas:
                    out_path = (
                        settings.ATLASES_DIR / f"atlas_{current_atlas_id:04d}.webp"
                    )
                    canvas.save(out_path, format="WEBP", quality=80, method=4)
                    current_atlas_id += 1
                    current_tile_idx = 0
                    canvas = Image.new("RGB", (canvas_px, canvas_px), color=(0, 0, 0))

    if current_tile_idx > 0:
        out_path = settings.ATLASES_DIR / f"atlas_{current_atlas_id:04d}.webp"
        canvas.save(out_path, format="WEBP", quality=80, method=4)
        total_atlases = current_atlas_id + 1
    else:
        total_atlases = current_atlas_id

    df["atlas_id"] = atlas_ids
    df["tile_index"] = tile_indices
    df.to_feather(settings.VIZ_DATA_PATH)

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
