import io
import os
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError

from be.config import settings
from be.domain.schemas import UserSampleResponse
from be.infrastructure.database import (
    UserSampleRecord,
    insert_user_sample,
    list_user_samples,
)
from be.services.ml_service import ml_service
from be.services.storage_service import StoredImageFiles, storage_service

router = APIRouter(prefix="/api")


def _to_response(record: UserSampleRecord) -> UserSampleResponse:
    return UserSampleResponse(
        id=record.id,
        x=record.x,
        y=record.y,
        z=record.z,
        category=record.category,
        caption=record.caption,
        image_url=f"/api/uploads/{record.filename}",
        thumb_url=f"/api/uploads/{record.thumb_filename}",
        created_at=record.created_at,
    )


def _decode_and_store(raw_bytes: bytes) -> StoredImageFiles:
    with Image.open(io.BytesIO(raw_bytes)) as img:
        return storage_service.save_image(img)


def _infer_stored_preview(filename: str) -> tuple[float, float, float]:
    preview_path = settings.UPLOADS_DIR / filename
    with Image.open(preview_path) as preview_img:
        return ml_service.infer_image(preview_img)


@router.get("/viz-data", summary="Pobranie pełnej mapy wektorów 3D")
async def get_visualization_data():
    if not settings.VIZ_DATA_PATH.exists():
        raise HTTPException(
            status_code=404, detail="Plik wizualizacji nie został jeszcze wygenerowany."
        )

    return FileResponse(
        path=settings.VIZ_DATA_PATH,
        media_type="application/vnd.apache.arrow.file",
        filename="viz_data.arrow",
    )


@router.get(
    "/user-samples",
    response_model=list[UserSampleResponse],
    summary="Pobranie próbek dodanych przez użytkowników",
)
async def get_user_samples():
    """Zwraca listę wszystkich obrazów przesłanych przez użytkowników."""
    records = await run_in_threadpool(list_user_samples)
    return [_to_response(record) for record in records]


@router.post(
    "/infer",
    status_code=201,
    response_model=UserSampleResponse,
    summary="Inferencja i zapis nowej próbki",
)
async def infer_new_sample(
    file: Annotated[UploadFile, File(...)],
    caption: Annotated[str, Form()] = "",
    category: Annotated[str, Form()] = "",
):
    """Zapisuje nowy obraz, wylicza współrzędne UMAP 3D i dodaje rekord do bazy."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Przesłany plik nie jest obrazem.")

    image_bytes = await file.read()
    if len(image_bytes) > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Plik za duży (max 10MB).")

    try:
        stored = await run_in_threadpool(_decode_and_store, image_bytes)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as e:
        raise HTTPException(
            status_code=400, detail="Uszkodzony lub nieobsługiwany format obrazu."
        ) from e

    try:
        x, y, z = await run_in_threadpool(_infer_stored_preview, stored.filename)
    except (RuntimeError, ValueError, OSError) as e:
        await run_in_threadpool(
            storage_service.delete_files, stored.filename, stored.thumb_filename
        )
        raise HTTPException(
            status_code=500, detail=f"Błąd przetwarzania AI: {e!s}"
        ) from e

    try:
        record = await run_in_threadpool(
            insert_user_sample,
            stored.sample_id,
            stored.filename,
            stored.thumb_filename,
            x,
            y,
            z,
            category[:100],
            caption[:500],
        )
    except sqlite3.Error as e:
        await run_in_threadpool(
            storage_service.delete_files, stored.filename, stored.thumb_filename
        )
        raise HTTPException(
            status_code=500, detail="Błąd zapisu metadanych do bazy danych."
        ) from e

    return _to_response(record)


@lru_cache(maxsize=1)
def _get_base_image_indexes() -> tuple[dict[str, Path], dict[str, Path]]:
    base_dir = settings.BASE_IMAGES_DIR.resolve()
    suffix_index: dict[str, Path] = {}
    basename_index: dict[str, Path] = {}

    if not base_dir.exists():
        return suffix_index, basename_index

    for root, _, files in os.walk(base_dir):
        root_path = Path(root)
        for fname in files:
            full_path = root_path / fname
            rel_posix = full_path.relative_to(base_dir).as_posix().lower()
            parts = rel_posix.split("/")
            suffix_index[rel_posix] = full_path
            if len(parts) >= 2:
                tail_two = "/".join(parts[-2:])
                suffix_index.setdefault(tail_two, full_path)
            basename_index.setdefault(fname.lower(), full_path)

    return suffix_index, basename_index


def _resolve_safe_base_image(rel_path: str) -> Path:
    base_dir = settings.BASE_IMAGES_DIR.resolve()
    clean_rel = rel_path.replace("\\", "/").lstrip("/")
    if not clean_rel:
        raise HTTPException(status_code=400, detail="Pusta ścieżka obrazu.")

    candidate = (base_dir / clean_rel).resolve()
    if not candidate.is_relative_to(base_dir):
        raise HTTPException(status_code=403, detail="Niedozwolona ścieżka pliku.")

    if candidate.is_file():
        return candidate

    suffix_index, basename_index = _get_base_image_indexes()
    norm = clean_rel.lower()
    if norm in suffix_index:
        return suffix_index[norm]

    parts = norm.split("/")
    if len(parts) >= 2:
        tail_two = "/".join(parts[-2:])
        if tail_two in suffix_index:
            return suffix_index[tail_two]

    resolved = basename_index.get(parts[-1])
    if resolved is not None and resolved.is_file():
        return resolved

    raise HTTPException(status_code=404, detail="Nie znaleziono wskazanego obrazu.")


def _render_base_preview(image_path: Path) -> bytes:
    with Image.open(image_path) as img:
        rgb_img = ImageOps.exif_transpose(img).convert("RGB")
        rgb_img.thumbnail(
            (settings.BASE_PREVIEW_MAX_WIDTH, settings.BASE_PREVIEW_MAX_WIDTH),
            resample=Image.Resampling.LANCZOS,
        )
        buffer = io.BytesIO()
        rgb_img.save(buffer, format="WEBP", quality=85, method=4)
        return buffer.getvalue()


@router.get("/image", summary="Pobranie podglądu kadru ze zbioru bazowego")
async def get_base_image(
    path: Annotated[
        str,
        Query(
            min_length=1,
            max_length=512,
            description="Względna ścieżka obrazu z kolumny image_path",
        ),
    ],
):
    """Zwraca pojedynczy zrzut ekranu przeskalowany do formatu WebP."""
    resolved_path = _resolve_safe_base_image(path)

    try:
        webp_bytes = await run_in_threadpool(_render_base_preview, resolved_path)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as e:
        raise HTTPException(
            status_code=500, detail="Błąd odczytu lub konwersji pliku obrazu."
        ) from e

    return Response(
        content=webp_bytes,
        media_type="image/webp",
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )
