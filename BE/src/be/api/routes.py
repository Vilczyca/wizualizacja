import io
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from be.config import settings
from be.domain.schemas import InferenceResult
from be.services.ml_service import ml_service

router = APIRouter(prefix="/api")


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


@router.post(
    "/infer", response_model=InferenceResult, summary="Inferencja nowej próbki (live)"
)
async def infer_new_sample(file: Annotated[UploadFile, File(...)]):
    """Ekstrakcja nowego zrzutu ekranu i rzutowanie go na istniejącą mapę UMAP."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Przesłany plik nie jest obrazem.")

    image_bytes = await file.read()
    if len(image_bytes) > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Plik za duży (max 10MB).")

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb_image = img.convert("RGB")
        x, y, z = await run_in_threadpool(ml_service.infer_image, rgb_image)
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(
            status_code=400, detail="Uszkodzony lub nieobsługiwany format obrazu."
        ) from e
    except (RuntimeError, ValueError) as e:
        raise HTTPException(
            status_code=500, detail=f"Błąd przetwarzania AI: {e!s}"
        ) from e

    return InferenceResult(x=x, y=y, z=z)
