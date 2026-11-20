import uuid
from dataclasses import dataclass

from PIL import Image, ImageOps

from be.config import settings


@dataclass(frozen=True)
class StoredImageFiles:
    sample_id: str
    filename: str
    thumb_filename: str


class ImageStorageService:
    def save_image(self, image: Image.Image) -> StoredImageFiles:
        settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

        sample_id = uuid.uuid4().hex
        filename = f"{sample_id}.webp"
        thumb_filename = f"{sample_id}_thumb.webp"

        transposed = ImageOps.exif_transpose(image)
        rgb_image = transposed.convert("RGB")

        preview_image = rgb_image.copy()
        preview_image.thumbnail(
            (settings.PREVIEW_MAX_WIDTH, settings.PREVIEW_MAX_WIDTH),
            resample=Image.Resampling.LANCZOS,
        )

        thumb_image = ImageOps.fit(
            rgb_image,
            settings.THUMB_SIZE,
            method=Image.Resampling.LANCZOS,
        )

        preview_path = settings.UPLOADS_DIR / filename
        thumb_path = settings.UPLOADS_DIR / thumb_filename

        preview_image.save(preview_path, format="WEBP", quality=85, method=4)
        thumb_image.save(thumb_path, format="WEBP", quality=80, method=4)

        return StoredImageFiles(
            sample_id=sample_id,
            filename=filename,
            thumb_filename=thumb_filename,
        )

    def delete_files(self, filename: str, thumb_filename: str) -> None:
        (settings.UPLOADS_DIR / filename).unlink(missing_ok=True)
        (settings.UPLOADS_DIR / thumb_filename).unlink(missing_ok=True)


storage_service = ImageStorageService()
