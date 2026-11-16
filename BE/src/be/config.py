import os
from pathlib import Path


def _get_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw and raw.strip() else default


IS_CLOUD = "SPACE_ID" in os.environ
DEFAULT_DATA_DIR = "/app/data" if IS_CLOUD else "./data"
DEFAULT_HF_HOME = "/app/cache" if IS_CLOUD else "./data/huggingface_cache"


class Settings:
    BASE_DIR = Path(os.getenv("VIZ_DATA_DIR", DEFAULT_DATA_DIR))
    HF_HOME = os.getenv("HF_HOME", DEFAULT_HF_HOME)

    DRY_RUN = _get_bool("DRY_RUN", default=False)
    DRY_RUN_LIMIT = _get_int("DRY_RUN_LIMIT", default=2048)
    MOCK_ML = _get_bool("MOCK_ML", default=False)

    BATCH_SIZE = _get_int("BATCH_SIZE", default=512)
    NUM_WORKERS = _get_int("NUM_WORKERS", default=6)
    CHECKPOINT_FREQ = 1024 if DRY_RUN else 10240

    RAW_DATA_DIR = BASE_DIR / "raw_data"
    CHECKPOINTS_DIR = BASE_DIR / ("checkpoints_dry_run" if DRY_RUN else "checkpoints")
    RAW_EMBEDDINGS_PATH = BASE_DIR / (
        "raw_embeddings_dry_run.parquet" if DRY_RUN else "raw_embeddings.parquet"
    )
    UMAP_MODEL_PATH = (
        BASE_DIR
        / "models"
        / ("umap_reducer_dry_run.pkl" if DRY_RUN else "umap_reducer.pkl")
    )
    VIZ_DATA_PATH = BASE_DIR / (
        "viz_data_dry_run.arrow" if DRY_RUN else "viz_data.arrow"
    )

    MAX_IMAGE_SIZE_MB = 10
    CORS_ORIGINS = tuple(
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if origin.strip()
    )

    CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
    DATASET_ID = "VideoGameBunny/Dataset"


settings = Settings()
os.environ["HF_HOME"] = settings.HF_HOME
