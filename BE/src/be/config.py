import os
from pathlib import Path


def _get_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw and raw.strip() else default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw and raw.strip() else default


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
    PREFETCH_FACTOR = _get_int("PREFETCH_FACTOR", default=4)
    CHECKPOINT_FREQ = 1024 if DRY_RUN else 10240

    UMAP_PCA_DIM = _get_int("UMAP_PCA_DIM", default=50)
    UMAP_N_NEIGHBORS = _get_int("UMAP_N_NEIGHBORS", default=30)
    UMAP_MIN_DIST = _get_float("UMAP_MIN_DIST", default=0.1)
    N_CLUSTERS = _get_int("N_CLUSTERS", default=40)

    RAW_DATA_DIR = BASE_DIR / "raw_data"
    METADATA_CACHE_PATH = BASE_DIR / (
        "metadata_cache_dry_run.parquet" if DRY_RUN else "metadata_cache.parquet"
    )
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
    METADATA_DB_PATH = BASE_DIR / ("metadata_dry_run.db" if DRY_RUN else "metadata.db")
    CLUSTERS_PATH = BASE_DIR / ("clusters_dry_run.json" if DRY_RUN else "clusters.json")
    UPLOADS_DIR = BASE_DIR / "uploads"
    USER_DB_PATH = BASE_DIR / "user_samples.db"
    PREVIEW_MAX_WIDTH = 1280
    THUMB_SIZE = (128, 128)

    BASE_IMAGES_DIR = RAW_DATA_DIR / ("eval_images" if DRY_RUN else "train_images")
    ATLASES_DIR = BASE_DIR / ("atlases_dry_run" if DRY_RUN else "atlases")
    ATLAS_GRID_SIZE = 32
    ATLAS_TILE_SIZE = 64
    BASE_PREVIEW_MAX_WIDTH = 800

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
