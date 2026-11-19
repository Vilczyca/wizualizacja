from pydantic import BaseModel, ConfigDict, Field


class InferenceResult(BaseModel):
    x: float = Field(..., description="Koordynata X w przestrzeni UMAP 3D")
    y: float = Field(..., description="Koordynata Y w przestrzeni UMAP 3D")
    z: float = Field(..., description="Koordynata Z w przestrzeni UMAP 3D")

    model_config = ConfigDict(
        json_schema_extra={"example": {"x": 4.12, "y": -1.34, "z": 8.01}}
    )


class UserSampleResponse(BaseModel):
    id: str = Field(..., description="Unikalny identyfikator próbki UUID")
    x: float = Field(..., description="Koordynata X w przestrzeni UMAP 3D")
    y: float = Field(..., description="Koordynata Y w przestrzeni UMAP 3D")
    z: float = Field(..., description="Koordynata Z w przestrzeni UMAP 3D")
    category: str = Field(default="", description="Opcjonalna kategoria obrazu")
    caption: str = Field(default="", description="Opcjonalny opis obrazu")
    image_url: str = Field(..., description="Adres URL pełnego podglądu WebP")
    thumb_url: str = Field(..., description="Adres URL miniatury WebP")
    created_at: str = Field(
        ..., description="Znacznik czasu utworzenia w formacie ISO 8601"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "a1b2c3d4e5f678901234567890abcdef",
                "x": 4.12,
                "y": -1.34,
                "z": 8.01,
                "category": "RPG",
                "caption": "Zrzut ekranu z walki z bossem",
                "image_url": "/api/uploads/a1b2c3d4e5f678901234567890abcdef.webp",
                "thumb_url": "/api/uploads/a1b2c3d4e5f678901234567890abcdef_thumb.webp",
                "created_at": "2026-11-18T18:30:00+00:00",
            }
        }
    )


class SampleDetailResponse(BaseModel):
    idx: int = Field(..., description="Indeks punktu w pliku viz_data.arrow")
    sample_idx: str = Field(..., description="Oryginalny identyfikator próbki")
    image_path: str = Field(..., description="Względna ścieżka obrazu w zbiorze")
    image_url: str = Field(..., description="Gotowy adres URL podglądu WebP")
    video_id: str = Field(default="", description="Identyfikator wideo YouTube")
    category: str = Field(default="", description="Kategoria obrazu")
    caption: str = Field(default="", description="Pełny opis tekstowy kadru")
    x: float
    y: float
    z: float
    cluster: int = Field(..., description="ID klastra KMeans")
    atlas_id: int = Field(..., description="ID pliku atlasu WebP (-1 jeśli brak)")
    tile_index: int = Field(..., description="Indeks kafelka w atlasie (-1 jeśli brak)")


class ClusterInfo(BaseModel):
    id: int
    color: str
    count: int


class DatasetBounds(BaseModel):
    min: list[float]
    max: list[float]


class DatasetMetaResponse(BaseModel):
    total_points: int
    atlas_grid_size: int
    atlas_tile_size: int
    atlas_count: int
    clusters_count: int
    bounds: DatasetBounds
