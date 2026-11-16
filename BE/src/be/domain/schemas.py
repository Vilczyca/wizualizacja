from pydantic import BaseModel, ConfigDict, Field


class InferenceResult(BaseModel):
    x: float = Field(..., description="Koordynata X w przestrzeni UMAP 3D")
    y: float = Field(..., description="Koordynata Y w przestrzeni UMAP 3D")
    z: float = Field(..., description="Koordynata Z w przestrzeni UMAP 3D")

    model_config = ConfigDict(
        json_schema_extra={"example": {"x": 4.12, "y": -1.34, "z": 8.01}}
    )
