from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope

from be.api.routes import router
from be.config import settings
from be.infrastructure.database import init_db
from be.services.ml_service import ml_service


class CachedStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=86400, immutable"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):

    settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    ml_service.initialize()
    yield


app = FastAPI(
    title="Wizualizacja - VideoGameBunny Backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

app.mount(
    "/api/uploads",
    CachedStaticFiles(directory=settings.UPLOADS_DIR, check_dir=False),
    name="uploads",
)

app.mount(
    "/api/atlases",
    CachedStaticFiles(directory=settings.ATLASES_DIR, check_dir=False),
    name="atlases",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("be.main:app", host="localhost", port=8000, reload=True)
