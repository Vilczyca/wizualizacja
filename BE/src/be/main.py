from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from be.api.routes import router
from be.config import settings
from be.services.ml_service import ml_service


@asynccontextmanager
async def lifespan(app: FastAPI):

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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("be.main:app", host="localhost", port=8000, reload=True)
