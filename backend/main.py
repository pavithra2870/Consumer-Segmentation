"""
Consumer Intelligence Platform — FastAPI Backend
Entry point: uvicorn main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from utils.logger import logger, setup_logger
from routers import data, ml_models, segments, personas, predictions, explanations, bias


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger(settings.debug)
    # Ensure directories exist
    Path("logs").mkdir(exist_ok=True)
    settings.data_path.mkdir(parents=True, exist_ok=True)
    settings.models_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    from utils.ollama_client import ollama
    available = await ollama.check_availability()
    logger.info(f"Ollama availability: {available} (model: {settings.ollama_model})")

    yield
    logger.info("Shutting down Consumer Intelligence Platform")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="End-to-end consumer segmentation, prediction, and persona platform with local LLM support.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router)
app.include_router(ml_models.router)
app.include_router(segments.router)
app.include_router(personas.router)
app.include_router(predictions.router)
app.include_router(explanations.router)
app.include_router(bias.router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    from utils.ollama_client import ollama
    return {
        "status": "ok",
        "ollama_available": ollama._available,
        "data_ready": (settings.data_path / "consumers.csv").exists(),
        "models_ready": (settings.models_path / "churn_model.pkl").exists(),
    }
