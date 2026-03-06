"""FastAPI backend for the CXR Diagnosis application."""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.config import settings
from app.pipeline import DiagnosisPipeline
from utils.logging_config import logger

# ---------------------------------------------------------------------------
# Application lifespan — load default model once at startup
# ---------------------------------------------------------------------------

_pipelines: dict[str, DiagnosisPipeline] = {}


def _get_pipeline(backend: str) -> DiagnosisPipeline:
    """Get or create a pipeline for the given backend."""
    if backend not in _pipelines:
        _pipelines[backend] = DiagnosisPipeline(backend=backend)
    return _pipelines[backend]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the default pipeline on startup; release on shutdown."""
    logger.info("Starting up — loading default model (%s)", settings.model_backend)
    _get_pipeline(settings.model_backend)
    logger.info("Models loaded — server ready")
    yield
    logger.info("Shutting down")
    _pipelines.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CXR Diagnosis API",
    description=(
        "Chest X-Ray diagnosis using Vision-Language Models. "
        "For educational use only."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query(default=None, description="Backend: qwen, chexagent, gpt4o"),
) -> dict[str, Any]:
    """Analyze an uploaded chest X-ray image.

    Parameters
    ----------
    file : UploadFile
        The chest X-ray image (PNG, JPEG).
    model : str, optional
        Model backend to use. Defaults to CXR_MODEL_BACKEND env var.

    Returns
    -------
    dict
        JSON response with ``diagnoses``, ``model``, and ``disclaimer``.
    """
    backend = model or settings.model_backend

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got {file.content_type}",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not read image: {exc}"
        ) from exc

    try:
        pipe = _get_pipeline(backend)
        result = pipe.diagnose(image)
    except Exception as exc:
        logger.exception("Diagnosis pipeline failed")
        raise HTTPException(
            status_code=500, detail=f"Inference error: {exc}"
        ) from exc

    return {
        "diagnoses": [
            {
                "disease": d.disease,
                "probability": d.probability,
                "confidence": d.confidence,
                "threshold": d.threshold,
            }
            for d in result.diagnoses
        ],
        "model": result.model_name,
        "disclaimer": result.disclaimer,
    }


# ---------------------------------------------------------------------------
# Entry-point for ``python -m app.main``
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
