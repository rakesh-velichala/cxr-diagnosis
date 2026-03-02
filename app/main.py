"""FastAPI backend for the CXR Diagnosis application."""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.config import settings
from app.pipeline import DiagnosisPipeline
from utils.logging_config import logger

# ---------------------------------------------------------------------------
# Application lifespan — load models once at startup
# ---------------------------------------------------------------------------

pipeline: DiagnosisPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the diagnosis pipeline on startup; release on shutdown."""
    global pipeline  # noqa: PLW0603
    logger.info("Starting up — loading models …")
    pipeline = DiagnosisPipeline()
    logger.info("Models loaded — server ready")
    yield
    logger.info("Shutting down")
    pipeline = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CXR Diagnosis API",
    description=(
        "Chest X-Ray diagnosis using Vision-Language Models with "
        "retrieval-augmented generation. For educational use only."
    ),
    version="0.1.0",
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
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    """Analyze an uploaded chest X-ray image.

    Parameters
    ----------
    file : UploadFile
        The chest X-ray image (PNG, JPEG).

    Returns
    -------
    dict
        JSON response with ``report``, ``similar_cases``, and ``disclaimer``.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Validate content type.
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
        result = pipeline.diagnose(image)
    except Exception as exc:
        logger.exception("Diagnosis pipeline failed")
        raise HTTPException(
            status_code=500, detail=f"Inference error: {exc}"
        ) from exc

    return {
        "report": result.report,
        "similar_cases": result.similar_cases,
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
