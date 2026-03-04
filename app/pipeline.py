"""Simplified diagnosis pipeline: image → VLM → structured diagnoses."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from PIL import Image

from app.config import settings
from models.base import BaseModel, Diagnosis, load_model
from prompts.prompt_builder import MEDICAL_DISCLAIMER
from utils.logging_config import logger


@dataclass
class DiagnosisResult:
    """Structured output from the diagnosis pipeline."""

    diagnoses: list[Diagnosis]
    model_name: str
    disclaimer: str = MEDICAL_DISCLAIMER


class DiagnosisPipeline:
    """Orchestrate VLM-based diagnosis.

    Lifecycle
    ---------
    1. ``__init__`` — loads the selected model backend.
    2. ``diagnose(image)`` — runs inference on one image.
    """

    def __init__(self, backend: str | None = None) -> None:
        self.backend_name = backend or settings.model_backend
        logger.info("Initializing DiagnosisPipeline (backend=%s)", self.backend_name)
        self.model: BaseModel = load_model(self.backend_name)
        logger.info("DiagnosisPipeline ready — model: %s", self.model.name)

    def diagnose(self, image: Image.Image) -> DiagnosisResult:
        """Run diagnosis on a single chest X-ray.

        Parameters
        ----------
        image : PIL.Image.Image
            Uploaded chest X-ray image (RGB).

        Returns
        -------
        DiagnosisResult
            Structured diagnoses and disclaimer.
        """
        image = image.convert("RGB")

        # Free GPU memory before inference.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Running %s inference", self.model.name)
        diagnoses = self.model.diagnose(image)
        logger.info("Diagnosis complete: %s", diagnoses)

        return DiagnosisResult(
            diagnoses=diagnoses,
            model_name=self.model.name,
        )
