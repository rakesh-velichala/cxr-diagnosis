"""End-to-end diagnosis pipeline: image → embedding → retrieval → VLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from PIL import Image

from app.config import settings
from data.loader import load_csv, load_embeddings
from data.retriever import DatasetRetriever, RetrievedCase
from models.clip_encoder import CLIPEncoder
from models.vlm_inference import VLMInference
from prompts.prompt_builder import build_prompt, MEDICAL_DISCLAIMER
from utils.logging_config import logger


@dataclass
class DiagnosisResult:
    """Structured output from the diagnosis pipeline."""

    report: str
    similar_cases: list[dict]
    disclaimer: str = MEDICAL_DISCLAIMER


class DiagnosisPipeline:
    """Orchestrate the full RAG-based diagnosis workflow.

    Lifecycle
    ---------
    1. ``__init__`` — loads models and dataset artefacts.
    2. ``diagnose(image)`` — runs the full pipeline on one image.
    """

    def __init__(self) -> None:
        logger.info("Initializing DiagnosisPipeline …")

        # Load dataset artefacts.
        self.df = load_csv()
        self.embeddings = load_embeddings()
        self.retriever = DatasetRetriever(self.embeddings, self.df)

        # Load ML models.
        self.clip_encoder = CLIPEncoder()
        self.vlm = VLMInference()

        logger.info("DiagnosisPipeline ready")

    def diagnose(self, image: Image.Image) -> DiagnosisResult:
        """Run the full RAG diagnosis on a single chest X-ray.

        Parameters
        ----------
        image : PIL.Image.Image
            Uploaded chest X-ray image (RGB).

        Returns
        -------
        DiagnosisResult
            Generated report, similar cases, and disclaimer.
        """
        # Step 1: Generate CLIP embedding for the uploaded image.
        logger.info("Step 1/4 — Generating CLIP embedding")
        query_embedding = self.clip_encoder.encode_image(image)

        # Step 2: Retrieve similar cases from the dataset.
        logger.info("Step 2/4 — Retrieving similar cases")
        similar_cases = self.retriever.retrieve(query_embedding)

        # Step 3: Build the prompt with few-shot context.
        logger.info("Step 3/4 — Building prompt")
        prompt = build_prompt(similar_cases)

        # Free GPU memory before VLM inference.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 4: Run VLM inference.
        logger.info("Step 4/4 — Running VLM inference")
        report = self.vlm.generate(image, prompt)

        # Package results.
        cases_dicts = [
            {
                "image_id": c.image_id,
                "similarity": round(c.similarity, 4),
                "findings": c.positive_findings,
            }
            for c in similar_cases
        ]

        logger.info("Diagnosis complete")
        return DiagnosisResult(
            report=report,
            similar_cases=cases_dicts,
        )
