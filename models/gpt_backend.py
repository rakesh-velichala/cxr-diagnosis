"""GPT-4o backend for chest X-ray diagnosis (via OpenAI API)."""

from __future__ import annotations

import base64
import io
import json
from typing import Optional

from PIL import Image

from app.config import settings
from models.base import BaseModel, Diagnosis, DISEASE_LABELS
from prompts.prompt_builder import build_diagnosis_prompt
from utils.logging_config import logger


class GPTBackend(BaseModel):
    """GPT-4o via OpenAI API with base64 image upload."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        import openai

        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var."
            )
        self.model_id = model or settings.openai_model
        self.client = openai.OpenAI(api_key=self.api_key)
        logger.info("GPT backend ready (model: %s)", self.model_id)

    @property
    def name(self) -> str:
        return f"GPT-4o ({self.model_id})"

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def diagnose(self, image: Image.Image) -> list[Diagnosis]:
        # Resize to reduce API payload size.
        max_pixels = 1024 * 1024
        w, h = image.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        image = image.convert("RGB")
        prompt = build_diagnosis_prompt()
        b64 = self._image_to_base64(image)

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=256,
            temperature=0.1,
        )

        text = response.choices[0].message.content.strip()
        logger.info("GPT raw output: %s", text[:200])
        return _parse_response(text)


def _parse_response(text: str) -> list[Diagnosis]:
    """Parse model output into Diagnosis objects."""
    try:
        cleaned = text
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        data = json.loads(cleaned.strip())
        diagnoses_raw = data.get("diagnoses", data) if isinstance(data, dict) else data
        results = []
        for i, item in enumerate(diagnoses_raw[:2]):
            disease = item.get("disease", "").strip()
            confidence = item.get("confidence", "Moderate").strip()
            if disease:
                results.append(Diagnosis(disease=disease, confidence=confidence, rank=i + 1))
        if results:
            return results
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    from difflib import get_close_matches
    text_lower = text.lower()
    found: list[Diagnosis] = []
    for label in DISEASE_LABELS:
        if label.lower() in text_lower:
            conf = "Moderate"
            if "high" in text_lower:
                conf = "High"
            elif "low" in text_lower:
                conf = "Low"
            found.append(Diagnosis(disease=label, confidence=conf, rank=len(found) + 1))
            if len(found) >= 2:
                break

    if not found:
        matches = get_close_matches(text_lower, [l.lower() for l in DISEASE_LABELS], n=1, cutoff=0.4)
        if matches:
            idx = [l.lower() for l in DISEASE_LABELS].index(matches[0])
            found.append(Diagnosis(disease=DISEASE_LABELS[idx], confidence="Low", rank=1))

    return found or [Diagnosis(disease="No Finding", confidence="Low", rank=1)]
