"""Qwen2.5-VL backend for chest X-ray diagnosis."""

from __future__ import annotations

import base64
import io
import json
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from app.config import settings
from models.base import BaseModel, Diagnosis, DISEASE_LABELS
from prompts.prompt_builder import build_diagnosis_prompt
from utils.logging_config import logger


class QwenBackend(BaseModel):
    """Qwen2.5-VL-7B with 4-bit quantization."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or settings.qwen_model_name
        self.device = device or settings.device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA unavailable, falling back to CPU")
            self.device = "cpu"

        logger.info("Loading Qwen model: %s (4-bit)", self.model_name)
        token = settings.hf_token or None

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True, token=token,
        )

        quantization_config = None
        if self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=True,
            token=token,
        )
        self.model.eval()
        logger.info("Qwen model loaded")

    @property
    def name(self) -> str:
        return "Qwen2.5-VL-7B"

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    @torch.no_grad()
    def diagnose(self, image: Image.Image) -> list[Diagnosis]:
        # Resize to limit VRAM.
        max_pixels = 512 * 512
        w, h = image.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        prompt = build_diagnosis_prompt()
        image_uri = self._image_to_base64(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text_input], images=[image], padding=True, return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs, max_new_tokens=256, temperature=0.1, do_sample=True,
        )

        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True,
        )[0].strip()

        logger.info("Qwen raw output: %s", response[:200])
        return _parse_response(response)


def _parse_response(text: str) -> list[Diagnosis]:
    """Parse model output into Diagnosis objects."""
    # Try JSON parsing first.
    try:
        # Extract JSON from possible markdown code blocks.
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
                results.append(Diagnosis(disease=disease, probability=0.0, confidence=confidence))
        if results:
            return results
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # Fallback: fuzzy match disease names from text.
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
            found.append(Diagnosis(disease=label, probability=0.0, confidence=conf))
            if len(found) >= 2:
                break

    if not found:
        matches = get_close_matches(text_lower, [l.lower() for l in DISEASE_LABELS], n=1, cutoff=0.4)
        if matches:
            idx = [l.lower() for l in DISEASE_LABELS].index(matches[0])
            found.append(Diagnosis(disease=DISEASE_LABELS[idx], probability=0.0, confidence="Low"))

    return found or [Diagnosis(disease="No Finding", probability=0.0, confidence="Low")]
