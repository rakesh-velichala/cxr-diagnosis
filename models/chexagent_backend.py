"""CheXagent-8b backend for chest X-ray diagnosis."""

from __future__ import annotations

import json
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from app.config import settings
from models.base import BaseModel, Diagnosis, DISEASE_LABELS
from prompts.prompt_builder import build_diagnosis_prompt
from utils.logging_config import logger


class CheXagentBackend(BaseModel):
    """CheXagent-8b — a medical-specialist VLM from Stanford AIMI."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or settings.chexagent_model_name
        self.device = device or settings.device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA unavailable, falling back to CPU")
            self.device = "cpu"

        logger.info("Loading CheXagent model: %s", self.model_name)
        token = settings.hf_token or None

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=token,
        )

        quantization_config = None
        if self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=True,
            token=token,
        )
        self.model.eval()
        logger.info("CheXagent model loaded")

    @property
    def name(self) -> str:
        return "CheXagent-8b"

    @torch.no_grad()
    def diagnose(self, image: Image.Image) -> list[Diagnosis]:
        # Resize to limit VRAM.
        max_pixels = 512 * 512
        w, h = image.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        image = image.convert("RGB")
        prompt = build_diagnosis_prompt()

        # CheXagent uses its own chat method via trust_remote_code.
        try:
            response = self.model.chat(
                self.tokenizer,
                image,
                prompt,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.1,
            )
        except AttributeError:
            # Fallback: use generate() with manual prompt formatting.
            response = self._generate_fallback(image, prompt)

        logger.info("CheXagent raw output: %s", response[:200])
        return _parse_response(response)

    def _generate_fallback(self, image: Image.Image, prompt: str) -> str:
        """Fallback generation if model.chat() is unavailable."""
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True,
        )
        formatted = f"USER: <s>{prompt} ASSISTANT: <s>"
        inputs = processor(
            text=formatted, images=image, return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs, max_new_tokens=256, temperature=0.1, do_sample=True,
        )
        input_len = inputs.get("input_ids", generated_ids).shape[1]
        output_ids = generated_ids[:, input_len:]
        return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def _parse_response(text: str) -> list[Diagnosis]:
    """Parse model output into Diagnosis objects."""
    # Try JSON parsing first.
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
            found.append(Diagnosis(disease=label, confidence=conf, rank=len(found) + 1))
            if len(found) >= 2:
                break

    if not found:
        matches = get_close_matches(text_lower, [l.lower() for l in DISEASE_LABELS], n=1, cutoff=0.4)
        if matches:
            idx = [l.lower() for l in DISEASE_LABELS].index(matches[0])
            found.append(Diagnosis(disease=DISEASE_LABELS[idx], confidence="Low", rank=1))

    return found or [Diagnosis(disease="No Finding", confidence="Low", rank=1)]
