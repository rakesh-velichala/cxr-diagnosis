"""Qwen2-VL Vision-Language Model inference for chest X-ray diagnosis."""

from __future__ import annotations

import base64
import io
from typing import Optional

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from app.config import settings
from utils.logging_config import logger


class VLMInference:
    """Run Qwen2-VL inference on chest X-ray images.

    Parameters
    ----------
    model_name : str, optional
        Hugging Face model ID for Qwen2-VL.
    device : str, optional
        Torch device string.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or settings.vlm_model_name
        self.device = device or settings.device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU")
            self.device = "cpu"

        logger.info("Loading Qwen2-VL model: %s", self.model_name)
        token = settings.hf_token or None
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=token,
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            token=token,
        )
        self.model.eval()
        logger.info("Qwen2-VL model loaded successfully")

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert a PIL image to a base64-encoded data URI."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Run VLM inference on an image with the given prompt.

        Parameters
        ----------
        image : PIL.Image.Image
            Chest X-ray image in RGB mode.
        prompt : str
            Fully constructed prompt including system instructions,
            few-shot examples, and the user query.
        max_new_tokens : int
            Maximum number of tokens to generate.
        temperature : float
            Sampling temperature (lower = more deterministic).

        Returns
        -------
        str
            Generated diagnosis text.
        """
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
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_input],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        # Trim the input tokens from the generated output.
        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()

        logger.info("VLM generated %d characters", len(response))
        return response
