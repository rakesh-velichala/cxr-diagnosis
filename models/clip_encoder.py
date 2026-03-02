"""CLIP-based image encoder for embedding generation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.config import settings
from utils.logging_config import logger


class CLIPEncoder:
    """Encode chest X-ray images into CLIP embeddings.

    Parameters
    ----------
    model_name : str, optional
        Hugging Face model ID for the CLIP model.
    device : str, optional
        Torch device string (``cuda`` or ``cpu``).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or settings.clip_model_name
        # CLIP runs on CPU to save GPU VRAM for the larger VLM model.
        self.device = device or "cpu"

        logger.info("Loading CLIP model: %s on %s", self.model_name, self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        logger.info("CLIP model loaded successfully")

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode a single PIL image into a CLIP embedding vector.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image in RGB mode.

        Returns
        -------
        np.ndarray
            Embedding vector of shape ``(D,)``.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        if not isinstance(image_features, torch.Tensor):
            image_features = image_features.pooler_output
        embedding = image_features.cpu().numpy().squeeze()
        return embedding

    @torch.no_grad()
    def encode_batch(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Encode a batch of PIL images into CLIP embeddings.

        Parameters
        ----------
        images : list[PIL.Image.Image]
            List of images in RGB mode.
        batch_size : int
            Batch size for processing.

        Returns
        -------
        np.ndarray
            Embedding matrix of shape ``(N, D)``.
        """
        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(
                self.device
            )
            image_features = self.model.get_image_features(**inputs)
            if not isinstance(image_features, torch.Tensor):
                image_features = image_features.pooler_output
            batch_emb = image_features.cpu().numpy()
            all_embeddings.append(batch_emb)
            logger.info(
                "Encoded batch %d-%d / %d",
                start,
                min(start + batch_size, len(images)),
                len(images),
            )

        return np.concatenate(all_embeddings, axis=0)
