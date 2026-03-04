"""TorchXRayVision DenseNet backend for chest X-ray diagnosis."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

from app.config import settings
from models.base import BaseModel, Diagnosis, DISEASE_LABELS
from utils.logging_config import logger

# Mapping from TorchXRayVision label names to our canonical labels.
_XRV_TO_CANONICAL = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Effusion": "Effusion",
    "Emphysema": "Emphysema",
    "Fibrosis": "Fibrosis",
    "Hernia": "Hernia",
    "Infiltration": "Infiltration",
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Pleural_Thickening": "Pleural_Thickening",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    "No Finding": "No Finding",
}


class DenseNetBackend(BaseModel):
    """Pre-trained DenseNet-121 from TorchXRayVision."""

    def __init__(self, device: Optional[str] = None) -> None:
        import torchxrayvision as xrv

        self.device = device or settings.device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA unavailable, falling back to CPU")
            self.device = "cpu"

        logger.info("Loading TorchXRayVision DenseNet (densenet121-res224-all)")
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Store the model's pathology list for label mapping.
        self.xrv_labels = self.model.pathologies
        logger.info("DenseNet loaded — pathologies: %s", self.xrv_labels)

    @property
    def name(self) -> str:
        return "DenseNet-121 (TorchXRayVision)"

    @torch.no_grad()
    def diagnose(self, image: Image.Image) -> list[Diagnosis]:
        import torchxrayvision as xrv
        import torchvision.transforms as transforms

        # Convert to grayscale numpy array normalized to [0, 255].
        image = image.convert("L")
        img_np = np.array(image).astype(np.float32)

        # TorchXRayVision expects images in [-1024, 1024] range.
        # Scale from [0, 255] to [-1024, 1024].
        img_np = (img_np / 255.0) * 2048.0 - 1024.0

        # Add channel dimension: (H, W) -> (1, H, W).
        img_np = img_np[np.newaxis, :, :]

        # Resize to 224x224 using xrv's transform.
        transform = transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ])
        img_np = transform(img_np)

        # Convert to tensor and add batch dimension.
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(self.device)

        # Run inference.
        output = self.model(img_tensor)
        probs = output[0].cpu().numpy()

        # Map xrv labels to our canonical labels and collect scores.
        scored: list[tuple[str, float]] = []
        for i, xrv_label in enumerate(self.xrv_labels):
            canonical = _XRV_TO_CANONICAL.get(xrv_label)
            if canonical and canonical in DISEASE_LABELS:
                scored.append((canonical, float(probs[i])))

        # Sort by descending probability.
        scored.sort(key=lambda x: -x[1])

        logger.info(
            "DenseNet top-5: %s",
            [(name, f"{score:.3f}") for name, score in scored[:5]],
        )

        # Check if "No Finding" should be the prediction.
        # If the highest pathology score is below a threshold, predict No Finding.
        top_pathologies = [(n, s) for n, s in scored if n != "No Finding"]
        no_finding_threshold = 0.5

        if not top_pathologies or top_pathologies[0][1] < no_finding_threshold:
            # Check if any pathology has a reasonable score.
            if top_pathologies and top_pathologies[0][1] > 0.3:
                # Borderline — return top pathology with Low confidence.
                name, score = top_pathologies[0]
                return [Diagnosis(disease=name, confidence="Low", rank=1)]
            return [Diagnosis(disease="No Finding", confidence="High", rank=1)]

        # Return top 1-2 diagnoses.
        results: list[Diagnosis] = []
        for rank, (name, score) in enumerate(top_pathologies[:2], start=1):
            if score < 0.2:
                break
            if score >= 0.7:
                conf = "High"
            elif score >= 0.5:
                conf = "Moderate"
            else:
                conf = "Low"
            results.append(Diagnosis(disease=name, confidence=conf, rank=rank))

        return results or [Diagnosis(disease="No Finding", confidence="Low", rank=1)]
