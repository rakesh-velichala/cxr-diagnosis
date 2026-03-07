"""Fine-tuned DenseNet-121 backend with calibrated per-class thresholds."""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from app.config import settings
from models.base import BaseModel, Diagnosis, DISEASE_LABELS
from utils.logging_config import logger

# The 3 disease labels (excludes "No Finding") that we score.
_SCORED_LABELS = [l for l in DISEASE_LABELS if l != "No Finding"]

# The 11 labels the fine-tuned model was trained on (output order).
_MODEL_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Fibrosis", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumothorax",
]


class _FineTunedDenseNet(nn.Module):
    """Architecture must match the training script exactly."""

    def __init__(self) -> None:
        super().__init__()
        import torchxrayvision as xrv

        base = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, len(_MODEL_LABELS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = self.pool(features)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class DenseNetBackend(BaseModel):
    """Fine-tuned DenseNet-121 with calibrated per-class thresholds."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or settings.device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA unavailable, falling back to CPU")
            self.device = "cpu"

        # Load thresholds.
        self.thresholds: dict[str, float] = {}
        thr_path = settings.thresholds_path
        if thr_path.exists():
            with open(thr_path) as f:
                data = json.load(f)
            for label in _SCORED_LABELS:
                if label in data and "threshold" in data[label]:
                    self.thresholds[label] = data[label]["threshold"]
            logger.info("Loaded thresholds for %d labels from %s", len(self.thresholds), thr_path)
        else:
            logger.warning("Thresholds file not found: %s — using 0.5 default", thr_path)
            self.thresholds = {label: 0.5 for label in _SCORED_LABELS}

        # Load fine-tuned model.
        checkpoint_path = settings.project_root / "models" / "checkpoints" / "densenet-finetuned-bce" / "best_model.pth"
        logger.info("Loading fine-tuned DenseNet from %s", checkpoint_path)
        self.model = _FineTunedDenseNet()
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Build label → model output index mapping for the 3 target diseases.
        self.label_to_idx: dict[str, int] = {}
        for label in _SCORED_LABELS:
            if label in _MODEL_LABELS:
                self.label_to_idx[label] = _MODEL_LABELS.index(label)
        logger.info(
            "Fine-tuned DenseNet loaded — scoring %d diseases", len(self.label_to_idx)
        )

    @property
    def name(self) -> str:
        return "DenseNet-121 (Fine-tuned)"

    @torch.no_grad()
    def diagnose(self, image: Image.Image) -> list[Diagnosis]:
        import torchxrayvision as xrv

        # Preprocess: grayscale, normalize to [-1024, 1024], resize to 224x224.
        image = image.convert("L")
        img_np = np.array(image).astype(np.float32)
        img_np = (img_np / 255.0) * 2048.0 - 1024.0
        img_np = img_np[np.newaxis, :, :]
        resize = xrv.datasets.XRayResizer(224)
        img_np = resize(img_np)

        img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(self.device)
        logits = self.model(img_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Check each disease against its calibrated threshold.
        findings: list[Diagnosis] = []
        for label, idx in self.label_to_idx.items():
            prob = float(probs[idx])
            thr = self.thresholds.get(label, 0.5)
            if prob >= thr:
                if prob >= thr + 0.10:
                    conf = "High"
                else:
                    conf = "Moderate"
                findings.append(Diagnosis(
                    disease=label,
                    probability=round(prob, 4),
                    confidence=conf,
                    threshold=round(thr, 4),
                ))

        # Sort by probability descending.
        findings.sort(key=lambda d: -d.probability)

        logger.info(
            "DenseNet findings: %s",
            [(f.disease, f"{f.probability:.3f}") for f in findings] or ["No Finding"],
        )

        if not findings:
            return [Diagnosis(
                disease="No Finding",
                probability=1.0,
                confidence="High",
                threshold=0.0,
            )]

        return findings
