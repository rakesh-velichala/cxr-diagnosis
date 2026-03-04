"""Abstract base model interface and shared types for CXR diagnosis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from PIL import Image

# All valid disease labels from the NIH CXR dataset.
DISEASE_LABELS: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
    "Pneumoperitoneum",
    "Pneumomediastinum",
    "Subcutaneous Emphysema",
    "Tortuous Aorta",
    "Calcification of the Aorta",
    "No Finding",
]


@dataclass
class Diagnosis:
    """A single predicted diagnosis."""

    disease: str
    confidence: str  # "High", "Moderate", or "Low"
    rank: int  # 1 = most likely, 2 = second most likely


class BaseModel(ABC):
    """Abstract interface that all diagnosis backends must implement."""

    @abstractmethod
    def diagnose(self, image: Image.Image) -> list[Diagnosis]:
        """Analyze a chest X-ray and return top diagnoses.

        Parameters
        ----------
        image : PIL.Image.Image
            Chest X-ray image in RGB mode.

        Returns
        -------
        list[Diagnosis]
            1-2 diagnoses ordered by descending probability.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name for display and logging."""


def load_model(backend: str) -> BaseModel:
    """Factory: instantiate a diagnosis model by backend name.

    Parameters
    ----------
    backend : str
        One of ``"qwen"``, ``"chexagent"``, ``"gpt4o"``.

    Returns
    -------
    BaseModel
    """
    backend = backend.lower().strip()

    if backend == "qwen":
        from models.qwen_backend import QwenBackend
        return QwenBackend()
    elif backend == "chexagent":
        from models.chexagent_backend import CheXagentBackend
        return CheXagentBackend()
    elif backend == "gpt4o":
        from models.gpt_backend import GPTBackend
        return GPTBackend()
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from: qwen, chexagent, gpt4o"
        )
