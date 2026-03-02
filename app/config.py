"""Centralized configuration for the CXR Diagnosis application."""

import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class AppConfig:
    """Application-wide configuration loaded from environment variables."""

    # ── Paths ──────────────────────────────────────────────────────────
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    data_dir: Path = field(init=False)
    embeddings_path: Path = field(init=False)
    csv_path: Path = field(init=False)
    image_dir: Path = field(init=False)

    # ── GCS ────────────────────────────────────────────────────────────
    gcs_bucket: str = field(
        default_factory=lambda: os.getenv("CXR_GCS_BUCKET", "")
    )
    gcs_image_prefix: str = field(
        default_factory=lambda: os.getenv("CXR_GCS_IMAGE_PREFIX", "images/")
    )
    gcs_csv_blob: str = field(
        default_factory=lambda: os.getenv(
            "CXR_GCS_CSV_BLOB", "nih-cxr-lt_single-label_balanced-test.csv"
        )
    )

    # ── Models ─────────────────────────────────────────────────────────
    clip_model_name: str = field(
        default_factory=lambda: os.getenv(
            "CXR_CLIP_MODEL", "openai/clip-vit-base-patch32"
        )
    )
    vlm_model_name: str = field(
        default_factory=lambda: os.getenv(
            "CXR_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"
        )
    )
    hf_token: str = field(
        default_factory=lambda: os.getenv("HF_TOKEN", "")
    )
    device: str = field(
        default_factory=lambda: os.getenv("CXR_DEVICE", "cuda")
    )

    # ── Retrieval ──────────────────────────────────────────────────────
    top_k: int = field(
        default_factory=lambda: int(os.getenv("CXR_TOP_K", "5"))
    )

    # ── Serving ────────────────────────────────────────────────────────
    api_host: str = field(
        default_factory=lambda: os.getenv("CXR_API_HOST", "0.0.0.0")
    )
    api_port: int = field(
        default_factory=lambda: int(os.getenv("CXR_API_PORT", "8000"))
    )
    ui_port: int = field(
        default_factory=lambda: int(os.getenv("CXR_UI_PORT", "7860"))
    )

    # ── CSV schema ─────────────────────────────────────────────────────
    image_col: str = "id"
    subject_col: str = "subject_id"
    # Disease columns are auto-detected at runtime (all columns except
    # image_col and subject_col).

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.embeddings_path = self.data_dir / "dataset_embeddings.npy"
        self.csv_path = self.data_dir / "dataset.csv"
        self.image_dir = self.data_dir / "images"


# Singleton used throughout the application.
settings = AppConfig()
