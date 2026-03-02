"""Dataset loading from local disk or Google Cloud Storage."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from app.config import settings
from utils.logging_config import logger


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_csv(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the dataset CSV from a local path.

    Parameters
    ----------
    csv_path : Path, optional
        Override for the default CSV location.

    Returns
    -------
    pd.DataFrame
        DataFrame with image IDs and binary disease columns.
    """
    path = csv_path or settings.csv_path
    logger.info("Loading CSV from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, columns: %s", len(df), list(df.columns))
    return df


def get_disease_columns(df: pd.DataFrame) -> list[str]:
    """Auto-detect disease label columns from the DataFrame.

    Excludes the image ID column and subject ID column; everything else
    that is numeric (0/1) is treated as a disease label.
    """
    exclude = {settings.image_col, settings.subject_col}
    disease_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    logger.info("Detected %d disease columns: %s", len(disease_cols), disease_cols)
    return disease_cols


def get_labels_for_image(df: pd.DataFrame, image_id: str) -> dict[str, int]:
    """Return the disease labels for a single image as a dict."""
    disease_cols = get_disease_columns(df)
    row = df.loc[df[settings.image_col] == image_id]
    if row.empty:
        return {}
    return row.iloc[0][disease_cols].to_dict()


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_local(image_name: str, image_dir: Optional[Path] = None) -> Image.Image:
    """Load a single image from local disk.

    Parameters
    ----------
    image_name : str
        Filename of the image (e.g. ``00000583_003.png``).
    image_dir : Path, optional
        Override for the default image directory.

    Returns
    -------
    PIL.Image.Image
        The loaded image in RGB mode.
    """
    directory = image_dir or settings.image_dir
    img_path = directory / image_name
    return Image.open(img_path).convert("RGB")


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def _get_gcs_client():
    """Lazy-import and return a GCS client."""
    from google.cloud import storage  # noqa: WPS433
    return storage.Client()


def load_csv_from_gcs(
    bucket_name: Optional[str] = None,
    blob_name: Optional[str] = None,
) -> pd.DataFrame:
    """Download the CSV from GCS into a DataFrame."""
    bucket_name = bucket_name or settings.gcs_bucket
    blob_name = blob_name or settings.gcs_csv_blob
    if not bucket_name:
        raise ValueError(
            "GCS bucket not configured. Set CXR_GCS_BUCKET env var."
        )
    logger.info("Downloading CSV from gs://%s/%s", bucket_name, blob_name)
    client = _get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(content))


def load_image_from_gcs(
    image_name: str,
    bucket_name: Optional[str] = None,
    prefix: Optional[str] = None,
) -> Image.Image:
    """Download a single image from GCS and return as PIL Image."""
    bucket_name = bucket_name or settings.gcs_bucket
    prefix = prefix or settings.gcs_image_prefix
    if not bucket_name:
        raise ValueError(
            "GCS bucket not configured. Set CXR_GCS_BUCKET env var."
        )
    blob_path = f"{prefix}{image_name}"
    logger.info("Downloading image from gs://%s/%s", bucket_name, blob_path)
    client = _get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    content = blob.download_as_bytes()
    return Image.open(io.BytesIO(content)).convert("RGB")


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def load_embeddings(path: Optional[Path] = None) -> np.ndarray:
    """Load pre-computed CLIP embeddings from disk.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, embedding_dim)``.
    """
    emb_path = path or settings.embeddings_path
    logger.info("Loading embeddings from %s", emb_path)
    embeddings = np.load(emb_path)
    logger.info("Loaded embeddings with shape %s", embeddings.shape)
    return embeddings
