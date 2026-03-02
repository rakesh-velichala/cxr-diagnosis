#!/usr/bin/env python3
"""Generate CLIP embeddings for every image listed in the dataset CSV.

Usage
-----
    python -m scripts.create_embeddings [--csv PATH] [--images PATH] [--out PATH]

The script:
1. Loads the dataset CSV.
2. Iterates over every image referenced in the CSV.
3. Generates a CLIP embedding for each image.
4. Saves the full matrix as ``dataset_embeddings.npy``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure the project root is on sys.path so relative imports work when
# running as ``python -m scripts.create_embeddings`` from the repo root.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.config import settings
from data.loader import load_csv
from models.clip_encoder import CLIPEncoder
from utils.logging_config import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CLIP embeddings for the CXR dataset")
    parser.add_argument("--csv", type=Path, default=None, help="Path to dataset CSV")
    parser.add_argument("--images", type=Path, default=None, help="Directory containing images")
    parser.add_argument("--out", type=Path, default=None, help="Output .npy path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding")
    args = parser.parse_args()

    csv_path = args.csv or settings.csv_path
    image_dir = args.images or settings.image_dir
    out_path = args.out or settings.embeddings_path

    # Load CSV.
    df = load_csv(csv_path)
    image_ids: list[str] = df[settings.image_col].tolist()
    logger.info("Found %d images in CSV", len(image_ids))

    # Load CLIP encoder.
    encoder = CLIPEncoder()

    # Load images, skipping any that are missing.
    images: list[Image.Image] = []
    valid_indices: list[int] = []
    for idx, img_name in enumerate(image_ids):
        img_path = image_dir / img_name
        if not img_path.exists():
            logger.warning("Image not found, skipping: %s", img_path)
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            valid_indices.append(idx)
        except Exception:
            logger.warning("Failed to open image, skipping: %s", img_path)

    if not images:
        logger.error("No images loaded — aborting")
        sys.exit(1)

    logger.info("Loaded %d / %d images", len(images), len(image_ids))

    # Encode.
    embeddings = encoder.encode_batch(images, batch_size=args.batch_size)
    logger.info("Embeddings shape: %s", embeddings.shape)

    # If some images were skipped, save a mapping alongside the embeddings.
    if len(valid_indices) < len(image_ids):
        index_path = out_path.parent / "embedding_index.npy"
        np.save(index_path, np.array(valid_indices))
        logger.info("Saved embedding-to-CSV index mapping at %s", index_path)

    # Save.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    logger.info("Saved embeddings to %s", out_path)


if __name__ == "__main__":
    main()
