#!/usr/bin/env python3
"""Sample a balanced subset of images from the full NIH CXR dataset.

Usage
-----
    python -m scripts.sample_dataset \\
        --csv /path/to/full_labels.csv \\
        --src-images /path/to/all_images/ \\
        --out-csv data/dataset.csv \\
        --out-images data/images/ \\
        --n 1000

The script:
1. Reads the full CSV of labels.
2. Performs stratified sampling so that each disease category is
   represented proportionally.
3. Copies the selected images into the project's ``data/images/`` dir.
4. Writes a trimmed CSV containing only the sampled rows.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.config import settings
from data.loader import get_disease_columns
from utils.logging_config import logger


def stratified_sample(df: pd.DataFrame, disease_cols: list[str], n: int) -> pd.DataFrame:
    """Sample *n* rows with balanced representation across disease columns.

    Strategy: assign each row to its primary label (the first positive
    column), then sample proportionally from each group.  Rows with no
    positive label (``No Finding``) are treated as their own group.
    """
    def primary_label(row: pd.Series) -> str:
        for col in disease_cols:
            if row[col] == 1:
                return col
        return "No Finding"

    df = df.copy()
    df["_primary"] = df.apply(primary_label, axis=1)

    sampled_parts: list[pd.DataFrame] = []
    group_counts = df["_primary"].value_counts()
    total = len(df)

    for label, count in group_counts.items():
        proportion = count / total
        k = max(1, round(proportion * n))
        group = df[df["_primary"] == label]
        k = min(k, len(group))
        sampled_parts.append(group.sample(n=k, random_state=42))

    result = pd.concat(sampled_parts).drop(columns=["_primary"])

    # Trim or pad to exactly n.
    if len(result) > n:
        result = result.sample(n=n, random_state=42)
    elif len(result) < n:
        remaining = df.drop(result.index)
        extra = remaining.sample(n=n - len(result), random_state=42)
        result = pd.concat([result, extra]).drop(columns=["_primary"], errors="ignore")

    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a subset of the CXR dataset")
    parser.add_argument("--csv", type=Path, required=True, help="Full labels CSV")
    parser.add_argument("--src-images", type=Path, required=True, help="Dir with all images")
    parser.add_argument("--out-csv", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--out-images", type=Path, default=None, help="Output image dir")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples")
    args = parser.parse_args()

    out_csv = args.out_csv or settings.csv_path
    out_images = args.out_images or settings.image_dir

    logger.info("Reading full CSV from %s", args.csv)
    df = pd.read_csv(args.csv)
    logger.info("Full dataset: %d rows", len(df))

    disease_cols = get_disease_columns(df)
    sampled = stratified_sample(df, disease_cols, args.n)
    logger.info("Sampled %d rows", len(sampled))

    # Write CSV.
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_csv, index=False)
    logger.info("Saved sampled CSV to %s", out_csv)

    # Copy images.
    out_images.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0
    for img_name in sampled[settings.image_col]:
        src = args.src_images / img_name
        dst = out_images / img_name
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            logger.warning("Source image not found: %s", src)
            missing += 1

    logger.info("Copied %d images (%d missing) to %s", copied, missing, out_images)


if __name__ == "__main__":
    main()
