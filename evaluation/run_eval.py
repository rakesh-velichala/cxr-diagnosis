"""CLI script to evaluate CXR diagnosis models against ground-truth labels.

Usage
-----
    python -m evaluation.run_eval --model qwen
    python -m evaluation.run_eval --model chexagent
    python -m evaluation.run_eval --model gpt4o
    python -m evaluation.run_eval --model all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

# Ensure project root is on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from evaluation.metrics import compute_all_metrics, save_confusion_matrix_plot
from models.base import DISEASE_LABELS, load_model
from utils.logging_config import logger

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _get_ground_truth_label(row: pd.Series) -> str:
    """Extract the single positive label from a dataset row."""
    for label in DISEASE_LABELS:
        col = label
        if col in row.index and row[col] == 1:
            return label
    return "No Finding"


def evaluate_model(
    backend: str,
    csv_path: Path,
    image_dir: Path,
    limit: int | None = None,
) -> dict:
    """Run a model on every image in the dataset and compute metrics.

    Parameters
    ----------
    backend : str
        Model backend name (qwen, chexagent, gpt4o).
    csv_path : Path
        Path to the dataset CSV.
    image_dir : Path
        Directory containing chest X-ray images.
    limit : int, optional
        Max number of images to evaluate (for quick testing).

    Returns
    -------
    dict
        Computed metrics.
    """
    logger.info("=== Evaluating %s ===", backend)
    model = load_model(backend)

    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    predictions: list[list[str]] = []
    ground_truths: list[str] = []
    prediction_rows: list[dict] = []
    skipped = 0

    total = len(df)
    for idx, row in df.iterrows():
        image_id = row[settings.image_col]
        image_path = image_dir / image_id
        if not image_path.exists():
            logger.warning("Image not found: %s (skipping)", image_path)
            skipped += 1
            continue

        gt_label = _get_ground_truth_label(row)

        try:
            image = Image.open(image_path).convert("RGB")
            diagnoses = model.diagnose(image)
            pred_labels = [d.disease for d in diagnoses]
            pred_confidence = diagnoses[0].confidence if diagnoses else "Low"
        except Exception as exc:
            logger.error("Error processing %s: %s", image_id, exc)
            pred_labels = ["No Finding"]
            pred_confidence = "Low"

        predictions.append(pred_labels)
        ground_truths.append(gt_label)
        prediction_rows.append({
            "image_id": image_id,
            "ground_truth": gt_label,
            "predicted_1": pred_labels[0] if pred_labels else "",
            "predicted_2": pred_labels[1] if len(pred_labels) > 1 else "",
            "confidence": pred_confidence,
        })

        done = idx + 1 - skipped
        if done % 10 == 0 or done == total:
            logger.info("Progress: %d/%d (skipped: %d)", done, total, skipped)

    if not ground_truths:
        logger.error("No valid samples found!")
        return {}

    # Compute metrics.
    metrics = compute_all_metrics(predictions, ground_truths)
    metrics["skipped"] = skipped
    metrics["model"] = model.name

    # Save results.
    out_dir = RESULTS_DIR / backend
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", out_dir / "metrics.json")

    # Save predictions CSV.
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=prediction_rows[0].keys())
        writer.writeheader()
        writer.writerows(prediction_rows)
    logger.info("Predictions saved to %s", out_dir / "predictions.csv")

    # Save confusion matrix plot.
    top1_preds = [p[0] if p else "No Finding" for p in predictions]
    try:
        save_confusion_matrix_plot(
            top1_preds, ground_truths,
            str(out_dir / "confusion_matrix.png"),
        )
        logger.info("Confusion matrix saved to %s", out_dir / "confusion_matrix.png")
    except Exception as exc:
        logger.warning("Could not save confusion matrix: %s", exc)

    # Print summary.
    print(f"\n{'='*50}")
    print(f"Model: {model.name}")
    print(f"{'='*50}")
    print(f"Samples evaluated: {metrics['total_samples']}")
    print(f"Top-1 Accuracy:    {metrics['top_1_accuracy']:.4f}")
    print(f"Top-2 Accuracy:    {metrics['top_2_accuracy']:.4f}")
    print(f"Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:       {metrics['weighted_f1']:.4f}")
    print(f"MCC:               {metrics['mcc']:.4f}")
    print(f"{'='*50}\n")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CXR diagnosis models")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen",
        help="Backend: qwen, chexagent, gpt4o, or 'all'",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to dataset CSV (default: data/dataset.csv)",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Path to image directory (default: data/images/)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Limit number of images for quick testing",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else settings.csv_path
    image_dir = Path(args.images) if args.images else settings.image_dir

    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        sys.exit(1)
    if not image_dir.exists():
        print(f"Error: Image directory not found at {image_dir}")
        sys.exit(1)

    backends = ["qwen", "chexagent", "gpt4o"] if args.model == "all" else [args.model]

    all_results = {}
    for backend in backends:
        try:
            start = time.time()
            metrics = evaluate_model(backend, csv_path, image_dir, args.limit)
            elapsed = time.time() - start
            logger.info("%s evaluation took %.1fs", backend, elapsed)
            all_results[backend] = metrics
        except Exception as exc:
            logger.error("Failed to evaluate %s: %s", backend, exc)

    # Print comparison table if multiple models.
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Top-1':>8} {'Top-2':>8} {'F1':>8} {'MCC':>8}")
        print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for name, m in all_results.items():
            print(
                f"{m.get('model', name):<20} "
                f"{m.get('top_1_accuracy', 0):>8.4f} "
                f"{m.get('top_2_accuracy', 0):>8.4f} "
                f"{m.get('macro_f1', 0):>8.4f} "
                f"{m.get('mcc', 0):>8.4f}"
            )
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
