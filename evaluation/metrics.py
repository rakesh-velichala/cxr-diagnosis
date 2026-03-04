"""Evaluation metrics for CXR diagnosis models."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from models.base import DISEASE_LABELS


def top_k_accuracy(
    predictions: list[list[str]],
    ground_truth: list[str],
    k: int = 1,
) -> float:
    """Compute top-k accuracy.

    Parameters
    ----------
    predictions : list[list[str]]
        Each element is a list of predicted disease names (ranked).
    ground_truth : list[str]
        True disease label for each sample.
    k : int
        Number of top predictions to consider.

    Returns
    -------
    float
        Fraction of samples where the ground truth appears in the top-k predictions.
    """
    correct = 0
    for preds, gt in zip(predictions, ground_truth):
        if gt in preds[:k]:
            correct += 1
    return correct / len(ground_truth) if ground_truth else 0.0


def per_class_metrics(
    predictions: list[str],
    ground_truth: list[str],
    labels: Optional[list[str]] = None,
) -> dict:
    """Compute per-class precision, recall, and F1.

    Parameters
    ----------
    predictions : list[str]
        Top-1 predicted disease for each sample.
    ground_truth : list[str]
        True disease label for each sample.
    labels : list[str], optional
        Label set. Defaults to DISEASE_LABELS.

    Returns
    -------
    dict
        ``{"per_class": {label: {precision, recall, f1, support}}, ...}``
    """
    labels = labels or DISEASE_LABELS
    # Filter to labels that appear in ground_truth or predictions.
    present_labels = sorted(
        set(ground_truth) | set(predictions),
        key=lambda x: labels.index(x) if x in labels else len(labels),
    )

    report = classification_report(
        ground_truth, predictions, labels=present_labels,
        output_dict=True, zero_division=0,
    )

    per_class = {}
    for label in present_labels:
        if label in report:
            per_class[label] = {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": int(report[label]["support"]),
            }

    return {"per_class": per_class}


def macro_f1(predictions: list[str], ground_truth: list[str]) -> float:
    """Compute macro-averaged F1 score."""
    return float(f1_score(ground_truth, predictions, average="macro", zero_division=0))


def weighted_f1(predictions: list[str], ground_truth: list[str]) -> float:
    """Compute weighted-averaged F1 score."""
    return float(f1_score(ground_truth, predictions, average="weighted", zero_division=0))


def mcc_score(predictions: list[str], ground_truth: list[str]) -> float:
    """Compute Matthews Correlation Coefficient (multi-class)."""
    return float(matthews_corrcoef(ground_truth, predictions))


def compute_confusion_matrix(
    predictions: list[str],
    ground_truth: list[str],
    labels: Optional[list[str]] = None,
) -> np.ndarray:
    """Compute confusion matrix.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (n_labels, n_labels).
    """
    labels = labels or sorted(set(ground_truth) | set(predictions))
    return sk_confusion_matrix(ground_truth, predictions, labels=labels)


def save_confusion_matrix_plot(
    predictions: list[str],
    ground_truth: list[str],
    output_path: str,
    labels: Optional[list[str]] = None,
) -> None:
    """Generate and save a confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = labels or sorted(set(ground_truth) | set(predictions))
    cm = sk_confusion_matrix(ground_truth, predictions, labels=labels)

    fig, ax = plt.subplots(figsize=(max(10, len(labels)), max(8, len(labels) * 0.8)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_all_metrics(
    predictions: list[list[str]],
    ground_truth: list[str],
) -> dict:
    """Compute all metrics in one call.

    Parameters
    ----------
    predictions : list[list[str]]
        Each element is a ranked list of predictions (top-1 first).
    ground_truth : list[str]
        True labels.

    Returns
    -------
    dict
        Dictionary with all computed metrics.
    """
    top1_preds = [p[0] if p else "No Finding" for p in predictions]

    results = {
        "top_1_accuracy": round(top_k_accuracy(predictions, ground_truth, k=1), 4),
        "top_2_accuracy": round(top_k_accuracy(predictions, ground_truth, k=2), 4),
        "macro_f1": round(macro_f1(top1_preds, ground_truth), 4),
        "weighted_f1": round(weighted_f1(top1_preds, ground_truth), 4),
        "mcc": round(mcc_score(top1_preds, ground_truth), 4),
        "total_samples": len(ground_truth),
    }
    results.update(per_class_metrics(top1_preds, ground_truth))
    return results
