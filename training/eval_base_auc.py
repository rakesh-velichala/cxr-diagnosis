"""Evaluate pretrained DenseNet-121 (TorchXRayVision) AUC per class."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import torchxrayvision as xrv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The 14 pathologies the pretrained model supports.
XRV_LABELS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
]

TEST_CSV = "data/dataset.csv"
IMAGE_DIRS = ["data/train_images", "data/images"]
OUTPUT_DIR = "evaluation/results/densenet-pretrained-auc"


def preprocess(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("L")
    img_np = np.array(img).astype(np.float32)
    img_np = (img_np / 255.0) * 2048.0 - 1024.0
    img_np = img_np[np.newaxis, :, :]
    resize = xrv.datasets.XRayResizer(224)
    img_np = resize(img_np)
    return torch.from_numpy(img_np).unsqueeze(0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load pretrained model.
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.to(device)
    model.eval()
    print(f"Model pathologies: {list(model.pathologies)}")

    # Map model output indices to our labels.
    model_pathologies = list(model.pathologies)
    label_to_idx = {}
    for label in XRV_LABELS:
        if label in model_pathologies:
            label_to_idx[label] = model_pathologies.index(label)
    print(f"Matched {len(label_to_idx)} labels")

    # Build image lookup.
    image_lookup = {}
    for d in IMAGE_DIRS:
        p = Path(d)
        if p.exists():
            for f in p.glob("*.png"):
                image_lookup[f.name] = f

    # Load test CSV.
    df = pd.read_csv(TEST_CSV)
    df = df[df["id"].apply(lambda x: x in image_lookup)].reset_index(drop=True)
    print(f"Test samples: {len(df)}")

    # Collect predictions and ground truth per class.
    all_probs = {label: [] for label in label_to_idx}
    all_gt = {label: [] for label in label_to_idx}

    with torch.no_grad():
        for i, (_, row) in enumerate(df.iterrows()):
            img_tensor = preprocess(str(image_lookup[row["id"]])).to(device)
            output = model(img_tensor)
            probs = torch.sigmoid(output).cpu().numpy()[0]

            for label, idx in label_to_idx.items():
                all_probs[label].append(float(probs[idx]))
                all_gt[label].append(int(row.get(label, 0)))

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(df)}")

    # Compute AUC per class.
    print(f"\n{'='*60}")
    print(f"{'Label':<25} {'AUC-ROC':>8} {'Positives':>10} {'Negatives':>10}")
    print("-" * 60)

    results = {}
    valid_labels = []
    for label in XRV_LABELS:
        if label not in label_to_idx:
            continue
        gt = np.array(all_gt[label])
        probs = np.array(all_probs[label])
        n_pos = int(gt.sum())
        n_neg = int(len(gt) - n_pos)

        if n_pos == 0 or n_neg == 0:
            auc = float("nan")
            print(f"{label:<25} {'N/A':>8} {n_pos:>10} {n_neg:>10}  (skipped)")
        else:
            auc = roc_auc_score(gt, probs)
            print(f"{label:<25} {auc:>8.4f} {n_pos:>10} {n_neg:>10}")
            valid_labels.append(label)

        results[label] = {
            "auc_roc": round(auc, 4) if not np.isnan(auc) else None,
            "positives": n_pos,
            "negatives": n_neg,
        }

    # Mean AUC across valid classes.
    valid_aucs = [results[l]["auc_roc"] for l in valid_labels]
    mean_auc = np.mean(valid_aucs)
    print(f"\n{'Mean AUC (valid classes)':<25} {mean_auc:>8.4f}")
    results["mean_auc"] = round(float(mean_auc), 4)

    # Save results.
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "auc_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot ROC curves.
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    axes = axes.flatten()
    for i, label in enumerate(valid_labels):
        gt = np.array(all_gt[label])
        probs = np.array(all_probs[label])
        fpr, tpr, _ = roc_curve(gt, probs)
        auc_val = results[label]["auc_roc"]

        axes[i].plot(fpr, tpr, linewidth=2, label=f"AUC={auc_val:.3f}")
        axes[i].plot([0, 1], [0, 1], "k--", alpha=0.3)
        axes[i].set_title(label, fontsize=12, fontweight="bold")
        axes[i].set_xlabel("FPR")
        axes[i].set_ylabel("TPR")
        axes[i].legend(loc="lower right")
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplots.
    for j in range(len(valid_labels), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"ROC Curves — Pretrained DenseNet-121 (Mean AUC: {mean_auc:.3f})", fontsize=16)
    plt.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=150)
    plt.close(fig)

    print(f"\nResults saved to {out}/")


if __name__ == "__main__":
    main()
