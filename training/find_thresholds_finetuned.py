"""Find optimal thresholds for the fine-tuned DenseNet on the validation set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import torchxrayvision as xrv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Fibrosis", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumothorax",
]
NUM_CLASSES = len(DISEASE_LABELS)

VAL_CSV = "data/val_12labels.csv"
IMAGE_DIRS = ["data/train_images", "data/images"]
CHECKPOINT = "models/checkpoints/densenet-finetuned-bce/best_model.pth"
OUTPUT_DIR = "evaluation/results/threshold-finetuned"


# ---------------------------------------------------------------------------
# Model (must match training architecture)
# ---------------------------------------------------------------------------
class FineTunedDenseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = self.pool(features)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# ---------------------------------------------------------------------------
def build_image_lookup(dirs: list[str]) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for d in dirs:
        p = Path(d)
        for f in p.glob("*.png"):
            if f.name not in lookup:
                lookup[f.name] = f
    return lookup


def preprocess(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    img_np = np.array(img).astype(np.float32)
    img_np = (img_np / 255.0) * 2048.0 - 1024.0
    img_np = img_np[np.newaxis, :, :]
    resize = xrv.datasets.XRayResizer(224)
    img_np = resize(img_np)
    return img_np


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model.
    print(f"Loading fine-tuned model from {CHECKPOINT}")
    model = FineTunedDenseNet()
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model = model.to(device)
    model.eval()

    # Load data.
    image_lookup = build_image_lookup(IMAGE_DIRS)
    df = pd.read_csv(VAL_CSV)
    df = df[df["id"].isin(image_lookup)].reset_index(drop=True)
    print(f"Validation samples: {len(df)}")

    # Run inference.
    all_probs = []
    all_labels = []
    batch_size = 32

    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            batch_imgs = []
            batch_labels = []

            for i in range(start, end):
                row = df.iloc[i]
                img_np = preprocess(image_lookup[row["id"]])
                batch_imgs.append(img_np)
                batch_labels.append(df.iloc[i][DISEASE_LABELS].values.astype(np.float32))

            batch_tensor = torch.from_numpy(np.stack(batch_imgs)).to(device)
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(np.stack(batch_labels))

            if (start // batch_size + 1) % 50 == 0:
                print(f"  Processed {end}/{len(df)} images")

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"Inference complete: {all_probs.shape}")

    # Compute AUC and thresholds.
    results = {}
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, label in enumerate(DISEASE_LABELS):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]

        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            print(f"  {label}: skipped (no positive/negative samples)")
            continue

        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        # Youden's J.
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thr = float(thresholds[best_idx])
        best_sens = float(tpr[best_idx])
        best_spec = float(1 - fpr[best_idx])

        results[label] = {
            "auc_roc": round(auc, 4),
            "threshold": round(best_thr, 4),
            "sensitivity": round(best_sens, 4),
            "specificity": round(best_spec, 4),
        }

        print(f"  {label}: AUC={auc:.4f}, thr={best_thr:.4f}, "
              f"sens={best_sens:.4f}, spec={best_spec:.4f}")

        # Plot ROC.
        ax = axes[i]
        ax.plot(fpr, tpr, "b-", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.plot(fpr[best_idx], tpr[best_idx], "ro", markersize=8)
        ax.set_title(f"{label}\nAUC={auc:.3f}, thr={best_thr:.3f}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.grid(True, alpha=0.3)

    # Hide unused subplot.
    axes[-1].set_visible(False)

    mean_auc = np.mean([v["auc_roc"] for v in results.values()])
    results["mean_auc"] = round(mean_auc, 4)
    print(f"\nMean AUC: {mean_auc:.4f}")

    fig.suptitle(f"Fine-tuned DenseNet — Val ROC Curves (Mean AUC: {mean_auc:.4f})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=150)
    print(f"ROC curves saved to {output_dir / 'roc_curves.png'}")

    # Save thresholds.
    with open(output_dir / "thresholds.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Thresholds saved to {output_dir / 'thresholds.json'}")


if __name__ == "__main__":
    main()
