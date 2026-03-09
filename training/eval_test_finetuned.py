"""Evaluate fine-tuned DenseNet on test set with fixed thresholds from validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
import torchxrayvision as xrv

# ---------------------------------------------------------------------------
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Fibrosis", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumothorax",
]
NUM_CLASSES = len(DISEASE_LABELS)

TEST_CSV = "data/test_12labels.csv"
IMAGE_DIRS = ["data/train_images", "data/images"]
CHECKPOINT = "models/checkpoints/densenet-finetuned-bce/best_model.pth"
THRESHOLDS_FILE = "evaluation/results/threshold-finetuned/thresholds.json"
OUTPUT_DIR = "evaluation/results/test-evaluation-finetuned"


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

    # Load thresholds.
    with open(THRESHOLDS_FILE) as f:
        thr_data = json.load(f)
    print("Loaded thresholds from", THRESHOLDS_FILE)

    # Load model.
    print(f"Loading fine-tuned model from {CHECKPOINT}")
    model = FineTunedDenseNet()
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model = model.to(device)
    model.eval()

    # Load data.
    image_lookup = build_image_lookup(IMAGE_DIRS)
    df = pd.read_csv(TEST_CSV)
    df = df[df["id"].isin(image_lookup)].reset_index(drop=True)
    print(f"Test samples: {len(df)}")

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

    # Evaluate with fixed thresholds.
    results = {}
    for i, label in enumerate(DISEASE_LABELS):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]

        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            print(f"  {label}: skipped")
            continue

        auc = roc_auc_score(y_true, y_score)
        thr = thr_data[label]["threshold"]

        y_pred = (y_score >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        results[label] = {
            "auc_roc": round(auc, 4),
            "threshold": round(thr, 4),
            "accuracy": round(accuracy, 4),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "ppv": round(ppv, 4),
            "npv": round(npv, 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "positives": int(y_true.sum()),
            "negatives": int(len(y_true) - y_true.sum()),
        }

        print(f"  {label}: AUC={auc:.4f}, sens={sensitivity:.4f}, "
              f"spec={specificity:.4f}, acc={accuracy:.4f}")

    # Summary.
    aucs = [v["auc_roc"] for v in results.values()]
    sens = [v["sensitivity"] for v in results.values()]
    specs = [v["specificity"] for v in results.values()]
    accs = [v["accuracy"] for v in results.values()]

    results["summary"] = {
        "mean_auc": round(np.mean(aucs), 4),
        "mean_sensitivity": round(np.mean(sens), 4),
        "mean_specificity": round(np.mean(specs), 4),
        "mean_accuracy": round(np.mean(accs), 4),
    }

    print(f"\nSummary:")
    print(f"  Mean AUC: {np.mean(aucs):.4f}")
    print(f"  Mean Sensitivity: {np.mean(sens):.4f}")
    print(f"  Mean Specificity: {np.mean(specs):.4f}")
    print(f"  Mean Accuracy: {np.mean(accs):.4f}")

    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'test_results.json'}")


if __name__ == "__main__":
    main()
