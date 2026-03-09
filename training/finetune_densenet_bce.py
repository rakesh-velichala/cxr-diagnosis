"""Fine-tune DenseNet-121 (TorchXRayVision) with multi-label BCE loss.

Trains on all 11 diseases from the 12-label splits, validates per epoch
with per-class AUC, saves best checkpoint by mean val AUC.

Usage
-----
    python3 -u finetune_densenet_bce.py \
        --train-csv data/train_12labels.csv \
        --val-csv data/val_12labels.csv \
        --images data/images/ \
        --epochs 10 \
        --batch-size 32 \
        --lr 1e-4 \
        --output models/checkpoints/densenet-finetuned-bce
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score
import torchxrayvision as xrv

# ---------------------------------------------------------------------------
# 11 scored disease labels (No Finding excluded from model output).
# ---------------------------------------------------------------------------
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Fibrosis", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumothorax",
]

NUM_CLASSES = len(DISEASE_LABELS)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CXRDataset(Dataset):
    """Multi-label chest X-ray dataset."""

    def __init__(self, csv_path: str, image_dirs: list[str], augment: bool = False) -> None:
        self.df = pd.read_csv(csv_path)
        self.augment = augment

        # Build image lookup across all directories.
        self.image_lookup: dict[str, Path] = {}
        for d in image_dirs:
            p = Path(d)
            for f in p.glob("*.png"):
                if f.name not in self.image_lookup:
                    self.image_lookup[f.name] = f

        before = len(self.df)
        self.df = self.df[self.df["id"].isin(self.image_lookup)].reset_index(drop=True)
        print(f"Dataset: {len(self.df)}/{before} images available from {csv_path}")

        # Extract multi-label targets.
        self.labels = self.df[DISEASE_LABELS].values.astype(np.float32)

        # Augmentation transforms (applied on numpy).
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = self.image_lookup[row["id"]]

        img = Image.open(image_path).convert("L")
        img_np = np.array(img).astype(np.float32)

        # Normalize to [-1024, 1024] (TorchXRayVision convention).
        img_np = (img_np / 255.0) * 2048.0 - 1024.0
        img_np = img_np[np.newaxis, :, :]  # (1, H, W)

        # Resize to 224x224.
        resize = xrv.datasets.XRayResizer(224)
        img_np = resize(img_np)

        # Simple augmentation: random horizontal flip.
        if self.augment and np.random.rand() > 0.5:
            img_np = img_np[:, :, ::-1].copy()

        img_tensor = torch.from_numpy(img_np)
        label_tensor = torch.from_numpy(self.labels[idx])

        return img_tensor, label_tensor


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FineTunedDenseNet(nn.Module):
    """DenseNet-121 with frozen early layers and new multi-label head."""

    def __init__(self, freeze_up_to: int = 7) -> None:
        super().__init__()
        base = xrv.models.DenseNet(weights="densenet121-res224-all")

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # New multi-label classifier (11 outputs, one per disease).
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, NUM_CLASSES),
        )

        # Freeze early feature layers.
        children = list(self.features.children())
        for i, child in enumerate(children):
            if i <= freeze_up_to:
                for param in child.parameters():
                    param.requires_grad = False

        # Count trainable vs frozen params.
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Parameters: {trainable:,} trainable / {total:,} total "
              f"({100*trainable/total:.1f}% trainable)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = self.pool(features)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# ---------------------------------------------------------------------------
# Compute pos_weight for BCE loss (handles class imbalance).
# ---------------------------------------------------------------------------
def compute_pos_weight(dataset: CXRDataset) -> torch.Tensor:
    """pos_weight = num_negatives / num_positives per class."""
    pos_counts = dataset.labels.sum(axis=0)
    neg_counts = len(dataset) - pos_counts
    # Clamp to avoid division by zero.
    pos_counts = np.maximum(pos_counts, 1.0)
    weights = neg_counts / pos_counts
    # Cap weights to avoid extreme values.
    weights = np.minimum(weights, 50.0)
    print("pos_weight per class:")
    for i, label in enumerate(DISEASE_LABELS):
        pos = int(dataset.labels[:, i].sum())
        print(f"  {label}: pos={pos}, weight={weights[i]:.1f}")
    return torch.FloatTensor(weights)


# ---------------------------------------------------------------------------
# Validation with per-class AUC.
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader, device: str) -> dict:
    """Compute per-class AUC on validation set."""
    model.eval()
    all_logits = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    probs = 1 / (1 + np.exp(-all_logits))  # sigmoid

    aucs = {}
    for i, label in enumerate(DISEASE_LABELS):
        y_true = all_labels[:, i]
        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            aucs[label] = roc_auc_score(y_true, probs[:, i])
        else:
            aucs[label] = 0.0

    mean_auc = np.mean(list(aucs.values()))
    return {"aucs": aucs, "mean_auc": float(mean_auc)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DenseNet-121 with BCE loss")
    parser.add_argument("--train-csv", type=str, default="data/train_12labels.csv")
    parser.add_argument("--val-csv", type=str, default="data/val_12labels.csv")
    parser.add_argument("--images", type=str, nargs="+", default=["data/train_images", "data/images"])
    parser.add_argument("--output", type=str, default="models/checkpoints/densenet-finetuned-bce")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets.
    print(f"\n=== Loading data ===")
    train_dataset = CXRDataset(args.train_csv, args.images, augment=True)
    val_dataset = CXRDataset(args.val_csv, args.images, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Create model.
    print(f"\n=== Creating model ===")
    model = FineTunedDenseNet(freeze_up_to=7).to(device)

    # BCE loss with pos_weight.
    pos_weight = compute_pos_weight(train_dataset).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer: higher LR for classifier, lower for fine-tuned features.
    feature_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and "classifier" not in n]
    classifier_params = [p for n, p in model.named_parameters()
                         if p.requires_grad and "classifier" in n]

    optimizer = torch.optim.AdamW([
        {"params": feature_params, "lr": args.lr * 0.1},   # 1e-5 for features
        {"params": classifier_params, "lr": args.lr},       # 1e-4 for classifier
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop.
    best_mean_auc = 0.0
    history = []

    print(f"\n=== Training: {args.epochs} epochs, batch_size={args.batch_size} ===")
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    print(f"Batches per epoch: {len(train_loader)}")
    print()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 200 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}, "
                      f"Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {running_loss/(batch_idx+1):.4f}")

        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        # Validate.
        val_metrics = validate(model, val_loader, device)
        mean_auc = val_metrics["mean_auc"]
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1}/{args.epochs} — "
              f"Loss: {avg_loss:.4f}, "
              f"Val Mean AUC: {mean_auc:.4f}, "
              f"Time: {epoch_time:.0f}s")

        # Print per-class AUCs.
        for label, auc in sorted(val_metrics["aucs"].items(), key=lambda x: -x[1]):
            print(f"    {label}: {auc:.4f}")

        # Save history.
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_mean_auc": mean_auc,
            "val_aucs": val_metrics["aucs"],
            "time_sec": epoch_time,
        })

        # Save best model.
        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print(f"  >> New best model saved (mean AUC: {mean_auc:.4f})")

        print()

    # Save final model.
    torch.save(model.state_dict(), output_dir / "final_model.pth")

    # Save training history and config.
    config = {
        "base_model": "densenet121-res224-all",
        "disease_labels": DISEASE_LABELS,
        "num_classes": NUM_CLASSES,
        "freeze_up_to": 7,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "best_val_mean_auc": best_mean_auc,
        "history": history,
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Training complete!")
    print(f"Best val mean AUC: {best_mean_auc:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
