"""Fine-tune DenseNet-121 (TorchXRayVision) on the 19-label CXR dataset.

Usage
-----
    python scripts/finetune_densenet.py \
        --train-csv data/train.csv \
        --test-csv data/dataset.csv \
        --images data/images/ \
        --epochs 10 \
        --batch-size 32 \
        --lr 1e-4 \
        --output models/checkpoints/densenet-finetuned
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
import torchvision.transforms as transforms
import torchxrayvision as xrv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from utils.logging_config import logger

# The 19 target labels in fixed order (Pneumomediastinum dropped — only 5 train samples).
TARGET_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
    "Pneumoperitoneum", "Subcutaneous Emphysema",
    "Tortuous Aorta", "Calcification of the Aorta", "No Finding",
]


class CXRDataset(Dataset):
    """Chest X-ray dataset for DenseNet fine-tuning."""

    def __init__(self, csv_path: str, image_dirs: list[str], augment: bool = False) -> None:
        self.df = pd.read_csv(csv_path)
        self.image_dirs = [Path(d) for d in image_dirs]
        self.augment = augment

        # Build image lookup across all directories.
        self.image_lookup: dict[str, Path] = {}
        for d in self.image_dirs:
            for f in d.glob("*.png"):
                self.image_lookup[f.name] = f

        # Filter to available images.
        before = len(self.df)
        self.df = self.df[self.df["id"].apply(lambda x: x in self.image_lookup)].reset_index(drop=True)
        logger.info("Dataset: %d/%d images available", len(self.df), before)

        # Drop rows where Pneumomediastinum is the label (too few samples).
        if "Pneumomediastinum" in self.df.columns:
            dropped = self.df["Pneumomediastinum"].sum()
            self.df = self.df[self.df["Pneumomediastinum"] != 1].reset_index(drop=True)
            if dropped > 0:
                logger.info("Dropped %d Pneumomediastinum rows", int(dropped))

        # Extract labels as numpy array.
        self.labels = self.df[TARGET_LABELS].values.astype(np.float32)

        # Transforms.
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, translate=(0.05, 0.05)),
            ])
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = self.image_lookup[row["id"]]

        # Load and preprocess image.
        img = Image.open(image_path).convert("L")

        # Apply augmentation on PIL image.
        if self.transform:
            img = self.transform(img)

        img_np = np.array(img).astype(np.float32)

        # Normalize to [-1024, 1024] (TorchXRayVision convention).
        img_np = (img_np / 255.0) * 2048.0 - 1024.0
        img_np = img_np[np.newaxis, :, :]  # (1, H, W)

        # Resize to 224x224.
        resize = xrv.datasets.XRayResizer(224)
        img_np = resize(img_np)

        img_tensor = torch.from_numpy(img_np)
        label_tensor = torch.from_numpy(self.labels[idx])

        return img_tensor, label_tensor


class FineTunedDenseNet(nn.Module):
    """DenseNet-121 with a new 19-class classification head."""

    def __init__(self) -> None:
        super().__init__()
        # Load pre-trained TorchXRayVision model.
        base = xrv.models.DenseNet(weights="densenet121-res224-all")

        # Keep the feature extractor (everything except the classifier).
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Get feature dimension from the base model.
        # DenseNet-121 outputs 1024 features.
        self.classifier = nn.Linear(1024, len(TARGET_LABELS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = self.pool(features)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def compute_class_weights(dataset: CXRDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    class_counts = dataset.labels.sum(axis=0)
    # Avoid division by zero.
    class_counts = np.maximum(class_counts, 1.0)
    total = len(dataset)
    weights = total / (len(TARGET_LABELS) * class_counts)
    logger.info("Class weights: %s", {TARGET_LABELS[i]: f"{w:.1f}" for i, w in enumerate(weights)})
    return torch.FloatTensor(weights)


def evaluate(model: nn.Module, dataloader: DataLoader, device: str) -> dict:
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            gt = labels.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(gt)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()

    return {"accuracy": float(accuracy), "total": len(all_labels)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DenseNet-121 for CXR")
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--test-csv", type=str, default="data/dataset.csv")
    parser.add_argument("--images", type=str, nargs="+", required=True,
                        help="One or more image directories")
    parser.add_argument("--output", type=str, default="models/checkpoints/densenet-finetuned")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets.
    logger.info("Loading training data from %s", args.train_csv)
    train_dataset = CXRDataset(args.train_csv, args.images, augment=True)
    logger.info("Loading test data from %s", args.test_csv)
    test_dataset = CXRDataset(args.test_csv, args.images, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Create model.
    logger.info("Creating fine-tuned DenseNet-121")
    model = FineTunedDenseNet().to(device)

    # Class-weighted loss.
    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop.
    best_acc = 0.0
    logger.info("Starting training: %d epochs, batch_size=%d, lr=%s", args.epochs, args.batch_size, args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            # Convert one-hot labels to class indices for CrossEntropyLoss.
            targets = labels.argmax(dim=1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            if (batch_idx + 1) % 100 == 0:
                logger.info(
                    "Epoch %d/%d, Batch %d/%d, Loss: %.4f, Acc: %.4f",
                    epoch + 1, args.epochs, batch_idx + 1, len(train_loader),
                    running_loss / (batch_idx + 1), correct / total,
                )

        scheduler.step()
        train_acc = correct / total
        avg_loss = running_loss / len(train_loader)

        # Evaluate on test set.
        test_metrics = evaluate(model, test_loader, device)
        test_acc = test_metrics["accuracy"]

        logger.info(
            "Epoch %d/%d — Train Loss: %.4f, Train Acc: %.4f, Test Acc: %.4f",
            epoch + 1, args.epochs, avg_loss, train_acc, test_acc,
        )

        # Save best model.
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            logger.info("New best model saved (test acc: %.4f)", test_acc)

    # Save final model.
    torch.save(model.state_dict(), output_dir / "final_model.pth")

    # Save training config.
    config = {
        "base_model": "densenet121-res224-all",
        "target_labels": TARGET_LABELS,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "best_test_accuracy": best_acc,
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best test accuracy: {best_acc:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
