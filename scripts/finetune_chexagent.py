"""LoRA fine-tuning script for CheXagent-8b on chest X-ray diagnosis.

Usage
-----
    python scripts/finetune_chexagent.py \
        --csv data/dataset.csv \
        --images data/images/ \
        --epochs 3 \
        --lr 2e-4 \
        --lora-rank 16 \
        --batch-size 2 \
        --output models/checkpoints/chexagent-finetuned
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from models.base import DISEASE_LABELS
from utils.logging_config import logger


class CXRDataset(Dataset):
    """Dataset for CheXagent fine-tuning."""

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Filter to images that exist.
        self.df = self.df[
            self.df[settings.image_col].apply(
                lambda x: (self.image_dir / x).exists()
            )
        ].reset_index(drop=True)

        logger.info("Fine-tuning dataset: %d samples", len(self.df))

    def __len__(self) -> int:
        return len(self.df)

    def _get_label(self, row: pd.Series) -> str:
        """Extract the single positive label from a row."""
        for label in DISEASE_LABELS:
            if label in row.index and row[label] == 1:
                return label
        return "No Finding"

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_id = row[settings.image_col]
        image_path = self.image_dir / image_id
        label = self._get_label(row)

        image = Image.open(image_path).convert("RGB")
        # Resize for memory efficiency.
        image = image.resize((384, 384), Image.LANCZOS)

        # Construct training prompt and target.
        prompt = (
            "Analyze this chest X-ray and provide the diagnosis. "
            "Respond with only the disease name from the valid list."
        )
        target = json.dumps({
            "diagnoses": [{"disease": label, "confidence": "High"}]
        })

        # Tokenize.
        input_text = f"USER: <s>{prompt} ASSISTANT: <s>{target}"
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze().clone(),
            "image": image,
        }


def setup_lora(model, lora_rank: int = 16, lora_alpha: int = 32) -> object:
    """Apply LoRA adapters to the model."""
    # Find linear layers to apply LoRA to.
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Target attention and MLP layers.
            if any(key in name for key in ["q_proj", "v_proj", "k_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"]):
                target_modules.append(name.split(".")[-1])

    target_modules = list(set(target_modules)) or ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CheXagent with LoRA")
    parser.add_argument("--csv", type=str, default="data/dataset.csv")
    parser.add_argument("--images", type=str, default="data/images/")
    parser.add_argument("--output", type=str, default="models/checkpoints/chexagent-finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    model_name = settings.chexagent_model_name
    token = settings.hf_token or None
    output_dir = Path(args.output)

    logger.info("Loading CheXagent for fine-tuning: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=token,
    )

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=token,
    )

    # Apply LoRA.
    logger.info("Applying LoRA (rank=%d, alpha=%d)", args.lora_rank, args.lora_alpha)
    model = setup_lora(model, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)

    # Create dataset.
    dataset = CXRDataset(
        csv_path=args.csv,
        image_dir=args.images,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # Training arguments.
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
    )

    # Train.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    logger.info("Starting LoRA fine-tuning (%d epochs, batch=%d, lr=%s)",
                args.epochs, args.batch_size, args.lr)
    trainer.train()

    # Save LoRA adapter weights.
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Fine-tuned adapter saved to %s", output_dir)

    # Save training config.
    config = {
        "base_model": model_name,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "dataset_size": len(dataset),
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nFine-tuning complete! Adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
