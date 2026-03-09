"""Create filtered datasets with 12 labels and 60/20/20 train/val/test split."""

import os
import pandas as pd
import numpy as np

LABELS_12 = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Fibrosis", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumothorax", "No Finding",
]

# Load all data.
train_imgs = set(f for f in os.listdir("data/train_images") if f.endswith(".png"))
test_imgs = set(f for f in os.listdir("data/images") if f.endswith(".png"))
all_imgs = train_imgs | test_imgs

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/dataset.csv")

# Merge into one pool.
all_df = pd.concat([train_df, test_df], ignore_index=True).drop_duplicates(subset="id")
all_df = all_df[all_df["id"].isin(all_imgs)].reset_index(drop=True)

# Keep only rows where one of the 12 labels is positive.
mask = all_df[LABELS_12].sum(axis=1) > 0
all_df = all_df[mask].reset_index(drop=True)
print(f"Total matched samples (12 labels): {len(all_df)}")

# Get primary label for stratified split.
def get_primary_label(row):
    for label in LABELS_12:
        if row[label] == 1:
            return label
    return "No Finding"

all_df["primary_label"] = all_df.apply(get_primary_label, axis=1)

# Stratified 60/20/20 split.
np.random.seed(42)
train_indices = []
val_indices = []
test_indices = []

for label in LABELS_12:
    label_rows = all_df[all_df["primary_label"] == label].index.tolist()
    np.random.shuffle(label_rows)
    n = len(label_rows)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train_indices.extend(label_rows[:n_train])
    val_indices.extend(label_rows[n_train:n_train + n_val])
    test_indices.extend(label_rows[n_train + n_val:])

train_split = all_df.loc[train_indices].drop(columns=["primary_label"]).reset_index(drop=True)
val_split = all_df.loc[val_indices].drop(columns=["primary_label"]).reset_index(drop=True)
test_split = all_df.loc[test_indices].drop(columns=["primary_label"]).reset_index(drop=True)

# Save.
train_split.to_csv("data/train_12labels.csv", index=False)
val_split.to_csv("data/val_12labels.csv", index=False)
test_split.to_csv("data/test_12labels.csv", index=False)

print(f"\nTrain: {len(train_split)}")
print(f"Validation: {len(val_split)}")
print(f"Test: {len(test_split)}")
print()
print(f"{'Label':<25} {'Train':>7} {'Val':>7} {'Test':>7}")
print("-" * 50)
for label in LABELS_12:
    t = int(train_split[label].sum())
    v = int(val_split[label].sum())
    te = int(test_split[label].sum())
    print(f"{label:<25} {t:>7} {v:>7} {te:>7}")
