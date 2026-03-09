# CXR Diagnosis — AI-Powered Chest X-Ray Screening Tool

AI-powered chest X-ray screening tool that detects Cardiomegaly, Edema, and Pleural Effusion using a fine-tuned DenseNet-121 CNN with calibrated per-disease thresholds.

Built on [TorchXRayVision](https://github.com/mlmed/torchxrayvision)'s DenseNet-121 pre-trained on 300k+ chest X-rays across 8 major CXR datasets, fine-tuned with multi-label BCE loss on 40k NIH CXR-14 images.

> **DISCLAIMER:** This application is for **educational and research purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment.

---

## Key Results

| Disease | AUC-ROC | Sensitivity | Specificity |
|---------|---------|-------------|-------------|
| Cardiomegaly | **0.938** | 87.4% | 84.5% |
| Edema | **0.892** | 77.4% | 84.6% |
| Pleural Effusion | **0.874** | 81.8% | 77.6% |
| **Mean (top 3)** | **0.901** | **82.2%** | **82.2%** |

Evaluated on 13,480 held-out test images. Full 11-disease results: mean AUC = 0.789 across Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Fibrosis, Infiltration, Mass, Nodule, Pleural Thickening, and Pneumothorax.

---

## Architecture

```
  Chest X-Ray Image (upload)
          │
          ▼
  ┌───────────────────┐
  │   Preprocessing   │  Grayscale → Normalize [-1024, 1024] → Resize 224×224
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │  DenseNet-121     │  TorchXRayVision backbone (pre-trained on 300k CXR)
  │  (frozen L0-L7)   │  → Fine-tuned deeper layers + new classifier head
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │  Sigmoid (×11)    │  Independent probability per disease
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │  Per-Disease      │  Calibrated via Youden's J on validation set
  │  Thresholds       │  (e.g., Cardiomegaly: 0.39, Effusion: 0.51)
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │  Results          │  Disease | Probability | Confidence
  │  (Gradio / API)   │  or "No Finding" if nothing above threshold
  └───────────────────┘
```

The system also includes experimental VLM backends (CheXagent-8b, Qwen2.5-VL-7B, GPT-4o) accessible via the API, though the fine-tuned DenseNet significantly outperforms all of them for structured classification.

---

## Repository Structure

```
cxr-diagnosis/
├── app/
│   ├── config.py              # Centralized settings (env vars, paths, defaults)
│   ├── main.py                # FastAPI backend (POST /predict, GET /health)
│   └── pipeline.py            # Orchestrates image → model → result
├── models/
│   ├── base.py                # Abstract interface, Diagnosis dataclass, model factory
│   ├── densenet_backend.py    # Production backend: fine-tuned DenseNet + thresholds
│   ├── chexagent_backend.py   # CheXagent-8b VLM backend (experimental)
│   ├── qwen_backend.py        # Qwen2.5-VL-7B backend (experimental)
│   └── gpt_backend.py         # GPT-4o backend via OpenAI API (experimental)
├── training/
│   ├── finetune_densenet_bce.py    # Fine-tuning script (BCE loss, 11 diseases)
│   ├── create_splits_12.py         # Train/val/test split generation (60/20/20)
│   ├── find_thresholds_finetuned.py # Per-disease threshold calibration
│   ├── eval_test_finetuned.py      # Final test set evaluation
│   └── eval_base_auc.py            # Pretrained baseline AUC evaluation
├── evaluation/
│   ├── metrics.py             # Metric computation (AUC, F1, MCC, confusion matrix)
│   └── run_eval.py            # CLI evaluation framework for any backend
├── prompts/
│   └── prompt_builder.py      # Structured prompt for VLM backends
├── ui/
│   └── gradio_app.py          # Gradio web interface
├── data/
│   ├── train_12labels.csv     # Training set (40,418 images, 12 labels)
│   ├── val_12labels.csv       # Validation set (13,471 images)
│   ├── test_12labels.csv      # Test set (13,480 images)
│   └── train.csv              # Original NIH CXR-14 labels (68,059 rows)
├── utils/
│   └── logging_config.py      # Logger configuration
├── run.py                     # Application entry point
├── Dockerfile                 # GPU-compatible container
└── requirements.txt           # Python dependencies
```

**Not tracked in git** (large files):
- `models/checkpoints/` — Fine-tuned model weights (~28 MB each)
- `evaluation/results/` — Evaluation metrics, predictions, plots
- `data/images/`, `data/train_images/` — X-ray image files

---

## Quick Start

### 1. Install dependencies

```bash
cd cxr-diagnosis
pip install -r requirements.txt
```

### 2. Download model checkpoint

The fine-tuned model checkpoint (`best_model.pth`, 28 MB) and calibrated thresholds (`thresholds.json`) must be placed at:

```
models/checkpoints/densenet-finetuned-bce/best_model.pth
evaluation/results/threshold-finetuned/thresholds.json
```

### 3. Run the application

```bash
# Launch Gradio UI on port 7860
python run.py

# Or with FastAPI backend (port 8000) + Gradio UI
python run.py --api
```

### 4. Open in browser

- **Web UI:** http://localhost:7860
- **API docs:** http://localhost:8000/docs (if started with `--api`)

---

## API Reference

### `POST /predict`

Upload a chest X-ray image for diagnosis.

**Request:** `multipart/form-data` with `file` field. Optional query param `model` (default: `densenet`).

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@chest_xray.png"
```

**Response:**

```json
{
  "diagnoses": [
    {
      "disease": "Cardiomegaly",
      "probability": 0.8234,
      "confidence": "High",
      "threshold": 0.3895
    }
  ],
  "model": "DenseNet-121 (Fine-tuned)",
  "disclaimer": "DISCLAIMER: This analysis is for educational..."
}
```

If no diseases exceed their thresholds:

```json
{
  "diagnoses": [
    {
      "disease": "No Finding",
      "probability": 1.0,
      "confidence": "High",
      "threshold": 0.0
    }
  ]
}
```

### `GET /health`

Returns `{"status": "ok"}`.

---

## Training Pipeline

Training was performed on a GCP VM with NVIDIA T4 GPU (16 GB VRAM).

### 1. Create data splits

```bash
python training/create_splits_12.py
```

Merges available CSVs, filters to 12 labels (11 diseases + No Finding), creates stratified 60/20/20 splits.

### 2. Fine-tune the model

```bash
python training/finetune_densenet_bce.py \
    --train-csv data/train_12labels.csv \
    --val-csv data/val_12labels.csv \
    --images data/train_images data/images \
    --epochs 10 --batch-size 32 --lr 1e-4
```

**Training details:**
- **Base model:** TorchXRayVision `densenet121-res224-all`
- **Frozen layers:** 0-7 (early feature extractors)
- **Classifier head:** Dropout(0.3) → Linear(1024, 11)
- **Loss:** BCEWithLogitsLoss with pos_weight for class imbalance
- **Optimizer:** AdamW with differential LR (features: 1e-5, classifier: 1e-4)
- **Scheduler:** CosineAnnealingLR
- **Best validation AUC:** 0.788 (saved as `best_model.pth`)

### 3. Calibrate thresholds

```bash
python training/find_thresholds_finetuned.py
```

Computes optimal per-disease thresholds on the validation set using Youden's J statistic (maximizes sensitivity + specificity).

### 4. Evaluate on test set

```bash
python training/eval_test_finetuned.py
```

Final unbiased evaluation on 13,480 held-out test images.

---

## Model Performance (Full 11-Disease Results)

Test set evaluation (13,480 images):

| Disease | AUC | Sens. | Spec. | Threshold |
|---------|-----|-------|-------|-----------|
| Cardiomegaly | 0.938 | 87.4% | 84.5% | 0.390 |
| Edema | 0.892 | 77.4% | 84.6% | 0.335 |
| Effusion | 0.874 | 81.8% | 77.6% | 0.512 |
| Mass | 0.803 | 67.0% | 80.2% | 0.463 |
| Consolidation | 0.784 | 67.4% | 78.1% | 0.463 |
| Atelectasis | 0.775 | 76.5% | 66.2% | 0.514 |
| Fibrosis | 0.773 | 85.6% | 53.7% | 0.235 |
| Pneumothorax | 0.755 | 57.4% | 79.5% | 0.590 |
| Pleural Thickening | 0.708 | 66.9% | 62.8% | 0.349 |
| Nodule | 0.708 | 69.8% | 59.9% | 0.456 |
| Infiltration | 0.667 | 76.2% | 47.3% | 0.457 |
| **Mean** | **0.789** | **73.9%** | **70.4%** | — |

Only the top 3 diseases (AUC >= 0.85, balanced sensitivity/specificity >= 75%) are surfaced in the production web app.

---

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CXR_MODEL_BACKEND` | `densenet` | Model backend (`densenet`, `chexagent`, `qwen`, `gpt4o`) |
| `CXR_DEVICE` | `cuda` | Torch device (`cuda` / `cpu`) |
| `CXR_THRESHOLDS_PATH` | `evaluation/results/threshold-finetuned/thresholds.json` | Per-disease threshold file |
| `CXR_API_PORT` | `8000` | FastAPI port |
| `CXR_UI_PORT` | `7860` | Gradio port |
| `HF_TOKEN` | — | HuggingFace token (for VLM backends) |
| `OPENAI_API_KEY` | — | OpenAI API key (for GPT-4o backend) |

---

## Docker

### Build

```bash
docker build -t cxr-diagnosis .
```

### Run

```bash
# With GPU
docker run --gpus all -p 8000:8000 -p 7860:7860 \
    -v /path/to/checkpoints:/app/models/checkpoints \
    -v /path/to/thresholds:/app/evaluation/results/threshold-finetuned \
    cxr-diagnosis

# CPU only
docker run -p 7860:7860 \
    -e CXR_DEVICE=cpu \
    -v /path/to/checkpoints:/app/models/checkpoints \
    cxr-diagnosis
```

---

## Dataset

**NIH CXR-14** (Clinical Center Chest X-Ray dataset):
- 67,369 images used (after filtering to available images)
- 12 labels: 11 diseases + No Finding
- Split: 40,418 train / 13,471 validation / 13,480 test (stratified 60/20/20)
- Labels extracted via NLP from radiology reports (~10-30% noise)

---

## Tech Stack

- **PyTorch** + **TorchXRayVision** — Model training and inference
- **scikit-learn** — Threshold calibration and evaluation metrics
- **Gradio** — Web interface
- **FastAPI** — REST API
- **transformers** + **bitsandbytes** — VLM backends (4-bit quantization)
- **Docker** — Containerized deployment

---

## License

This project is for educational and research purposes. The NIH CXR-14 dataset is publicly available from the [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC).
