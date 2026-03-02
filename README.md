# CXR Diagnosis — Chest X-Ray Vision-LLM Application

AI-powered chest X-ray diagnosis using Vision-Language Models (Qwen2-VL) with retrieval-augmented generation (RAG) via CLIP embeddings.

> **DISCLAIMER:** This application is for **educational and research purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## Architecture

```
Upload Image
     │
     ▼
┌──────────────┐     ┌──────────────┐     ┌─────────────────┐
│ CLIP Encoder │────▶│  Retriever   │────▶│  Prompt Builder  │
│  (embedding) │     │ (top-5 sim.) │     │  (few-shot ctx)  │
└──────────────┘     └──────────────┘     └────────┬────────┘
                                                    │
                                                    ▼
                                          ┌─────────────────┐
                                          │   Qwen2-VL      │
                                          │   (diagnosis)    │
                                          └────────┬────────┘
                                                    │
                                                    ▼
                                            JSON Response
```

## Repository Structure

```
cxr-diagnosis/
├── app/
│   ├── config.py          # Centralized configuration
│   ├── main.py            # FastAPI backend (POST /predict)
│   └── pipeline.py        # End-to-end RAG diagnosis pipeline
├── models/
│   ├── clip_encoder.py    # CLIP image embedding
│   └── vlm_inference.py   # Qwen2-VL inference
├── data/
│   ├── loader.py          # CSV / image loading (local + GCS)
│   └── retriever.py       # Cosine similarity retrieval
├── prompts/
│   └── prompt_builder.py  # Radiology prompt with few-shot examples
├── scripts/
│   ├── create_embeddings.py  # Generate dataset_embeddings.npy
│   └── sample_dataset.py     # Sample N images from full dataset
├── ui/
│   └── gradio_app.py      # Gradio web interface
├── utils/
│   └── logging_config.py  # Logging setup
├── run.py                 # Launch API + UI together
├── Dockerfile             # GPU-compatible container
├── requirements.txt
└── README.md
```

## Quick Start (Local)

### 1. Install dependencies

```bash
cd cxr-diagnosis
pip install -r requirements.txt
```

### 2. Prepare the dataset

Place your CSV and images in the `data/` directory:

```
data/
├── dataset.csv            # Label CSV (columns: id, disease labels, subject_id)
└── images/                # Chest X-ray PNGs
    ├── 00000583_003.png
    ├── ...
```

Or sample from the full NIH dataset:

```bash
python -m scripts.sample_dataset \
    --csv /path/to/full_labels.csv \
    --src-images /path/to/all_images/ \
    --n 1000
```

### 3. Generate CLIP embeddings

```bash
python -m scripts.create_embeddings
```

This produces `data/dataset_embeddings.npy`.

### 4. Run the application

```bash
# Both FastAPI (port 8000) and Gradio UI (port 7860)
python run.py

# Or run them separately:
python -m app.main      # FastAPI only
python -m ui.gradio_app # Gradio only
```

### 5. Test the API

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@test_xray.png"
```

---

## Configuration

All settings are driven by environment variables:

| Variable              | Default                          | Description                     |
|-----------------------|----------------------------------|---------------------------------|
| `CXR_GCS_BUCKET`     | *(empty)*                        | GCS bucket name                 |
| `CXR_GCS_IMAGE_PREFIX` | `images/`                      | GCS prefix for images           |
| `CXR_GCS_CSV_BLOB`   | `nih-cxr-lt_single-label_balanced-test.csv` | CSV blob name in GCS |
| `CXR_CLIP_MODEL`     | `openai/clip-vit-base-patch32`   | CLIP model HF ID                |
| `CXR_VLM_MODEL`      | `Qwen/Qwen2-VL-7B-Instruct`     | Qwen2-VL model HF ID           |
| `CXR_DEVICE`         | `cuda`                           | Torch device (`cuda` / `cpu`)   |
| `CXR_TOP_K`          | `5`                              | Number of similar cases to retrieve |
| `CXR_API_PORT`       | `8000`                           | FastAPI port                    |
| `CXR_UI_PORT`        | `7860`                           | Gradio port                     |

---

## Docker

### Build

```bash
docker build -t cxr-diagnosis .
```

### Run (GPU)

```bash
docker run --gpus all -p 8000:8000 -p 7860:7860 \
    -v /path/to/data:/app/data \
    cxr-diagnosis
```

---

## Google Cloud Platform Deployment

### 1. Create a GPU VM

```bash
gcloud compute instances create cxr-diagnosis-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=100GB \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release
```

**Recommended GPU options:**
- **Development/testing:** NVIDIA T4 (16 GB VRAM) — cost-effective
- **Production:** NVIDIA L4 (24 GB VRAM) — better for Qwen2-VL 7B
- **High performance:** NVIDIA A100 (40/80 GB VRAM)

### 2. Configure firewall rules

```bash
# Allow Gradio UI access
gcloud compute firewall-rules create allow-gradio \
    --allow=tcp:7860 \
    --target-tags=cxr-server \
    --description="Allow Gradio UI"

# Allow FastAPI access
gcloud compute firewall-rules create allow-fastapi \
    --allow=tcp:8000 \
    --target-tags=cxr-server \
    --description="Allow FastAPI"

# Tag your VM
gcloud compute instances add-tags cxr-diagnosis-vm \
    --tags=cxr-server \
    --zone=us-central1-a
```

### 3. SSH into the VM

```bash
gcloud compute ssh cxr-diagnosis-vm --zone=us-central1-a
```

### 4. Install dependencies on the VM

```bash
# Verify GPU
nvidia-smi

# Clone the repository
git clone https://github.com/<your-username>/cxr-diagnosis.git
cd cxr-diagnosis

# Install Python dependencies
pip install -r requirements.txt
```

### 5. Upload data to GCS and configure

```bash
# Create a bucket
gsutil mb -l us-central1 gs://your-cxr-bucket

# Upload dataset
gsutil -m cp data/dataset.csv gs://your-cxr-bucket/
gsutil -m cp -r data/images/ gs://your-cxr-bucket/images/

# Set environment variable
export CXR_GCS_BUCKET=your-cxr-bucket
```

### 6. Generate embeddings and launch

```bash
# Generate embeddings (run once)
python -m scripts.create_embeddings

# Start the application
python run.py
```

### 7. Access the application

Get the external IP:

```bash
gcloud compute instances describe cxr-diagnosis-vm \
    --zone=us-central1-a \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

Open in your browser:
- **Gradio UI:** `http://<EXTERNAL_IP>:7860`
- **FastAPI docs:** `http://<EXTERNAL_IP>:8000/docs`

---

## API Reference

### `POST /predict`

Upload a chest X-ray image for diagnosis.

**Request:** `multipart/form-data` with `file` field.

**Response:**

```json
{
  "report": "## Findings\n...",
  "similar_cases": [
    {
      "image_id": "00000583_003.png",
      "similarity": 0.9421,
      "findings": ["Atelectasis"]
    }
  ],
  "disclaimer": "DISCLAIMER: This analysis is for educational..."
}
```

### `GET /health`

Returns `{"status": "ok"}`.

---

## Disease Labels

The system auto-detects label columns from the CSV. The NIH CXR dataset includes:

Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax, Pneumoperitoneum, Pneumomediastinum, Subcutaneous Emphysema, Tortuous Aorta, Calcification of the Aorta, No Finding.
