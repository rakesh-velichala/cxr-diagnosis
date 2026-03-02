# ── GPU-compatible Dockerfile for CXR Diagnosis ──────────────────────────
# Base: NVIDIA CUDA runtime with Python 3.11
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git curl && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Pre-download models during build (optional — comment out for smaller image)
# RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
#     CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
#     CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

# Expose API and UI ports
EXPOSE 8000 7860

# Default: start both FastAPI and Gradio via a small launcher
CMD ["python", "run.py"]
