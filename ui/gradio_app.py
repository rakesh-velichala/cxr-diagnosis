"""Gradio web interface for the CXR Diagnosis application."""

from __future__ import annotations

from typing import Optional

import gradio as gr
from PIL import Image

from app.config import settings
from app.pipeline import DiagnosisPipeline
from models.base import Diagnosis, DISEASE_LABELS
from prompts.prompt_builder import MEDICAL_DISCLAIMER
from utils.logging_config import logger

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_pipelines: dict[str, DiagnosisPipeline] = {}

_SCORED_LABELS = [l for l in DISEASE_LABELS if l != "No Finding"]

# Display-friendly names for diseases.
_DISPLAY_NAMES = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Effusion": "Pleural Effusion",
    "Fibrosis": "Fibrosis",
    "Infiltration": "Infiltration",
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Pleural_Thickening": "Pleural Thickening",
    "Pneumothorax": "Pneumothorax",
}

# Custom CSS for a polished medical-app look.
_CUSTOM_CSS = """
.findings-detected {
    background: linear-gradient(135deg, #fff5f5 0%, #fff 100%);
    border-left: 4px solid #e53e3e;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 12px;
}
.no-findings {
    background: linear-gradient(135deg, #f0fff4 0%, #fff 100%);
    border-left: 4px solid #38a169;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 12px;
}
.screening-summary {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px;
    margin-top: 8px;
}
.app-header {
    text-align: center;
    padding: 8px 0 16px 0;
}
.app-header h1 {
    color: #2b6cb0;
    margin-bottom: 4px;
}
.confidence-high {
    color: #e53e3e;
    font-weight: 600;
}
.confidence-moderate {
    color: #dd6b20;
    font-weight: 600;
}
"""


def get_pipeline(backend: str) -> DiagnosisPipeline:
    """Lazy-load a pipeline for the given backend (cached per backend)."""
    if backend not in _pipelines:
        logger.info("Loading pipeline for backend: %s", backend)
        _pipelines[backend] = DiagnosisPipeline(backend=backend)
    return _pipelines[backend]


# ---------------------------------------------------------------------------
# Inference callback
# ---------------------------------------------------------------------------

def analyze_xray(image: Image.Image) -> str:
    """Run the diagnosis pipeline and return formatted results."""
    if image is None:
        return "**Please upload a chest X-ray image.**"

    try:
        pipe = get_pipeline("densenet")
        result = pipe.diagnose(image)
        return _format_report(result.diagnoses, result.model_name)
    except Exception as exc:
        logger.exception("Gradio inference error")
        return f"**Error during analysis:** {exc}"


def _format_report(diagnoses: list[Diagnosis], model_name: str) -> str:
    """Render multi-label results as a polished markdown report."""
    lines: list[str] = []

    if not diagnoses:
        lines.append("### No diagnoses could be determined.")
        return "\n".join(lines)

    # Check if it's a "No Finding" result.
    is_normal = len(diagnoses) == 1 and diagnoses[0].disease == "No Finding"

    if is_normal:
        lines.append("### No Significant Findings Detected")
        lines.append("")
        lines.append("All 11 conditions screened were below their detection thresholds.")
    else:
        n = len(diagnoses)
        lines.append(f"### {n} Possible Finding{'s' if n > 1 else ''} Detected")
        lines.append("")
        lines.append("| Disease | Probability | Confidence |")
        lines.append("|---------|-------------|------------|")
        for d in diagnoses:
            display = _DISPLAY_NAMES.get(d.disease, d.disease)
            pct = f"{d.probability * 100:.1f}%"
            lines.append(f"| **{display}** | {pct} | {d.confidence} |")

    lines.append("")
    lines.append(f"*Model: {model_name}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""
    with gr.Blocks(
        title="Chest X-Ray Diagnosis",
    ) as demo:

        # Header.
        gr.Markdown(
            "# Chest X-Ray Diagnosis Assistant\n\n"
            "AI-powered multi-label screening using a pretrained DenseNet-121 model "
            "with calibrated per-disease thresholds.\n\n"
            "**Screens for 11 conditions:** "
            "Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion, "
            "Fibrosis, Infiltration, Mass, Nodule, Pleural Thickening, Pneumothorax."
        )

        with gr.Row():
            # Left column: inputs.
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Chest X-Ray",
                    sources=["upload", "clipboard"],
                    height=380,
                )
                analyze_btn = gr.Button(
                    "Analyze X-Ray",
                    variant="primary",
                    size="lg",
                )

            # Right column: results.
            with gr.Column(scale=2):
                report_output = gr.Markdown(
                    value="Upload an image and click **Analyze X-Ray** to begin.",
                    label="Analysis Report",
                )

        # Disclaimer at the bottom.
        with gr.Accordion("Disclaimer", open=False):
            gr.Markdown(f"*{MEDICAL_DISCLAIMER}*")

        analyze_btn.click(
            fn=analyze_xray,
            inputs=[image_input],
            outputs=[report_output],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the Gradio interface."""
    demo = build_ui()
    demo.launch(
        server_name=settings.api_host,
        server_port=settings.ui_port,
        share=False,
        theme=gr.themes.Soft(),
        css=_CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
