"""Gradio web interface for the CXR Diagnosis application."""

from __future__ import annotations

from typing import Optional

import gradio as gr
from PIL import Image

from app.config import settings
from app.pipeline import DiagnosisPipeline
from prompts.prompt_builder import MEDICAL_DISCLAIMER
from utils.logging_config import logger

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_pipelines: dict[str, DiagnosisPipeline] = {}


def get_pipeline(backend: str) -> DiagnosisPipeline:
    """Lazy-load a pipeline for the given backend (cached per backend)."""
    if backend not in _pipelines:
        logger.info("Loading pipeline for backend: %s", backend)
        _pipelines[backend] = DiagnosisPipeline(backend=backend)
    return _pipelines[backend]


# ---------------------------------------------------------------------------
# Inference callback
# ---------------------------------------------------------------------------

def analyze_xray(image: Image.Image, model_choice: str) -> tuple[str, str]:
    """Run the diagnosis pipeline and return results for the UI.

    Returns
    -------
    tuple[str, str]
        (diagnosis_markdown, disclaimer)
    """
    if image is None:
        return "Please upload a chest X-ray image.", ""

    backend_map = {
        "Qwen2.5-VL-7B": "qwen",
        "CheXagent-8b": "chexagent",
        "GPT-4o": "gpt4o",
    }
    backend = backend_map.get(model_choice, "qwen")

    try:
        pipe = get_pipeline(backend)
        result = pipe.diagnose(image)
        diagnosis_text = _format_diagnoses(result.diagnoses, result.model_name)
        return diagnosis_text, result.disclaimer
    except Exception as exc:
        logger.exception("Gradio inference error")
        return f"Error during analysis: {exc}", MEDICAL_DISCLAIMER


def _format_diagnoses(diagnoses: list, model_name: str) -> str:
    """Render diagnoses as readable markdown."""
    lines = [f"### Diagnosis Results — *{model_name}*\n"]

    if not diagnoses:
        lines.append("No diagnoses could be determined.")
        return "\n".join(lines)

    lines.append("| Rank | Disease | Confidence |")
    lines.append("|------|---------|------------|")
    for d in diagnoses:
        lines.append(f"| {d.rank} | **{d.disease}** | {d.confidence} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""
    with gr.Blocks(
        title="Chest X-Ray Diagnosis",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Chest X-Ray Diagnosis Assistant\n"
            "Upload a chest X-ray image for AI-assisted diagnostic analysis "
            "powered by Vision-Language Models.\n\n"
            f"> {MEDICAL_DISCLAIMER}"
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Chest X-Ray",
                    sources=["upload", "clipboard"],
                )
                model_selector = gr.Dropdown(
                    choices=["Qwen2.5-VL-7B", "CheXagent-8b", "GPT-4o"],
                    value="Qwen2.5-VL-7B",
                    label="Model",
                )
                analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

            with gr.Column(scale=2):
                report_output = gr.Markdown(label="Diagnosis Report")
                disclaimer_output = gr.Textbox(
                    label="Disclaimer",
                    interactive=False,
                    lines=3,
                )

        analyze_btn.click(
            fn=analyze_xray,
            inputs=[image_input, model_selector],
            outputs=[report_output, disclaimer_output],
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
    )


if __name__ == "__main__":
    main()
