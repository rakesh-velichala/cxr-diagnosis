"""Gradio web interface for the CXR Diagnosis application."""

from __future__ import annotations

import json
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

pipeline: Optional[DiagnosisPipeline] = None


def get_pipeline() -> DiagnosisPipeline:
    """Lazy-load the diagnosis pipeline (loaded once on first call)."""
    global pipeline  # noqa: PLW0603
    if pipeline is None:
        logger.info("Loading pipeline for Gradio UI …")
        pipeline = DiagnosisPipeline()
    return pipeline


# ---------------------------------------------------------------------------
# Inference callback
# ---------------------------------------------------------------------------

def analyze_xray(image: Image.Image) -> tuple[str, str, str]:
    """Run the diagnosis pipeline and return results for the UI.

    Returns
    -------
    tuple[str, str, str]
        (diagnosis_report, similar_cases_json, disclaimer)
    """
    if image is None:
        return "Please upload a chest X-ray image.", "", ""

    try:
        pipe = get_pipeline()
        result = pipe.diagnose(image)

        cases_text = _format_similar_cases(result.similar_cases)
        return result.report, cases_text, result.disclaimer
    except Exception as exc:
        logger.exception("Gradio inference error")
        return f"Error during analysis: {exc}", "", MEDICAL_DISCLAIMER


def _format_similar_cases(cases: list[dict]) -> str:
    """Render similar cases as readable markdown."""
    if not cases:
        return "No similar cases found."

    lines = ["### Retrieved Similar Cases\n"]
    for i, case in enumerate(cases, start=1):
        findings = ", ".join(case["findings"]) if case["findings"] else "No Finding"
        lines.append(
            f"**Case {i}** — `{case['image_id']}` "
            f"(similarity: {case['similarity']:.4f})\n"
            f"  Findings: {findings}\n"
        )
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
            "powered by Vision-Language Models with retrieval-augmented generation.\n\n"
            f"> {MEDICAL_DISCLAIMER}"
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Chest X-Ray",
                    sources=["upload", "clipboard"],
                )
                analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

            with gr.Column(scale=2):
                report_output = gr.Markdown(label="Diagnosis Report")
                cases_output = gr.Markdown(label="Similar Cases")
                disclaimer_output = gr.Textbox(
                    label="Disclaimer",
                    interactive=False,
                    lines=3,
                )

        analyze_btn.click(
            fn=analyze_xray,
            inputs=[image_input],
            outputs=[report_output, cases_output, disclaimer_output],
        )

        # Also trigger on image upload.
        image_input.change(
            fn=analyze_xray,
            inputs=[image_input],
            outputs=[report_output, cases_output, disclaimer_output],
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
