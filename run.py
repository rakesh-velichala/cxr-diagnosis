#!/usr/bin/env python3
"""Launch the CXR Diagnosis application.

By default, starts only the Gradio UI (which loads the pipeline once).
Use --api to also start the FastAPI backend in a separate process.
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.logging_config import logger


def start_api() -> None:
    """Start the FastAPI server."""
    import uvicorn
    from app.config import settings

    logger.info("Starting FastAPI on %s:%d", settings.api_host, settings.api_port)
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


def start_ui() -> None:
    """Start the Gradio UI (loads the pipeline directly)."""
    from app.config import settings
    from ui.gradio_app import build_ui

    logger.info("Starting Gradio UI on %s:%d", settings.api_host, settings.ui_port)
    demo = build_ui()
    demo.launch(
        server_name=settings.api_host,
        server_port=settings.ui_port,
        share=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch CXR Diagnosis")
    parser.add_argument(
        "--api",
        action="store_true",
        help="Also start the FastAPI backend (uses extra GPU memory)",
    )
    args = parser.parse_args()

    if args.api:
        api_proc = multiprocessing.Process(target=start_api, daemon=True)
        api_proc.start()
        logger.info("FastAPI started in background")

    # Run Gradio in the main process.
    start_ui()


if __name__ == "__main__":
    main()
