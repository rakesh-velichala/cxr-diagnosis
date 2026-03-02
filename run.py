#!/usr/bin/env python3
"""Launch both the FastAPI backend and Gradio UI concurrently."""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

# Ensure project root is on sys.path.
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
    """Start the Gradio UI."""
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
    api_proc = multiprocessing.Process(target=start_api, daemon=True)
    ui_proc = multiprocessing.Process(target=start_ui, daemon=True)

    api_proc.start()
    ui_proc.start()

    logger.info("Both services started. Press Ctrl+C to stop.")

    try:
        api_proc.join()
        ui_proc.join()
    except KeyboardInterrupt:
        logger.info("Shutting down …")
        api_proc.terminate()
        ui_proc.terminate()


if __name__ == "__main__":
    main()
