"""CLI entry point: python -m rag_api."""

from __future__ import annotations

import logging
import sys

import uvicorn

from .config import Settings


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)


def main() -> None:
    settings = Settings()
    setup_logging(settings.verbose)

    uvicorn.run(
        "rag_api.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,  # MUST be 1 — multi-worker breaks shared state
        log_level="debug" if settings.verbose else "info",
    )


if __name__ == "__main__":
    main()
