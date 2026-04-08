"""Configuration via CLI args and environment variables."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

# Load project-root .env so POSTGRES_* vars are available
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


def get_default_dsn() -> str:
    """Build PostgreSQL URL from env vars. No hardcoded credentials."""
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return database_url
    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "qorvault")
    user = os.environ.get("POSTGRES_USER", "qorvault")
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError(
            "POSTGRES_PASSWORD environment variable is not set. " "Copy .env.example to .env and fill in credentials."
        )
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="document_processor",
        description="Extract text and chunk documents for RAG pipeline",
    )
    parser.add_argument(
        "--dsn",
        default=get_default_dsn(),
        help="PostgreSQL DSN (default: $DATABASE_URL or localhost)",
    )
    parser.add_argument(
        "--ocr-url",
        default=os.environ.get("OCR_URL", "http://localhost:8001"),
        help="OCR service base URL (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent document processing limit (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process 5 docs, print results, no DB writes",
    )
    parser.add_argument(
        "--document-type",
        choices=["agenda_item", "agenda", "attachment"],
        default=None,
        help="Only process this document type",
    )
    parser.add_argument(
        "--tenant",
        default="kent_sd",
        help="Tenant ID (default: kent_sd)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)
