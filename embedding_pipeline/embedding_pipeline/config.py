"""CLI argument parsing."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

# Load project-root .env so POSTGRES_* vars are available
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


def _get_default_dsn() -> str:
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
        prog="embedding_pipeline",
        description="Embed document chunks and upsert to Qdrant",
    )
    parser.add_argument(
        "--dsn",
        default=_get_default_dsn(),
        help="PostgreSQL DSN",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
        help="Qdrant server URL",
    )
    parser.add_argument(
        "--collection",
        default="boarddocs_chunks",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of chunks to fetch per batch from PostgreSQL (default: 256)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch 3 chunks, embed, print vectors and payloads, write nothing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N chunks total (0 = unlimited)",
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
