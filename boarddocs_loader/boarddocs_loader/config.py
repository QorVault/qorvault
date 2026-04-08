"""Configuration via Pydantic Settings."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load project-root .env so POSTGRES_* vars are available
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


def _default_database_url() -> str:
    """Build PostgreSQL URL from POSTGRES_* env vars. No hardcoded credentials."""
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


class Settings(BaseSettings):
    data_dir: Path = Path("/home/qorvault/projects/ksd_forensic/boarddocs/data")
    pg_dsn: str = Field(default_factory=_default_database_url)
    tenant: str = "kent_sd"
    dry_run: bool = False
    limit: int = 0
    verbose: bool = False
    format_report: bool = False

    model_config = {"env_prefix": "BOARDDOCS_"}
