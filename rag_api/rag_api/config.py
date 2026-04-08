"""Configuration via environment variables with Pydantic Settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load project-root .env so POSTGRES_* vars are available
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# Default: reuse the shared model cache from embedding_pipeline.
_DEFAULT_MODEL_CACHE = str(
    Path(__file__).resolve().parent.parent.parent / "embedding_pipeline" / "model_cache" / "mxbai-embed-large-v1-onnx"
)

# Cross-encoder reranker model cache.  bge-reranker-v2-m3 (278M params,
# MIT license) provides significant precision improvement at ~3.5s
# added latency on CPU.  See research/contextual_retrieval.md Phase 1.
_DEFAULT_RERANKER_CACHE = str(Path(__file__).resolve().parent.parent / "model_cache" / "bge-reranker-v2-m3-onnx")


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
    """Application settings loaded from RAG_* environment variables."""

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    # PostgreSQL (127.0.0.1, never localhost — Podman IPv4 only)
    database_url: str = Field(default_factory=_default_database_url)

    # Qdrant (127.0.0.1, never localhost)
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "boarddocs_chunks"

    # Embedding model
    model_cache_dir: str = _DEFAULT_MODEL_CACHE

    # Anthropic
    anthropic_api_key: str = ""

    # BoardDocs permalink base (override per-tenant via RAG_BOARDDOCS_BASE_URL)
    boarddocs_base_url: str = "https://go.boarddocs.com/wa/ksdwa/Board.nsf"

    # RAG parameters
    default_top_k: int = 10
    max_top_k: int = 25
    excluded_document_types: list[str] = ["research_analysis"]
    recency_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight of recency boost in hybrid retrieval (0.0 = disabled, 1.0 = strong)",
    )
    recency_half_life_days: int = Field(
        default=365,
        gt=0,
        description="Half-life in days for recency decay (365 = 1 year half-life)",
    )
    anthropic_model: str = "claude-opus-4-6"
    max_response_tokens: int = 4096
    tenant_id: str = "kent_sd"

    # Cross-encoder reranker (Phase 1 retrieval improvement)
    reranker_enabled: bool = Field(
        default=True,
        description="Enable cross-encoder reranking after RRF fusion",
    )
    reranker_cache_dir: str = _DEFAULT_RERANKER_CACHE
    reranker_model_filename: str = Field(
        default="model_quantized.onnx",
        description="ONNX model variant: model_quantized.onnx (INT8) or model.onnx (FP32)",
    )
    reranker_model_sha256: str = Field(
        default="912fc1215c2dbff6499700534bd8d31253af01573861abbfc43afd1fab6cce5d",
        description="Expected SHA-256 digest of the ONNX model file. Verified at load time to detect tampering.",
    )

    # Logging
    verbose: bool = False

    model_config = {"env_prefix": "RAG_"}
