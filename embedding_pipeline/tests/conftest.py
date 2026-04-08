"""Shared fixtures for embedding pipeline tests."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def embedder():
    """Load the embedding model once per test session."""
    from embedding_pipeline.embedder import Embedder

    e = Embedder()
    e.load()
    return e
