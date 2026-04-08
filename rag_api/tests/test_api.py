"""Tests for the FastAPI endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import rag_api.main as main_module


@asynccontextmanager
async def _null_lifespan(app):
    yield


@pytest.fixture
def client(mock_embedder, mock_retriever, mock_llm_client, mock_db_pool):
    """Create a TestClient with mocked dependencies via monkeypatching."""
    # Save originals
    orig_embedder = main_module.embedder
    orig_retriever = main_module.retriever
    orig_llm = main_module.llm_client
    orig_pool = main_module.db_pool

    # Replace globals
    main_module.embedder = mock_embedder
    main_module.retriever = mock_retriever
    main_module.llm_client = mock_llm_client
    main_module.db_pool = mock_db_pool

    main_module.app.router.lifespan_context = _null_lifespan

    yield TestClient(main_module.app, raise_server_exceptions=False)

    # Restore
    main_module.embedder = orig_embedder
    main_module.retriever = orig_retriever
    main_module.llm_client = orig_llm
    main_module.db_pool = orig_pool


# ---------------------------------------------------------------------------
# POST /api/v1/query
# ---------------------------------------------------------------------------


def test_query_returns_answer_with_citations(client):
    resp = client.post("/api/v1/query", json={"query": "What policies were approved?"})

    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "[Source 1]" in data["answer"]
    assert len(data["citations"]) == 3
    assert data["citations"][0]["source_number"] == 1
    assert data["query"] == "What policies were approved?"
    assert data["chunks_retrieved"] == 3
    assert data["model"] == "claude-opus-4-6"


def test_query_empty_results_returns_no_info_message(
    mock_embedder,
    mock_db_pool,
):
    mock_ret = MagicMock()
    mock_ret.search.return_value = []
    mock_llm = MagicMock()

    orig_embedder = main_module.embedder
    orig_retriever = main_module.retriever
    orig_llm = main_module.llm_client
    orig_pool = main_module.db_pool

    main_module.embedder = mock_embedder
    main_module.retriever = mock_ret
    main_module.llm_client = mock_llm
    main_module.db_pool = mock_db_pool
    main_module.app.router.lifespan_context = _null_lifespan

    try:
        tc = TestClient(main_module.app, raise_server_exceptions=False)
        resp = tc.post("/api/v1/query", json={"query": "Something obscure?"})

        assert resp.status_code == 200
        data = resp.json()
        assert "don't have enough information" in data["answer"]
        assert data["citations"] == []
        assert data["chunks_retrieved"] == 0
        mock_llm.generate.assert_not_called()
    finally:
        main_module.embedder = orig_embedder
        main_module.retriever = orig_retriever
        main_module.llm_client = orig_llm
        main_module.db_pool = orig_pool


def test_query_validates_empty_query(client):
    resp = client.post("/api/v1/query", json={"query": ""})
    assert resp.status_code == 422


def test_query_validates_top_k_bounds(client):
    resp = client.post("/api/v1/query", json={"query": "test", "top_k": 100})
    assert resp.status_code == 422


def test_query_latency_fields_present(client):
    resp = client.post("/api/v1/query", json={"query": "test"})

    assert resp.status_code == 200
    data = resp.json()
    assert "latency_seconds" in data
    assert "embedding_latency_seconds" in data
    assert "retrieval_latency_seconds" in data
    assert "llm_latency_seconds" in data


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------


def test_health_all_healthy(client):
    resp = client.get("/api/v1/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["database"] is True
    assert data["qdrant"] is True
    assert data["embedder"] is True
    assert data["qdrant_collection_count"] == 8422


class _FailingPoolAcquire:
    async def __aenter__(self):
        raise ConnectionError("DB down")

    async def __aexit__(self, *args):
        return False


def test_health_degraded_when_db_down(
    mock_embedder,
    mock_retriever,
    mock_llm_client,
):
    # Create a pool that raises on acquire
    mock_pool = MagicMock()
    mock_pool.acquire.return_value = _FailingPoolAcquire()
    mock_pool.close = AsyncMock()

    orig_embedder = main_module.embedder
    orig_retriever = main_module.retriever
    orig_llm = main_module.llm_client
    orig_pool = main_module.db_pool

    main_module.embedder = mock_embedder
    main_module.retriever = mock_retriever
    main_module.llm_client = mock_llm_client
    main_module.db_pool = mock_pool
    main_module.app.router.lifespan_context = _null_lifespan

    try:
        tc = TestClient(main_module.app, raise_server_exceptions=False)
        resp = tc.get("/api/v1/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["database"] is False
    finally:
        main_module.embedder = orig_embedder
        main_module.retriever = orig_retriever
        main_module.llm_client = orig_llm
        main_module.db_pool = orig_pool
