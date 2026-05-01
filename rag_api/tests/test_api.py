"""Tests for the FastAPI endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio
import rag_api.main as main_module


@asynccontextmanager
async def _null_lifespan(app):
    yield


@pytest.fixture(autouse=True)
def _inline_to_thread(monkeypatch):
    """Avoid local Python 3.14 threadpool deadlocks during API tests."""

    async def _run_inline(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(main_module.asyncio, "to_thread", _run_inline)


@pytest.fixture
def mock_hybrid_retriever(sample_chunks):
    """Mock async hybrid retriever used by query route tests."""
    hybrid = MagicMock()
    hybrid.search = AsyncMock(return_value=sample_chunks)
    return hybrid


@asynccontextmanager
async def _test_client():
    transport = httpx.ASGITransport(
        app=main_module.app,
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as async_client:
        yield async_client


@pytest_asyncio.fixture
async def client(mock_embedder, mock_retriever, mock_hybrid_retriever, mock_llm_client, mock_db_pool):
    """Create an AsyncClient with mocked dependencies via monkeypatching."""
    orig_embedder = main_module.embedder
    orig_retriever = main_module.retriever
    orig_hybrid = main_module.hybrid_retriever
    orig_llm = main_module.llm_client
    orig_pool = main_module.db_pool

    main_module.embedder = mock_embedder
    main_module.retriever = mock_retriever
    main_module.hybrid_retriever = mock_hybrid_retriever
    main_module.llm_client = mock_llm_client
    main_module.db_pool = mock_db_pool

    main_module.app.router.lifespan_context = _null_lifespan

    try:
        async with _test_client() as async_client:
            yield async_client
    finally:
        main_module.embedder = orig_embedder
        main_module.retriever = orig_retriever
        main_module.hybrid_retriever = orig_hybrid
        main_module.llm_client = orig_llm
        main_module.db_pool = orig_pool


# ---------------------------------------------------------------------------
# POST /api/v1/query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_returns_answer_with_citations(client):  # noqa: D103
    resp = await client.post("/api/v1/query", json={"query": "What policies were approved?"})

    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "[Source 1]" in data["answer"]
    assert len(data["citations"]) == 3
    assert data["citations"][0]["source_number"] == 1
    assert data["query"] == "What policies were approved?"
    assert data["chunks_retrieved"] == 3
    assert data["model"] == "claude-opus-4-6"


@pytest.mark.asyncio
async def test_query_hybrid_route_uses_retrieval_without_database_sql(
    client,
    monkeypatch,
    mock_hybrid_retriever,
    sample_chunks,
):  # noqa: D103
    """Hybrid route uses retrieval and does not call database SQL."""
    monkeypatch.setattr(
        main_module,
        "classify_query",
        MagicMock(
            return_value={
                "route": "hybrid",
                "confidence": 0.9,
                "reasoning": "test hybrid route",
                "extracted_filters": {"person_names": []},
            }
        ),
    )
    execute_database_query = AsyncMock()
    monkeypatch.setattr(main_module, "execute_database_query", execute_database_query)

    resp = await client.post(
        "/api/v1/query",
        json={"query": "Compare budget topics over time", "enable_routing": True},
    )

    assert resp.status_code == 200
    data = resp.json()
    execute_database_query.assert_not_called()
    mock_hybrid_retriever.search.assert_awaited_once()
    assert data["chunks_retrieved"] == len(sample_chunks)
    assert len(data["citations"]) == len(sample_chunks)
    assert data["citations"][0]["chunk_id"] == sample_chunks[0].chunk_id
    assert data["routing_decision"]["route"] == "hybrid"


@pytest.mark.asyncio
async def test_query_database_route_calls_database_sql_and_skips_retrieval(
    client,
    monkeypatch,
    mock_hybrid_retriever,
):  # noqa: D103
    """Database route calls SQL handling and skips hybrid retrieval."""
    monkeypatch.setattr(
        main_module,
        "classify_query",
        MagicMock(
            return_value={
                "route": "database",
                "confidence": 0.9,
                "reasoning": "test database route",
                "extracted_filters": {},
            }
        ),
    )
    execute_database_query = AsyncMock(
        return_value={
            "answer": "Database answer",
            "sql_used": "SELECT 1",
            "row_count": 1,
        }
    )
    monkeypatch.setattr(main_module, "execute_database_query", execute_database_query)

    resp = await client.post(
        "/api/v1/query",
        json={"query": "How many meetings were held?", "enable_routing": True},
    )

    assert resp.status_code == 200
    data = resp.json()
    execute_database_query.assert_awaited_once()
    mock_hybrid_retriever.search.assert_not_called()
    assert data["answer"] == "Database answer"
    assert data["citations"] == []
    assert data["chunks_retrieved"] == 0
    assert data["routing_decision"]["route"] == "database"


@pytest.mark.asyncio
async def test_query_empty_results_returns_no_info_message(  # noqa: D103
    mock_embedder,
    mock_db_pool,
):
    mock_ret = MagicMock()
    mock_hybrid = MagicMock()
    mock_hybrid.search = AsyncMock(return_value=[])
    mock_llm = MagicMock()

    orig_embedder = main_module.embedder
    orig_retriever = main_module.retriever
    orig_hybrid = main_module.hybrid_retriever
    orig_llm = main_module.llm_client
    orig_pool = main_module.db_pool

    main_module.embedder = mock_embedder
    main_module.retriever = mock_ret
    main_module.hybrid_retriever = mock_hybrid
    main_module.llm_client = mock_llm
    main_module.db_pool = mock_db_pool
    main_module.app.router.lifespan_context = _null_lifespan

    try:
        async with _test_client() as async_client:
            resp = await async_client.post("/api/v1/query", json={"query": "Something obscure?"})

        assert resp.status_code == 200
        data = resp.json()
        assert "don't have enough information" in data["answer"]
        assert data["citations"] == []
        assert data["chunks_retrieved"] == 0
        mock_llm.generate.assert_not_called()
    finally:
        main_module.embedder = orig_embedder
        main_module.retriever = orig_retriever
        main_module.hybrid_retriever = orig_hybrid
        main_module.llm_client = orig_llm
        main_module.db_pool = orig_pool


@pytest.mark.asyncio
async def test_query_validates_empty_query(client):  # noqa: D103
    resp = await client.post("/api/v1/query", json={"query": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_query_validates_top_k_bounds(client):  # noqa: D103
    resp = await client.post("/api/v1/query", json={"query": "test", "top_k": 100})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_query_latency_fields_present(client):  # noqa: D103
    resp = await client.post("/api/v1/query", json={"query": "test"})

    assert resp.status_code == 200
    data = resp.json()
    assert "latency_seconds" in data
    assert "embedding_latency_seconds" in data
    assert "retrieval_latency_seconds" in data
    assert "llm_latency_seconds" in data


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_all_healthy(client):  # noqa: D103
    resp = await client.get("/api/v1/health")

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


@pytest.mark.asyncio
async def test_health_degraded_when_db_down(  # noqa: D103
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
        async with _test_client() as async_client:
            resp = await async_client.get("/api/v1/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["database"] is False
    finally:
        main_module.embedder = orig_embedder
        main_module.retriever = orig_retriever
        main_module.llm_client = orig_llm
        main_module.db_pool = orig_pool
