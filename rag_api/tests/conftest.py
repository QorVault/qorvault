"""Shared test fixtures for the RAG API tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from rag_api.retriever import RetrievedChunk


@pytest.fixture
def sample_chunks() -> list[RetrievedChunk]:
    """Three realistic chunks for testing."""
    return [
        RetrievedChunk(
            chunk_id="chunk-aaa",
            document_id="doc-111",
            content=(
                "The Board of Directors approved Policy 3210 regarding "
                "student nondiscrimination at the December 10, 2025 meeting."
            ),
            score=0.92,
            title="Second Reading and Approval of Policy 3210",
            meeting_date="2025-12-10",
            committee_name="Regular Meeting",
            document_type="agenda_item",
            source_url="https://go.boarddocs.com/wa/ksdwa/Board.nsf/goto?open&id=AAA",
            chunk_index=0,
            meeting_id="MTG-001",
            agenda_item_id="ITEM-001",
        ),
        RetrievedChunk(
            chunk_id="chunk-bbb",
            document_id="doc-222",
            content=(
                "Motion & Voting — A motion was made to approve the 2025-2026 "
                "budget as presented. The motion passed unanimously."
            ),
            score=0.85,
            title="2025-2026 Budget Approval",
            meeting_date="2025-06-15",
            committee_name="Regular Meeting",
            document_type="agenda_item",
            source_url="https://go.boarddocs.com/wa/ksdwa/Board.nsf/goto?open&id=BBB",
            chunk_index=0,
            meeting_id="MTG-002",
            agenda_item_id="ITEM-002",
        ),
        RetrievedChunk(
            chunk_id="chunk-ccc",
            document_id="doc-333",
            content=(
                "Superintendent presented enrollment figures for 2024-2025: "
                "total enrollment of 26,543 students across 42 schools."
            ),
            score=0.78,
            title="Superintendent Report",
            meeting_date="2024-09-11",
            committee_name="Regular Meeting",
            document_type="agenda_item",
            source_url="https://go.boarddocs.com/wa/ksdwa/Board.nsf/goto?open&id=CCC",
            chunk_index=1,
            meeting_id="MTG-003",
            agenda_item_id="ITEM-003",
        ),
    ]


@pytest.fixture
def mock_embedder():
    """Mock Embedder that returns a fixed 1024-dim vector."""
    emb = MagicMock()
    emb.ready = True
    emb.embed_query.return_value = [0.1] * 1024
    return emb


@pytest.fixture
def mock_retriever(sample_chunks):
    """Mock Retriever that returns sample_chunks."""
    ret = MagicMock()
    ret.search.return_value = sample_chunks
    ret.get_collection_info.return_value = {"points_count": 8422}
    return ret


@pytest.fixture
def mock_llm_client():
    """Mock LLMClient returning a canned answer."""
    from rag_api.llm import LLMResponse

    client = MagicMock()
    client.generate.return_value = LLMResponse(
        content=(
            "The Board approved Policy 3210 on student nondiscrimination "
            "[Source 1]. The 2025-2026 budget was also approved unanimously "
            "[Source 2]."
        ),
        model="claude-opus-4-6",
        input_tokens=1500,
        output_tokens=80,
        latency_seconds=3.5,
        stop_reason="end_turn",
    )
    return client


class _FakePoolAcquire:
    """Fake async context manager returned by pool.acquire()."""

    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        return False


@pytest.fixture
def mock_db_pool():
    """Mock asyncpg pool."""
    conn = AsyncMock()
    conn.fetchval.return_value = 1

    pool = MagicMock()
    pool.acquire.return_value = _FakePoolAcquire(conn)
    pool.close = AsyncMock()
    return pool
