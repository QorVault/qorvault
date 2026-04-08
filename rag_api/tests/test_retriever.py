"""Tests for the Qdrant retriever."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rag_api.retriever import Retriever


def _mock_search_response(results: list[dict]) -> MagicMock:
    """Build a mock httpx response for Qdrant search."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"result": results}
    return resp


def _make_result(pid: str, score: float, payload: dict) -> dict:
    return {"id": pid, "version": 0, "score": score, "payload": payload}


@patch("rag_api.retriever.httpx.Client")
def test_search_returns_retrieved_chunks(mock_http_cls):
    mock_http = MagicMock()
    mock_http_cls.return_value = mock_http

    mock_http.post.return_value = _mock_search_response(
        [
            _make_result(
                "p1",
                0.95,
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "content": "Test content",
                    "title": "Test Title",
                    "meeting_date": "2025-01-15",
                    "committee_name": "Regular Meeting",
                    "document_type": "agenda_item",
                    "source_url": "https://example.com",
                    "chunk_index": 0,
                    "meeting_id": "m1",
                    "agenda_item_id": "a1",
                },
            ),
        ]
    )

    ret = Retriever("http://127.0.0.1:6333", "boarddocs_chunks")
    results = ret.search([0.1] * 1024, top_k=5)

    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert results[0].score == 0.95
    assert results[0].content == "Test content"


@patch("rag_api.retriever.httpx.Client")
def test_search_builds_tenant_filter(mock_http_cls):
    mock_http = MagicMock()
    mock_http_cls.return_value = mock_http
    mock_http.post.return_value = _mock_search_response([])

    ret = Retriever("http://127.0.0.1:6333", "boarddocs_chunks")
    ret.search([0.1] * 1024, tenant_id="kent_sd")

    call_args = mock_http.post.call_args
    body = call_args[1]["json"]
    must = body["filter"]["must"]
    assert must[0]["key"] == "tenant_id"
    assert must[0]["match"]["value"] == "kent_sd"


@patch("rag_api.retriever.httpx.Client")
def test_search_with_document_type_filter(mock_http_cls):
    mock_http = MagicMock()
    mock_http_cls.return_value = mock_http
    mock_http.post.return_value = _mock_search_response([])

    ret = Retriever("http://127.0.0.1:6333", "boarddocs_chunks")
    ret.search([0.1] * 1024, document_type="agenda_item")

    call_args = mock_http.post.call_args
    body = call_args[1]["json"]
    must = body["filter"]["must"]
    # Should have 2 conditions: tenant_id + document_type
    assert len(must) == 2


@patch("rag_api.retriever.httpx.Client")
def test_search_with_date_range_filters_post_retrieval(mock_http_cls):
    """Date filtering is done post-retrieval (Qdrant Range is numeric-only)."""
    mock_http = MagicMock()
    mock_http_cls.return_value = mock_http

    mock_http.post.return_value = _mock_search_response(
        [
            _make_result(
                "p1",
                0.95,
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "content": "In range",
                    "document_type": "agenda_item",
                    "chunk_index": 0,
                    "meeting_date": "2024-06-15",
                },
            ),
            _make_result(
                "p2",
                0.90,
                {
                    "chunk_id": "c2",
                    "document_id": "d2",
                    "content": "Out of range",
                    "document_type": "agenda_item",
                    "chunk_index": 0,
                    "meeting_date": "2023-06-15",
                },
            ),
            _make_result(
                "p3",
                0.85,
                {
                    "chunk_id": "c3",
                    "document_id": "d3",
                    "content": "Also in range",
                    "document_type": "agenda_item",
                    "chunk_index": 0,
                    "meeting_date": "2024-12-01",
                },
            ),
        ]
    )

    ret = Retriever("http://127.0.0.1:6333", "boarddocs_chunks")
    results = ret.search(
        [0.1] * 1024,
        date_from="2024-01-01",
        date_to="2025-01-01",
    )

    # Only the two in-range chunks should be returned
    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[1].chunk_id == "c3"

    # Qdrant should have been asked for more results (top_k * 3)
    call_args = mock_http.post.call_args
    body = call_args[1]["json"]
    assert body["limit"] == 30  # 10 * 3


@patch("rag_api.retriever.httpx.Client")
def test_search_empty_results(mock_http_cls):
    mock_http = MagicMock()
    mock_http_cls.return_value = mock_http
    mock_http.post.return_value = _mock_search_response([])

    ret = Retriever("http://127.0.0.1:6333", "boarddocs_chunks")
    results = ret.search([0.1] * 1024)

    assert results == []
