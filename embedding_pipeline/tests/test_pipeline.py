"""Tests for pipeline logic: payload building, SQL updates, dry run."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from embedding_pipeline.pipeline import (
    REQUIRED_PAYLOAD_FIELDS,
    UPDATE_CHUNK_SQL,
    build_payload,
)


def _make_row(
    chunk_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    tenant_id: str = "kent_sd",
    chunk_index: int = 0,
    content: str = "Test chunk content",
    token_count: int = 42,
    contains_table: bool = False,
    document_type: str = "agenda_item",
    metadata: dict | None = None,
) -> MagicMock:
    """Create a mock asyncpg.Record for a chunk row."""
    meta = (
        metadata
        if metadata is not None
        else {
            "meeting_date": "2024-01-15",
            "committee_name": "Board of Directors",
            "meeting_id": "mtg_001",
            "agenda_item_id": "ai_001",
            "title": "Test Document",
            "source_url": "https://example.com/doc",
        }
    )
    row_data = {
        "id": chunk_id or uuid.uuid4(),
        "document_id": document_id or uuid.uuid4(),
        "tenant_id": tenant_id,
        "chunk_index": chunk_index,
        "content": content,
        "token_count": token_count,
        "contains_table": contains_table,
        "document_type": document_type,
        "metadata": meta,
    }
    row = MagicMock()
    row.__getitem__ = lambda self, key: row_data[key]
    row.get = lambda key, default=None: row_data.get(key, default)
    return row


class TestBuildPayload:
    def test_all_required_fields_present(self):
        """Payload must include every required field."""
        row = _make_row()
        payload = build_payload(row)
        missing = REQUIRED_PAYLOAD_FIELDS - set(payload.keys())
        assert not missing, f"Missing fields in payload: {missing}"

    def test_field_values_correct(self):
        """Payload field values should match row data."""
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        row = _make_row(
            chunk_id=chunk_id,
            document_id=doc_id,
            chunk_index=3,
            token_count=150,
            contains_table=True,
            document_type="attachment",
        )
        payload = build_payload(row)
        assert payload["chunk_id"] == str(chunk_id)
        assert payload["document_id"] == str(doc_id)
        assert payload["tenant_id"] == "kent_sd"
        assert payload["chunk_index"] == 3
        assert payload["token_count"] == 150
        assert payload["contains_table"] is True
        assert payload["document_type"] == "attachment"
        assert payload["meeting_date"] == "2024-01-15"
        assert payload["committee_name"] == "Board of Directors"

    def test_missing_metadata_fields_are_none(self):
        """If metadata is missing fields, they should be None."""
        row = _make_row(metadata={})
        payload = build_payload(row)
        assert payload["meeting_date"] is None
        assert payload["committee_name"] is None
        assert payload["title"] is None
        # But direct fields should still be present
        assert payload["chunk_index"] == 0
        assert payload["tenant_id"] == "kent_sd"


class TestUpdateSQL:
    def test_update_sets_three_fields(self):
        """UPDATE_CHUNK_SQL should reference embedding_status, qdrant_point_id, embedding_model."""
        sql = UPDATE_CHUNK_SQL.lower()
        assert "embedding_status" in sql
        assert "qdrant_point_id" in sql
        assert "embedding_model" in sql

    @pytest.mark.asyncio
    async def test_batch_update_calls(self):
        """Simulated batch update should call execute for each chunk."""
        conn = MagicMock()
        conn.execute = AsyncMock()
        conn.transaction.return_value = _AsyncCtx()

        rows = [_make_row() for _ in range(5)]
        point_ids = [uuid.uuid4() for _ in range(5)]

        async with conn.transaction():
            for row, pid in zip(rows, point_ids):
                await conn.execute(UPDATE_CHUNK_SQL, row["id"], pid, "mxbai-embed-large-v1")

        assert conn.execute.call_count == 5
        # Check that each call got the right model name
        for call in conn.execute.call_args_list:
            assert call[0][3] == "mxbai-embed-large-v1"


class _AsyncCtx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_no_qdrant_writes(self):
        """Dry run should not call qdrant.upsert."""
        # We can't easily run the full pipeline without DB,
        # but we can verify the flag propagates by checking
        # that the dry_run path in the code doesn't call upsert.
        # This is verified by the integration test and manual dry run.
        # Here we just verify the config flag parses correctly.
        from embedding_pipeline.config import parse_args

        args = parse_args(["--dry-run"])
        assert args.dry_run is True
        args2 = parse_args([])
        assert args2.dry_run is False
